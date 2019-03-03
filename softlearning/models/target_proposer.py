import numpy as np


from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as GymHalfCheetahEnv
from gym.envs.mujoco.ant import AntEnv as GymAntEnv
from gym.envs.mujoco.humanoid import HumanoidEnv as GymHumanoidEnv
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv

from softlearning.environments.gym.mujoco.swimmer import (
    SwimmerEnv as CustomSwimmerEnv)
from softlearning.environments.gym.mujoco.ant import (
    AntEnv as CustomAntEnv)
from softlearning.environments.gym.mujoco.humanoid import (
    HumanoidEnv as CustomHumanoidEnv)
from softlearning.environments.gym.mujoco.half_cheetah import (
    HalfCheetahEnv as CustomHalfCheetahEnv)
from softlearning.environments.gym.mujoco.hopper import (
    HopperEnv as CustomHopperEnv)
from softlearning.environments.gym.mujoco.walker2d import (
    Walker2dEnv as CustomWalker2dEnv)

from softlearning.utils.numpy import softmax


class BaseTargetProposer(object):
    def __init__(self, env, pool):
        self._env = env
        self._pool = pool

    def set_distance_fn(self, distance_fn):
        self.distance_fn = distance_fn

    def propose_target(self, paths):
        raise NotImplementedError


class UnsupervisedTargetProposer(BaseTargetProposer):
    def __init__(self, target_proposal_rule, *args, **kwargs):
        super(UnsupervisedTargetProposer, self).__init__(*args, **kwargs)
        self._first_observation = None
        self._target_proposal_rule = target_proposal_rule

    def propose_target(self, paths):
        if self._first_observation is None:
            self._first_observation = paths[0].get(
                'observations.observation', paths[0].get('observations'))[0]

        if self._target_proposal_rule == 'closest_l2_from_goal':
            ultimate_goal = getattr(self._env.unwrapped, 'ultimate_goal', None)
            new_observations = np.concatenate([
                path.get('observations.observation', path.get('observations'))
                for path in paths
            ], axis=0)
            new_distances = np.linalg.norm(
                new_observations - ultimate_goal, axis=1)

            min_distance_idx = np.argmin(new_distances)
            best_observation = new_observations[min_distance_idx]

        elif (self._target_proposal_rule ==
              'farthest_l2_from_first_observation'):
            new_observations = np.concatenate([
                path.get('observations.observation', path.get('observations'))
                for path in paths
            ], axis=0)
            new_distances = np.linalg.norm(
                new_observations - self._first_observation, axis=1)

            max_distance_idx = np.argmax(new_distances)
            best_observation = new_observations[max_distance_idx]

        elif (self._target_proposal_rule in
              ('farthest_estimate_from_first_observation',
               'random_weighted_estimate_from_first_observation')):
            new_observations = self._pool.last_n_batch(
                min(self._pool.size, int(1e5)),
                field_name_filter='observations',
                observation_keys=getattr(self._env, 'observation_keys', None),
            )['observations']
            new_distances = self.distance_fn(
                np.tile(self._first_observation[None, :],
                        (new_observations.shape[0], 1)),
                new_observations)

            if (self._target_proposal_rule
                == 'farthest_estimate_from_first_observation'):
                max_distance_idx = np.argmax(new_distances)
                best_observation = new_observations[max_distance_idx]
            elif (self._target_proposal_rule
                  == 'random_weighted_estimate_from_first_observation'):
                raise NotImplementedError("TODO: check this")
                best_observation = new_observations[np.random.choice(
                    new_distances.size, p=softmax(new_distances))]
        else:
            raise NotImplementedError(self._target_proposal_rule)

        return best_observation


class SemiSupervisedTargetProposer(BaseTargetProposer):
    def __init__(self, *args, proposal_scheduler=None, **kwargs):
        super(SemiSupervisedTargetProposer, self).__init__(*args, **kwargs)
        self._proposal_scheduler = proposal_scheduler

    def propose_target(self, paths):
        env = self._env.unwrapped

        ultimate_goal = getattr(self._env.unwrapped, 'ultimate_goal', None)

        new_observations = np.concatenate([
            path.get('observations.observation', path.get('observations'))
            for path in paths
        ], axis=0)
        path_last_observations = np.concatenate([
            path.get('observations.observation', path.get('observations'))[-1:]
            for path in paths
        ], axis=0)

        if isinstance(env, (Point2DEnv, Point2DWallEnv)):
            goals = np.tile(
                ultimate_goal, (path_last_observations.shape[0], 1))
            last_observations_distances = (
                env.get_optimal_paths(
                    path_last_observations, goals))

            min_distance_idx = np.argmin(last_observations_distances)
            best_observation = path_last_observations[min_distance_idx]

        elif isinstance(env, (GymAntEnv, GymHalfCheetahEnv, GymHumanoidEnv)):
            velocity_indices = {
                GymAntEnv:
                slice(env.sim.data.qpos.size - 2, env.sim.data.qpos.size),
                GymHalfCheetahEnv:
                slice(env.sim.data.qpos.size - 1, env.sim.data.qpos.size),
                GymHumanoidEnv:
                slice(env.sim.data.qpos.size - 2, env.sim.data.qpos.size),
            }[type(env)]
            new_velocities = new_observations[:, velocity_indices]
            new_velocities = np.linalg.norm(new_velocities, ord=2, axis=1)

            max_velocity_idx = np.argmax(new_velocities)
            best_observation = new_observations[max_velocity_idx]

        elif isinstance(
                env,
                (CustomSwimmerEnv,
                 CustomAntEnv,
                 CustomHumanoidEnv,
                 CustomHalfCheetahEnv,
                 CustomHopperEnv,
                 CustomWalker2dEnv)):
            if env._exclude_current_positions_from_observation:
                raise NotImplementedError
            position_idx = slice(0, 2)
            last_observations_positions = path_last_observations[
                :, position_idx]
            last_observations_distances = np.linalg.norm(
                last_observations_positions, ord=2, axis=1)

            max_distance_idx = np.argmax(last_observations_distances)
            best_observation = path_last_observations[max_distance_idx]

        else:
            raise NotImplementedError

        return best_observation


class RandomTargetProposer(BaseTargetProposer):
    def propose_target(self, paths):
        target = self._env.unwrapped.sample_metric_goal()
        return target
