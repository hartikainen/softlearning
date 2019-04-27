import numpy as np


from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as GymHalfCheetahEnv
from gym.envs.mujoco.ant import AntEnv as GymAntEnv
from gym.envs.mujoco.humanoid import HumanoidEnv as GymHumanoidEnv
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

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
    def __init__(self, target_proposal_rule, last_n_batch=int(1e5),
                 random_weighted_scale=1.0, *args, **kwargs):
        super(UnsupervisedTargetProposer, self).__init__(*args, **kwargs)
        self._first_observation = None
        self._target_proposal_rule = target_proposal_rule
        self._last_n_batch = last_n_batch
        self._random_weighted_scale = random_weighted_scale

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
                min(self._pool.size, self._last_n_batch),
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
                best_observation = new_observations[np.random.choice(
                    new_distances.size,
                    p=softmax(
                        new_distances * self._random_weighted_scale
                    ).ravel()
                )]
        else:
            raise NotImplementedError(self._target_proposal_rule)

        return best_observation


class SemiSupervisedTargetProposer(BaseTargetProposer):
    def __init__(self, *args, proposal_scheduler=None, **kwargs):
        super(SemiSupervisedTargetProposer, self).__init__(*args, **kwargs)
        self._proposal_scheduler = proposal_scheduler

    def propose_target(self, paths):
        env = self._env.unwrapped

        new_observations = np.concatenate([
            path.get('observations.observation', path.get('observations'))
            for path in paths
        ], axis=0)
        path_last_observations = np.concatenate([
            path.get('observations.observation', path.get('observations'))[-1:]
            for path in paths
        ], axis=0)

        if isinstance(env, (Point2DEnv, Point2DWallEnv)):
            ultimate_goal = self._env.unwrapped.ultimate_goal
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
                (SwimmerEnv,
                 AntEnv,
                 HumanoidEnv,
                 HalfCheetahEnv,
                 HopperEnv,
                 Walker2dEnv)):
            if env._exclude_current_positions_from_observation:
                raise NotImplementedError

            position_slice = {
                SwimmerEnv: slice(0, 2),
                AntEnv: slice(0, 2),
                HumanoidEnv: slice(0, 2),
                HalfCheetahEnv: slice(0, 1),
                HopperEnv: slice(0, 1),
                Walker2dEnv: slice(0, 1),
            }[type(env)]

            last_observations_positions = path_last_observations[
                :, position_slice]
            last_observations_distances = np.linalg.norm(
                last_observations_positions, ord=2, axis=1)

            max_distance_idx = np.argmax(last_observations_distances)
            best_observation = path_last_observations[max_distance_idx]

        else:
            raise NotImplementedError

        return best_observation


class RandomTargetProposer(BaseTargetProposer):
    def __init__(self, target_proposal_rule='uniform_from_environment',
                 last_n_batch=int(1e5), *args, **kwargs):
        super(RandomTargetProposer, self).__init__(*args, **kwargs)
        self._target_proposal_rule = target_proposal_rule
        self._last_n_batch = last_n_batch
    
    def propose_target(self, paths):
        if self._target_proposal_rule == 'uniform_from_environment':
            try:
                target = self._env._env.env.sample_metric_goal()
            except Exception as e:
                target = self._env.unwrapped.sample_metric_goal()
        elif self._target_proposal_rule == 'uniform_from_pool':
            size = min(self._pool.size, self._last_n_batch)
            new_observations = self._pool.last_n_batch(
                size,
                field_name_filter='observations',
                observation_keys=getattr(self._env, 'observation_keys', None),
            )['observations']
            
            target = new_observations[np.random.randint(size)]
        else:
            raise NotImplementedError
            

        return target
