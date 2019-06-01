import numpy as np


from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from softlearning.utils.numpy import softmax
from softlearning.environments.utils import is_point_2d_env


class BaseTargetProposer(object):
    def __init__(self, env, pool):
        self._env = env
        self._pool = pool
        self._current_target = None

    def set_distance_fn(self, distance_fn):
        self.distance_fn = distance_fn

    def propose_target(self):
        raise NotImplementedError


class UnsupervisedTargetProposer(BaseTargetProposer):
    def __init__(self,
                 target_proposal_rule,
                 target_candidate_strategy='all_steps',
                 random_weighted_scale=1.0,
                 *args,
                 **kwargs):
        super(UnsupervisedTargetProposer, self).__init__(*args, **kwargs)
        self._first_observation = None
        self._first_state_observation = None
        self._target_proposal_rule = target_proposal_rule
        self._random_weighted_scale = random_weighted_scale
        self._target_candidate_strategy = target_candidate_strategy

    def propose_target(self, epoch):
        past_batch = self._pool.last_n_batch(1e5)
        past_observations = past_batch['observations']

        if self._first_observation is None:
            self._first_observation = type(past_observations)(
                (key, values[0])
                for key, values in past_observations.items()
            )

        if self._target_candidate_strategy == 'last_steps':
            episode_end_indices = np.flatnonzero(
                past_batch['episode_index_backwards'])
            new_observations = type(past_observations)(
                (key, values[episode_end_indices])
                for key, values in past_observations.items()
            )
        elif self._target_candidate_strategy == 'all_steps':
            new_observations = past_observations
        else:
            raise NotImplementedError(self._target_candidate_strategy)

        if (self._target_proposal_rule in
            ('farthest_estimate_from_first_observation',
             'random_weighted_estimate_from_first_observation')):
            first_observations = type(self._first_observation)(
                (key, np.tile(value[None, :],
                              (new_observations[key].shape[0], 1)))
                for key, value in self._first_observation.items()
            )
            new_distances = self.distance_fn(
                first_observations, new_observations)

            if (self._target_proposal_rule
                == 'farthest_estimate_from_first_observation'):
                best_observation_index = np.argmax(new_distances)

            elif (self._target_proposal_rule
                  == 'random_weighted_estimate_from_first_observation'):
                best_observation_index = np.random.choice(
                    new_distances.size,
                    p=softmax(
                        new_distances * self._random_weighted_scale
                    ).ravel()
                )

        elif self._target_proposal_rule == 'random':
            best_observation_index = np.random.randint(
                new_observations[next(iter(new_observations.keys()))].shape[0])
        else:
            raise NotImplementedError(self._target_proposal_rule)

        best_observation = type(new_observations)(
            (key, values[best_observation_index])
            for key, values in new_observations.items())

        self._current_target = best_observation

        return best_observation


class LogarithmicLabelScheduler(object):
    def __init__(self,
                 decay_steps,
                 end_labels,
                 start_labels=None,
                 start_labels_frac=None,
                 decay_rate=1e-2):
        assert (start_labels is None) or (start_labels_frac is None)
        self._start_labels = int(
            start_labels
            if start_labels is not None
            else start_labels_frac * end_labels)
        self._decay_steps = decay_steps
        self._end_labels = end_labels
        self._decay_rate = decay_rate

    @property
    def num_pretrain_labels(self):
        return self._start_labels

    def num_labels(self, time_step):
        """Return the number of labels desired at this point in time."""
        decayed_labels = (
            (self._start_labels - self._end_labels)
            * self._decay_rate ** (time_step / self._decay_steps)
            + self._end_labels)

        return int(decayed_labels)


class LinearLabelScheduler(object):
    def __init__(self,
                 decay_steps,
                 end_labels,
                 start_labels=None,
                 start_labels_frac=None,
                 power=1.0):
        assert (start_labels is None) or (start_labels_frac is None)
        self._start_labels = int(
            start_labels
            if start_labels is not None
            else start_labels_frac * end_labels)
        self._decay_steps = decay_steps
        self._end_labels = end_labels
        self._power = power

    @property
    def num_pretrain_labels(self):
        return self._start_labels

    def num_labels(self, time_step):
        time_step = min(time_step, self._decay_steps)
        decayed_labels = (
            (self._start_labels - self._end_labels)
            * (1 - time_step / self._decay_steps) ** (self._power)
            + self._end_labels)

        return int(decayed_labels)


PREFERENCE_SCHEDULERS = {
    'linear': LinearLabelScheduler,
    'logarithmic': LogarithmicLabelScheduler,
}


class SemiSupervisedTargetProposer(BaseTargetProposer):
    def __init__(self,
                 supervision_schedule_params,
                 epoch_length,
                 max_path_length,
                 *args,
                 **kwargs):
        super(SemiSupervisedTargetProposer, self).__init__(*args, **kwargs)
        assert supervision_schedule_params is not None
        self._supervision_schedule_params = supervision_schedule_params

        self._supervision_scheduler = PREFERENCE_SCHEDULERS[
            supervision_schedule_params['type']](
                **supervision_schedule_params['kwargs'])
        self._supervision_labels_used = 0
        self._last_supervision_epoch = -1
        self._best_observation_value = -float('inf')

        self._epoch_length = epoch_length
        self._max_path_length = max_path_length

    def propose_target(self, epoch):
        env = self._env.unwrapped

        expected_num_supervision_labels = (
            self._supervision_scheduler.num_labels(epoch))
        should_supervise = (
            expected_num_supervision_labels > self._supervision_labels_used)

        if self._current_target is None:
            self._current_target = (
                paths[0]
                .get('observations.observation', paths[0].get('observations'))
                [-1])
            self._current_state_target = (
                paths[0]
                .get('observations.state_observation', paths[0].get('observations'))
                [-1])

        num_epochs_since_last_supervision = (
            epoch - self._last_supervision_epoch)

        if (not should_supervise) or num_epochs_since_last_supervision < 1:
            return self._current_state_target

        self._last_supervision_epoch = epoch
        self._supervision_labels_used += 1

        use_last_n_paths = num_epochs_since_last_supervision * (
            self._epoch_length // self._max_path_length)

        paths_observations = [
            path.get(
                'observations.observation', path.get('observations')
            )[-1:]
            for path in paths[:use_last_n_paths]
        ]
        paths_state_observations = [
            path.get(
                'observations.state_observation', path.get('observations')
            )[-1:]
            for path in paths[:use_last_n_paths]
        ]

        new_observations = np.concatenate(paths_observations, axis=0)
        new_state_observations = np.concatenate(paths_state_observations, axis=0)

        if is_point_2d_env(env):
            ultimate_goal = self._env.unwrapped.ultimate_goal
            goals = np.tile(
                ultimate_goal, (new_state_observations.shape[0], 1))
            last_observations_distances = env.get_optimal_paths(
                new_state_observations, goals)

            best_observation_index = np.argmin(last_observations_distances)
            if (-last_observations_distances[best_observation_index]
                > self._best_observation_value):
                best_observation = new_observations[best_observation_index]
                best_state_observation = new_state_observations[
                    best_observation_index]
                self._best_observation_value = -last_observations_distances[
                    best_observation_index]
            else:
                best_observation = self._current_target
                best_state_observation = self._current_state_target

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

            last_observations_x_positions = new_state_observations[:, 0:1]
            last_observations_distances = last_observations_x_positions

            best_observation_index = np.argmax(last_observations_distances)
            if (last_observations_distances[best_observation_index]
                > self._best_observation_value):
                best_observation = new_observations[best_observation_index]
                best_state_observation = new_state_observations[
                    best_observation_index]
                self._best_observation_value = last_observations_distances[
                    best_observation_index]
            else:
                best_observation = self._current_target
                best_state_observation = self._current_state_target

        else:
            raise NotImplementedError

        self._current_target = best_observation
        self._current_state_target = best_state_observation

        return best_state_observation


class RandomTargetProposer(BaseTargetProposer):
    def __init__(self,
                 target_proposal_rule='uniform_from_environment',
                 *args,
                 **kwargs):
        super(RandomTargetProposer, self).__init__(*args, **kwargs)
        self._target_proposal_rule = target_proposal_rule

    def propose_target(self, epoch):
        if self._target_proposal_rule == 'uniform_from_environment':
            try:
                target = self._env._env.env.sample_metric_goal()
            except Exception as e:
                target = self._env.unwrapped.sample_metric_goal()
        elif self._target_proposal_rule == 'uniform_from_pool':
            new_observations = np.concatenate([
                path.get('observations.observation', path.get('observations'))
                for path in paths
            ], axis=0)

            target = new_observations[
                np.random.randint(new_observations.shape[0])]
        else:
            raise NotImplementedError

        self._current_target = target

        return target
