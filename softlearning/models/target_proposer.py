from collections import OrderedDict
import os
import json
import time
import pickle

import numpy as np
import skimage.io

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from softlearning.utils.numpy import softmax
from softlearning.environments.utils import is_point_2d_env


class BaseTargetProposer(object):
    def __init__(self,
                 env,
                 pool,
                 target_candidate_strategy='all_steps',
                 target_candidate_window=int(1e5)):
        self._env = env
        self._pool = pool
        self._target_candidate_strategy = target_candidate_strategy
        self._target_candidate_window = target_candidate_window
        self._current_target = None

    def set_distance_fn(self, distance_fn):
        self.distance_fn = distance_fn

    def propose_target(self):
        raise NotImplementedError

    def _get_new_observations(self):
        past_batch = self._pool.last_n_batch(self._target_candidate_window)
        past_observations = past_batch['observations']
        if self._target_candidate_strategy == 'last_steps':
            episode_end_indices = np.flatnonzero(
                past_batch['episode_index_backwards'] == 0)
            new_observations = type(past_observations)(
                (key, values[episode_end_indices])
                for key, values in past_observations.items()
            )
        elif self._target_candidate_strategy == 'all_steps':
            new_observations = past_observations
        else:
            raise NotImplementedError(self._target_candidate_strategy)

        return new_observations

    def get_diagnostics(self):
        return OrderedDict()


class UnsupervisedTargetProposer(BaseTargetProposer):
    def __init__(self,
                 target_proposal_rule,
                 random_weighted_scale=1.0,
                 *args,
                 **kwargs):
        super(UnsupervisedTargetProposer, self).__init__(*args, **kwargs)
        self._first_observation = None
        self._first_state_observation = None
        self._target_proposal_rule = target_proposal_rule
        self._random_weighted_scale = random_weighted_scale

    def propose_target(self, epoch):
        past_batch = self._pool.last_n_batch(1e5)
        past_observations = past_batch['observations']

        if self._first_observation is None:
            self._first_observation = type(past_observations)(
                (key, values[0])
                for key, values in past_observations.items()
            )

        new_observations = self._get_new_observations()

        if (self._target_proposal_rule in
            ('farthest_estimate_from_first_observation',
             'random_weighted_estimate_from_first_observation')):
            first_observations = type(self._first_observation)(
                (key, np.repeat(value[None, ...],
                                new_observations[key].shape[0],
                                axis=0))
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


def synthetic_goal_query(env,
                         new_observations,
                         current_target,
                         previous_best_observation_value):
    assert set(new_observations.keys()) == set(current_target.keys())

    new_observations = type(new_observations)((
        (name, np.concatenate((
            new_observations[name], current_target[name][None, ...],
        ), axis=0))
        for name in new_observations.keys()
    ))

    if is_point_2d_env(env):
        ultimate_goal = env.ultimate_goal
        new_positions = new_observations['state_observation']
        goals = np.tile(ultimate_goal, (new_positions.shape[0], 1))
        last_observations_distances = env.get_optimal_paths(
            new_positions, goals)

        best_observation_index = np.argmin(last_observations_distances)
        best_observation = type(new_observations)(
            (key, values[best_observation_index])
            for key, values in new_observations.items())
        best_observation_value = -last_observations_distances[
            best_observation_index]
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

        last_observations_x_positions = new_observations[
            'observations'][:, 0:1]
        new_observations_distances = last_observations_x_positions

        best_observation_index = np.argmax(new_observations_distances)
        best_observation = type(new_observations)(
            (key, values[best_observation_index])
            for key, values in new_observations.items())
        best_observation_value = new_observations_distances[
            best_observation_index]

    elif 'dclaw3' in type(env).__name__.lower():
        from sac_envs.utils.unit_circle_math import angle_distance_from_positions

        assert np.unique(env.target_initial_position_range).size == 1, (
            env.target_initial_position_range)

        object_positions = new_observations['object_position']
        desired_object_positions = np.array(
            env.target_initial_position_range[0]).reshape(1, -1)
        new_observations_distances = angle_distance_from_positions(
            [np.sin(object_positions), np.cos(object_positions)],
            [np.sin(desired_object_positions),
             np.cos(desired_object_positions)])

        best_observation_index = np.argmin(new_observations_distances)
        best_observation = type(new_observations)(
            (key, values[best_observation_index])
            for key, values in new_observations.items())
        best_observation_value = -new_observations_distances[
            best_observation_index]

    elif type(env).__name__ == 'DClawTurnFixed':
        from sac_envs.utils.unit_circle_math import angle_distance_from_positions
        new_observations_distances = new_observations[
            'object_to_target_angle_dist']

        if np.any(new_observations_distances < 0.1):
            candidate_indices = np.flatnonzero(new_observations_distances < 0.1)
            candidate_actions = new_observations['last_action'][candidate_indices]
            best_observation_index = candidate_indices[np.argmin(
                np.mean(candidate_actions, axis=-1))]
        else:
            best_observation_index = np.argmin(new_observations_distances)

        best_observation = type(new_observations)(
            (key, values[best_observation_index])
            for key, values in new_observations.items())
        best_observation_value = -new_observations_distances[
            best_observation_index]
    else:
        raise NotImplementedError

    return (best_observation.copy(),
            best_observation_value,
            best_observation_index)


def human_goal_query(env,
                     new_observations,
                     best_observation_value,
                     current_target):
    trial_directory = os.getcwd()
    preference_directory = os.path.join(trial_directory, 'preferences')

    metadata_path = os.path.join(preference_directory, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    query_id = len(metadata['queries'])

    assert all(query_id != query['query_id'] for query in metadata['queries'])

    (_,
     synthetic_best_observation_value,
     synthetic_best_observation_index) = synthetic_goal_query(
         env, new_observations, current_target, best_observation_value)

    query_directory = os.path.join(preference_directory, f'query-{query_id}')

    assert not os.path.exists(query_directory), query_directory
    os.makedirs(query_directory)

    print(query_directory)

    pickle_path = os.path.join(query_directory, 'observations.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(new_observations, f)

    metadata['queries'].append({
        'query_id': query_id,
        'query_time': time.time(),
        'response_time': None,
        'best_observation_index': 'PENDING',
        'best_observation_value': 'PENDING',
        'pickle_path': pickle_path,

        'num_observations': (
            1 + new_observations[next(iter(new_observations))].shape[0]
        ), # + 1 for current_target
        'syntetic_query': {
            'best_observation_index': int(synthetic_best_observation_index),
            'best_observation_value': float(synthetic_best_observation_value),
        }
    })

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    image_key = next(
        key for key, values in new_observations.items()
        if values.ndim == 4 and values.dtype == np.uint8)

    images = np.concatenate((
        new_observations[image_key],
        current_target[image_key][None, ...],
    ), axis=0)
    image_row = np.transpose(np.concatenate(np.transpose(images, axes=(0, 2, 1, 3))), axes=(1, 0, 2))

    query_images_path = os.path.join(query_directory, 'query_images.png')
    preferred_height = 320
    repeat_times = int(np.ceil(preferred_height / image_row.shape[0]))
    skimage.io.imsave(query_images_path, image_row.repeat(10, axis=0).repeat(10, axis=1))

    for i, image in enumerate(images):
        image_path = os.path.join(query_directory, f'query_image_{i}.png')
        skimage.io.imsave(image_path, image)


def evaluate_human_response(env,
                            current_target,
                            best_observation_value):
    trial_directory = os.getcwd()
    preference_directory = os.path.join(trial_directory, 'preferences')

    metadata_path = os.path.join(preference_directory, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    query_id = len(metadata['queries']) - 1

    query = metadata['queries'][-1]
    assert query_id == query['query_id'], (query_id, query['query_id'])

    if query['best_observation_index'] == 'PENDING':
        assert query['best_observation_value'] == 'PENDING', query
        return (current_target, 'PENDING', 'PENDING')

    best_observation_index = query['best_observation_index']
    best_observation_value = query['best_observation_value']

    query_directory = os.path.join(preference_directory, f'query-{query_id}')
    pickle_path = os.path.join(query_directory, 'observations.pkl')
    with open(pickle_path, 'rb') as f:
        new_observations = pickle.load(f)

    new_observations = type(new_observations)((
        (name, np.concatenate((
            new_observations[name], current_target[name][None, ...],
        ), axis=0))
        for name in new_observations.keys()
    ))

    best_observation = type(new_observations)((
        (key, value[best_observation_index].copy())
        for key, value in new_observations.items()
    ))

    return (best_observation,
            best_observation_value,
            best_observation_index)


class SemiSupervisedTargetProposer(BaseTargetProposer):
    def __init__(self,
                 supervision_schedule_params,
                 epoch_length,
                 max_path_length,
                 supervision_type='synthetic',
                 *args,
                 **kwargs):
        super(SemiSupervisedTargetProposer, self).__init__(*args, **kwargs)
        assert supervision_schedule_params is not None
        self._supervision_schedule_params = supervision_schedule_params

        self._supervision_scheduler = PREFERENCE_SCHEDULERS[
            supervision_schedule_params['type']](
                **supervision_schedule_params['kwargs'])
        self._supervision_type = supervision_type

        if self._supervision_type == 'human':
            trial_directory = os.getcwd()
            preference_directory = os.path.join(trial_directory, 'preferences')
            if not os.path.exists(preference_directory):
                os.makedirs(preference_directory)

            metadata_path = os.path.join(preference_directory, 'metadata.json')
            if not os.path.exists(metadata_path):
                with open(metadata_path, 'w') as f:
                    json.dump({'queries': []}, f)

        self._supervision_labels_used = 0
        self._observations_seen = 0
        self._last_supervision_epoch = -1
        self._best_observation_value = -float('inf')

        self._epoch_length = epoch_length
        self._max_path_length = max_path_length

    def evaluate_human_response(self):
        (current_target,
         best_observation_value,
         best_observation_index) = evaluate_human_response(
             self._env.unwrapped,
             self._current_target,
             self._best_observation_value)

        if (best_observation_index != 'PENDING'
            and best_observation_index is not None):
            (self._current_target, self._best_observation_value) = (
                 current_target, best_observation_value)
            return True
        return False

    def propose_target(self, epoch):
        env = self._env.unwrapped

        expected_num_supervision_labels = (
            self._supervision_scheduler.num_labels(epoch))
        should_supervise = (
            expected_num_supervision_labels > self._supervision_labels_used)

        if self._current_target is None:
            last_observation = self._pool.last_n_batch(1)['observations']
            self._current_target = type(last_observation)(
                (key, values[-1])
                for key, values in last_observation.items()
            )

        num_epochs_since_last_supervision = (
            epoch - self._last_supervision_epoch)

        if (not should_supervise) or num_epochs_since_last_supervision < 1:
            if self._supervision_type == 'human':
                assert self.evaluate_human_response()

            return self._current_target.copy()


        self._last_supervision_epoch = epoch
        self._supervision_labels_used += 1

        new_observations = self._get_new_observations()

        self._observations_seen += new_observations[
            next(iter(new_observations.keys()))].shape[0]

        if self._supervision_type == 'synthetic':
            (self._current_target,
             self._best_observation_value,
             _) = synthetic_goal_query(
                env,
                new_observations,
                self._current_target,
                self._best_observation_value)

            # if best_observation_index is not None:
            #     (self._current_target,
            #      self._best_observation_value,
            #      best_observation_index) = (
            #          current_target,
            #          best_observation_value,
            #          best_observation_index)

        elif self._supervision_type == 'human':
            human_goal_query(
                env,
                new_observations,
                self._best_observation_value,
                self._current_target)
            while not self.evaluate_human_response():
                sleep(10) # Sleep for a moment to wait for the query
        else:
            raise NotImplementedError(self._supervision_type)

        return self._current_target

    def get_diagnostics(self):
        return OrderedDict((
            ('observations_seen', self._observations_seen),
            ('supervision_labels_used', self._supervision_labels_used),
        ))


class RandomTargetProposer(BaseTargetProposer):
    def __init__(self,
                 target_proposal_rule='uniform_from_environment',
                 *args,
                 **kwargs):
        super(RandomTargetProposer, self).__init__(*args, **kwargs)
        self._target_proposal_rule = target_proposal_rule

    def propose_target(self, epoch):
        new_observations = self._get_new_observations()

        if self._target_proposal_rule == 'uniform_from_environment':
            try:
                best_observation = self._env._env.env.sample_metric_goal()
            except AttributeError:
                best_observation = self._env.unwrapped.sample_metric_goal()

        elif self._target_proposal_rule == 'uniform_from_pool':
            best_observation_index = np.random.randint(
                new_observations[next(iter(new_observations.keys()))].shape[0])
            best_observation = type(new_observations)(
                (key, values[best_observation_index])
                for key, values in new_observations.items())

        else:
            raise NotImplementedError(self._target_proposal_rule)

        self._current_target = best_observation

        return best_observation
