from collections import deque
import math
from itertools import islice

import numpy as np

from softlearning.utils.numpy import softmax
from .simple_replay_pool import SimpleReplayPool


def random_int_with_variable_range(mins, maxs):
    result = np.floor(np.random.uniform(mins, maxs)).astype(int)
    return result


class DistancePool(SimpleReplayPool):
    def __init__(self,
                 path_length,
                 *args,
                 on_policy_window=None,
                 max_pair_distance=float('inf'),
                 max_size=None,
                 her_strategy=None,
                 fixed_path_length=False,
                 use_distances=True,
                 **kwargs):
        self._on_policy_window = on_policy_window or max_size
        self._max_pair_distance = max_pair_distance or float('inf')
        self._path_length = path_length
        self._fixed_path_length = fixed_path_length

        self.paths = deque(maxlen=10 * math.ceil(max_size//path_length))
        self.path_lengths = deque(maxlen=10 * math.ceil(max_size//path_length))

        self._her_strategy = her_strategy
        self._use_distances = use_distances

        super(DistancePool, self).__init__(*args, max_size=max_size, **kwargs)

    def add_path(self, path):
        self.paths.append(path)
        path_length = path[next(iter(path.keys()))].shape[0]
        self.path_lengths.append(path_length)
        return super(DistancePool, self).add_path(path)

    def distance_batch(self, batch_size, *args, **kwargs):
        cumulative_samples = 0
        for path_index in range(len(self.paths)-1, -1, -1):
            path = self.paths[path_index]
            path_length = path[list(path.keys())[0]].shape[0]
            cumulative_samples += path_length

            if cumulative_samples >= self._on_policy_window:
                break

        possible_path_lengths = np.array(tuple(islice(
            self.path_lengths, path_index, len(self.path_lengths))))
        path_weights = possible_path_lengths / np.sum(possible_path_lengths)
        path_probabilities = softmax(path_weights)

        random_rollout_indices = np.random.choice(
            np.arange(path_index, len(self.path_lengths)),
            size=batch_size,
            replace=True,
            p=path_probabilities)

        pairs_observations = []
        pairs_goals = []
        pairs_actions = []
        pairs_distances = []

        for random_rollout_index in random_rollout_indices:
            path = self.paths[random_rollout_index]
            path_observations = path.get(
                'observations', path.get('observations.observation'))
            path_length = path_observations.shape[0]
            if path_length < 2:
                continue
            random_start_index = np.random.randint(0, path_length-1)
            random_offset = np.random.randint(
                1, 1 + np.minimum(
                    path_length - 1 - random_start_index,
                    self._max_pair_distance))
            random_end_index = random_start_index + random_offset

            assert np.all(random_start_index <= random_end_index)
            assert np.all(random_end_index - random_start_index
                          <= self._max_pair_distance)

            pair_index = [random_start_index, random_end_index]

            assert np.all(path['observations.desired_goal'] == path['observations.desired_goal'][0])

            pairs_observations.append(path_observations[pair_index])
            pairs_goals.append(path['observations.desired_goal'][random_start_index])
            pairs_actions.append(path['actions'][pair_index])
            pairs_distances.append(random_end_index - random_start_index)
            assert pairs_distances[-1] >= 0

        pairs_observations = np.array(pairs_observations)
        pairs_goals = np.array(pairs_goals)
        pairs_actions = np.array(pairs_actions)
        pairs_distances = np.array(pairs_distances)

        triples_idx = np.random.randint(0, self._size, (batch_size, 3))
        assert np.all(triples_idx < self._size)

        triples_batch = self.batch_by_indices(
            triples_idx,
            *args,
            field_name_filter=('observations', 'actions'),
            **kwargs)

        triples_observations = triples_batch['observations']
        triples_actions = triples_batch['actions']

        objectives_idx = np.random.randint(0, self._size, (batch_size, 2))
        assert np.all(objectives_idx < self._size)

        objectives_batch = self.batch_by_indices(
            objectives_idx,
            *args,
            field_name_filter=('observations', 'actions'),
            **kwargs)

        objectives_observations = objectives_batch['observations']
        objectives_actions = objectives_batch['actions']

        return {
            'distance_pairs_observations': pairs_observations,
            'distance_pairs_goals': pairs_goals,
            'distance_pairs_actions': pairs_actions,
            'distance_pairs_distances': pairs_distances,

            'distance_triples_observations': triples_observations,
            'distance_triples_actions': triples_actions,

            'distance_objectives_observations': objectives_observations,
            'distance_objectives_actions': objectives_actions,
        }

    def variable_length_random_batch(self, batch_size, *args, **kwargs):
        path_lengths = np.array(self.path_lengths)
        path_weights = path_lengths / np.sum(path_lengths)
        path_probabilities = softmax(path_weights)

        random_rollout_indices = np.random.choice(
            np.arange(len(self.paths)),
            size=batch_size,
            replace=True,
            p=path_probabilities)

        batch_observations = []
        batch_next_observations = []
        batch_actions = []
        batch_terminals = []
        batch_rewards = []
        batch_goals = []

        for random_rollout_index in random_rollout_indices:
            path = self.paths[random_rollout_index]

            path_observations = path.get(
                'observations', path.get('observations.observation'))
            path_next_observations = path.get(
                'next_observations', path.get('next_observations.observation'))

            path_length = path_observations.shape[0]
            random_start_index = np.random.randint(0, path_length)

            assert np.all(
                path['observations.desired_goal']
                == path['observations.desired_goal'][0])

            if self._her_strategy:
                her_strategy_type = self._her_strategy['type']
                goal_resampling_probability = self._her_strategy[
                    'resampling_probability']

                if np.random.rand() < 1.0 - goal_resampling_probability:
                    goal = path['observations.desired_goal'][0]
                elif her_strategy_type == 'random':
                    sample = super(DistancePool, self).random_batch(1)
                    goal = sample.get(
                        'observations.observation', sample.get('observations')
                    )[0]
                elif her_strategy_type == 'goal':
                    goal = path['observations.desired_goal'][0]
                else:
                    if her_strategy_type == 'final':
                        end_index = -1
                    elif her_strategy_type == 'episode':
                        end_index = np.random.randint(0, path_length)
                    elif her_strategy_type == 'future':
                        max_offset = min(
                            path_length - random_start_index,
                            self._max_pair_distance)

                        offset = np.random.randint(0, max_offset)
                        end_index = random_start_index + offset

                    assert end_index < path_length, end_index
                    goal = path_observations[end_index]
            else:
                goal = path['observations.desired_goal'][0]

            batch_observations.append(path_observations[random_start_index])
            batch_next_observations.append(
                path_next_observations[random_start_index])
            batch_actions.append(path['actions'][random_start_index])
            terminal = np.linalg.norm(
                path_next_observations[random_start_index] - goal,
                ord=2,
                keepdims=True,
            ) < 0.1
            # terminal = path['terminals'][random_start_index]
            batch_terminals.append(terminal)
            batch_rewards.append(path['rewards'][random_start_index])
            batch_goals.append(goal)

        distance_batch = (
            self.distance_batch(batch_size, *args, **kwargs)
            if self._use_distances
            else {}
        )

        batch = {
            **distance_batch,
            'observations': np.array(batch_observations),
            'next_observations': np.array(batch_next_observations),
            'actions': np.array(batch_actions),
            'terminals': np.array(batch_terminals),
            'rewards': np.array(batch_rewards),
            'goals': np.array(batch_goals),
        }

        return batch

    def random_batch(self, batch_size, *args, **kwargs):
        if True or not self._fixed_path_length:
            return self.variable_length_random_batch(
                batch_size, *args, **kwargs)

        raise NotImplementedError(
            "TODO(hartikainen): This does not have HER-strategy implemented."
            " Change the `metric_learning.variants.fixed_path_length` so that it"
            " returns `False` or manually set the above condition to always"
            " evaluate to `True`.")

        assert self._path_length is not None
        num_full_rollouts = self.size // self._path_length
        last_full_rollout_index = self._pointer // self._path_length

        num_possible_rollouts = min(
            self._on_policy_window // self._path_length,
            num_full_rollouts)

        random_rollout_idx = (np.random.randint(
            num_full_rollouts - num_possible_rollouts,
            num_full_rollouts,
            batch_size
        ) + last_full_rollout_index) % num_full_rollouts

        if self._max_pair_distance is None:
            random_steps_idx = np.random.randint(
                0, self._path_length, (batch_size, 2))

            sorted_random_steps_idx = np.sort(random_steps_idx)

            assert np.all(sorted_random_steps_idx < self._path_length)

        else:
            random_start_idx = np.random.randint(
                0, self._path_length-1, (batch_size, 1))
            random_offset = np.floor(np.random.uniform(
                np.ones_like(random_start_idx),
                np.minimum(self._path_length - random_start_idx,
                           self._max_pair_distance)
            )).astype(int)
            random_end_idx = random_start_idx + random_offset
            assert np.all(random_end_idx < self._path_length)
            assert np.all(random_start_idx < random_end_idx)
            assert np.all(random_end_idx - random_start_idx
                          <= self._max_pair_distance)
            assert np.all(random_end_idx - random_start_idx > 0)

            sorted_random_steps_idx = np.concatenate(
                [random_start_idx, random_end_idx], axis=-1)

        rollout_offsets = random_rollout_idx * self._path_length
        pairs_idx = sorted_random_steps_idx + rollout_offsets[:, None]
        assert np.all(pairs_idx < self.size)

        pairs_batch = self.batch_by_indices(
            pairs_idx,
            *args,
            **kwargs)

        pairs_observations = pairs_batch['observations']
        pairs_actions = pairs_batch['actions']
        pairs_goals = pairs_batch['observations.desired_goal'][:, 0, :]
        pairs_distances = pairs_idx[:, 1] - pairs_idx[:, 0]

        distances_sanity_check = (
            sorted_random_steps_idx[:, 1] - sorted_random_steps_idx[:, 0])
        assert np.all(pairs_distances >= 0)
        assert np.all(pairs_distances == distances_sanity_check)

        triples_idx = np.random.randint(0, self._size, (batch_size, 3))
        assert np.all(triples_idx < self._size)

        triples_batch = self.batch_by_indices(
            triples_idx,
            *args,
            field_name_filter=('observations', 'actions'),
            **kwargs)

        triples_observations = triples_batch['observations']
        triples_actions = triples_batch['actions']

        objectives_idx = np.random.randint(0, self._size, (batch_size, 2))
        assert np.all(objectives_idx < self._size)

        objectives_batch = self.batch_by_indices(
            objectives_idx,
            *args,
            field_name_filter=('observations', 'actions'),
            **kwargs)

        objectives_observations = objectives_batch['observations']
        objectives_actions = objectives_batch['actions']

        batch_indices = self.random_indices(batch_size)
        batch = self.batch_by_indices(
            batch_indices, *args, **kwargs)

        goals = batch.pop('observations.desired_goal')

        if self._her_strategy:
            her_strategy_type = self._her_strategy['type']
            goal_resampling_probability = self._her_strategy[
                'resampling_probability']

            resample_index = (
                np.random.rand(batch_size) < goal_resampling_probability)
            where_resampled = np.where(resample_index)

            num_resamples = np.sum(resample_index)

            if her_strategy_type == 'random':
                samples = super(DistancePool, self).random_batch(num_resamples)
                goals = samples.get(
                    'observations.observation', samples.get('observations'))
                goals[where_resampled] = goals
            elif her_strategy_type == 'goal':
                pass
            else:
                if her_strategy_type == 'final':
                    goal_indices = (
                        (batch_indices[where_resampled] // self._path_length + 1)
                        * self._path_length)
                elif her_strategy_type == 'episode':
                    goal_indices = (
                        ((batch_indices[where_resampled] // self._path_length) * self._path_length)
                        + np.random.randint(0, self._path_length, num_resamples))
                elif her_strategy_type == 'future':
                    goal_indices = (
                        ((batch_indices[where_resampled] // self._path_length) * self._path_length)
                        + random_int_with_variable_range(
                            (batch_indices[where_resampled] % self._path_length), self._path_length))

                assert np.all(
                    (goal_indices // self._path_length)
                    == (batch_indices[where_resampled] // self._path_length))
                assert np.all((goal_indices - batch_indices[where_resampled]) < self._path_length)
                assert np.all(goal_indices >= 0)

                goals_batch = super(DistancePool, self).batch_by_indices(
                    *args,
                    indices=goal_indices,
                    **kwargs,
                    field_name_filter=lambda x: 'observations' in x)

                goals[where_resampled] = goals_batch.get(
                    'observations.observation', goals_batch.get('observations'))

        batch.update({
            'distance_pairs_observations': pairs_observations,
            'distance_pairs_goals': pairs_goals,
            'distance_pairs_actions': pairs_actions,
            'distance_pairs_distances': pairs_distances,

            'distance_triples_observations': triples_observations,
            'distance_triples_actions': triples_actions,

            'distance_objectives_observations': objectives_observations,
            'distance_objectives_actions': objectives_actions,

            'goals': goals,
        })

        return batch

    def get_new_observations(self, n, observation_keys=None):
        if self._fixed_path_length:
            new_observations = self.last_n_batch(
                min(self.size, n),
                field_name_filter=lambda x: 'observations' in x,
                observation_keys=observation_keys,
            )['observations']

            return new_observations

        min_path_length = np.min(50, self._path_length)

        path_lengths = 0
        for path_index in range(-1, -len(self.paths)-1, -1):
            # if path_lengths >= self._pool.on_policy_window:
            if path_lengths >= 5e4:
                break

            path = self.paths[path_index]
            path_length = path[list(path.keys())[0]].shape[0]
            if path_length >= min_path_length:
                path_lengths += path_length

        new_observations = []
        for i in range(path_index, 0):
            path = self.paths[i]
            path_observations = path.get(
                'observations', path.get('observations.observation'))
            path_length = path_observations.shape[0]
            if path_length < min_path_length: continue
            max_index = (
                path_length
                if path_length >= self._path_length
                else int(path_length * 0.8))
            new_observations.append(path_observations[:max_index])

        if not new_observations:
            return None

        new_observations = np.concatenate(new_observations)
        return new_observations
