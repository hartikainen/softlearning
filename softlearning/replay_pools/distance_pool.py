import numpy as np

from serializable import Serializable
from .simple_replay_pool import SimpleReplayPool


class DistancePool(SimpleReplayPool, Serializable):
    def __init__(self,
                 path_length,
                 *args,
                 on_policy_window=None,
                 max_pair_distance=None,
                 max_size=None,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._on_policy_window = on_policy_window or max_size
        self._max_pair_distance = max_pair_distance

        self._path_length = path_length
        super(DistancePool, self).__init__(
            *args, max_size=max_size, **kwargs)

    def random_batch(self, batch_size, *args, **kwargs):
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
            field_name_filter=('observations', 'actions'),
            **kwargs)

        pairs_observations = pairs_batch['observations']
        pairs_actions = pairs_batch['actions']
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

        batch = super(DistancePool, self).random_batch(
            batch_size, *args, **kwargs)

        batch.update({
            'distance_pairs_observations': pairs_observations,
            'distance_pairs_actions': pairs_actions,
            'distance_pairs_distances': pairs_distances,

            'distance_triples_observations': triples_observations,
            'distance_triples_actions': triples_actions,

            'distance_objectives_observations': objectives_observations,
            'distance_objectives_actions': objectives_actions,
        })

        return batch
