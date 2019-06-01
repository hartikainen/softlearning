import numpy as np

from .hindsight_experience_replay_pool import ResamplingReplayPool


class DistancePool(ResamplingReplayPool):
    def __init__(self, *args, max_pair_distance=None, **kwargs):
        assert max_pair_distance is None
        self._max_pair_distance = max_pair_distance
        super(DistancePool, self).__init__(*args, **kwargs)

    def batch_by_indices(self, indices, *args, **kwargs):
        batch1 = super(DistancePool, self).batch_by_indices(
            indices, *args, **kwargs)

        resampled_indices, resampled_distances = self._resample_indices(
            indices,
            -1 * batch1['episode_index_forwards'],
            batch1['episode_index_backwards'],
            resampling_strategy='future')

        assert np.all(resampled_distances >= 0), resampled_distances

        batch2 = super(DistancePool, self).batch_by_indices(
            indices, *args, **kwargs)

        batch = {
            'observations1': batch1['observations'],
            'actions1': batch1['actions'],
            'observations2': batch2['observations'],
            'distances': resampled_distances,
        }

        return batch
