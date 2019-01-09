"""Pool that adds randomly chosen observations as goals to the batch."""

from .simple_replay_pool import SimpleReplayPool

import numpy as np


def random_int_with_variable_range(mins, maxs):
    result = np.floor(np.random.uniform(mins, maxs)).astype(int)
    return result


class GoalPool(SimpleReplayPool):
    def __init__(self, path_length, *args, **kwargs):
        self._path_length = path_length
        super(GoalPool, self).__init__(*args, **kwargs)

    def batch_by_indices(self,
                         indices,
                         observation_keys=None,
                         **kwargs):
        batch = super(GoalPool, self).batch_by_indices(
            indices, observation_keys=observation_keys, **kwargs)

        goal_indices = (
            (indices // self._path_length)
            + random_int_with_variable_range(
                (indices % self._path_length), self._path_length))
        # goal_indices = self.random_indices(indices.size)

        assert np.all((goal_indices - indices) < self._path_length)
        assert np.all(goal_indices >= 0)

        goal_batch = super(GoalPool, self).batch_by_indices(
            indices=goal_indices,
            observation_keys=observation_keys,
            field_name_filter=lambda x: 'observations' in x)

        batch['goals'] = goal_batch.get(
            'observations',
            goal_batch.get('observations.observation'))

        return batch
