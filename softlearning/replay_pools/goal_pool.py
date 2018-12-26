"""Pool that adds randomly chosen observations as goals to the batch."""

from .simple_replay_pool import SimpleReplayPool


class GoalPool(SimpleReplayPool):
    def batch_by_indices(self,
                         indices,
                         observation_keys=None,
                         **kwargs):
        batch = super(GoalPool, self).batch_by_indices(
            indices, observation_keys=observation_keys, **kwargs)

        goal_indices = self.random_indices(indices.size)

        goal_batch = super(GoalPool, self).batch_by_indices(
            indices=goal_indices,
            observation_keys=observation_keys,
            field_name_filter=lambda x: x == 'observations')

        batch['goals'] = goal_batch['observations']

        return batch
