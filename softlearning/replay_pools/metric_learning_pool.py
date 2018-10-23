from .simple_replay_pool import SimpleReplayPool


class MetricLearningPool(SimpleReplayPool):
    def batch_by_indices(self,
                         indices,
                         observation_keys=None,
                         **kwargs):
        batch = super(MetricLearningPool, self).batch_by_indices(
            indices, observation_keys=observation_keys, **kwargs)

        goal_indices = self.random_indices(indices.size)

        goal_batch = super(MetricLearningPool, self).batch_by_indices(
            indices=goal_indices,
            observation_keys=observation_keys,
            field_name_filter=lambda x: x == 'observations')

        batch['goals'] = goal_batch['observations']

        return batch
