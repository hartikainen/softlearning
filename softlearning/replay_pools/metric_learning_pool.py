from gym.spaces import Dict

from .simple_replay_pool import SimpleReplayPool


class MetricLearningPool(SimpleReplayPool):
    def batch_by_indices(self,
                         indices,
                         observation_keys=None,
                         **kwargs):
        batch = super(MetricLearningPool, self).batch_by_indices(
            indices, observation_keys=observation_keys, **kwargs)

        if isinstance(self._observation_space, Dict):
            raise NotImplementedError

        random_indices = self.random_indices(indices.size)
        batch['goals'] = self.observations[random_indices]

        return batch
