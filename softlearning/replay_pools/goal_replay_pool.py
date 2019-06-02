from gym.spaces import Dict

from .flexible_replay_pool import Field
from .simple_replay_pool import SimpleReplayPool


class GoalReplayPool(SimpleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 **kwargs):
        observation_space = environment.observation_space
        assert isinstance(observation_space, Dict), observation_space

        extra_fields = {
            'goals': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
                if name in (environment.goal_key_map.values()
                            or environment.observation_keys)
            },
        }

        return super(GoalReplayPool, self).__init__(
            environment, *args, **kwargs, extra_fields=extra_fields)

    def add_samples(self, samples, *args, **kwargs):
        full_observations = samples['observations']
        observations = type(full_observations)(
            (key, values)
            for key, values in full_observations.items()
            if key not in self._environment.goal_key_map.keys()
        )
        goals = type(full_observations)(
            (goal_key, full_observations[observation_key])
            for observation_key, goal_key
            in self._environment.goal_key_map.items()
        )
        samples.update({
            'observations': observations,
            'goals': goals,
        })
        return super(GoalReplayPool, self).add_samples(
            samples, *args, **kwargs)
