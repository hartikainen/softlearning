import numpy as np

import gym
from gym import spaces


def rescale_values(values, old_low, old_high, new_low, new_high):
    rescaled_values = new_low + (new_high - new_low) * (
        (values - old_low) / (old_high - old_low))
    rescaled_values = np.clip(rescaled_values, new_low, new_high)
    return rescaled_values


class RescaleObservation(gym.ObservationWrapper):
    r"""Rescales the continuous observation space of the environment to a range [a,b].

    Example::

        >>> RescaleObservation(env, a, b).observation_space == Box(a,b)
        True

    """
    def __init__(self, env, a, b):
        assert isinstance(env.observation_space, spaces.Box), (
            "expected Box observation space, got {}".format(type(env.observation_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleObservation, self).__init__(env)
        self.a = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + a
        self.b = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + b
        self.observation_space = spaces.Box(low=a, high=b, shape=env.observation_space.shape, dtype=env.observation_space.dtype)

    def observation(self, observation):
        old_low = self.env.observation_space.low
        old_high = self.env.observation_space.high
        new_low = self.a
        new_high = self.b
        observation = rescale_values(
            observation, old_low, old_high, new_low, new_high)

        return observation
