import gym
from gym import spaces
import numpy as np


__all__ = ['PerturbNoisyActionWrapper']


class PerturbNoisyActionWrapper(gym.ActionWrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, noise_scale=0, **kwargs):
        super(PerturbNoisyActionWrapper, self).__init__(*args, **kwargs)
        self._noise_scale = noise_scale

    def action(self, action):
        if not isinstance(self.env.action_space, spaces.Box):
            raise NotImplementedError(self.env.action_space)

        noise = np.random.normal(
            scale=self._noise_scale, size=action.shape)

        action = action + noise

        return action

    def reverse_action(self, action):
        raise NotImplementedError

