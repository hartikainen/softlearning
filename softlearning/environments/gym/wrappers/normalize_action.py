import gym
from gym import spaces
import numpy as np


__all__ = ['NormalizeActionWrapper']


class NormalizeActionWrapper(gym.ActionWrapper):
    """Rescale the action space of the environment."""
    def __init__(self, env, *args, **kwargs):
        super(NormalizeActionWrapper, self).__init__(env, *args, **kwargs)

        if isinstance(self.env.action_space, spaces.Box):
            self.action_space = spaces.Box(
                dtype=self.action_space.dtype,
                low=-1.0,
                high=1.0,
                shape=self.action_space.shape)

    def action(self, action):
        if not isinstance(self.env.action_space, spaces.Box):
            return action

        # rescale the action
        low, high = self.env.action_space.low, self.env.action_space.high
        scaled_action = low + (action + 1.0) * (high - low) / 2.0
        scaled_action = np.clip(scaled_action, low, high)

        return scaled_action

    def reverse_action(self, action):
        raise NotImplementedError


normalize = NormalizeActionWrapper
