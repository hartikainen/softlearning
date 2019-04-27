import gym
from gym import spaces
import numpy as np


__all__ = ['PerturbActionWrapper']


class PerturbActionWrapper(gym.ActionWrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, perturbation_probability=0, **kwargs):
        super(PerturbActionWrapper, self).__init__(*args, **kwargs)
        self._perturbation_probability = perturbation_probability

    def action(self, action):
        if not isinstance(self.env.action_space, spaces.Box):
            raise NotImplementedError(self.env.action_space)

        if np.random.rand() < self._perturbation_probability:
            action = np.random.uniform(
                self.action_space.low, self.action_space.high)

        return action

    def reverse_action(self, action):
        raise NotImplementedError


normalize = PerturbActionWrapper
