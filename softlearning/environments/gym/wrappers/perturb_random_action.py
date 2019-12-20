import gym
from gym import spaces
import numpy as np


__all__ = ['PerturbRandomActionWrapper']


class PerturbRandomActionWrapper(gym.ActionWrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, perturbation_probability=0, **kwargs):
        super(PerturbRandomActionWrapper, self).__init__(*args, **kwargs)
        self._perturbation_probability = perturbation_probability

    def step(self, action):
        perturbed_action = self.action(action)
        perturbed = not np.array_equal(action, perturbed_action)
        observation, reward, done, info = self.env.step(perturbed_action)
        info.update({
            'perturbed': perturbed,
            'original_action': action,
            'perturbed_action': perturbed_action,
        })
        return observation, reward, done, info

    def action(self, action):
        if not isinstance(self.env.action_space, spaces.Box):
            raise NotImplementedError(self.env.action_space)

        if np.random.rand() < self._perturbation_probability:
            action = np.random.uniform(
                self.action_space.low, self.action_space.high)

        return action

    def reverse_action(self, action):
        raise NotImplementedError
