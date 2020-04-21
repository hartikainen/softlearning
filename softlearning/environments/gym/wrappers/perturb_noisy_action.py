import gym
from gym import spaces
import numpy as np


__all__ = ['PerturbNoisyActionWrapper']


class PerturbNoisyActionWrapper(gym.ActionWrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, noise_scale=0, **kwargs):
        super(PerturbNoisyActionWrapper, self).__init__(*args, **kwargs)
        if not isinstance(self.env.action_space, spaces.Box):
            raise NotImplementedError(self.env.action_space)
        self._noise_scale = noise_scale

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
        noise = np.random.normal(
            scale=self._noise_scale, size=action.shape)

        action = action + noise

        return action

    def reverse_action(self, action):
        raise NotImplementedError
