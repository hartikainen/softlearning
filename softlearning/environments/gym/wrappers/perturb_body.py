import gym
from gym import spaces
import numpy as np

from softlearning.utils.random import random_three_vector


__all__ = ['PerturbBodyWrapper']


class PerturbBodyWrapper(gym.Wrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, perturbation_strength=0.0, **kwargs):
        self._perturbation_strength = perturbation_strength
        return super(PerturbBodyWrapper, self).__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._step_counter = 0
        return super(PerturbBodyWrapper, self).reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if self._step_counter % 100 == 0:
            torso_index = self.sim.model.body_name2id('torso')
            perturbation_direction = random_three_vector()
            perturbation  = (
                perturbation_direction * self._perturbation_strength)
            self.sim.data.xfrc_applied[torso_index][0:3] = perturbation

        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.vopt.flags[:] = 0
            self.viewer.vopt.flags[11] = 1
            self.viewer.vopt.flags[12] = 1
            self.viewer.vopt.flags[13] = 1

        result = super(PerturbBodyWrapper, self).step(*args, **kwargs)

        if (self._step_counter % 105) == 0:
            self.sim.data.xfrc_applied[:] = 0.0

        return result
