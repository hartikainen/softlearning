import gym
from gym import spaces
import numpy as np


__all__ = ['PerturbBodyWrapper']


class PerturbBodyWrapper(gym.Wrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, **kwargs):
        self._perturbation_probability = perturbation_probability
        return super(PerturbBodyWrapper, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        old_xfrc_applied = self.sim.data.xfrc_applied[:]
        torso_index = self.sim.model.body_name2id('torso')
        if np.random.rand() < self._perturbation_probability:
            # self.sim.data.xfrc_applied[torso_index][0] = -5.0
            # self.sim.data.xfrc_applied[torso_index][0:3] = np.random.uniform(-50, 50, 3)
            self.sim.data.xfrc_applied[torso_index][np.random.choice((0,1,2))] = (
                np.random.choice((-1, 1)) * 5.0)
        result = super(PerturbBodyWrapper, self).step(*args, **kwargs)
        self.sim.data.xfrc_applied[:] = old_xfrc_applied
        return result

