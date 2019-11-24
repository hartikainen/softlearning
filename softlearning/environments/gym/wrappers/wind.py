import gym
from gym import spaces
import numpy as np

from softlearning.utils.random import spherical as random_spherical

from .perturb_body import num_dimensions


__all__ = ['WindWrapper']


class WindWrapper(gym.Wrapper):
    """Perturb the agent by applying wind to the environment."""
    def __init__(self, *args, wind_strength=0, **kwargs):
        self._wind_strength = wind_strength
        result = super(WindWrapper, self).__init__(*args, **kwargs)
        self.sim.model.opt.density = 1.2
        return result

    def reset(self, *args, **kwargs):
        self._step_counter = 0
        return super(WindWrapper, self).reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if self._step_counter % 150 == 0:
            perturbation_size = num_dimensions(self.unwrapped) - 1
            wind_direction = random_spherical(ndim=perturbation_size)
            wind_xy = wind_direction * self._wind_strength
            self.sim.model.opt.wind[0:perturbation_size] = wind_xy

        result = super(WindWrapper, self).step(*args, **kwargs)

        if (self._step_counter % 200) == 0:
            self.sim.model.opt.wind[:] = 0.0

        return result
