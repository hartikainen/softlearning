import gym
from gym import spaces
import numpy as np


__all__ = ['WindWrapper']


class WindWrapper(gym.Wrapper):
    """Rescale the action space of the environment."""
    def __init__(self, *args, wind_strength=0, **kwargs):
        self._wind_strength = wind_strength
        result = super(WindWrapper, self).__init__(*args, **kwargs)

        self.sim.model.opt.density = 1.2
        self._wind_started = None
        self._wind_length = 50

        return result

    def reset(self, *args, **kwargs):
        self._step_counter = 0
        return super(WindWrapper, self).reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        if self._step_counter % 150 == 0:
            self._wind_started = self._step_counter
            wind_direction = np.random.choice((-1, 1))
            wind_x = wind_direction * self._wind_strength
            self.sim.model.opt.wind[:] = [wind_x, 0.0, 0.0]

        wind_on = np.any(self.sim.model.opt.wind[:] != 0)
        wind_should_stop = (
            self._step_counter >= self._wind_started + self._wind_length)
        if wind_on and wind_should_stop:
            self.sim.model.opt.wind[:] = 0
            # env.unwrapped.sim.model.opt.gravity[:] = 0

        result = super(WindWrapper, self).step(*args, **kwargs)

        self._step_counter += 1

        return result
