import gym
from gym import spaces
import numpy as np

from softlearning.utils.random import spherical as random_spherical

from .perturb_body import num_dimensions


__all__ = ['WindWrapper']


class WindWrapper(gym.Wrapper):
    """Perturb the agent by applying wind to the environment."""
    def __init__(self,
                 *args,
                 wind_strength=0,
                 wind_direction='random',
                 **kwargs):
        self._wind_strength = wind_strength
        self._wind_direction = wind_direction
        result = super(WindWrapper, self).__init__(*args, **kwargs)
        return result

    def reset(self, *args, **kwargs):
        self.sim.model.opt.density = 0.0
        self.sim.model.opt.wind[:] = 0.0
        self._step_counter = 0
        return super(WindWrapper, self).reset(*args, **kwargs)

    @property
    def wind_direction(self):
        if isinstance(self._wind_direction, np.ndarray):
            return self._wind_direction

        elif self._wind_direction == 'random':
            perturbation_size = num_dimensions(self.unwrapped) - 1
            perturbation = random_spherical(ndim=perturbation_size)

            if perturbation_size == 1:
                perturbation = np.array((perturbation[0], 0.0, 0.0))
            elif perturbation_size == 2:
                perturbation = np.array((*perturbation, 0.0))
            else:
                raise NotImplementedError(perturbation_size)

            return perturbation

        elif isinstance(self._wind_direction, dict):
            if self._wind_direction['type'] == 'towards':
                perturbation_target = self._wind_direction['target']
                if perturbation_target == 'pond_center':
                    perturbation_target = self.env.pond_center
                else:
                    raise NotImplementedError(self._wind_direction)

                xy = self.sim.data.qpos.flat[:2]
                wind_direction = perturbation_target - xy
                wind_direction /= np.linalg.norm(wind_direction)
                wind_direction = np.array((*wind_direction, 0.0))

            return wind_direction

        raise NotImplementedError(type(self._wind_direction))

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if self._step_counter % 150 == 0:
            wind_direction = self.wind_direction
            wind = wind_direction * self._wind_strength
            self.sim.model.opt.density = 1.2
            self.sim.model.opt.wind[0:wind.size] = wind

        observation, reward, done, info = super(WindWrapper, self).step(
            *args, **kwargs)

        info.update({
            'perturbed': np.any(0 != self.sim.model.opt.wind),
            'perturbation': self.sim.model.opt.wind[:3].copy(),
        })

        if (self._step_counter % 200) == 0:
            self.sim.model.opt.density = 0.0
            self.sim.model.opt.wind[:] = 0.0

        return observation, reward, done, info
