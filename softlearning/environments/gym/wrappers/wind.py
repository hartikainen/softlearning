import gym
import numpy as np

from softlearning.utils.random import spherical as random_spherical

from .perturb_body import num_dimensions


__all__ = ['WindWrapper']


class WindWrapper(gym.Wrapper):
    """Perturb the agent by applying wind to the environment."""
    def __init__(self,
                 *args,
                 wind_strength=0,
                 wind_frequency=200,
                 wind_probability=None,
                 wind_length=50,
                 wind_direction='random',
                 **kwargs):
        self._wind_strength = wind_strength
        self._wind_frequency = wind_frequency
        self._wind_probability = wind_probability
        self._wind_length = wind_length
        self._wind_direction = wind_direction
        self._wind_started_at = None

        if not ((wind_frequency is None) or (wind_probability is None)):
            raise ValueError(
                "Either `wind_probability` or `wind_frequency`"
                " should be `None`.")

        result = super(WindWrapper, self).__init__(*args, **kwargs)
        return result

    def reset(self, *args, **kwargs):
        self.sim.model.opt.density = 0.0
        self.sim.model.opt.wind[:] = 0.0
        self._wind_started_at = None
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

    @property
    def should_start_wind(self):
        if self._wind_frequency is not None:
            return self._step_counter % self._wind_frequency == 0
        elif self._wind_probability is not None:
            return np.random.rand() < self._wind_probability

        raise ValueError

    @property
    def should_end_wind(self):
        if self._wind_started_at is None:
            return True

        return ((self._step_counter - self._wind_started_at)
                > self._wind_length - 2)

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if self.should_start_wind:
            self._wind_started_at = self._step_counter
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

        if self.should_end_wind:
            self.sim.model.opt.density = 0.0
            self.sim.model.opt.wind[:] = 0.0

        return observation, reward, done, info
