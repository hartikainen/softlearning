import dm_env
import numpy as np

from softlearning.utils.random import spherical as random_spherical

from .body_perturbation import num_dimensions


__all__ = ['Wrapper']


class Wrapper(dm_env.Environment):
    """Perturb the agent by applying wind to the environment."""
    def __init__(self,
                 env,
                 *args,
                 wind_strength=0,
                 wind_frequency=200,
                 wind_probability=None,
                 wind_length=50,
                 wind_direction='random',
                 **kwargs):
        self._env = env
        self._wind_strength = wind_strength
        self._wind_direction = wind_direction
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

        result = super(Wrapper, self).__init__(*args, **kwargs)
        return result

    def reset(self, *args, **kwargs):
        self.physics.model.opt.density = 0.0
        self.physics.model.opt.wind[:] = 0.0
        self._wind_started_at = None
        self._step_counter = 0
        return self._env.reset(*args, **kwargs)

    @property
    def wind_direction(self):
        if isinstance(self._wind_direction, np.ndarray):
            return self._wind_direction

        elif self._wind_direction == 'random':
            # TODO(hartikainen): Is -1 correct here?
            perturbation_size = num_dimensions(self._env.physics) - 1
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
                    perturbation_target = self._env.physics.pond_center_xyz[:2]
                else:
                    raise NotImplementedError(self._wind_direction)

                # xy = self.physics.data.qpos.flat[:2]
                xy = self.physics.position()[:2]
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
            self.physics.model.opt.density = 1.2
            self.physics.model.opt.wind[0:wind.size] = wind

        time_step = self._env.step(*args, **kwargs)

        assert not any(x in time_step.observation for x in (
            'perturbed', 'perturbation'))
        time_step.observation.update({
            'perturbed': np.any(0 != self.physics.model.opt.wind),
            'perturbation': self.physics.model.opt.wind[:3].copy(),
        })

        if self.should_end_wind:
            self.physics.model.opt.density = 0.0
            self.physics.model.opt.wind[:] = 0.0

        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
