import dm_env
from gym import spaces
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
                 wind_direction='random',
                 **kwargs):
        self._env = env
        self._wind_strength = wind_strength
        self._wind_direction = wind_direction
        result = super(Wrapper, self).__init__(*args, **kwargs)
        return result

    @property
    def wind_direction(self):
        if isinstance(self._wind_direction, np.ndarray):
            return self._wind_direction

        elif self._wind_direction == 'random':
            # TODO(hartikainen): Should be -1?
            perturbation_size = num_dimensions(self._env.physics)
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

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if self._step_counter % 150 == 0:
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

        if (self._step_counter % 200) == 0:
            self.physics.model.opt.density = 0.0
            self.physics.model.opt.wind[:] = 0.0

        return time_step

    def reset(self, *args, **kwargs):
        self.physics.model.opt.density = 0.0
        self.physics.model.opt.wind[:] = 0.0
        self._step_counter = 0
        return self._env.reset(*args, **kwargs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
