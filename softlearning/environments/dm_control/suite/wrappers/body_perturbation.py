import dm_env
import numpy as np

from gym.envs.mujoco import hopper_v3
from gym.envs.mujoco import walker2d_v3
from gym.envs.mujoco import ant_v3
from gym.envs.mujoco import humanoid_v3


from softlearning.utils.random import spherical as random_spherical
from softlearning.environments.dm_control.suite.point_mass import (
    PointMassPhysics,
)


__all__ = ['Wrapper']


def num_dimensions(physics):
    position_size = physics.position().size
    assert position_size in (2, 3), position_size
    return position_size


class Wrapper(dm_env.Environment):
    """Perturb the agent by applying a force to its torso."""
    def __init__(self,
                 env,
                 *args,
                 perturbation_strength=0.0,
                 perturbation_frequency=100,
                 perturbation_probability=None,
                 perturbation_length=1,
                 perturbation_direction='random',
                 **kwargs):
        self._env = env
        self._perturbation_strength = perturbation_strength
        self._perturbation_length = perturbation_length
        if not ((perturbation_frequency is None)
                or (perturbation_probability is None)):
            raise ValueError(
                "Either `perturbation_probability` or `perturbation_frequency`"
                " should be `None`.")

        self._perturbation_frequency = perturbation_frequency
        self._perturbation_probability = perturbation_probability

        if isinstance(perturbation_direction, (tuple, list, np.ndarray)):
            perturbation_direction = np.array(perturbation_direction)
            assert perturbation_direction.size == 3, perturbation_direction

        self._perturbation_direction = perturbation_direction

        self._perturbation_started_at = None
        return super(Wrapper, self).__init__(*args, **kwargs)

    @property
    def perturbation_direction(self):
        if isinstance(self._perturbation_direction, np.ndarray):
            return self._perturbation_direction

        elif self._perturbation_direction == 'random':
            perturbation_size = num_dimensions(self.physics)
            perturbation = random_spherical(ndim=perturbation_size)
            if perturbation_size == 2:
                if not isinstance(self.physics, PointMassPhysics):
                    raise NotImplementedError("TODO(hartikainen)")
                # perturbation = np.array(
                #     (perturbation[0], 0.0, perturbation[1]))
                perturbation = np.array((*perturbation, 0.0))
            return perturbation

        elif isinstance(self._perturbation_direction, dict):
            if self._perturbation_direction['type'] == 'towards':
                perturbation_target = self._perturbation_direction['target']
                if perturbation_target == 'pond_center':
                    perturbation_target = self._env.physics.pond_center_xyz[:2]
                else:
                    raise NotImplementedError(self._perturbation_direction)

                xy = self.physics.position()[:2]
                perturbation_direction = perturbation_target - xy
                perturbation_direction /= np.linalg.norm(
                    perturbation_direction)
                perturbation_direction = np.array(
                    (*perturbation_direction[:2], 0.0))

            return perturbation_direction

        raise NotImplementedError(type(self._perturbation_direction))

    @property
    def should_start_perturbation(self):
        if self._perturbation_frequency is not None:
            return self._step_counter % self._perturbation_frequency == 0
        elif self._perturbation_probability is not None:
            return np.random.rand() < self._perturbation_probability

        raise ValueError

    @property
    def should_end_perturbation(self):
        if self._perturbation_started_at is None:
            return True

        return ((self._step_counter - self._perturbation_started_at)
                > self._perturbation_length - 2)

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if isinstance(self.physics, PointMassPhysics):
            torso_key = 'pointmass'
        else:
            raise NotImplementedError("TODO(hartikainen)")

        if self.should_start_perturbation:
            perturbation_direction = self.perturbation_direction

            perturbation = (
                perturbation_direction * self._perturbation_strength)
            self._perturbation_started_at = self._step_counter
            self.physics.named.data.xfrc_applied[torso_key][
                0:perturbation.size] = perturbation

        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.vopt.flags[11:13] = 1

        time_step = self._env.step(*args, **kwargs)
        assert not any(x in time_step.observation for x in (
            'perturbed', 'perturbation'))
        time_step.observation.update({
            'perturbed': np.any(
                0 != self.physics.named.data.xfrc_applied[torso_key][0:3]),
            'perturbation': self.physics.named.data.xfrc_applied[
                torso_key][0:3].copy(),
        })

        if self.should_end_perturbation:
            self.physics.data.xfrc_applied[:] = 0.0
            self._perturbation_started_at = None

        return time_step

    def reset(self, *args, **kwargs):
        self.physics.data.xfrc_applied[:] = 0.0
        self._perturbation_started_at = None
        self._step_counter = 0
        return self._env.reset(*args, **kwargs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
