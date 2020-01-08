import gym
import numpy as np

from gym.envs.mujoco import hopper_v3
from gym.envs.mujoco import walker2d_v3
from gym.envs.mujoco import ant_v3
from gym.envs.mujoco import humanoid_v3


from softlearning.utils.random import spherical as random_spherical


__all__ = ['PerturbBodyWrapper']


def num_dimensions(environment):
    two_dimensional_envs = (
        hopper_v3.HopperEnv,
        walker2d_v3.Walker2dEnv,
    )
    three_dimensional_envs = (
        ant_v3.AntEnv,
        humanoid_v3.HumanoidEnv,
    )
    if isinstance(environment, two_dimensional_envs):
        return 2
    elif isinstance(environment, three_dimensional_envs):
        return 3

    raise NotImplementedError(type(environment))


class PerturbBodyWrapper(gym.Wrapper):
    """Perturb the agent by applying a force to its torso."""
    def __init__(self,
                 *args,
                 perturbation_strength=0.0,
                 perturbation_frequency=100,
                 perturbation_probability=None,
                 perturbation_length=1,
                 perturbation_direction='random',
                 **kwargs):
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
        return super(PerturbBodyWrapper, self).__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._step_counter = 0
        return super(PerturbBodyWrapper, self).reset(*args, **kwargs)

    @property
    def perturbation_direction(self):
        if isinstance(self._perturbation_direction, np.ndarray):
            return self._perturbation_direction

        elif self._perturbation_direction == 'random':
            perturbation_size = num_dimensions(self.unwrapped)
            perturbation = random_spherical(ndim=perturbation_size)
            if perturbation_size == 2:
                perturbation = np.array(
                    (perturbation[0], 0.0, perturbation[1]))
            return perturbation

        elif isinstance(self._perturbation_direction, dict):
            if self._perturbation_direction['type'] == 'towards':
                perturbation_target = self._perturbation_direction['target']
                if perturbation_target == 'pond_center':
                    perturbation_target = self.env.pond_center
                else:
                    raise NotImplementedError(self._perturbation_direction)

                xy = self.sim.data.qpos.flat[:2]
                perturbation_direction = perturbation_target - xy
                perturbation_direction /= np.linalg.norm(perturbation_direction)
                perturbation_direction = np.array(
                    (perturbation_direction[0], perturbation_direction[1], 0.0)
                )

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

        if self.should_start_perturbation:
            torso_index = self.sim.model.body_name2id('torso')
            perturbation_direction = self.perturbation_direction

            perturbation  = (
                perturbation_direction * self._perturbation_strength)
            self._perturbation_started_at = self._step_counter
            self.sim.data.xfrc_applied[torso_index][
                0:perturbation.size] = perturbation

        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.vopt.flags[11:13] = 1

        result = super(PerturbBodyWrapper, self).step(*args, **kwargs)

        if self.should_end_perturbation:
            self.sim.data.xfrc_applied[:] = 0.0
            self._perturbation_started_at = None

        return result
