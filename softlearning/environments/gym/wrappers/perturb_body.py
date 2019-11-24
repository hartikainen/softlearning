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
                 perturbation_length=1,
                 **kwargs):
        self._perturbation_strength = perturbation_strength
        self._perturbation_length = perturbation_length
        self._perturbation_frequency = perturbation_frequency
        return super(PerturbBodyWrapper, self).__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._step_counter = 0
        return super(PerturbBodyWrapper, self).reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._step_counter += 1

        if self._step_counter % self._perturbation_frequency == 0:
            torso_index = self.sim.model.body_name2id('torso')
            perturbation_size = num_dimensions(self.unwrapped)
            perturbation_direction = random_spherical(ndim=perturbation_size)

            perturbation  = (
                perturbation_direction * self._perturbation_strength)
            self.sim.data.xfrc_applied[torso_index][
                0:perturbation_size] = perturbation

        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.vopt.flags[:] = 0
            self.viewer.vopt.flags[11] = 1
            self.viewer.vopt.flags[12] = 1
            self.viewer.vopt.flags[13] = 1

        result = super(PerturbBodyWrapper, self).step(*args, **kwargs)

        should_stop_perturbation = (
            (self._step_counter % self._perturbation_frequency)
            == self._perturbation_length - 1)

        if should_stop_perturbation:
            self.sim.data.xfrc_applied[:] = 0.0

        return result


        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.vopt.flags[:] = 0
            self.viewer.vopt.flags[11] = 1
            self.viewer.vopt.flags[12] = 1
            self.viewer.vopt.flags[13] = 1

        result = super(PerturbBodyWrapper, self).step(*args, **kwargs)

        should_stop_perturbation = (
            (self._step_counter % self._perturbation_frequency)
            == self._perturbation_length - 1)

        if should_stop_perturbation:
            self.sim.data.xfrc_applied[:] = 0.0

        return result
