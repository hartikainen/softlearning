import abc
import numpy as np
from lxml import etree
from dm_control.suite import base
from dm_control.utils import rewards
import skimage.measure

from scipy.spatial.transform import Rotation

DEFAULT_FREEZE_INTERVAL = 100
DEFAULT_CONSTANT_REWARD = 10.0
DEFAULT_FREEZE_REWARD_WEIGHT = 10.0


def make_model(base_model_string):
    mjcf = etree.fromstring(base_model_string)
    sensor_element = mjcf.find('sensor')
    return etree.tostring(mjcf, pretty_print=True)


class FreezeStepPhysicsMixin:
    @property
    def agent_geom_ids(self):
        raise NotImplementedError

    @property
    def floor_geom_id(self):
        if getattr(self, '_floor_geom_id', None) is None:
            self._floor_geom_id = self.model.name2id('floor', 'geom')
        return self._floor_geom_id


class FreezeStepTaskMixin(base.Task):
    """A task solved by running across a bridge."""

    def __init__(self,
                 freeze_interval=DEFAULT_FREEZE_INTERVAL,
                 constant_reward=DEFAULT_CONSTANT_REWARD,
                 freeze_reward_weight=DEFAULT_FREEZE_REWARD_WEIGHT,
                 feet_com_target_range=1.0,
                 **kwargs):
        """Initializes an instance of `FreezeStepTask`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a
            seed automatically (default).
        """
        self._freeze_interval = freeze_interval
        self._constant_reward = constant_reward
        self._freeze_reward_weight = freeze_reward_weight
        self._feet_com_target_range = feet_com_target_range
        self._feet_target_position = None
        return super(FreezeStepTaskMixin, self).__init__(**kwargs)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        self._current_timestep = -1
        return super(FreezeStepTaskMixin, self).initialize_episode(physics)

    def before_step(self, action, physics):
        self._current_timestep += 1

        freeze_started = (
            (self._freeze_interval // 2)
            == (self._current_timestep % self._freeze_interval))

        freeze_ended = (
            0 == self._current_timestep % self._freeze_interval)

        if freeze_started:
            feet_orientation = physics.named.data.xmat[
                ['left_foot', 'right_foot']].reshape(2, 3, 3)
            torso_com_xy = physics.center_of_mass_position()[:2]
            feet_com_target_xy_offset = np.random.uniform(
                low=-self._feet_com_target_range,
                high=+self._feet_com_target_range,
                size=(2, ))
            feet_com_target_xy = torso_com_xy + feet_com_target_xy_offset
            feet_com_target = np.array((*feet_com_target_xy, 0))

            default_feet_position = np.array(
                ((-0.00267608, +0.09, 0.02834923),
                 (-0.00267608, -0.09, 0.02834923)))

            random_feet_rotation = np.random.uniform(-np.pi, np.pi)
            self._feet_target_position = (
                feet_com_target
                + (Rotation.from_euler('z', random_feet_rotation)
                   .apply(default_feet_position)))
        elif freeze_ended:
            self._feet_target_position = None

        return super(FreezeStepTaskMixin, self).before_step(action, physics)

    def after_step(self, physics):
        return super(FreezeStepTaskMixin, self).after_step(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        observation = (
            super(FreezeStepTaskMixin, self).get_observation(physics))
        observation['feet_velocity'] = physics.named.data.subtree_linvel[
            ['left_foot', 'right_foot']]

        feet_position = physics.named.data.xpos[
            ['left_foot', 'right_foot']]

        if self._feet_target_position is not None:
            observation['feet_target_offset'] = (
                feet_position - self._feet_target_position)
            observation['feet_target_position'] = self._feet_target_position
        else:
            observation['feet_target_offset'] = np.zeros((2, 3))
            observation['feet_target_position'] = np.zeros((2, 3))

        observation['feet_position'] = feet_position

        return observation

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        print(self._feet_target_com)

        if (self._current_timestep % self._freeze_interval
            < self._freeze_interval // 2):
            reward = self._constant_reward

        elif (self._freeze_interval // 2
              <= self._current_timestep % self._freeze_interval):
            feet_position = physics.named.data.xpos[
                ['left_foot', 'right_foot']]
            reward = -1.0 * self._freeze_reward_weight * np.sum(
                np.linalg.norm(
                    feet_position - self._feet_target_position,
                    axis=1))
        else:
            raise ValueError(self._current_timestep, self._freeze_interval)

        return reward
