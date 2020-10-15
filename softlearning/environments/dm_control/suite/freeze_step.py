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
        return super(FreezeStepTaskMixin, self).__init__(**kwargs)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        self._current_timestep = -1
        return super(FreezeStepTaskMixin, self).initialize_episode(physics)

    def before_step(self, *args, **kwargs):
        self._current_timestep += 1
        return super(FreezeStepTaskMixin, self).before_step(*args, **kwargs)

    def after_step(self, physics):
        return super(FreezeStepTaskMixin, self).after_step(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        observation = (
            super(FreezeStepTaskMixin, self).get_observation(physics))
        observation['feet_velocity'] = physics.named.data.subtree_linvel[
            ['left_foot', 'right_foot']]
        return observation

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        if (self._current_timestep % self._freeze_interval
            < self._freeze_interval / 2):
            reward = self._constant_reward
        elif (self._freeze_interval / 2
              <= self._current_timestep % self._freeze_interval):
            feet_velocity = physics.named.data.subtree_linvel[
                ['left_foot', 'right_foot']]
            left_foot_velocity, right_foot_velocity = feet_velocity
            reward = -1.0 * self._freeze_reward_weight * np.sum(
                np.linalg.norm(feet_velocity, axis=1))
        else:
            raise ValueError(self._current_timestep, self._freeze_interval)

        return reward
