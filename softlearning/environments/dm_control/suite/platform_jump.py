import abc
import numpy as np
from lxml import etree
from dm_control.suite import base
from dm_control.utils import rewards
import skimage.measure

from scipy.spatial.transform import Rotation
from softlearning.environments.dm_control.suite.bridge import stringify

DEFAULT_PLATFORM_SIZE = 0.1
DEFAULT_JUMP_REWARD_WEIGHT = 1.0
DEFAULT_CONSTANT_REWARD = 0.0


def make_model(base_model_string,
               platform_size=DEFAULT_PLATFORM_SIZE,
               extra_position_attrib=None,
               friction=None,
               hang=False):
    extra_position_attrib = extra_position_attrib or {}
    mjcf = etree.fromstring(base_model_string)
    sensor_element = mjcf.find('sensor')
    worldbody = mjcf.find('worldbody')
    default_element = mjcf.find('default')
    default_element.insert(-1, etree.Element(
        'position',
        # ctrlrange='-1 1',
        ctrllimited='true',
    ))

    if friction is not None:
        default_element.find('default').find('geom').attrib['friction'] = friction
        floor_geom = mjcf.find(".//geom[@name='floor']")
        floor_geom.attrib['friction'] = friction


    if hang:
        torso_body_element = worldbody.find(".//body[@name='torso']")
        torso_body_element.remove(
            torso_body_element.find(".//freejoint[@name='root']"))

    default_feet_position = np.array(
        ((-0.00267608, +0.09, 0.02834923),
         (-0.00267608, -0.09, 0.02834923)))

    left_foot_platform_element = etree.Element(
        "geom",
        type="cylinder",
        name="left-foot-platform",
        pos=stringify((*default_feet_position[0][:2], 0.0)),
        size=stringify((platform_size, 0.01)),
        contype="0",
        conaffinity="0",
        rgba="1 0 0 0.1")
    right_foot_platform_element = etree.Element(
        "geom",
        type="cylinder",
        name="right-foot-platform",
        pos=stringify((*default_feet_position[1][:2], 0.0)),
        size=stringify((platform_size, 0.01)),
        contype="0",
        conaffinity="0",
        rgba="1 0 0 0.1")
    worldbody.insert(0, left_foot_platform_element)
    worldbody.insert(1, right_foot_platform_element)

    all_joints = {
        x.attrib['name']: x
        for x in worldbody.iter('joint')
    }

    if extra_position_attrib:
        actuator_element = mjcf.find('actuator')
        motor_elements = actuator_element.findall('motor')
        for motor_element in motor_elements:
            position_attrib = {
                **motor_element.attrib,
                'gear': "1",
                **extra_position_attrib,
            }

            joint_element = all_joints[position_attrib['joint']]
            joint_range = joint_element.attrib['range']
            position_attrib['ctrlrange'] = " ".join([
                f"{(float(x) / (180.0 / np.pi)):.2f}"
                for x in joint_range.split(' ')
            ])

            position_element = etree.Element("position", **position_attrib)
            actuator_element.remove(motor_element)
            actuator_element.insert(-1, position_element)

    return etree.tostring(mjcf, pretty_print=True)


class PlatformJumpPhysicsMixin:
    @property
    def agent_geom_ids(self):
        raise NotImplementedError

    @property
    def floor_geom_id(self):
        if getattr(self, '_floor_geom_id', None) is None:
            self._floor_geom_id = self.model.name2id('floor', 'geom')
        return self._floor_geom_id

    def feet_platform_difference(self):
        left_platform_pos = np.array([-0.00267608, 0.09, 0.24202798])
        right_platform_pos = np.array([-0.00267608, -0.09, 0.24202798])

        # left_platform_pos = self.named.model.geom_pos['left-foot-platform']
        # right_platform_pos = self.named.model.geom_pos['right-foot-platform']

        left_foot_pos = self.named.data.xpos['left_foot']
        right_foot_pos = self.named.data.xpos['right_foot']

        feet_platform_difference = np.array((
            left_foot_pos - left_platform_pos,
            right_foot_pos - right_platform_pos,
        ))

        return feet_platform_difference


class PlatformJumpTaskMixin(base.Task):
    """A task solved by running across a bridge."""

    def __init__(self,
                 constant_reward=DEFAULT_CONSTANT_REWARD,
                 jump_reward_weight=DEFAULT_JUMP_REWARD_WEIGHT,
                 **kwargs):
        """Initializes an instance of `PlatformJumpTask`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a
            seed automatically (default).
        """
        self._constant_reward = constant_reward
        self._jump_reward_weight = jump_reward_weight
        return super(PlatformJumpTaskMixin, self).__init__(**kwargs)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        self._current_timestep = -1
        return super(PlatformJumpTaskMixin, self).initialize_episode(physics)

    def before_step(self, action, physics):
        self._current_timestep += 1
        return super(PlatformJumpTaskMixin, self).before_step(action, physics)

    def after_step(self, physics):
        return super(PlatformJumpTaskMixin, self).after_step(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        observation = (
            super(PlatformJumpTaskMixin, self).get_observation(physics))

        observation.update((
            ('feet_velocity', physics.named.data.subtree_linvel[
                ['left_foot', 'right_foot']]),
            ('feet_platform_difference', physics.feet_platform_difference()),
        ))

        return observation

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        torso_height = physics.named.data.xpos['torso'][-1]
        feet_height = physics.named.data.xpos[
            ['left_foot', 'right_foot']][:, -1]
        minimum_feet_height = np.min(feet_height)

        feet_platform_difference_xy = physics.feet_platform_difference()[:, :2]
        feet_platform_difference_xy_cost = np.sum(np.linalg.norm(
            feet_platform_difference_xy, ord=2, axis=-1))

        reward = (
            (minimum_feet_height <= torso_height) * (
                self._constant_reward
                + self._jump_reward_weight * minimum_feet_height
                - self._jump_reward_weight * feet_platform_difference_xy_cost))

        return reward


class PlatformDropTaskMixin(base.Task):
    """A task solved by running across a bridge."""

    def __init__(self,
                 constant_reward=DEFAULT_CONSTANT_REWARD,
                 drop_reward_weight=DEFAULT_JUMP_REWARD_WEIGHT,
                 **kwargs):
        """Initializes an instance of `PlatformDropTask`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a
            seed automatically (default).
        """
        self._constant_reward = constant_reward
        self._drop_reward_weight = drop_reward_weight
        return super(PlatformDropTaskMixin, self).__init__(**kwargs)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        self._current_timestep = -1
        return super(PlatformDropTaskMixin, self).initialize_episode(physics)

    def before_step(self, action, physics):
        self._current_timestep += 1
        return super(PlatformDropTaskMixin, self).before_step(action, physics)

    def after_step(self, physics):
        return super(PlatformDropTaskMixin, self).after_step(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        observation = (
            super(PlatformDropTaskMixin, self).get_observation(physics))

        observation.update((
            ('feet_velocity', physics.named.data.subtree_linvel[
                ['left_foot', 'right_foot']]),
            ('feet_platform_difference', physics.feet_platform_difference()),
        ))

        return observation

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        torso_height = physics.named.data.xpos['torso'][-1]
        feet_height = physics.named.data.xpos[
            ['left_foot', 'right_foot']][:, -1]
        minimum_feet_height = np.min(feet_height)

        feet_platform_difference_xy = physics.feet_platform_difference()[:, :3]
        feet_platform_difference_xy_cost = np.sum(np.linalg.norm(
            feet_platform_difference_xy, ord=2, axis=-1))

        reward = self._constant_reward - (
            self._drop_reward_weight * (feet_platform_difference_xy_cost))

        # print(minimum_feet_height, feet_platform_difference_xy_cost, reward)
        # print(feet_platform_difference_xy_cost)

        # reward = (
        #     (0.2 < minimum_feet_height)
        #     * (minimum_feet_height <= torso_height) * (
        #         self._constant_reward
        #         + self._jump_reward_weight * minimum_feet_height
        #         - self._jump_reward_weight * feet_platform_difference_xy_cost))

        return reward
