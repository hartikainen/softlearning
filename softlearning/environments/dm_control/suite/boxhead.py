import collections
import sys

from dm_control import mujoco
from dm_control.rl import control
from dm_control.locomotion.soccer import boxhead
from dm_control import suite
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import xml_tools, containers

from lxml import etree

import numpy as np
from scipy.spatial.transform import Rotation

from .pond import (
    PondPhysicsMixin,
    make_pond_model,
    quaternion_multiply,
    DEFAULT_POND_XY,
    OrbitTaskMixin)
from . import bridge, visualization

suite._DOMAINS['boxhead'] = sys.modules[__name__]
SUITE = containers.TaggedTasks()

DEFAULT_TIME_LIMIT = 25

DEFAULT_POND_RADIUS = 10.0
DEFAULT_DESIRED_ANGULAR_VELOCITY = 5.0
DEFAULT_DESIRED_ANGULAR_VELOCITY = 10.0
DEFAULT_ANGULAR_VELOCITY_REWARD_WEIGHT = 1.0
DEFAULT_CONTROL_COST_WEIGHT = 0.0
DEFAULT_DESIRED_SPEED_ON_BRIDGE = 10.0
DEFAULT_DESIRED_SPEED_AFTER_BRIDGE = 1.0


def make_model(friction=None):
    walker = boxhead.BoxHead(name='boxhead')
    xml_string = walker.mjcf_model.to_xml_string()

    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    head_body = mjcf.find(".//body[@name='head_body']")
    mjcf.find(".//worldbody").remove(head_body)

    main_body = etree.Element('body',  name='main_body')
    main_body.insert(0, head_body)
    x_slide = etree.Element(
        'joint',
        name='root_x',
        type='slide',
        axis='1 0 0',
    )
    y_slide = etree.Element(
        'joint',
        name='root_y',
        type='slide',
        axis='0 1 0',
    )
    z_slide = etree.Element(
        'joint',
        name='root_z',
        type='slide',
        axis='0 0 1',
    )

    main_body.insert(0, z_slide)
    main_body.insert(0, y_slide)
    main_body.insert(0, x_slide)
    mjcf.find(".//worldbody").insert(0, main_body)

    compiler_element = etree.Element(
        'compiler',
        boundmass="1e-05",
        boundinertia="1e-11",
        coordinate="local",
        angle="radian",
        eulerseq="xyz",
    )
    option_element = etree.Element(
        'option',
        timestep="0.005",
        cone="elliptic",
        noslip_iterations="5",
        noslip_tolerance="0.0",
    )
    mjcf.insert(0, compiler_element)
    mjcf.insert(1, option_element)

    if friction is not None:
        shell_geom = mjcf.find(".//geom[@name='shell']")
        shell_geom.attrib['friction'] = " ".join(
            np.array(friction).astype(str))

    kick_actuator = mjcf.find(".//general[@name='kick']")
    kick_actuator.getparent().remove(kick_actuator)

    egocentric_camera_body = mjcf.find(".//body[@name='egocentric_camera']")
    egocentric_camera_body.getparent().remove(egocentric_camera_body)
    # print(etree.tostring(mjcf, pretty_print=True, encoding='unicode', method='xml'))

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()
def orbit_pond(time_limit=DEFAULT_TIME_LIMIT,
               desired_angular_velocity=DEFAULT_DESIRED_ANGULAR_VELOCITY,
               angular_velocity_reward_weight=(
                   DEFAULT_ANGULAR_VELOCITY_REWARD_WEIGHT),
               control_cost_weight=DEFAULT_CONTROL_COST_WEIGHT,
               pond_radius=DEFAULT_POND_RADIUS,
               friction=None,
               random=None,
               environment_kwargs=None):
    """Returns the Orbit task."""
    environment_kwargs = environment_kwargs or {}
    pond_xy = environment_kwargs.get('pond_xy', DEFAULT_POND_XY)
    # base_model_string, assets = get_model_and_assets_common()
    base_model_string = make_model(friction=friction)
    xml_string = make_pond_model(
        base_model_string,
        pond_radius=pond_radius,
        pond_xy=pond_xy)
    physics = PondPhysics.from_xml_string(xml_string, common.ASSETS)
    task = Orbit(
        desired_angular_velocity=desired_angular_velocity,
        angular_velocity_reward_weight=angular_velocity_reward_weight,
        control_cost_weight=control_cost_weight,
        random=random)
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=0.025,
        **environment_kwargs)


@SUITE.add()
def bridge_run(time_limit=DEFAULT_TIME_LIMIT,
               random=None,
               environment_kwargs=None):
    """Returns the BridgeRun task."""
    environment_kwargs = environment_kwargs or {}
    bridge_length = environment_kwargs.get(
        'bridge_length', bridge.DEFAULT_BRIDGE_LENGTH)
    bridge_width = environment_kwargs.get(
        'bridge_width', bridge.DEFAULT_BRIDGE_WIDTH)
    base_model_string = make_model()
    xml_string = bridge.make_model(
        base_model_string,
        size_multiplier=1.0,
        bridge_length=bridge_length,
        bridge_width=bridge_width)
    physics = BridgeMovePhysics.from_xml_string(xml_string, common.ASSETS)
    task = BridgeMove(
        random=random,
        water_map_length=4,
        water_map_width=4,
        water_map_dx=0.5,
        water_map_dy=0.5,
        desired_speed_on_bridge=DEFAULT_DESIRED_SPEED_ON_BRIDGE,
        desired_speed_after_bridge=DEFAULT_DESIRED_SPEED_AFTER_BRIDGE)
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


def _common_observations(physics):
    acceleration = 15.0 * np.tanh(physics.acceleration() / 50.0)
    observation = collections.OrderedDict((
        ('position', physics.position()),
        ('velocity', physics.velocity()),
        ('roll_velocity', physics.named.data.qvel['roll']),
        ('acceleration', acceleration),
        ('gyro', physics.gyro()),
        ('global_velocity', physics.global_velocity()),
        ('orientation', physics.orientation()),
    ))
    return observation


class Physics(mujoco.Physics):
    def set_orientation(self, quaternion):
        normalised_quaternion = quaternion / np.linalg.norm(quaternion)
        xmat = Rotation.from_quat(
            np.roll(normalised_quaternion, -1)
        ).as_matrix().ravel()
        self.named.data.xmat['torso'] = xmat
        return self.orientation()

    def orientation(self):
        orientation = np.roll(Rotation.from_matrix(
            self.named.data.xmat['torso'].reshape(3, 3)
        ).as_quat(), 1)
        return orientation

    def set_position(self, position):
        self.named.data.qpos['root_x'] = position[0]
        self.named.data.qpos['root_y'] = position[1]
        return self.position()

    def position(self):
        return self.named.data.qpos[['root_x', 'root_y']].copy()

    def global_velocity(self):
        return self.named.data.qvel[['root_x', 'root_y']].copy()

    def velocity(self):
        return self.named.data.sensordata['sensor_torso_vel'].copy()

    def acceleration(self):
        return self.named.data.sensordata['sensor_torso_accel'].copy()

    def gyro(self):
        return self.named.data.sensordata['sensor_torso_gyro'].copy()


class PondPhysics(PondPhysicsMixin, Physics):
    def torso_velocity(self):
        return self.velocity()

    def center_of_mass(self):
        return self.named.data.xpos['torso'].copy()

    def orientation_to_pond(self):
        orientation_to_origin = self.orientation()

        xy_from_pond_center = self.position()[:2] - self.pond_center_xyz[:2]
        angle_to_pond_center = np.arctan2(*xy_from_pond_center[::-1])
        origin_to_pond_transformation = np.roll(
            Rotation.from_euler('z', angle_to_pond_center).inv().as_quat(), 1)

        orientation_to_pond = quaternion_multiply(
            origin_to_pond_transformation, orientation_to_origin)

        # TODO(hartikainen): Check if this has some negative side effects on
        # other rotation axes.
        orientation_to_pond[-1] = np.abs(orientation_to_pond[-1])

        return orientation_to_pond

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_orbit_pond(self, *args, **kwargs)


class Orbit(OrbitTaskMixin):
    def common_observations(self, physics):
        return _common_observations(physics)

    def upright_reward(self, physics):
        return 1.0

    def initialize_episode(self, physics):
        pond_radius = physics.named.model.geom_size['pond'][0]
        pond_center_x, pond_center_y = physics.pond_center_xyz[:2]

        random_angle = np.random.uniform(0, 2 * np.pi)
        distance_from_pond = pond_radius / 10.0
        distance_from_origin = pond_radius + distance_from_pond
        x = pond_center_x + distance_from_origin * np.cos(random_angle)
        y = pond_center_y + distance_from_origin * np.sin(random_angle)
        physics.set_position((x, y))

        tangent_to_pond_rotation = np.pi - random_angle
        physics.named.data.qpos['steer'] = tangent_to_pond_rotation

        result = super(Orbit, self).initialize_episode(physics)

        return result


class BridgeMovePhysics(bridge.MovePhysicsMixin, Physics):
    def torso_velocity(self):
        return self.velocity()

    def key_geom_positions(self):
        return self.named.data.xpos['head'][np.newaxis, ...]

    def center_of_mass(self):
        return self.named.data.geom_xpos['head']

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_bridge_move(self, *args, **kwargs)


class BridgeMove(bridge.MoveTaskMixin):
    def common_observations(self, physics):
        return _common_observations(physics)

    def upright_reward(self, physics):
        return 1.0

    def initialize_episode(self, physics):
        physics.named.data.geom_xpos['head'][:2] = 0.0
        physics.named.model.geom_pos['head'][:2] = 0.0
        return super(BridgeMove, self).initialize_episode(physics)
