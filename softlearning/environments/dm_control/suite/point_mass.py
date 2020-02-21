import collections
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.suite.point_mass import (
    _DEFAULT_TIME_LIMIT,
    SUITE,
    get_model_and_assets as get_model_and_assets_common,
    Physics as PointMassPhysics)
from dm_control.utils import xml_tools

from lxml import etree

import numpy as np
from .pond import (
    PondPhysicsMixin,
    make_pond_model,
    DEFAULT_POND_XY,
    DEFAULT_POND_RADIUS,
    OrbitTaskMixin)
from . import bridge, visualization


_WALLS = ('wall_x', 'wall_y', 'wall_neg_x', 'wall_neg_y')
DEFAULT_DESIRED_ANGULAR_VELOCITY = 3.0
DEFAULT_DESIRED_SPEED_ON_BRIDGE = 3.0
DEFAULT_DESIRED_SPEED_AFTER_BRIDGE = 1.0


def make_model(walls_and_target=False):
    xml_string = common.read_model('point_mass.xml')
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)
    # Remove walls, ball and target.
    if not walls_and_target:
        for wall in _WALLS:
            wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
            wall_geom.getparent().remove(wall_geom)

        # Remove target.
        target_site = xml_tools.find_element(mjcf, 'geom', 'target')
        target_site.getparent().remove(target_site)

    # Rename ground geom to floor
    mjcf.find(".//geom[@name='ground']").attrib['name'] = 'floor'

    # Unlimit the joints
    default_joint_element = mjcf.find('.//default').find(".//joint")
    default_joint_element.attrib['limited'] = 'false'
    default_joint_element.attrib.pop('range')

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()
def orbit_pond(time_limit=_DEFAULT_TIME_LIMIT,
               angular_velocity_reward_weight=1.0,
               random=None,
               environment_kwargs=None):
    """Returns the Orbit task."""
    environment_kwargs = environment_kwargs or {}
    pond_radius = environment_kwargs.get(
        'pond_radius', DEFAULT_POND_RADIUS * 0.1)
    pond_xy = environment_kwargs.get('pond_xy', DEFAULT_POND_XY)
    # base_model_string, assets = get_model_and_assets_common()
    base_model_string = make_model(walls_and_target=False)
    xml_string = make_pond_model(
        base_model_string,
        pond_radius=pond_radius,
        pond_xy=pond_xy)
    physics = PondPhysics.from_xml_string(xml_string, common.ASSETS)
    task = Orbit(
        desired_angular_velocity=DEFAULT_DESIRED_ANGULAR_VELOCITY,
        angular_velocity_reward_weight=angular_velocity_reward_weight,
        random=random)
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def bridge_run(time_limit=_DEFAULT_TIME_LIMIT,
               random=None,
               environment_kwargs=None):
    """Returns the BridgeRun task."""
    environment_kwargs = environment_kwargs or {}
    bridge_length = environment_kwargs.get(
        'bridge_length', bridge.DEFAULT_BRIDGE_LENGTH)
    bridge_width = environment_kwargs.get(
        'bridge_width', bridge.DEFAULT_BRIDGE_WIDTH)
    base_model_string = make_model(walls_and_target=False)
    xml_string = bridge.make_model(
        base_model_string,
        size_multiplier=0.1,
        bridge_length=bridge_length * 0.05,
        bridge_width=bridge_width * 0.01)
    physics = BridgeMovePhysics.from_xml_string(xml_string, common.ASSETS)
    task = BridgeMove(
        random=random,
        water_map_length=4 * 0.1,
        water_map_width=4 * 0.1,
        water_map_dx=0.5 * 0.1,
        water_map_dy=0.5 * 0.1,
        desired_speed_on_bridge=DEFAULT_DESIRED_SPEED_ON_BRIDGE,
        desired_speed_after_bridge=DEFAULT_DESIRED_SPEED_AFTER_BRIDGE)
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


def _common_observations(physics):
    observation = collections.OrderedDict((
        ('position', physics.position()),
        ('velocity', physics.velocity()),
    ))
    return observation


class PondPhysics(PondPhysicsMixin, PointMassPhysics):
    def torso_velocity(self):
        return self.velocity()

    def center_of_mass(self):
        return self.named.data.geom_xpos['pointmass']

    def orientation_to_pond(self):
        x = self.named.data.qpos['root_x']
        y = self.named.data.qpos['root_y']
        angle_to_pond_center = np.arctan2(y, x)
        sin_cos_encoded_angle_to_pond_center = np.array((
            np.sin(angle_to_pond_center),
            np.cos(angle_to_pond_center)))
        return sin_cos_encoded_angle_to_pond_center

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
        distance_from_pond = pond_radius / np.random.uniform(2.0, 10.0)
        distance_from_origin = pond_radius + distance_from_pond
        x = pond_center_x + distance_from_origin * np.cos(random_angle)
        y = pond_center_y + distance_from_origin * np.sin(random_angle)

        physics.named.data.qpos['root_x'] = x
        physics.named.data.qpos['root_y'] = y
        physics.named.data.geom_xpos['pointmass'][:2] = (x, y)

        return super(Orbit, self).initialize_episode(physics)


class BridgeMovePhysics(bridge.MovePhysicsMixin, PointMassPhysics):
    def torso_velocity(self):
        return self.velocity()

    def key_geom_positions(self):
        return self.named.data.xpos['pointmass'][np.newaxis, ...]

    def center_of_mass(self):
        return self.named.data.geom_xpos['pointmass']

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_bridge_move(self, *args, **kwargs)


class BridgeMove(bridge.MoveTaskMixin):
    def common_observations(self, physics):
        return _common_observations(physics)

    def upright_reward(self, physics):
        return 1.0

    def initialize_episode(self, physics):
        physics.named.data.qpos['root_x'] = 0.0
        physics.named.data.qpos['root_y'] = 0.0
        physics.named.data.geom_xpos['pointmass'][:2] = 0.0
        return super(BridgeMove, self).initialize_episode(physics)
