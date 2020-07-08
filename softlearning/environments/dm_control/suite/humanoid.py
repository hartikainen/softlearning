import collections

from dm_control.utils import rewards
from dm_control.rl import control
from dm_control.suite.humanoid import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    _RUN_SPEED,
    _WALK_SPEED,
    _STAND_HEIGHT,
    SUITE,
    get_model_and_assets as get_model_and_assets_common,
    Physics as HumanoidPhysics,
)

from dm_control.suite.quadruped import _find_non_contacting_height


import numpy as np
from scipy.spatial.transform import Rotation
from .pond import (
    PondPhysicsMixin,
    make_pond_model,
    quaternion_multiply,
    DEFAULT_POND_XY,
    DEFAULT_POND_RADIUS,
    OrbitTaskMixin)
from . import bridge, visualization


KEY_GEOM_NAMES = [
    # 'torso',
    # 'head',
    # 'lower_waist',
    # 'pelvis',
    # 'right_thigh',
    # 'right_shin',
    'right_foot',
    # 'left_thigh',
    # 'left_shin',
    'left_foot',
    # 'right_upper_arm',
    # 'right_lower_arm',
    # 'right_hand',
    # 'left_upper_arm',
    # 'left_lower_arm',
    # 'left_hand',
]


def _common_observations(physics):
    com_velocity = physics.center_of_mass_velocity()
    com_velocity = np.concatenate((
        _localize_xy_value(physics, com_velocity[:2]),
        com_velocity[2:],
    ))

    velocity = physics.velocity()
    velocity = np.concatenate((
        _localize_xy_value(physics, velocity[:2]), velocity[2:],
    ))

    common_observations = collections.OrderedDict((
        ('joint_angles', physics.joint_angles()),
        ('head_height', physics.head_height()),
        ('extremities', physics.extremities()),
        ('torso_vertical', physics.torso_vertical_orientation()),
        ('com_velocity', com_velocity),
        ('velocity', velocity),
    ))

    return common_observations


def _upright_reward(physics):
    standing = rewards.tolerance(physics.head_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)
    upright = rewards.tolerance(physics.torso_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
    reward = standing * upright
    return reward


def _localize_xy_value(physics, value):
    assert value.shape == (2, )
    xy_from_pond_center = (
        physics.center_of_mass_position()[:2]
        - physics.pond_center_xyz[:2])

    angle_to_pond_center = np.arctan2(*xy_from_pond_center[::-1])
    sin_cos_angle_to_pond = np.array((
            np.sin(angle_to_pond_center),
            np.cos(angle_to_pond_center)))
    angle_to_pond = np.arctan2(*sin_cos_angle_to_pond)

    rotation_matrix = np.array((
        (np.cos(angle_to_pond), np.sin(angle_to_pond)),
        (-np.sin(angle_to_pond), np.cos(angle_to_pond)),
    ))

    rotated_value = rotation_matrix @ value

    return rotated_value


@SUITE.add()
def orbit_pond(time_limit=_DEFAULT_TIME_LIMIT,
               angular_velocity_reward_weight=1.0,
               default_quaternion=None,
               random=None,
               environment_kwargs=None):
    """Returns the Orbit task."""
    environment_kwargs = environment_kwargs or {}
    pond_radius = environment_kwargs.get('pond_radius', DEFAULT_POND_RADIUS)
    pond_xy = environment_kwargs.get('pond_xy', DEFAULT_POND_XY)
    base_model_string, common_assets = get_model_and_assets_common()
    # base_model_string = make_common_model(floor_size=pond_radius * 2)
    xml_string = make_pond_model(
        base_model_string,
        pond_radius=pond_radius,
        pond_xy=pond_xy)
    physics = PondPhysics.from_xml_string(xml_string, common_assets)
    task = Orbit(
        desired_angular_velocity=_WALK_SPEED,
        angular_velocity_reward_weight=angular_velocity_reward_weight,
        default_quaternion=default_quaternion,
        random=random)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


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
    base_model_string, common_assets = get_model_and_assets_common()
    xml_string = bridge.make_model(
        base_model_string,
        bridge_length=bridge_length,
        bridge_width=bridge_width)
    physics = BridgeMovePhysics.from_xml_string(xml_string, common_assets)
    task = BridgeMove(
        random=random,
        desired_speed_on_bridge=_RUN_SPEED,
        desired_speed_after_bridge=_WALK_SPEED)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class PondPhysics(PondPhysicsMixin, HumanoidPhysics):
    def key_geom_positions(self):
        key_geom_positions = self.named.data.xpos[KEY_GEOM_NAMES]
        return key_geom_positions

    def center_of_mass(self):
        np.testing.assert_equal(
            self.named.data.xpos['torso'],
            self.named.data.geom_xpos['torso'])
        np.testing.assert_equal(
            self.named.data.xpos['torso'],
            self.named.data.qpos['root'][:3])
        return self.named.data.geom_xpos['torso'].copy()

    def imu(self, *args, **kwargs):
        imu = super(PondPhysics, self).imu(*args, **kwargs)
        imu = 50.0 * np.tanh(imu / 100.0)
        return imu

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_orbit_pond(self, *args, **kwargs)


class Orbit(OrbitTaskMixin):
    def __init__(self, default_quaternion, *args, **kwargs):
        self._default_quaternion = default_quaternion
        return super(Orbit, self).__init__(*args, **kwargs)

    def common_observations(self, physics):
        common_observations = _common_observations(physics)
        common_observations['position'] = physics.center_of_mass()
        return common_observations

    def upright_reward(self, physics):
        upright_reward = _upright_reward(physics)
        return upright_reward

    def initialize_episode(self, physics):
        # Make the agent initially face tangent relative to pond.
        pond_radius = physics.named.model.geom_size['pond'][0].copy()
        pond_center_x, pond_center_y = physics.pond_center_xyz[:2]

        random_angle = np.random.uniform(0, 2 * np.pi)
        orientation = np.array(self._default_quaternion)
        orientation = np.roll(Rotation.from_euler('z', np.pi/2).as_quat(), 1)
        orientation /= np.linalg.norm(orientation)

        rotate_by_angle_quaternion = np.roll(
            Rotation.from_euler('z', random_angle).as_quat(), 1)
        orientation = quaternion_multiply(
            rotate_by_angle_quaternion, orientation)

        distance_from_pond = pond_radius / np.random.uniform(5.0, 10.0)
        distance_from_origin = pond_radius + distance_from_pond
        x = pond_center_x + distance_from_origin * np.cos(random_angle)
        y = pond_center_y + distance_from_origin * np.sin(random_angle)
        _find_non_contacting_height(physics, orientation, x_pos=x, y_pos=y)

        return super(Orbit, self).initialize_episode(physics)

    def get_termination(self, physics):
        fell_over = physics.head_height() < _STAND_HEIGHT / 3
        return fell_over or super(Orbit, self).get_termination(physics)


class BridgeMovePhysics(bridge.MovePhysicsMixin, HumanoidPhysics):
    def any_key_geom_in_water(self):
        raise NotImplementedError("TODO(hartikainen)")
        key_geoms_xy = self.key_geom_positions()[..., :2]
        key_geoms_z = self.key_geom_positions()[..., 2:3]

        pond_center = self.pond_center_xyz[:2]
        pond_radius = self.pond_radius

        any_key_geom_in_water = np.any(np.logical_and(
            TODO,
            key_geoms_z < 0.1,
        ))

        return any_key_geom_in_water

    def key_geom_positions(self):
        key_geom_positions = self.named.data.xpos[KEY_GEOM_NAMES]
        return key_geom_positions

    def center_of_mass(self):
        return self.named.data.geom_xpos['torso'].copy()

    def torso_xmat(self):
        return self.named.data.geom_xmat['torso'].copy()

    def imu(self, *args, **kwargs):
        imu = super(BridgeMovePhysics, self).imu(*args, **kwargs)
        imu = 50.0 * np.tanh(imu / 100.0)
        return imu


class BridgeMove(bridge.MoveTaskMixin):
    def common_observations(self, physics):
        return _common_observations(physics)

    def upright_reward(self, physics):
        return _upright_reward(physics)

    def initialize_episode(self, physics):
        # Make the agent initially face tangent relative to pond.
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        _find_non_contacting_height(physics, orientation, x_pos=0)
        return super(BridgeMove, self).initialize_episode(physics)
