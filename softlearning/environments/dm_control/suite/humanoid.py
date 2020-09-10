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
    Humanoid as HumanoidTask,
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
    # 'pond',
    # 'floor',
    'torso',
    'upper_waist',
    'head',
    'lower_waist',
    'butt',
    'right_thigh',
    'right_shin',
    'right_right_foot',
    'left_right_foot',
    'left_thigh',
    'left_shin',
    'left_left_foot',
    'right_left_foot',
    'right_upper_arm',
    'right_lower_arm',
    'right_hand',
    'left_upper_arm',
    'left_lower_arm',
    'left_hand',
]


def _common_observations(physics):
    com_velocity = physics.center_of_mass_velocity()
    velocity = physics.velocity()

    if isinstance(physics, PondPhysics):
        com_velocity = np.concatenate((
            _localize_xy_value(physics, com_velocity[:2]), com_velocity[2:],
        ))
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


class SimpleResetHumanoid(HumanoidTask):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        # Find a collision-free random initial configuration.
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        _find_non_contacting_height(physics, orientation)
        return super(HumanoidTask, self).initialize_episode(physics)


@SUITE.add()
def custom_stand(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the Stand task."""
    physics = HumanoidPhysics.from_xml_string(*get_model_and_assets_common())
    task = SimpleResetHumanoid(move_speed=0, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


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
def bridge_run(*args, bridge_width=bridge.DEFAULT_BRIDGE_WIDTH, **kwargs):
    """Returns the BridgeRun task."""
    return tapering_bridge_run(
        *args,
        bridge_start_width=bridge_width,
        bridge_end_width=bridge_width,
        **kwargs)


@SUITE.add()
def tapering_bridge_run(time_limit=_DEFAULT_TIME_LIMIT,
                        random=None,
                        bridge_length=bridge.DEFAULT_BRIDGE_LENGTH,
                        bridge_start_width=bridge.DEFAULT_BRIDGE_WIDTH,
                        bridge_end_width=0.0,
                        on_bridge_reward_type='x_velocity',
                        on_bridge_reward_weight=5.0,
                        after_bridge_reward_type='constant',
                        after_bridge_reward_weight=5.0,
                        desired_speed_on_bridge=_WALK_SPEED,
                        desired_speed_after_bridge=_WALK_SPEED,
                        terminate_outside_of_reward_bounds=False,
                        randomize_initial_x_position=False,
                        environment_kwargs=None):
    """Returns the BridgeRun task."""
    environment_kwargs = environment_kwargs or {}
    base_model_string, common_assets = get_model_and_assets_common()
    size_multiplier = 1.0
    water_map_length = 4 * size_multiplier
    water_map_width = 4 * size_multiplier
    water_map_dx = 0.5 * size_multiplier / 2
    water_map_dy = 0.5 * size_multiplier / 2
    xml_string = bridge.make_model(
        base_model_string,
        size_multiplier=size_multiplier,
        bridge_length=bridge_length * size_multiplier,
        bridge_start_width=bridge_start_width * size_multiplier,
        bridge_end_width=bridge_end_width * size_multiplier,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy,
    )
    physics = BridgeMovePhysics.from_xml_string(
        xml_string, common_assets)
    task = BridgeMove(
        random=random,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy,
        water_map_offset=(water_map_length / 4, 0.0),
        on_bridge_reward_type=on_bridge_reward_type,
        on_bridge_reward_weight=on_bridge_reward_weight,
        after_bridge_reward_type=after_bridge_reward_type,
        after_bridge_reward_weight=after_bridge_reward_weight,
        desired_speed_on_bridge=desired_speed_on_bridge,
        desired_speed_after_bridge=desired_speed_after_bridge,
        terminate_outside_of_reward_bounds=terminate_outside_of_reward_bounds,
        randomize_initial_x_position=randomize_initial_x_position)

    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class PondPhysics(PondPhysicsMixin, HumanoidPhysics):
    @property
    def agent_geom_ids(self):
        if self._agent_geom_ids is None:
            self._agent_geom_ids = np.array([
                self.model.name2id(geom_name, 'geom')
                for geom_name in KEY_GEOM_NAMES
            ])
        return self._agent_geom_ids

    def any_key_geom_in_water(self):
        floor_geom_id = self.floor_geom_id
        agent_geom_ids = self.agent_geom_ids

        contacts = self.data.contact
        if contacts.size == 0:
            return False

        water_contacts_index = (
            (np.isin(contacts.geom1, floor_geom_id)
             & np.isin(contacts.geom2, agent_geom_ids))
            | (np.isin(contacts.geom2, floor_geom_id)
               & np.isin(contacts.geom1, agent_geom_ids)))
        water_contacts = contacts[np.where(water_contacts_index)]

        key_geoms_xy = water_contacts.pos[:, :2]

        pond_center = self.pond_center_xyz[:2]
        pond_radius = self.pond_radius

        key_geoms_distance_to_pond_center = np.linalg.norm(
            key_geoms_xy - pond_center,
            ord=2,
            keepdims=True,
            axis=-1)

        any_key_geom_in_water = np.any(
            key_geoms_distance_to_pond_center < pond_radius)

        return any_key_geom_in_water

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


class BridgeMovePhysics(
        bridge.MovePhysicsMixin, HumanoidPhysics):
    def __init__(self, *args,  **kwargs):
        self._agent_geom_ids = None
        self._fell_over_geom_ids = None
        return super(BridgeMovePhysics, self).__init__(*args, **kwargs)

    @property
    def agent_geom_ids(self):
        if self._agent_geom_ids is None:
            self._agent_geom_ids = np.array([
                self.model.name2id(geom_name, 'geom')
                for geom_name in KEY_GEOM_NAMES
            ])
        return self._agent_geom_ids

    @property
    def fell_over_geom_ids(self):
        if self._fell_over_geom_ids is None:
            self._fell_over_geom_ids = np.array([
                self.model.name2id(geom_name, 'geom')
                for geom_name in KEY_GEOM_NAMES
                if 'foot' not in geom_name
            ])
        return self._fell_over_geom_ids

    def torso_velocity(self, *args, **kwargs):
        return self.center_of_mass_velocity(*args, **kwargs)

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

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_bridge_move(self, *args, **kwargs)

    def fell_over(self):
        floor_geom_id = self.floor_geom_id
        agent_geom_ids = self.fell_over_geom_ids

        # breakpoint()

        contacts = self.data.contact
        if contacts.size == 0:
            return False

        floor_contacts_index = (
            (np.isin(contacts.geom1, floor_geom_id)
             & np.isin(contacts.geom2, agent_geom_ids))
            | (np.isin(contacts.geom2, floor_geom_id)
               & np.isin(contacts.geom1, agent_geom_ids)))
        any_agent_geom_contact = np.any(floor_contacts_index)
        return any_agent_geom_contact


class BridgeMove(bridge.MoveTaskMixin):
    def common_observations(self, physics):
        common_observations = _common_observations(physics)
        common_observations['position'] = physics.center_of_mass()
        return common_observations

    def upright_reward(self, physics):
        return _upright_reward(physics)

    def initialize_episode(self, physics):
        # Make the agent initially face tangent relative to pond.
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        bridge_pos = physics.named.model.geom_pos['bridge']
        bridge_size = physics.named.model.geom_size['bridge']
        np.testing.assert_equal(
            physics.named.model.geom_quat['bridge'], (1, 0, 0, 0))

        if self._randomize_initial_x_position:
            x_pos_max = (
                bridge_pos[0]
                + bridge_size[0]
                - np.atleast_1d(self._water_map_offset)[0])
            x_pos = np.random.uniform(0, x_pos_max)
        else:
            x_pos = 0.0

        _find_non_contacting_height(physics, orientation, x_pos=x_pos)
        return super(BridgeMove, self).initialize_episode(physics)

    def get_termination(self, physics):
        super_termination = super(BridgeMove, self).get_termination(physics)
        fell_over = physics.fell_over()
        terminated = fell_over or super_termination
        return terminated
