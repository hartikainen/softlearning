from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.quadruped import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    _RUN_SPEED,
    _WALK_SPEED,
    _TOES,
    SUITE,
    make_model as make_common_model,
    Physics as QuadrupedPhysics,
    _find_non_contacting_height,
    _common_observations,
    _upright_reward)


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
    # 'eye_r',
    # 'eye_l',
    'torso',
    'thigh_front_left',
    'shin_front_left',
    'foot_front_left',
    'toe_front_left',
    'thigh_front_right',
    'shin_front_right',
    'foot_front_right',
    'toe_front_right',
    'thigh_back_right',
    'shin_back_right',
    'foot_back_right',
    'toe_back_right',
    'thigh_back_left',
    'shin_back_left',
    'foot_back_left',
    'toe_back_left',
]


@SUITE.add()
def orbit_pond(time_limit=_DEFAULT_TIME_LIMIT,
               angular_velocity_reward_weight=1.0,
               control_range_multiplier=None,
               friction=None,
               random=None,
               environment_kwargs=None):
    """Returns the Orbit task."""
    environment_kwargs = environment_kwargs or {}
    pond_radius = environment_kwargs.get('pond_radius', DEFAULT_POND_RADIUS)
    pond_xy = environment_kwargs.get('pond_xy', DEFAULT_POND_XY)
    base_model_string = make_common_model(floor_size=pond_radius * 2)
    xml_string = make_pond_model(
        base_model_string,
        pond_radius=pond_radius,
        pond_xy=pond_xy,
        control_range_multiplier=control_range_multiplier,
        friction=friction)
    physics = PondPhysics.from_xml_string(xml_string, common.ASSETS)
    task = Orbit(
        desired_angular_velocity=_WALK_SPEED,
        angular_velocity_reward_weight=angular_velocity_reward_weight,
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
                        bridge_offset=-1.0,
                        on_bridge_reward_type='x_velocity',
                        on_bridge_reward_weight=5.0,
                        after_bridge_reward_type='constant',
                        after_bridge_reward_weight=5.0,
                        terminate_outside_of_reward_bounds=False,
                        environment_kwargs=None):
    """Returns the BridgeRun task."""
    environment_kwargs = environment_kwargs or {}
    base_model_string = make_common_model()

    water_map_length = 4
    water_map_width = 4
    water_map_dx = 0.25
    water_map_dy = 0.25

    xml_string = bridge.make_model(
        base_model_string,
        bridge_length=bridge_length,
        bridge_start_width=bridge_start_width,
        bridge_end_width=bridge_end_width,
        bridge_offset=bridge_offset,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy)
    physics = BridgeMovePhysics.from_xml_string(xml_string, common.ASSETS)
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
        terminate_outside_of_reward_bounds=terminate_outside_of_reward_bounds,
        desired_speed_on_bridge=_WALK_SPEED,
        desired_speed_after_bridge=_WALK_SPEED)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class PondPhysics(PondPhysicsMixin, QuadrupedPhysics):
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
        toes = self.named.data.xpos[_TOES]
        return toes

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
    def common_observations(self, physics):
        common_observations = _common_observations(physics)
        common_observations['position'] = physics.center_of_mass()
        return common_observations

    def upright_reward(self, physics):
        return _upright_reward(physics)

    def initialize_episode(self, physics):
        # Make the agent initially face tangent relative to pond.
        pond_radius = physics.named.model.geom_size['pond'][0]
        pond_center_x, pond_center_y = physics.pond_center_xyz[:2]

        random_angle = np.random.uniform(0, 2 * np.pi)
        # orientation = np.roll((
        #     Rotation.from_euler('z', random_angle)
        #     * Rotation.from_euler('z', np.pi / 2)
        # ).as_quat(), 1)
        # Initial configuration.
        orientation = np.roll(Rotation.from_euler('z', np.pi/2).as_quat(), 1)

        rotate_by_angle_quaternion = np.roll(
            Rotation.from_euler('z', random_angle).as_quat(), 1)
        orientation = quaternion_multiply(
            rotate_by_angle_quaternion, orientation)

        distance_from_pond = np.maximum(
            0.75, pond_radius / np.random.uniform(4.0, 6.0))

        distance_from_origin = pond_radius + distance_from_pond
        x = pond_center_x + distance_from_origin * np.cos(random_angle)
        y = pond_center_y + distance_from_origin * np.sin(random_angle)
        _find_non_contacting_height(physics, orientation, x_pos=x, y_pos=y)

        return super(Orbit, self).initialize_episode(physics)


class BridgeMovePhysics(bridge.MovePhysicsMixin, QuadrupedPhysics):
    @property
    def agent_geom_ids(self):
        if self._agent_geom_ids is None:
            self._agent_geom_ids = np.array([
                self.model.name2id(geom_name, 'geom')
                for geom_name in KEY_GEOM_NAMES
            ])
        return self._agent_geom_ids

    def key_geom_positions(self):
        toes = self.named.data.xpos[_TOES]
        return toes

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
        _find_non_contacting_height(physics, orientation, x_pos=0)
        return super(BridgeMove, self).initialize_episode(physics)
