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
    DEFAULT_POND_XY,
    DEFAULT_POND_RADIUS,
    OrbitTaskMixin)
from . import bridge


@SUITE.add()
def orbit_pond(time_limit=_DEFAULT_TIME_LIMIT,
               angular_velocity_reward_weight=1.0,
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
        pond_xy=pond_xy)
    physics = PondPhysics.from_xml_string(xml_string, common.ASSETS)
    task = Orbit(
        desired_angular_velocity=_WALK_SPEED,
        angular_velocity_reward_weight=angular_velocity_reward_weight,
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
    base_model_string = make_common_model()
    xml_string = bridge.make_model(
        base_model_string,
        bridge_length=bridge_length,
        bridge_width=bridge_width)
    physics = BridgeMovePhysics.from_xml_string(xml_string, common.ASSETS)
    task = BridgeMove(
        random=random,
        desired_speed_on_bridge=_RUN_SPEED,
        desired_speed_after_bridge=_WALK_SPEED)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class PondPhysics(PondPhysicsMixin, QuadrupedPhysics):
    def center_of_mass(self):
        np.testing.assert_equal(
            self.named.data.xpos['torso'],
            self.named.data.geom_xpos['torso'])
        return self.named.data.geom_xpos['torso']

    def imu(self, *args, **kwargs):
        imu = super(PondPhysics, self).imu(*args, **kwargs)
        imu = 50.0 * np.tanh(imu / 100.0)
        return imu


class Orbit(OrbitTaskMixin):
    def common_observations(self, physics):
        return _common_observations(physics)

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
        orientation = self.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        distance_from_pond = pond_radius / np.random.uniform(2.0, 10.0)
        distance_from_origin = pond_radius + distance_from_pond
        x = pond_center_x + distance_from_origin * np.cos(random_angle)
        y = pond_center_y + distance_from_origin * np.sin(random_angle)
        _find_non_contacting_height(physics, orientation, x_pos=x, y_pos=y)

        return super(Orbit, self).initialize_episode(physics)


class BridgeMovePhysics(bridge.MovePhysicsMixin, QuadrupedPhysics):
    def key_geom_positions(self):
        toes = self.named.data.xpos[_TOES]
        return toes

    def center_of_mass(self):
        return self.named.data.geom_xpos['torso']

    def imu(self, *args, **kwargs):
        imu = super(PondPhysics, self).imu(*args, **kwargs)
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
