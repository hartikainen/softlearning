import collections

import numpy as np
from scipy.spatial.transform import Rotation
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.point_mass import (
    _DEFAULT_TIME_LIMIT,
    Physics as PointMassPhysics)

from softlearning.environments.dm_control.suite.pond import (
    PondPhysicsMixin,
    make_pond_model,
    quaternion_multiply,
    DEFAULT_POND_XY,
    DEFAULT_POND_RADIUS,
    OrbitTaskMixin)
from softlearning.environments.dm_control.suite import visualization

from .common import make_model, KEY_GEOM_NAMES, SUITE


DEFAULT_DESIRED_ANGULAR_VELOCITY = 3.0


@SUITE.add()
def orbit_pond(time_limit=_DEFAULT_TIME_LIMIT,
               angular_velocity_reward_weight=1.0,
               make_1d=False,
               actuator_type='motor',
               random=None,
               environment_kwargs=None):
    """Returns the Orbit task."""
    environment_kwargs = environment_kwargs or {}
    pond_radius = environment_kwargs.get(
        'pond_radius', DEFAULT_POND_RADIUS * 0.05)
    pond_xy = environment_kwargs.get('pond_xy', DEFAULT_POND_XY)
    # base_model_string, assets = get_model_and_assets_common()
    size_multiplier = 0.05
    base_model_string = make_model(
        walls_and_target=False, actuator_type=actuator_type)

    water_map_length = 2 * size_multiplier
    water_map_width = 2 * size_multiplier
    water_map_dx = 0.2 * size_multiplier / 2
    water_map_dy = 0.2 * size_multiplier / 2

    xml_string = make_pond_model(
        base_model_string,
        pond_radius=pond_radius,
        pond_xy=pond_xy,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy,
    )
    physics = PondPhysics.from_xml_string(xml_string, common.ASSETS)
    task = Orbit(
        desired_angular_velocity=DEFAULT_DESIRED_ANGULAR_VELOCITY,
        angular_velocity_reward_weight=angular_velocity_reward_weight,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy,
        water_map_offset=0,
        make_1d=make_1d,
        random=random)
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class PondPhysics(PondPhysicsMixin, PointMassPhysics):
    @property
    def agent_geom_ids(self):
        if self._agent_geom_ids is None:
            self._agent_geom_ids = np.array([
                self.model.name2id(geom_name, 'geom')
                for geom_name in KEY_GEOM_NAMES
            ])
        return self._agent_geom_ids

    def velocity_to_pond(self):
        velocity = self.velocity()
        sin_cos_angle_to_pond = self.sin_cos_angle_to_pond()
        angle_to_pond = np.arctan2(*sin_cos_angle_to_pond)

        rotation_matrix = np.array((
            (np.cos(angle_to_pond), -np.sin(angle_to_pond)),
            (np.sin(angle_to_pond), np.cos(angle_to_pond)),
        )).T

        rotated_velocity = rotation_matrix @ velocity
        return rotated_velocity

    def torso_velocity(self):
        torso_velocity = self.named.data.sensordata['sensor_torso_vel'].copy()
        assert np.all(torso_velocity[:2] == self.velocity())
        return torso_velocity

    def global_velocity(self, *args, **kwargs):
        return self.torso_velocity(*args, **kwargs)

    def center_of_mass(self):
        return self.named.data.geom_xpos['pointmass']

    def orientation(self):
        orientation = np.roll(Rotation.from_matrix(
            self.named.data.xmat['pointmass'].reshape(3, 3)
        ).as_quat(), 1)
        return orientation

    def angle_to_pond(self):
        xy_from_pond_center = self.position()[:2] - self.pond_center_xyz[:2]
        angle_to_pond_center = np.arctan2(*xy_from_pond_center[::-1])
        return angle_to_pond_center

    def sin_cos_angle_to_pond(self):
        angle_to_pond_center = self.angle_to_pond()

        sin_cos_encoded_angle_to_pond_center = np.array((
            np.sin(angle_to_pond_center),
            np.cos(angle_to_pond_center)))
        return sin_cos_encoded_angle_to_pond_center

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

        return self.sin_cos_angle_to_pond()
        return orientation_to_pond

    def any_key_geom_in_water(self):
        point_in_water = self.distance_from_pond() < 0.0
        return point_in_water

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_orbit_pond(self, *args, **kwargs)

    def _get_orientation(self):
        return self.angle_to_pond()


class Orbit(OrbitTaskMixin):
    def __init__(self, *args, make_1d=False, **kwargs):
        self._make_1d = make_1d
        return super(Orbit, self).__init__(*args, **kwargs)

    def common_observations(self, physics):
        observation = collections.OrderedDict((
            ('position', physics.position()),
            # ('velocity', physics.velocity()),
            ('velocity', physics.velocity_to_pond()),
        ))
        return observation

    def upright_reward(self, physics):
        return 1.0

    def initialize_episode(self, physics):
        pond_radius = physics.named.model.geom_size['pond'][0]
        pond_center_x, pond_center_y = physics.pond_center_xyz[:2]

        random_angle = np.random.uniform(0, 2 * np.pi)
        distance_from_pond = pond_radius / 5.0
        distance_from_origin = pond_radius + distance_from_pond
        x = pond_center_x + distance_from_origin * np.cos(random_angle)
        y = pond_center_y + distance_from_origin * np.sin(random_angle)

        physics.named.data.qpos['root_x'] = x
        physics.named.data.qpos['root_y'] = y
        physics.named.data.geom_xpos['pointmass'][:2] = (x, y)

        return super(Orbit, self).initialize_episode(physics)

    def before_step(self, action, physics):
        if self._make_1d:
            action = np.array((np.sum(action), 1.0 / 5.0))

        sin_cos_angle_to_pond = physics.sin_cos_angle_to_pond()
        angle_to_pond = np.arctan2(*sin_cos_angle_to_pond)

        rotation_matrix = np.array((
            (np.cos(angle_to_pond), -np.sin(angle_to_pond)),
            (np.sin(angle_to_pond), np.cos(angle_to_pond)),
        ))

        rotated_action = rotation_matrix @ action

        return super(Orbit, self).before_step(
            rotated_action, physics)
