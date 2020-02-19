from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.quadruped import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    _RUN_SPEED,
    _WALK_SPEED,
    _TOES,
    SUITE,
    make_model as make_common_model,
    Physics,
    _find_non_contacting_height,
    _common_observations,
    _upright_reward)
from dm_control.utils import rewards


from lxml import etree
import numpy as np
from scipy.spatial.transform import Rotation

_DEFAULT_POND_RADIUS = 5
_DEFAULT_POND_XY = (0, 0)
_DEFAULT_BRIDGE_LENGTH = 10.0
_DEFAULT_BRIDGE_WIDTH = 2.0

_TODO_ANGULAR_VELOCITY = _WALK_SPEED


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), dtype=np.float64)


def make_pond_model(pond_radius, pond_xy=_DEFAULT_POND_XY, *args, **kwargs):
    common_model_string = make_common_model(*args, **kwargs)
    mjcf = etree.fromstring(common_model_string)

    worldbody = mjcf.find('worldbody')

    pond_element = etree.Element(
        "geom",
        type="cylinder",
        name="pond",
        pos=" ".join((str(x) for x in (*pond_xy, 0))),
        size=f"{pond_radius} 0.01",
        contype="96",
        conaffinity="66",
        rgba="0 0 1 1")
    worldbody.insert(0, pond_element)

    return etree.tostring(mjcf, pretty_print=True)


def make_bridge_model(bridge_length=_DEFAULT_BRIDGE_LENGTH,
                      bridge_width=_DEFAULT_BRIDGE_WIDTH,
                      *args,
                      **kwargs):
    floor_size = 15 + bridge_length
    common_model_string = make_common_model(
        floor_size=floor_size, *args, **kwargs)
    mjcf = etree.fromstring(common_model_string)
    worldbody = mjcf.find('worldbody')

    floor_geom = mjcf.find('.//geom[@name={!r}]'.format('floor'))
    floor_geom.attrib['size'] = '{} {} .5'.format(floor_size, floor_size)

    floor_x = floor_size - 2
    floor_geom.attrib['pos'] = f"{floor_x} 0 0"

    bridge_x = (bridge_length + 2) / 2
    water_width = floor_size - bridge_width / 2
    bridge_y_abs = (bridge_width + water_width) / 2

    left_water_element = etree.Element(
        "geom",
        type="box",
        name="water-left",
        pos=f"{bridge_x} {bridge_y_abs} 0",
        size=f"{bridge_length / 2} {water_width / 2} 0.01",
        contype="96",
        conaffinity="66",
        rgba="0 0 1 1")
    right_water_element = etree.Element(
        "geom",
        type="box",
        name="water-right",
        pos=f"{bridge_x} {-bridge_y_abs} 0",
        size=f"{bridge_length / 2} {water_width / 2} 0.01",
        contype="96",
        rgba="0 0 1 1")
    bridge_element = etree.Element(
        "geom",
        type="box",
        name="bridge",
        pos=f"{bridge_x} 0 0",
        contype="97",
        conaffinity="67",
        size=f"{bridge_length / 2} {bridge_width / 2} 0.01",
        rgba="0.247 0.165 0.078 1")
    grass_length = floor_size - (bridge_length + 2) / 2
    grass_width = floor_size
    grass_x = 1 + bridge_length + grass_length
    grass_element = etree.Element(
        "geom",
        type="box",
        name="grass",
        pos=f"{grass_x} 0 0",
        size=f"{grass_length} {grass_width} 0.01",
        contype="98",
        conaffinity="68",
        rgba="0 0.502 0 1")
    worldbody.insert(0, left_water_element)
    worldbody.insert(1, right_water_element)
    worldbody.insert(2, bridge_element)
    worldbody.insert(3, grass_element)

    size_element = etree.Element(
        "size",
        njmax="5000",
        nconmax="2000")

    # <size njmax="8000" nconmax="4000"/>
    mjcf.insert(0, size_element)

    water_map_length = 4
    water_map_width = 4
    water_map_dx = 0.5
    water_map_dy = 0.5

    for x in range(int(water_map_length / water_map_dx)):
        for y in range(int(water_map_width / water_map_dy)):
            water_map_cell_element = etree.Element(
                "geom",
                type="box",
                contype="99",
                conaffinity="69",
                name=f"water-map-{x}-{y}",
                pos=f"0 0 0.2",
                size=f"{water_map_dx / 2} {water_map_dy / 2} 0.01",
                rgba="0 0 0 1")
            worldbody.insert(-1, water_map_cell_element)

    return etree.tostring(mjcf, pretty_print=True)


class PondPhysics(Physics):
    @property
    def pond_radius(self):
        return self.named.model.geom_size['pond'][0]

    @property
    def pond_center_xyz(self):
        return self.named.data.geom_xpos['pond']

    def distances_from_pond_center(self):
        state = self.named.data.geom_xpos['torso'][:2]
        states = np.atleast_2d(state)
        pond_center = self.pond_center_xyz[:2]
        distances_from_pond_center = np.linalg.norm(
            states - pond_center, ord=2, keepdims=True, axis=-1)
        return distances_from_pond_center

    def distance_from_pond_center(self):
        distance_from_pond_center = self.distances_from_pond_center()[0]
        return distance_from_pond_center

    def distances_from_pond(self):
        distances_from_pond_center = self.distances_from_pond_center()
        distances_from_pond = distances_from_pond_center - self.pond_radius
        return distances_from_pond

    def distance_from_pond(self):
        distance_from_pond = self.distances_from_pond()[0]
        return distance_from_pond

    # def angular_velocities(self, positions1, positions2):
    def angular_velocities(self):
        velocity = self.torso_velocity()[:2]
        positions2 = self.named.data.xpos['torso'][:2][None]
        positions1 = positions2 - velocity

        positions1 = positions1 - self.pond_center_xyz[:2]
        positions2 = positions2 - self.pond_center_xyz[:2]
        angles1 = np.arctan2(positions1[..., 1], positions1[..., 0])
        angles2 = np.arctan2(positions2[..., 1], positions2[..., 0])
        angles = np.arctan2(
            np.sin(angles2 - angles1),
            np.cos(angles2 - angles1)
        )[..., np.newaxis]

        angular_velocities = angles * self.pond_radius

        return angular_velocities

    def angular_velocity(self):
        angular_velocity = self.angular_velocities()[0]
        return angular_velocity

    def orientation_to_pond(self):
        root_qpos = self.named.data.qpos['root']
        xyz, orientation_to_origin = root_qpos[:3], root_qpos[3:7]

        xy_from_pond_center = xyz[:2] - self.pond_center_xyz[:2]
        angle_to_pond_center = np.arctan2(*xy_from_pond_center[::-1])
        origin_to_pond_transformation = np.roll(
            Rotation.from_euler('z', angle_to_pond_center).as_quat(), 1)

        # NOTE(hartikainen): This makes the whole 2 * pi rotation around the
        # z-axis continuous.
        # TODO(hartikainen): Check if this has some negative side effects on
        # other rotation axes.
        if origin_to_pond_transformation[-1] < 0:
            origin_to_pond_transformation[-1] *= -1

        orientation_to_pond = quaternion_multiply(
            origin_to_pond_transformation, orientation_to_origin)

        return orientation_to_pond


class BridgeMovePhysics(Physics):
    def before_bridge(self):
        toes_x = self.named.data.xpos[_TOES][:, :1]
        #
        bridge_x = self.named.data.geom_xpos['bridge'][:1]
        bridge_length = self.named.model.geom_size['bridge'][:1]
        toes_after_bridge = toes_x < bridge_x - bridge_length

        after_bridge = np.all(toes_after_bridge)

        return after_bridge

    def after_bridge(self):
        toes_x = self.named.data.xpos[_TOES][:, :1]

        bridge_x = self.named.data.geom_xpos['bridge'][:1]
        bridge_length = self.named.model.geom_size['bridge'][:1]
        toes_after_bridge = bridge_x + bridge_length < toes_x

        after_bridge = np.all(toes_after_bridge)

        return after_bridge

    def on_bridge(self):
        toes_xy = self.named.data.xpos[_TOES][:, :2]

        bridge_xy = self.named.data.geom_xpos['bridge'][:2]
        bridge_size = self.named.model.geom_size['bridge'][:2]

        toes_on_bridge = point_inside_2d_rectangle(
            toes_xy, bridge_xy, bridge_size)

        on_bridge = np.all(toes_on_bridge)

        return on_bridge

    def any_toe_in_water(self):
        toes_xy = self.named.data.xpos[_TOES][:, :2]

        water_left_xy = self.named.model.geom_pos['water-left'][:2]
        water_left_size = self.named.model.geom_size['water-left'][:2]
        water_right_xy = self.named.model.geom_pos['water-right'][:2]
        water_right_size = self.named.model.geom_size['water-right'][:2]

        toes_in_waters = point_inside_2d_rectangle(
            toes_xy,
            np.stack((water_left_xy, water_right_xy)),
            np.stack((water_left_size, water_right_size)))

        assert toes_in_waters.shape == (4, 2), (
            toes_in_waters, toes_in_waters.shape)

        any_toe_in_water = np.any(toes_in_waters)

        return any_toe_in_water

    def water_map(self, length, width, dx, dy):
        """Create a water map around the egocentric view.

        Water map is a boolean array of shape (length / dx, width / dy)
        with element value `True` if it overlaps with water and value `False`
        otherwise. The water map is located around the agent, although not
        necessarily centered around it. There are more cells in the egocentric
        forward direction and fewer cells in the egocentric backward direction.
        """
        torso_x, torso_y = self.named.data.geom_xpos['torso'][:2]
        water_map_x = torso_x + length / 4
        water_map_y = torso_y
        water_map_xy = np.stack(np.meshgrid(
            np.arange(water_map_x - length / 2, water_map_x + length / 2, dx),
            np.arange(water_map_y - width / 2, water_map_y + width / 2, dy),
            indexing='ij',
        ), axis=-1)

        water_left_xy = self.named.model.geom_pos['water-left'][:2]
        water_left_size = self.named.model.geom_size['water-left'][:2]
        water_right_xy = self.named.model.geom_pos['water-right'][:2]
        water_right_size = self.named.model.geom_size['water-right'][:2]

        cell_centers = water_map_xy + (dx / 2, dy / 2)
        cells_in_waters = point_inside_2d_rectangle(
            cell_centers,
            np.stack((water_left_xy, water_right_xy)),
            np.stack((water_left_size, water_right_size)))

        water_map = np.any(cells_in_waters, axis=-1)

        return water_map


def point_inside_2d_rectangle(points, rectangle_positions, rectangle_sizes):
    points = np.atleast_2d(points)
    rectangle_positions = np.atleast_2d(rectangle_positions)
    rectangle_sizes = np.atleast_2d(rectangle_sizes)
    result_shape = (*points.shape[:-1], *rectangle_positions.shape[:-1])
    rectangles_top_right_xy = rectangle_positions + rectangle_sizes
    rectangles_bottom_left_xy = rectangle_positions - rectangle_sizes

    point_inside_2d_rectangle = np.logical_and(
        np.logical_and.reduce(
            points[..., None, :] <= rectangles_top_right_xy, axis=-1),
        np.logical_and.reduce(
            rectangles_bottom_left_xy <= points[..., None, :], axis=-1))

    assert point_inside_2d_rectangle.shape == (
        *points.shape[:-1], *rectangle_positions.shape[:-1]), (
        point_inside_2d_rectangle, point_inside_2d_rectangle.shape)

    return np.reshape(point_inside_2d_rectangle, result_shape)


@SUITE.add()
def orbit_pond(time_limit=_DEFAULT_TIME_LIMIT,
               random=None,
               environment_kwargs=None):
    """Returns the Orbit task."""
    environment_kwargs = environment_kwargs or {}
    pond_radius = environment_kwargs.get('pond_radius', _DEFAULT_POND_RADIUS)
    pond_xy = environment_kwargs.get('pond_xy', _DEFAULT_POND_XY)
    xml_string = make_pond_model(
        pond_radius=pond_radius,
        pond_xy=pond_xy,
        floor_size=pond_radius * 2)
    physics = PondPhysics.from_xml_string(xml_string, common.ASSETS)
    task = Orbit(
        desired_angular_velocity=_TODO_ANGULAR_VELOCITY, random=random)
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
        'bridge_length', _DEFAULT_BRIDGE_LENGTH)
    bridge_width = environment_kwargs.get(
        'bridge_width', _DEFAULT_BRIDGE_WIDTH)
    xml_string = make_bridge_model(
        bridge_length=bridge_length,
        bridge_width=bridge_width)
    physics = BridgeMovePhysics.from_xml_string(xml_string, common.ASSETS)
    task = BridgeMove(random=random)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class BridgeMove(base.Task):
    """A quadruped task solved by running across a bridge."""

    def __init__(self,
                 desired_speed_on_bridge=_RUN_SPEED,
                 desired_speed_after_bridge=_WALK_SPEED,
                 water_map_length=4,
                 water_map_width=4,
                 water_map_dx=0.5,
                 water_map_dy=0.5,
                 random=None):
        """Initializes an instance of `BridgeMove`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a
            seed automatically (default).
        """
        self._desired_speed_on_bridge = desired_speed_on_bridge
        self._desired_speed_after_bridge = desired_speed_after_bridge
        self._water_map_length = water_map_length
        self._water_map_width = water_map_width
        self._water_map_dx = water_map_dx
        self._water_map_dy = water_map_dy
        return super(BridgeMove, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        # Make the agent initially face tangent relative to pond.
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        _find_non_contacting_height(physics, orientation, x_pos=0)
        return super(BridgeMove, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        common_observations = _common_observations(physics)
        water_map = physics.water_map(
            length=self._water_map_length,
            width=self._water_map_width,
            dx=self._water_map_dx,
            dy=self._water_map_dy)
        pond_observations = type(common_observations)((
            *common_observations.items(),
            ('water_map', water_map),
        ))

        # for i in range(int(self._water_map_length / self._water_map_dx)):
        #     for j in range(int(self._water_map_width / self._water_map_dy)):
        #         cell_id = f'water-map-{i}-{j}'
        #         physics.named.data.geom_xpos[
        #             cell_id][:2] = water_map_xy[i, j] - (self._water_map_dx / 2, self._water_map_dy / 2)
        #         physics.named.model.geom_rgba[cell_id] = (
        #             (1, 0, 0, 1)
        #             if water_map[i, j]
        #             else (1, 1, 1, 0.5))

        return pond_observations

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        if physics.after_bridge():
            xy_velocity = np.linalg.norm(physics.torso_velocity(), ord=2)
            move_reward = rewards.tolerance(
                xy_velocity,
                bounds=(self._desired_speed_after_bridge, float('inf')),
                margin=self._desired_speed_after_bridge,
                value_at_margin=0.5,
                sigmoid='linear')
        else:
            x_velocity = physics.torso_velocity()[0]
            move_reward = rewards.tolerance(
                x_velocity,
                bounds=(self._desired_speed_on_bridge, float('inf')),
                margin=self._desired_speed_on_bridge,
                value_at_margin=0.5,
                sigmoid='linear')

        return _upright_reward(physics) * move_reward

    def get_termination(self, physics):
        """Terminates when the state norm is smaller than epsilon."""
        if physics.any_toe_in_water():
            return 0


class Orbit(base.Task):
    """A quadruped task to orbit around a pond with designated speed."""

    def __init__(self, desired_angular_velocity, random=None):
        """Initializes an instance of `Orbit`.

        Args:
          desired_angular_velocity: A float. If this value is zero, reward is
            given simply for standing upright. Otherwise this specifies the
            horizontal velocity at which the velocity-dependent reward
            component is maximized.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a
            seed automatically (default).
        """
        self._desired_angular_velocity = desired_angular_velocity
        return super(Orbit, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        # Make the agent initially face tangent relative to pond.
        orientation = np.roll(Rotation.from_euler('z', np.pi/2).as_quat(), 1)
        _find_non_contacting_height(physics, orientation, x_pos=6.0)
        return super(Orbit, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        common_observations = _common_observations(physics)

        orientation_to_pond = physics.orientation_to_pond()
        distance_from_pond = physics.distance_from_pond()

        pond_observations = type(common_observations)((
            *common_observations.items(),
            ('orientation_to_pond', orientation_to_pond),
            ('distance_from_pond', distance_from_pond),
        ))
        return pond_observations

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        angular_velocity_reward = rewards.tolerance(
            physics.angular_velocity(),
            bounds=(self._desired_angular_velocity, float('inf')),
            margin=self._desired_angular_velocity,
            value_at_margin=0.0,
            sigmoid='linear')

        return _upright_reward(physics) * angular_velocity_reward

    def get_termination(self, physics):
        """Terminates when the state norm is smaller than epsilon."""
        if physics.distance_from_pond() < 0:
            return 0
