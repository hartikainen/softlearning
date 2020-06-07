import abc
import numpy as np
from lxml import etree
from dm_control.suite import base
from dm_control.utils import rewards
import skimage.measure

from scipy.spatial.transform import Rotation

DEFAULT_BRIDGE_LENGTH = 10.0
DEFAULT_BRIDGE_WIDTH = 2.0


def stringify(value):
    return ' '.join(np.array(value).astype(str))


def rotate_around_z(positions, quaternion):
    z_rotation = Rotation.from_quat(
        np.roll(quaternion, -1)
    ).as_euler('xyz', degrees=False)[-1]

    sin = np.sin(z_rotation)
    cos = np.cos(z_rotation)

    M = np.array(((cos, -sin), (sin, cos)))
    new_positions = positions @ M.T

    return new_positions


def make_model(base_model_string,
               size_multiplier=1.0,
               bridge_length=DEFAULT_BRIDGE_LENGTH,
               bridge_width=DEFAULT_BRIDGE_WIDTH,
               floor_geom_name='floor',
               *args,
               **kwargs):
    size_multiplier = np.array(size_multiplier)

    mjcf = etree.fromstring(base_model_string)
    worldbody = mjcf.find('worldbody')

    floor_geom = mjcf.find(".//geom[@name='floor']")
    floor_size = float(size_multiplier * 15 + bridge_length)
    floor_geom.attrib['size'] = f'{floor_size} {floor_size} .1'

    before_bridge_length = 0 * size_multiplier

    bridge_offset = -1.0 * size_multiplier

    floor_x = floor_size - before_bridge_length + bridge_offset
    floor_geom.attrib['pos'] = f"{floor_x} 0 0"

    bridge_x = (bridge_length + before_bridge_length) / 2 + bridge_offset
    water_width = floor_size - bridge_width / 2
    bridge_y_abs = (bridge_width + water_width) / 2

    left_water_element = etree.Element(
        "geom",
        type="box",
        name="water-left",
        pos=stringify((bridge_x, bridge_y_abs, 0)),
        size=stringify((
            bridge_length / 2, water_width / 2, 0.01)),
        contype="96",
        conaffinity="66",
        rgba="0 0 1 1")
    right_water_element = etree.Element(
        "geom",
        type="box",
        name="water-right",
        pos=stringify((bridge_x, -bridge_y_abs, 0)),
        size=stringify((bridge_length / 2, water_width / 2, 0.01)),
        contype="96",
        rgba="0 0 1 1")
    bridge_element = etree.Element(
        "geom",
        type="box",
        name="bridge",
        pos=stringify((bridge_x, 0, 0)),
        size=stringify((bridge_length / 2, bridge_width / 2, 0.01)),
        contype="97",
        conaffinity="67",
        # rgba="0.247 0.165 0.078 1",
        rgba="0 0 0 0",
    )
    # grass_length = floor_size - (bridge_length + before_bridge_length) / 2
    # grass_width = floor_size
    # grass_x = before_bridge_length / 2 + bridge_length + grass_length + bridge_offset
    # grass_element = etree.Element(
    #     "geom",
    #     type="box",
    #     name="grass",
    #     pos=stringify((grass_x, 0, 0)),
    #     size=stringify((grass_length, grass_width, 0.01)),
    #     contype="98",
    #     conaffinity="68",
    #     rgba="0 0.502 0 1")
    worldbody.insert(0, left_water_element)
    worldbody.insert(1, right_water_element)
    worldbody.insert(2, bridge_element)
    # worldbody.insert(3, grass_element)

    size_element = etree.Element(
        "size",
        njmax="5000",
        nconmax="2000")

    mjcf.insert(0, size_element)

    water_map_length = 5 * size_multiplier
    water_map_width = 5 * size_multiplier
    water_map_dx = 0.5 * size_multiplier
    water_map_dy = 0.5 * size_multiplier

    for x in range(int(water_map_length / water_map_dx)):
        for y in range(int(water_map_width / water_map_dy)):
            water_map_cell_element = etree.Element(
                "geom",
                type="box",
                contype="99",
                conaffinity="69",
                name=f"water-map-{x}-{y}",
                pos="0 0 0.1",
                size=stringify((water_map_dx, water_map_dy, 0.01)),
                rgba="0 0 0 1")
            worldbody.insert(-1, water_map_cell_element)

    return etree.tostring(mjcf, pretty_print=True)


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


class MovePhysicsMixin:
    def before_bridge(self):
        key_geoms_x = self.key_geom_positions()[..., :1]
        bridge_x = self.named.data.geom_xpos['bridge'][:1]
        bridge_length = self.named.model.geom_size['bridge'][:1]
        toes_before_bridge = key_geoms_x < bridge_x - bridge_length

        before_bridge = np.all(toes_before_bridge)

        return before_bridge

    def after_bridge(self):
        key_geoms_x = self.key_geom_positions()[..., :1]

        bridge_x = self.named.data.geom_xpos['bridge'][:1]
        bridge_length = self.named.model.geom_size['bridge'][:1]
        toes_after_bridge = bridge_x + bridge_length < key_geoms_x

        after_bridge = np.any(toes_after_bridge)

        return after_bridge

    def on_bridge(self):
        key_geoms_xy = self.key_geom_positions()[..., :2]

        bridge_xy = self.named.data.geom_xpos['bridge'][:2]
        bridge_size = self.named.model.geom_size['bridge'][:2]

        toes_on_bridge = point_inside_2d_rectangle(
            key_geoms_xy, bridge_xy, bridge_size)

        on_bridge = np.any(toes_on_bridge)

        return on_bridge

    def any_key_geom_in_water(self):
        key_geoms_xy = self.key_geom_positions()[..., :2]

        water_left_xy = self.named.model.geom_pos['water-left'][:2]
        water_left_size = self.named.model.geom_size['water-left'][:2]
        water_right_xy = self.named.model.geom_pos['water-right'][:2]
        water_right_size = self.named.model.geom_size['water-right'][:2]

        key_geoms_in_waters = point_inside_2d_rectangle(
            key_geoms_xy,
            np.stack((water_left_xy, water_right_xy)),
            np.stack((water_left_size, water_right_size)))

        assert key_geoms_in_waters.shape == (
            *key_geoms_xy.shape[:-1], 2), (
                key_geoms_in_waters, key_geoms_in_waters.shape)

        any_key_geom_in_water = np.any(key_geoms_in_waters)

        return any_key_geom_in_water

    def _get_orientation(self):
        return self.named.data.qpos['root'][3:]

    def water_map(self, length, width, dx, dy, density=10):
        """Create a water map around the egocentric view.

        Water map is a float array of shape (length / dx, width / dy)
        with element in [0, 1] representing the proportion of the are of
        the cell in water. The water map is located around the agent, although
        not necessarily centered around it. There are more cells in the
        egocentric forward direction and fewer cells in the egocentric backward
        direction.
        """
        com_x, com_y = self.center_of_mass()[:2]
        nx = int(length / dx)
        ny = int(width / dy)
        water_map_origin_xy = np.stack(np.meshgrid(
            np.linspace(
                - length / 2,
                + length / 2 - dx / density,
                density * nx),
            np.linspace(
                - width / 2,
                + width / 2 - dy / density,
                density * ny),
            indexing='ij',
        ), axis=-1)
        water_map_origin_xy += (length / 4, 0.0)
        mini_cell_centers_origin_xy = water_map_origin_xy + (
            dx / (density * 2), dy / (density * 2))

        cell_centers_xy = rotate_around_z(
            water_map_origin_xy[::density, ::density] + (dx / 2, dy / 2),
            self._get_orientation()
        ) + (com_x, com_y)

        mini_cell_centers_xy = rotate_around_z(
            mini_cell_centers_origin_xy, self._get_orientation())
        mini_cell_centers_xy += (com_x, com_y)

        water_left_xy = self.named.model.geom_pos['water-left'][:2]
        water_left_size = self.named.model.geom_size['water-left'][:2]
        water_right_xy = self.named.model.geom_pos['water-right'][:2]
        water_right_size = self.named.model.geom_size['water-right'][:2]

        cells_in_waters = point_inside_2d_rectangle(
            mini_cell_centers_xy,
            np.stack((water_left_xy, water_right_xy)),
            np.stack((water_left_size, water_right_size)))

        water_map = np.any(cells_in_waters, axis=-1)
        water_map = skimage.measure.block_reduce(
            water_map, (density, density), np.mean)

        cell_centers_xy_v0 = skimage.measure.block_reduce(
            mini_cell_centers_xy, (density, density, 1), np.mean)

        try:
            np.testing.assert_allclose(
                cell_centers_xy_v0,
                cell_centers_xy,
                atol=1e-10)
        except Exception as e:
            breakpoint()
            pass

        return cell_centers_xy, water_map


class MoveTaskMixin(base.Task):
    """A task solved by running across a bridge."""

    def __init__(self,
                 desired_speed_on_bridge,
                 desired_speed_after_bridge,
                 water_map_length=10,
                 water_map_width=10,
                 water_map_dx=1.0,
                 water_map_dy=1.0,
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
        return super(MoveTaskMixin, self).__init__(random=random)

    @abc.abstractmethod
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        pass

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        common_observations = self.common_observations(physics)
        # water_map_xy corresponds to the center of each map cell
        water_map_xy, water_map = physics.water_map(
            length=self._water_map_length,
            width=self._water_map_width,
            dx=self._water_map_dx,
            dy=self._water_map_dy,
            density=10)
        bridge_observations = type(common_observations)((
            *common_observations.items(),
            ('water_map', water_map),
        ))
        for i in range(int(self._water_map_length / self._water_map_dx)):
            for j in range(int(self._water_map_width / self._water_map_dy)):
                cell_id = f'water-map-{i}-{j}'
                physics.named.data.geom_xpos[cell_id][:2] = water_map_xy[i, j]
                physics.named.data.geom_xmat[cell_id][:-3] = (
                    physics.named.data.geom_xmat['torso'][:-3])
                physics.named.model.geom_rgba[cell_id] = (
                    water_map[i, j], 0, 0, 1)

        return bridge_observations

    @abc.abstractmethod
    def upright_reward(self, physics):
        pass

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        if physics.before_bridge():
            move_reward = -1.0
        elif physics.on_bridge():
            x_velocity = physics.torso_velocity()[0]
            move_reward = rewards.tolerance(
                x_velocity,
                bounds=(self._desired_speed_on_bridge, float('inf')),
                margin=self._desired_speed_on_bridge,
                value_at_margin=0.0,
                sigmoid='linear')

        elif physics.after_bridge():
            x_velocity = physics.torso_velocity()[0]
            move_reward = rewards.tolerance(
                x_velocity,
                bounds=(self._desired_speed_after_bridge, float('inf')),
                margin=self._desired_speed_after_bridge,
                value_at_margin=0.0,
                sigmoid='linear')
        elif physics.any_key_geom_in_water():
            move_reward = 0.0
        else:
            raise ValueError("The agent has to be somewhere!")

        return self.upright_reward(physics) * move_reward

    def get_termination(self, physics):
        """Terminates when any of the key geoms are in the water."""
        if physics.any_key_geom_in_water():
            return 0
