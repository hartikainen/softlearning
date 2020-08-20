import abc
import numpy as np
from lxml import etree
from dm_control.suite import base
from dm_control.utils import rewards
import skimage.measure

from scipy.spatial.transform import Rotation

from .bridge import rotate_around_z, stringify, point_inside_2d_rectangle

DEFAULT_BRIDGE_LENGTH = 10.0
DEFAULT_BRIDGE_WIDTH = 2.0


def make_model(base_model_string,
               size_multiplier=1.0,
               bridge_length=DEFAULT_BRIDGE_LENGTH,
               bridge_start_width=DEFAULT_BRIDGE_WIDTH,
               bridge_end_width=0.0,
               water_map_length=5,
               water_map_width=5,
               water_map_dx=0.5,
               water_map_dy=0.5,
               floor_geom_name='floor',
               *args,
               **kwargs):
    size_multiplier = np.array(size_multiplier)

    mjcf = etree.fromstring(base_model_string)
    worldbody = mjcf.find('worldbody')

    floor_geom = mjcf.find(".//geom[@name='floor']")
    floor_size = float(size_multiplier * 15 + bridge_length)
    floor_geom.attrib['size'] = f'{floor_size} {floor_size} .1'

    bridge_offset = -1.0 * size_multiplier
    # bridge_offset = 0.0

    floor_x = floor_size + bridge_offset
    floor_geom.attrib['pos'] = f"{floor_x} 0 0"

    bridge_x = (bridge_length) / 2 + bridge_offset
    # bridge_y_abs = (bridge_start_width + water_width) / 2
    bridge_y = 0.0

    water_width = floor_size - bridge_start_width / 2
    water_length = np.linalg.norm((
        (bridge_start_width - bridge_end_width) / 2, bridge_length))

    water_angle = np.arctan2(
        (bridge_start_width - bridge_end_width) / 2,
        # water_length,
        bridge_length,
    )

    # water_x = bridge_x
    # water_y_abs = (bridge_start_width + water_width) / 2.0

    left_water_bottom_right_corner_xy = (bridge_offset, bridge_start_width / 2.0)
    left_water_xy = (
        left_water_bottom_right_corner_xy
        + rotate_around_z((water_length / 2.0, water_width / 2.0), -water_angle))

    left_water_position = np.array((*left_water_xy, 0.0))
    right_water_position = left_water_position * np.array((1.0, -1.0, 1.0))

    left_water_quat = np.roll(Rotation.from_euler('z', -water_angle).as_quat(), 1)
    right_water_quat = np.roll(Rotation.from_euler('z', water_angle).as_quat(), 1)

    left_water_element = etree.Element(
        "geom",
        type="box",
        name="water-left",
        pos=stringify(left_water_position),
        quat=stringify((left_water_quat)),
        size=stringify((water_length / 2, water_width / 2, 0.01)),
        contype="0",
        conaffinity="0",
        rgba="0 0 1 0.1")
    right_water_element = etree.Element(
        "geom",
        type="box",
        name="water-right",
        pos=stringify(right_water_position),
        quat=stringify((right_water_quat)),
        size=stringify((water_length / 2, water_width / 2, 0.01)),
        contype="0",
        conaffinity="0",
        rgba="0 0 1 0.1")
    bridge_element = etree.Element(
        "geom",
        type="box",
        name="bridge",
        pos=stringify((bridge_x, bridge_y, 0)),
        size=stringify((bridge_length / 2, bridge_start_width / 2, 0.01)),
        contype="0",
        conaffinity="0",
        rgba="0 0 0 0")
    worldbody.insert(0, left_water_element)
    worldbody.insert(1, right_water_element)
    worldbody.insert(2, bridge_element)

    for x in range(int(water_map_length / water_map_dx)):
        for y in range(int(water_map_width / water_map_dy)):
            water_map_cell_element = etree.Element(
                "geom",
                type="box",
                contype="0",
                conaffinity="0",
                name=f"water-map-{x}-{y}",
                pos=stringify((0, 0, 0.01 * size_multiplier)),
                size=stringify(
                    (water_map_dx, water_map_dy, 0.01 * size_multiplier)),
                rgba="0 0 0 1")
            worldbody.insert(-1, water_map_cell_element)

    return etree.tostring(mjcf, pretty_print=True)


class MovePhysicsMixin:
    def __init__(self, *args, **kwargs):
        self._floor_geom_id = None
        result = super(MovePhysicsMixin, self).__init__(*args, **kwargs)

        bridge_size = self.named.model.geom_size['bridge']
        bridge_pos = self.named.model.geom_pos['bridge']
        bridge_quat = self.named.model.geom_quat['bridge']
        np.testing.assert_equal(bridge_quat, (1, 0, 0, 0))

        bridge_top_right_pos = bridge_pos + (1, +1, 1) * bridge_size
        bridge_bottom_right_pos = bridge_pos + (1, -1, 1) * bridge_size
        bridge_top_left_pos = bridge_pos + (-1, +1, 1) * bridge_size
        bridge_bottom_left_pos = bridge_pos + (-1, -1, 1) * bridge_size


        water_left_size = self.named.model.geom_size['water-left']
        water_left_pos = self.named.model.geom_pos['water-left']
        water_left_quat = self.named.model.geom_quat['water-left']

        np.testing.assert_equal(water_left_quat[1:3], 0.0)
        water_left_euler = Rotation.from_quat(np.roll(water_left_quat, -1)).as_euler('xyz')
        np.testing.assert_equal(water_left_euler[:2], 0.0)

        np.testing.assert_equal(water_left_quat[1:3], 0.0)
        water_left_euler = Rotation.from_quat(np.roll(water_left_quat, -1)).as_euler('xyz')
        np.testing.assert_equal(water_left_euler[:2], 0.0)

        water_left_bottom_left_pos = water_left_pos + Rotation.from_quat(
            np.roll(water_left_quat, -1)).apply(
                (-1, -1, 1) * water_left_size)

        water_left_bottom_right_pos = water_left_pos + Rotation.from_quat(
            np.roll(water_left_quat, -1)).apply(
                (1, -1, 1) * water_left_size)

        water_right_size = self.named.model.geom_size['water-right']
        water_right_pos = self.named.model.geom_pos['water-right']
        water_right_quat = self.named.model.geom_quat['water-right']

        np.testing.assert_equal(water_right_quat[1:3], 0.0)
        water_right_euler = Rotation.from_quat(np.roll(water_right_quat, -1)).as_euler('xyz')
        np.testing.assert_equal(water_right_euler[:2], 0.0)

        water_right_top_left_pos = water_right_pos + Rotation.from_quat(
            np.roll(water_right_quat, -1)).apply(
                (-1, 1, 1) * water_right_size)

        water_right_top_right_pos = water_right_pos + Rotation.from_quat(
            np.roll(water_right_quat, -1)).apply(
                (1, 1, 1) * water_right_size)

        np.testing.assert_allclose(
            Rotation.from_quat(water_right_quat).inv().as_quat(),
            water_left_quat)

        np.testing.assert_allclose(water_left_pos, (1, -1, 1) * water_right_pos)
        
        np.testing.assert_allclose(
            water_left_bottom_left_pos, (1, -1, 1) * water_right_top_left_pos)

        np.testing.assert_allclose(
            water_left_bottom_left_pos, bridge_top_left_pos)
        np.testing.assert_allclose(
            water_right_top_left_pos, bridge_bottom_left_pos)

        # # Should equal ``bridge_end_width``
        # np.testing.assert_allclose(
        #     water_left_bottom_right_pos[1] - water_right_top_right_pos[1], 0.3)

        return result

    @property
    def agent_geom_ids(self):
        raise NotImplementedError

    @property
    def floor_geom_id(self):
        if self._floor_geom_id is None:
            self._floor_geom_id = self.model.name2id('floor', 'geom')
        return self._floor_geom_id

    def before_bridge(self):
        agent_x = self.center_of_mass()[0]
        bridge_x = self.named.data.geom_xpos['bridge'][0]
        bridge_length = self.named.model.geom_size['bridge'][0]
        before_bridge = agent_x < bridge_x - bridge_length

        return before_bridge

    def after_bridge(self):
        agent_x = self.center_of_mass()[0]
        bridge_x = self.named.data.geom_xpos['bridge'][0]
        bridge_length = self.named.model.geom_size['bridge'][0]
        after_bridge = bridge_x + bridge_length < agent_x

        return after_bridge

    def on_bridge(self):
        agent_xy = self.center_of_mass()[:2]
        bridge_xy = self.named.data.geom_xpos['bridge'][:2]
        bridge_size = self.named.model.geom_size['bridge'][:2]

        on_bridge = point_inside_2d_rectangle(
            agent_xy, bridge_xy, bridge_size)

        return on_bridge

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

        water_names = ('water-left', 'water-right')
        waters_xy = np.stack([
            self.named.model.geom_pos[water_name][:2]
            for water_name in water_names
        ], axis=0)
        waters_size = np.stack([
            self.named.model.geom_size[water_name][:2]
            for water_name in water_names
        ], axis=0)
        waters_angle = np.stack([
            Rotation.from_quat(
                np.roll(self.named.model.geom_quat[water_name], -1)
            ).as_euler('xyz')[-1:]
            for water_name in water_names
        ], axis=0)

        key_geoms_in_waters = point_inside_2d_rectangle(
            key_geoms_xy, waters_xy, waters_size, waters_angle)

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
        nx = int(round(length / dx, 12))
        ny = int(round(width / dy, 12))
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

        water_names = ('water-left', 'water-right')
        waters_xy = np.stack([
            self.named.model.geom_pos[water_name][:2]
            for water_name in water_names
        ], axis=0)
        waters_size = np.stack([
            self.named.model.geom_size[water_name][:2]
            for water_name in water_names
        ], axis=0)
        waters_angle = np.stack([
            Rotation.from_quat(
                np.roll(self.named.model.geom_quat[water_name], -1)
            ).as_euler('xyz')[-1:]
            for water_name in water_names
        ], axis=0)

        cells_in_waters = point_inside_2d_rectangle(
            mini_cell_centers_xy,
            waters_xy,
            waters_size,
            waters_angle)

        water_map = np.any(cells_in_waters, axis=-1)
        water_map = skimage.measure.block_reduce(
            water_map, (density, density), np.mean)

        cell_centers_xy_v0 = skimage.measure.block_reduce(
            mini_cell_centers_xy, (density, density, 1), np.mean)

        np.testing.assert_allclose(
            cell_centers_xy_v0,
            cell_centers_xy,
            atol=1e-10)

        return cell_centers_xy, water_map

    def reward_bounds(self):
        bridge_x = self.named.data.geom_xpos['bridge'][0]
        bridge_length = 2 * self.named.model.geom_size['bridge'][0]
        reward_bounds_x_low = bridge_x + bridge_length / 2
        reward_bounds_x_high = reward_bounds_x_low + 3 * bridge_length
        # reward_bounds_x_high = 5.0

        reward_bounds_y_low = (
            self.named.model.geom_pos['water-right'][1]
            - self.named.model.geom_size['water-right'][1])
        reward_bounds_y_high = (
            self.named.model.geom_pos['water-left'][1]
            + self.named.model.geom_size['water-left'][1])

        reward_bounds = (
            reward_bounds_x_low,
            reward_bounds_x_high,
            reward_bounds_y_low,
            reward_bounds_y_high)
        return reward_bounds


class MoveTaskMixin(base.Task):
    """A task solved by running across a bridge."""

    def __init__(self,
                 desired_speed_on_bridge,
                 desired_speed_after_bridge,
                 after_bridge_reward_type='constant',
                 after_bridge_reward_weight=1.0,
                 terminate_outside_of_reward_bounds=False,
                 randomize_initial_x_position=False,
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
        self._after_bridge_reward_type = after_bridge_reward_type
        self._after_bridge_reward_weight = after_bridge_reward_weight
        self._terminate_outside_of_reward_bounds = (
            terminate_outside_of_reward_bounds)
        self._randomize_initial_x_position = randomize_initial_x_position
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
                # physics.named.data.geom_xmat[cell_id][-3:] = (
                #     physics.torso_xmat()[-3:])
                physics.named.model.geom_rgba[cell_id] = (
                    water_map[i, j], 0, 0, 0.1)

        return bridge_observations

    @abc.abstractmethod
    def upright_reward(self, physics):
        pass

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        if physics.any_key_geom_in_water():
            move_reward = -1.0
        elif physics.before_bridge():
            move_reward = -1.0
        elif physics.on_bridge():
            x_velocity = physics.torso_velocity()[0]
            move_reward = np.minimum(x_velocity, self._desired_speed_on_bridge)
            # move_reward = rewards.tolerance(
            #     x_velocity,
            #     bounds=(self._desired_speed_on_bridge, float('inf')),
            #     margin=self._desired_speed_on_bridge,
            #     value_at_margin=0.0,
            #     sigmoid='linear')
        elif physics.after_bridge():
            (reward_bounds_x_low,
             reward_bounds_x_high,
             reward_bounds_y_low,
             reward_bounds_y_high) = physics.reward_bounds()

            x_position, y_position = physics.center_of_mass()[:2]
            within_reward_bounds = np.logical_and.reduce((
                reward_bounds_x_low <= x_position,
                x_position <= reward_bounds_x_high,
                reward_bounds_y_low <= y_position,
                y_position <= reward_bounds_y_high))

            if within_reward_bounds:
                if self._after_bridge_reward_type == 'x_velocity':
                    x_velocity = physics.torso_velocity()[0]
                    move_reward = (
                        self._after_bridge_reward_weight * np.minimum(
                            x_velocity, self._desired_speed_after_bridge))
                elif self._after_bridge_reward_type == 'xy_velocity':
                    xy_velocity = physics.torso_velocity()
                    velocity = np.linalg.norm(xy_velocity)
                    velocity /= np.maximum(velocity, 1.0)
                    move_reward = (
                        self._after_bridge_reward_weight * np.minimum(
                            velocity, self._desired_speed_after_bridge))
                elif self._after_bridge_reward_type == 'constant':
                    move_reward = self._after_bridge_reward_weight
                else:
                    raise ValueError(self._after_bridge_reward_type)
            else:
                move_reward = 0.0
        else:
            move_reward = 0.0

        return self.upright_reward(physics) * move_reward

    def get_termination(self, physics):
        """Terminates when any of the key geoms are in the water."""
        if physics.any_key_geom_in_water():
            return 0

        if physics.before_bridge():
            return 0

        if self._terminate_outside_of_reward_bounds and not physics.on_bridge():
            (reward_bounds_x_low,
             reward_bounds_x_high,
             reward_bounds_y_low,
             reward_bounds_y_high) = physics.reward_bounds()
            x_position, y_position = physics.center_of_mass()[:2]
            within_reward_bounds = np.logical_and.reduce((
                reward_bounds_x_low <= x_position,
                x_position <= reward_bounds_x_high,
                reward_bounds_y_low <= y_position,
                y_position <= reward_bounds_y_high))

            if not within_reward_bounds:
                return 0
