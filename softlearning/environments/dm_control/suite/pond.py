import abc

from lxml import etree
import numpy as np
from scipy.spatial.transform import Rotation
import skimage.measure
from dm_control.suite import base
from dm_control.utils import rewards

from softlearning.environments.dm_control.suite.bridge import (
    rotate_around_z,
    point_inside_2d_rectangle)


DEFAULT_POND_XY = (0, 0)
DEFAULT_POND_RADIUS = 5


def stringify(value):
    return ' '.join(np.array(value).astype(str))


def make_pond_model(base_model_string,
                    pond_radius,
                    size_multiplier=1.0,
                    pond_xy=DEFAULT_POND_XY,
                    control_range_multiplier=None,
                    friction=None,
                    water_map_length=5,
                    water_map_width=5,
                    water_map_dx=0.5,
                    water_map_dy=0.5):
    size_multiplier = np.array(size_multiplier)

    mjcf = etree.fromstring(base_model_string)
    worldbody = mjcf.find('worldbody')

    floor_geom = mjcf.find(".//geom[@name='floor']")
    floor_size = float(pond_radius * 4)
    floor_size_str = f'{floor_size} {floor_size} .1'

    if floor_geom is None:
        floor_geom = etree.Element(
            'geom',
            name='floor',
            type='plane',
            # material='grid',
            rgba=stringify([.1, .1, .1, .8]),
            pos=stringify((*pond_xy, 0)),
            size=floor_size_str,
        )
        worldbody.insert(0, floor_geom)

    floor_geom.attrib['size'] = floor_size_str

    pond_element = etree.Element(
        "geom",
        type="cylinder",
        name="pond",
        pos=stringify((*pond_xy, 0)),
        # pos=" ".join((str(x) for x in (*pond_xy, 0))),
        size=f"{pond_radius} 0.01",
        contype="96",
        conaffinity="66",
        rgba="0 0 1 1")
    worldbody.insert(0, pond_element)

    if control_range_multiplier is not None:
        for actuator_name in ('yaw_act', 'lift_act', 'extend_act'):
            actuator = (mjcf
                        .find(f".//default[@class='{actuator_name}']")
                        .find('.//general'))
            ctrlrange_str = actuator.attrib['ctrlrange']
            ctrlrange = np.array([float(x) for x in ctrlrange_str.split(' ')])
            ctrlrange *= control_range_multiplier
            actuator.attrib['ctrlrange'] = stringify(ctrlrange)

    if friction is not None:
        model_name = mjcf.attrib['model']
        if model_name == 'humanoid':
            raise NotImplementedError
            default_capsule_element = (
                mjcf
                .find(".//default")
                .find(".//geom[@type='capsule']"))
            default_capsule_element.attrib['friction'] = (
                stringify(np.atleast_1d(friction)))
        elif model_name == 'quadruped':
            default_toe_element = (
                mjcf
                .find(".//default")
                .find(".//default[@class='body']")
                .find(".//default[@class='toe']")
                .find(".//geom[@friction]"))
            default_toe_element.attrib['friction'] = (
                stringify(np.atleast_1d(friction)))
        else:
            raise NotImplementedError((model_name, friction))

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


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), dtype=np.float64)


def compute_angular_deltas(positions1, positions2, center=0.0):
    center = np.array(center)
    positions1 = positions1 - center[..., :positions1.shape[-1]]
    positions2 = positions2 - center[..., :positions1.shape[-1]]
    angles1 = np.arctan2(positions1[..., 1], positions1[..., 0])
    angles2 = np.arctan2(positions2[..., 1], positions2[..., 0])
    angular_deltas = np.arctan2(
        np.sin(angles2 - angles1),
        np.cos(angles2 - angles1)
    )[..., np.newaxis]
    return angular_deltas


class PondPhysicsMixin:
    def __init__(self, *args, **kwargs):
        self._floor_geom_id = None
        self._agent_geom_ids = None
        return super(PondPhysicsMixin, self).__init__(*args, **kwargs)

    @property
    def agent_geom_ids(self):
        raise NotImplementedError

    @property
    def floor_geom_id(self):
        if self._floor_geom_id is None:
            self._floor_geom_id = self.model.name2id('floor', 'geom')
        return self._floor_geom_id

    @property
    def pond_radius(self):
        return self.named.model.geom_size['pond'][0]

    @property
    def pond_center_xyz(self):
        return self.named.data.geom_xpos['pond']

    @abc.abstractmethod
    def center_of_mass(self, physics):
        pass

    def any_key_geom_in_water(self):
        raise NotImplementedError

    def _get_orientation(self):
        return self.named.data.qpos['root'][3:]

    def water_map(self, length, width, dx, dy, offset=0, density=10):
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
        water_map_origin_xy += offset
        mini_cell_centers_origin_xy = water_map_origin_xy + (
            dx / (density * 2), dy / (density * 2))

        cell_centers_xy = rotate_around_z(
            water_map_origin_xy[::density, ::density] + (dx / 2, dy / 2),
            self._get_orientation()
        ) + (com_x, com_y)

        mini_cell_centers_xy = rotate_around_z(
            mini_cell_centers_origin_xy, self._get_orientation()
        ) + (com_x, com_y)

        pond_center_xy = self.pond_center_xyz[:2]

        cells_in_waters = (
            np.linalg.norm(
                mini_cell_centers_xy - pond_center_xy,
                ord=2,
                axis=-1
            ) < self.pond_radius)

        # water_map = np.any(cells_in_waters, axis=-1)
        water_map = skimage.measure.block_reduce(
            cells_in_waters, (density, density), np.mean)

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

    def distances_from_pond_center(self):
        state = self.center_of_mass()[:2]
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

    def angular_velocities(self):
        # global_velocity = self.global_velocity()[:2]
        positions2 = self.center_of_mass()[:2][None]
        # positions1 = positions2 - global_velocity
        positions1 = self._previous_center_of_mass[:2].copy()[None]

        angular_deltas = compute_angular_deltas(
            positions1, positions2, self.pond_center_xyz)

        angular_velocities = angular_deltas
        # angular_velocities = angular_deltas * self.pond_radius

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
            Rotation.from_euler('z', angle_to_pond_center).inv().as_quat(), 1)

        orientation_to_pond = quaternion_multiply(
            origin_to_pond_transformation, orientation_to_origin)

        # TODO(hartikainen): Check if this has some negative side effects on
        # other rotation axes.
        # orientation_to_pond = np.abs(orientation_to_pond[-1])
        orientation_to_pond = (
            np.sign(orientation_to_pond[0])
            * orientation_to_pond)

        return orientation_to_pond


class OrbitTaskMixin(base.Task):
    """A task to orbit around a pond with designated speed."""

    def __init__(self,
                 desired_angular_velocity,
                 angular_velocity_reward_weight,
                 control_cost_weight=0.0,
                 water_map_length=10,
                 water_map_width=10,
                 water_map_dx=1.0,
                 water_map_dy=1.0,
                 water_map_offset=0,
                 random=None):
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
        self._angular_velocity_reward_weight = angular_velocity_reward_weight
        self._control_cost_weight = control_cost_weight
        self._water_map_length = water_map_length
        self._water_map_width = water_map_width
        self._water_map_dx = water_map_dx
        self._water_map_dy = water_map_dy
        self._water_map_offset = water_map_offset
        self._previous_action = None
        return super(OrbitTaskMixin, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        self._previous_action = np.zeros(
            shape=self.action_spec(physics).shape,
            dtype=self.action_spec(physics).dtype)

    @abc.abstractmethod
    def common_observations(self, physics):
        pass

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        common_observations = self.common_observations(physics)

        water_map_xy, water_map = physics.water_map(
            length=self._water_map_length,
            width=self._water_map_width,
            dx=self._water_map_dx,
            dy=self._water_map_dy,
            offset=self._water_map_offset,
            density=10)

        previous_action = (
            self._previous_action
            if self._previous_action is not None
            else np.zeros_like(physics.control()))

        pond_observations = type(common_observations)((
            *common_observations.items(),
            ('water_map', water_map),
            ('previous_action', previous_action),
        ))

        for i in range(int(self._water_map_length / self._water_map_dx)):
            for j in range(int(self._water_map_width / self._water_map_dy)):
                cell_id = f'water-map-{i}-{j}'
                physics.named.data.geom_xpos[cell_id][:2] = water_map_xy[i, j]
                # physics.named.data.geom_xmat[cell_id][-3:] = (
                #     physics.torso_xmat()[-3:])
                physics.named.model.geom_rgba[cell_id] = (
                    water_map[i, j], 0, 0, 0.1)

        return pond_observations

    def before_step(self, action, physics):
        physics._previous_center_of_mass = physics.center_of_mass().copy()
        return super(OrbitTaskMixin, self).before_step(action, physics)

    def after_step(self, physics, *args, **kwargs):
        self._previous_action[:] = physics.control().copy()
        return super(OrbitTaskMixin, self).after_step(physics, *args, **kwargs)

    @abc.abstractmethod
    def upright_reward(self, physics):
        pass

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        control_cost = np.sum(self._control_cost_weight * physics.control())
        angular_velocity_reward = (
            self._angular_velocity_reward_weight
            * np.minimum(
                physics.angular_velocity(), self._desired_angular_velocity))

        reward = (
            self.upright_reward(physics) * angular_velocity_reward
            + control_cost)
        return reward

    def get_termination(self, physics):
        """Terminates when the agent falls into pond."""
        if physics.any_key_geom_in_water():
            return 0
