import abc

from lxml import etree
import numpy as np
from scipy.spatial.transform import Rotation
from dm_control.suite import base
from dm_control.utils import rewards


DEFAULT_POND_XY = (0, 0)
DEFAULT_POND_RADIUS = 5


def stringify(value):
    return ' '.join(np.array(value).astype(str))


def make_pond_model(base_model_string,
                    pond_radius,
                    size_multiplier=1.0,
                    pond_xy=DEFAULT_POND_XY):
    size_multiplier = np.array(size_multiplier)

    mjcf = etree.fromstring(base_model_string)
    worldbody = mjcf.find('worldbody')

    floor_geom = mjcf.find(".//geom[@name='floor']")
    floor_size = float(pond_radius * 4)
    floor_size_str = f'{floor_size} {floor_size} .1'

    if floor_geom is None:
        floor_element = etree.Element(
            'geom',
            type='plane',
            # material='grid',
            rgba=stringify([.1, .1, .1, .8]),
            pos=stringify((*pond_xy, 0)),
            size=floor_size_str,
        )
        worldbody.insert(0, floor_element)

    floor_element.attrib['size'] = floor_size_str

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

    return etree.tostring(mjcf, pretty_print=True)


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), dtype=np.float64)


class PondPhysicsMixin:
    @property
    def pond_radius(self):
        return self.named.model.geom_size['pond'][0]

    @property
    def pond_center_xyz(self):
        return self.named.data.geom_xpos['pond']

    @abc.abstractmethod
    def center_of_mass(self, physics):
        pass

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
        global_velocity = self.global_velocity()[:2]
        positions2 = self.center_of_mass()[:2][None]
        positions1 = positions2 - global_velocity

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
            Rotation.from_euler('z', angle_to_pond_center).inv().as_quat(), 1)

        orientation_to_pond = quaternion_multiply(
            origin_to_pond_transformation, orientation_to_origin)

        # TODO(hartikainen): Check if this has some negative side effects on
        # other rotation axes.
        orientation_to_pond[-1] = np.abs(orientation_to_pond[-1])

        return orientation_to_pond


class OrbitTaskMixin(base.Task):
    """A task to orbit around a pond with designated speed."""

    def __init__(self,
                 desired_angular_velocity,
                 angular_velocity_reward_weight,
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
        return super(OrbitTaskMixin, self).__init__(random=random)

    @abc.abstractmethod
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        pass

    @abc.abstractmethod
    def common_observations(self, physics):
        pass

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        common_observations = self.common_observations(physics)

        orientation_to_pond = physics.orientation_to_pond()
        distance_from_pond = physics.distance_from_pond()

        pond_observations = type(common_observations)((
            *common_observations.items(),
            ('orientation_to_pond', orientation_to_pond),
            ('distance_from_pond', distance_from_pond),
        ))

        return pond_observations

    @abc.abstractmethod
    def upright_reward(self, physics):
        pass

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        angular_velocity_reward = (
            self._angular_velocity_reward_weight
            * np.sign(physics.torso_velocity()[0])
            * rewards.tolerance(
                physics.angular_velocity(),
                bounds=(self._desired_angular_velocity, float('inf')),
                margin=self._desired_angular_velocity,
                value_at_margin=0.0,
                sigmoid='linear'))

        return self.upright_reward(physics) * angular_velocity_reward

    def get_termination(self, physics):
        """Terminates when the agent falls into pond."""
        if physics.distance_from_pond() < 0:
            return 0
