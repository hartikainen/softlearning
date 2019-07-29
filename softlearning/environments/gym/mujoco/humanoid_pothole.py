import os
import inspect
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

from xml.etree import ElementTree
from xml.dom import minidom


FALL_LENGTH = 2


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class HumanoidPotholeEnv(HumanoidEnv):
    def __init__(self,
                 pothole_depth=0.025,
                 pothole_length=0.25,
                 pothole_distance=5.0,
                 x_reset_range=(-3, 3),
                 *args,
                 **kwargs):
        self._pothole_depth = pothole_depth
        self._pothole_length = pothole_length
        self._pothole_distance = pothole_distance
        self._x_reset_range = x_reset_range

        humanoid_xml_path = os.path.join(
            os.path.dirname(inspect.getfile(HumanoidEnv)),
            "assets",
            "humanoid.xml")
        tree = ET.parse(humanoid_xml_path)
        # <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>
        worldbody = tree.find(".//worldbody")
        floor = worldbody.find(".//geom[@name='floor']")

        floor_position = np.array([float(x) for x in floor.get('pos').split(' ')])
        floor_position[-1] -= pothole_depth
        floor.set('pos', ' '.join([str(x) for x in floor_position]))

        floor_size = np.array([float(x) for x in floor.get('size').split(' ')])

        surface_before_pothole_size = (
            (floor_size[0] + pothole_distance) / 2,
            floor_size[1],
            pothole_depth/2)
        surface_before_pothole_position = floor_position + (
            -floor_size[0] / 2 + pothole_distance / 2,
            0,
            pothole_depth/2)

        extra_length = 100
        surface_after_pothole_size = (
            (floor_size[0] - pothole_distance - pothole_length + extra_length) / 2,
            floor_size[1],
            pothole_depth/2)
        surface_after_pothole_position = floor_position + (
            (floor_size[0] + pothole_distance + pothole_length + extra_length) / 2,
            0,
            pothole_depth/2)

        start_platform = ET.SubElement(
            worldbody,
            "geom",
            name=f"start_platform",
            material='MatPlane',
            pos=" ".join(map(str, surface_before_pothole_position)),
            size=" ".join(map(str, surface_before_pothole_size)),
            type="box",
            rgba=floor.get('rgba'),
            condim=floor.get('condim'),
            friction=floor.get('friction'),
        )

        # second_platform = ET.SubElement(
        #     worldbody,
        #     "geom",
        #     name=f"second_platform",
        #     material='MatPlane',
        #     pos=" ".join(map(str, surface_after_pothole_position)),
        #     size=" ".join(map(str, surface_after_pothole_size)),
        #     type="box",
        #     rgba=floor.get('rgba'),
        #     condim=floor.get('condim'),
        #     friction=floor.get('friction'),
        # )

        # floor.set('rgba', '0 0 0 0')

        xml_path = tempfile.mkstemp(text=True, suffix='.xml')[1]
        tree.write(xml_path)

        super(HumanoidPotholeEnv, self).__init__(
            xml_file=xml_path, *args, **kwargs)

    @property
    def is_healthy(self):
        x, z = self.sim.data.qpos[[0, 2]]

        min_z, max_z = self._healthy_z_range

        should_be_falling = (
            self._pothole_distance
            < x
            < self._pothole_distance + FALL_LENGTH)
        # Adjust z to match drop
        if should_be_falling:
            min_z -= self._pothole_depth

        after_fall = self._pothole_distance + FALL_LENGTH < x
        if after_fall:
            min_z -= self._pothole_depth
            max_z -= self._pothole_depth

        is_healthy = min_z < z < max_z

        return is_healthy

    def _get_obs(self, *args, **kwargs):
        observation = super(HumanoidPotholeEnv, self)._get_obs(*args, **kwargs)

        after_fall = self._pothole_distance < self.sim.data.qpos[0]
        if after_fall:
            observation[0] = self.sim.data.qpos[1] + self._pothole_depth

        return observation

    def step(self, *args, **kwargs):
        observation, reward, done, info = super(HumanoidPotholeEnv, self).step(
            *args, **kwargs)
        info['past_pothole'] = (
            self._pothole_distance + 3 < self.sim.data.qpos[0])
        info['distance_after_pothole'] = (
            self.sim.data.qpos[0] - self._pothole_distance)

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        qpos[0] = np.random.uniform(*self._x_reset_range)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
