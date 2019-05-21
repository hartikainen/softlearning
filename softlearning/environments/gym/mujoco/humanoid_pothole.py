import os
import inspect
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

from xml.etree import ElementTree
from xml.dom import minidom


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
                 *args,
                 **kwargs):
        self._pothole_depth = pothole_depth
        self._pothole_length = pothole_length
        self._pothole_distance = pothole_distance

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

        fall_length = 2
        should_be_falling = (
            self._pothole_distance
            < x
            < self._pothole_distance + fall_length)
        # Adjust z to match drop
        if should_be_falling:
            min_z -= self._pothole_depth

        after_fall = self._pothole_distance + fall_length < x
        if after_fall:
            min_z -= self._pothole_depth
            max_z -= self._pothole_depth

        is_healthy = min_z < z < max_z

        return is_healthy
