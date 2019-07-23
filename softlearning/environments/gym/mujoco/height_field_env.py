import os
import inspect
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

from xml.etree import ElementTree
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class HeightFieldEnv(object):
    xml_path = None

    def __init__(self,
                 *args,
                 field_z_range=(0, 0.2),
                 **kwargs):
        self._field_z_range = field_z_range

        xml_path = self.xml_path
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        asset = tree.find(".//asset")
        floor = worldbody.find(".//geom[@name='floor']")

        height_field_asset = ET.SubElement(
            asset,
            "hfield",
            name="perturbation-field",
            nrow="2",
            ncol="50",
            size="10 0.5 1 0.1",
        )

        height_field_geom = ET.SubElement(
            worldbody,
            "geom",
            name="perturbation-field",
            type="hfield",
            hfield="perturbation-field",
            pos="11 0 0",
            rgba=floor.get('rgba'),
            condim=floor.get('condim'),
            conaffinity=floor.get('conaffinity'),
            material=floor.get('material'),
        )

        xml_path = tempfile.mkstemp(text=True, suffix='.xml')[1]

        tree.write(xml_path)

        result = super(HeightFieldEnv, self).__init__(
            xml_file=xml_path, *args, **kwargs)

        return result

    def reset_model(self, *args, **kwargs):
        ncol = self.sim.model.hfield_ncol
        nrow = self.sim.model.hfield_nrow

        self.sim.model.hfield_data[:] = np.repeat(
            np.concatenate((
                np.zeros(1),
                np.random.uniform(*self._field_z_range, ncol - 1),
            ))[None],
            nrow,
            axis=0,
        ).ravel()

        return super(HeightFieldEnv, self).reset_model(*args, **kwargs)


class HopperHeightFieldEnv(HeightFieldEnv, HopperEnv):
    xml_path = os.path.join(
        os.path.dirname(inspect.getfile(HopperEnv)),
        "assets",
        "hopper.xml")


class Walker2dHeightFieldEnv(HeightFieldEnv, Walker2dEnv):
    xml_path = os.path.join(
        os.path.dirname(inspect.getfile(Walker2dEnv)),
        "assets",
        "walker2d.xml")


class HumanoidHeightFieldEnv(HeightFieldEnv, HumanoidEnv):
    xml_path = os.path.join(
        os.path.dirname(inspect.getfile(HumanoidEnv)),
        "assets",
        "humanoid.xml")
