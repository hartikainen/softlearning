import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from .utils import create_maze


class BaseMazeEnv():
    MODEL_CLASS = None

    def __init__(
            self,
            # goal_generator,
            sensor_bins=20,
            sensor_range=10.0,
            sensor_span=math.pi,
            maze_type='u-shape',
            maze_kwargs=None,
            *args,
            **kwargs):
        utils.EzPickle.__init__(**locals())

        self._sensor_bins = sensor_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span
        self._maze_type = maze_type
        self._maze_kwargs = maze_kwargs

        assert self.MODEL_CLASS is not None, (
            "You must specify MODEL_CLASS attribute when extending MazeEnv.")

        model_class = self.MODEL_CLASS
        xml_file = self.MODEL_XML_FILE
        xml_path = os.path.join(
            os.path.dirname(mujoco_env.__file__), "assets", xml_file)

        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.maze_blueprint = create_maze(
            maze_type=maze_type, maze_kwargs=maze_kwargs)

        maze_height = 1
        maze_scale = 4

        start_positions = np.stack(
            np.where(self.maze_blueprint == 's'), axis=-1)
        goal_positions = np.stack(
            np.where(self.maze_blueprint == 'g'), axis=-1)

        assert start_positions.shape[0] == 1, start_positions
        assert goal_positions.shape[0] == 1, goal_positions

        start_position = start_positions[0]
        start_y, start_x = start_position * maze_scale
        goal_position = goal_positions[0]

        maze_xml = self.create_maze_xml()

        for j, row in enumerate(self.maze_blueprint):
            for i, col in enumerate(row):
                ii = i * maze_scale
                jj = j * maze_scale

                element_position = (ii-start_x, jj-start_y, maze_height/2)
                if col == '#':
                    element_size = (maze_scale/2, maze_scale/2, maze_height/2)
                    ET.SubElement(
                        worldbody,
                        "geom",

                        name=f"block_{i}_{j}",
                        pos=" ".join(map(str, element_position)),
                        size=" ".join(map(str, element_size)),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        density="0.00001",
                        rgba="0 0 0 1"
                        # rgba="0.4 0.4 0.4 0.5"
                    )
                elif col == "g":
                    element_size = (maze_scale/3, maze_scale/3, maze_height/3)
                    ET.SubElement(
                        worldbody,
                        "geom",

                        name=f"target_{i}_{j}",
                        pos=" ".join(map(str, element_position)),
                        size=" ".join(map(str, element_size)),
                        type="sphere",
                        rgba="0 1 0 0.2"
                    )

        # import xml.dom.minidom
        # dom_str = xml.dom.minidom.parseString(ET.tostring(worldbody))

        make_contacts = False
        if make_contacts:
            torso = tree.find(".//body[@name='torso']")
            geoms = torso.findall(".//geom")
            for geom in geoms:
                if 'name' not in geom.attrib:
                    from pprint import pprint; import ipdb; ipdb.set_trace(context=30)

                    raise Exception("Every geom of the torso must have a name "
                                    "defined")

            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for j, row in enumerate(self.maze_blueprint):
                for i, col in enumerate(row):
                    if col == '#':
                        for geom in geoms:
                            ET.SubElement(
                                contact,
                                "pair",
                                geom1=geom.attrib["name"],
                                geom2=f"block_{i}_{j}")

        _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(xml_path)

        self.inner_env = model_class(xml_file=xml_path, *args, **kwargs)  # file to the robot specifications

    def create_maze_xml(self):
        pass

    def reset(self, *args, **kwargs):
        return self.inner_env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.inner_env.step(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.inner_env.render(*args, **kwargs)

    @property
    def observation_space(self):
        return self.inner_env.observation_space

    @property
    def action_space(self):
        return self.inner_env.action_space

    @property
    def reward_range(self):
        return self.inner_env.reward_range

    @property
    def metadata(self):
        return self.inner_env.metadata
