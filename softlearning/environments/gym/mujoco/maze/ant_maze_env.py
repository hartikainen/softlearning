from .base_maze_env import BaseMazeEnv
from softlearning.environments.gym.mujoco.ant_env import AntEnv


class AntMazeEnv(BaseMazeEnv):
    MODEL_CLASS = AntEnv
    MODEL_XML_FILE = 'ant.xml'
