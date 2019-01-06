from .base_maze_env import BaseMazeEnv
from softlearning.environments.gym.mujoco.swimmer_env import SwimmerEnv


class SwimmerMazeEnv(BaseMazeEnv):
    MODEL_CLASS = SwimmerEnv
    MODEL_XML_FILE = 'swimmer.xml'
