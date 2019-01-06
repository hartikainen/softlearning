from .base_maze_env import BaseMazeEnv
from softlearning.environments.gym.mujoco.humanoid_env import HumanoidEnv


class HumanoidMazeEnv(BaseMazeEnv):
    MODEL_CLASS = HumanoidEnv
    MODEL_XML_FILE = 'humanoid.xml'
