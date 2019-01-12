"""Implements an RllabAdapter that converts Rllab envs into SoftlearningEnv."""

import gym.spaces

import rllab.spaces
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.normalized_env import NormalizedEnv

from .softlearning_env import SoftlearningEnv


def raise_on_use_wrapper(e):
    def raise_on_use(*args, **kwargs):
        raise e
    return raise_on_use


try:
    from curriculum.envs.maze.point_env import PointEnv
    from curriculum.envs.maze.point_maze_env import PointMazeEnv
    from curriculum.envs.maze.maze_ant.ant_env import AntEnv
    from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv
except Exception as e:
    PointEnv = raise_on_use_wrapper(e)
    PointMazeEnv = raise_on_use_wrapper(e)
    AntEnv = raise_on_use_wrapper(e)
    AntMazeEnv = raise_on_use_wrapper(e)


RLLAB_ENVIRONMENTS = {
    'Swimmer': {
        'Default': SwimmerEnv,
    },
    'Ant': {
        'Default': AntEnv,
    },
    'Humanoid': {
        'Default': HumanoidEnv,
    },
    'Multigoal': {
    },
    'CurriculumPointEnv': {
        'Default': PointEnv,
        'Maze': PointMazeEnv
    },
    'CurriculumAntEnv': {
        'Default': AntEnv,
        'Maze': AntMazeEnv
    },
}


def convert_rllab_space_to_gym_space(space):
    dtype = space.sample().dtype
    if isinstance(space, rllab.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high, dtype=dtype)
    elif isinstance(space, rllab.spaces.Discrete):
        return gym.spaces.Discrete(n=space.n, dtype=dtype)
    elif isinstance(space, rllab.spaces.Product):
        return gym.spaces.Tuple([
            convert_rllab_space_to_gym_space(x)
            for x in space.components
        ])

    raise NotImplementedError(space)


class RllabAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Rllab envs."""

    def __init__(self, domain, task, *args, normalize=True, **kwargs):
        self._Serializable__initialize(locals())
        super(RllabAdapter, self).__init__(domain, task, *args, **kwargs)

        env = RLLAB_ENVIRONMENTS[domain][task](*args, **kwargs)

        if normalize:
            env = NormalizedEnv(env)

        self._env = env

    @property
    def observation_space(self):
        observation_space = convert_rllab_space_to_gym_space(
            self._env.observation_space)
        if len(observation_space.shape) > 1:
            raise NotImplementedError(
                "Observation space ({}) is not flat, make sure to check the"
                " implemenation. ".format(observation_space))
        return observation_space

    @property
    def action_space(self):
        action_space = convert_rllab_space_to_gym_space(
            self._env.action_space)
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation. ".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return OrderedDict,
        # such that the observation for all the envs is consistent. Right now
        # all the rllab envs return np.array whereas some Gym envs return
        # gym.spaces.Dict type.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.terminate(*args, **kwargs)

    def seed(self, *args, **kwargs):
        pass  # Nothing to seed

    @property
    def unwrapped(self):
        return getattr(self._env, 'wrapped_env', self._env)

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
