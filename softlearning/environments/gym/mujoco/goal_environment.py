from collections import OrderedDict

import numpy as np

import gym
from gym import spaces

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class ProxyEnv(object):
    @property
    def wrapped_env(self):
        return self.env

    def __getattr__(self, name):
        if name == 'spec':
            from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
            pass
        return getattr(self.env, name)

    # def __getattribute__(self, name):
    #     if name == 'spec':
    #         from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
    #         pass
    #     return object.__getattribute__(self, name)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.env)


class GoalEnvironment(gym.Env, ProxyEnv):
    def __init__(self, *args, **kwargs):
        self.env = self.ENVIRONMENT_CLASS(*args, **kwargs)
        self.current_goal = self.env._get_obs()
        self.action_space = self.env.action_space
        observation = self._get_obs()
        self.observation_space = spaces.Dict({
            'desired_goal': spaces.Box(
                -np.inf,
                np.inf,
                shape=observation['desired_goal'].shape,
                dtype='float32'),
            'observation': spaces.Box(
                -np.inf,
                np.inf,
                shape=observation['observation'].shape,
                dtype='float32'),
        })

    def _set_observation_space(self):
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.current_goal = super(GoalEnvironment, self)._get_obs()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done

        def get_observation_box(observation):
            high = np.inf * np.ones(observation.size)
            low = -high
            observation_box = spaces.Box(low, high, dtype=np.float32)
            return observation_box

        self.observation_space = spaces.Dict({
            key: get_observation_box(value)
            for key, value in observation.items()
        })

    def sample_metric_goal(self):
        raise NotImplementedError

    def set_goal(self, goal, dtype=np.float32):
        self.current_goal = np.array(goal, dtype=dtype)

    def _get_goal_info(self, observation, reward, done, base_info):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        observation, reward, done, base_info = self.env.step(*args, **kwargs)
        goal_observation = self._get_obs()

        assert np.allclose(goal_observation['observation'], observation)

        goal_info = self._get_goal_info(
            goal_observation, reward, done, base_info)
        info = {**base_info, **goal_info}

        return goal_observation, reward, done, info

    def _get_obs(self, *args, **kwargs):
        base_observation = self.env._get_obs(*args, **kwargs)

        observation = OrderedDict((
            ('observation', base_observation),
            ('desired_goal', self.current_goal),
        ))

        return observation

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self._get_obs()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec


def one_dimensional_goal_info(observation, reward, done, base_info):
    x_distance_to_goal = (
        observation['observation'][0] - observation['desired_goal'][0])
    distance_to_goal = np.abs(x_distance_to_goal)
    info = {
        'distance_to_goal': distance_to_goal,
        'x_distance_to_goal': x_distance_to_goal,
        'goal_x': observation['desired_goal'][0],
    }
    return info


def two_dimensional_goal_info(observation, reward, done, base_info):
    xy_distance_to_goal = (
        observation['observation'][:2] - observation['desired_goal'][:2])
    distance_to_goal = np.linalg.norm(xy_distance_to_goal, ord=2)
    info = {
        'distance_to_goal': distance_to_goal,
        'x_distance_to_goal': xy_distance_to_goal[0],
        'y_distance_to_goal': xy_distance_to_goal[1],
        'goal_x': observation['desired_goal'][0],
        'goal_y': observation['desired_goal'][1],
    }
    return info


class GoalSwimmerEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = SwimmerEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        return two_dimensional_goal_info(observation, reward, done, base_info)


class GoalAntEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = AntEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        return two_dimensional_goal_info(observation, reward, done, base_info)


class GoalHumanoidEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = HumanoidEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        return two_dimensional_goal_info(observation, reward, done, base_info)


class GoalHopperEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = HopperEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        return one_dimensional_goal_info(observation, reward, done, base_info)


class GoalWalker2dEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = Walker2dEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        return one_dimensional_goal_info(observation, reward, done, base_info)


class GoalHalfCheetahEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = HalfCheetahEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        return one_dimensional_goal_info(observation, reward, done, base_info)
