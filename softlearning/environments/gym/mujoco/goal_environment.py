import os
from collections import OrderedDict
import inspect
import tempfile
import xml.etree.ElementTree as ET
from xml import etree

import numpy as np

import gym
from gym import spaces
from gym import utils

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.classic_control.pendulum import PendulumEnv


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


class TargetlessReacherEnv(ReacherEnv):
    def __init__(self, fixed_goal=None, *args, **kwargs):
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(
            os.path.dirname(inspect.getfile(ReacherEnv)),
            'assets',
            'reacher.xml')

        self.set_goal(fixed_goal)

        tree = ET.parse(xml_path)
        world_body = tree.find(".//worldbody")

        target_body = world_body.find("body[@name='target']")
        world_body.remove(target_body)

        reacher_body = world_body.find("body[@name='body0']")
        shadow_body = ET.fromstring(ET.tostring(reacher_body))
        shadow_body.set('name', f"shadow_{shadow_body.get('name')}")

        for geom in shadow_body.findall(".//geom"):
            geom_rgba = geom.get('rgba')
            geom.set('conaffinity', '0')
            geom.set('contype', '0')
            geom.set('name', f"shadow_{geom.get('name')}")

            if geom_rgba is None: continue
            geom_rgba_array = geom_rgba.split(" ")
            new_geom_rgba = " ".join(geom_rgba_array[:3] + ["0.25"])
            geom.set('rgba', new_geom_rgba)

        for body in shadow_body.findall(".//body"):
            body.set('name', f"shadow_{body.get('name')}")

        for joint in shadow_body.findall(".//joint"):
            joint.set('name', f"shadow_{joint.get('name')}")
            joint.set('stiffness', '0')

        world_body.append(shadow_body)

        _, tmp_xml_path = tempfile.mkstemp(suffix='.xml', text=True)
        tree.write(tmp_xml_path)

        result = super(TargetlessReacherEnv, self).__init__(
            *args, xml_file=tmp_xml_path, **kwargs)

        return result

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        target_position = getattr(self, '_target_position', np.zeros(2))[:2]

        reward = - np.linalg.norm(
            self.sim.data.qpos[:2] - target_position, ord=2)
        ob = self._get_obs()
        done = False
        info = {}
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -90
        self.viewer.cam.distance = 0.8
        self.viewer.cam.trackbodyid = -1

    def random_reacher_position_and_velocity(self):
        init_qpos = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=self.init_qpos.size - 2)
        qpos = init_qpos + self.np_random.uniform(
            low=-self._position_noise_scale,
            high=self._position_noise_scale,
            size=self.model.nq - 2)

        init_qvel = np.random.uniform(
            low=0,
            high=0,
            size=self.init_qvel.size - 2)
        qvel = init_qvel + self.np_random.uniform(
            low=-self._velocity_noise_scale,
            high=self._velocity_noise_scale,
            size=self.model.nv - 2)

        return qpos, qvel

    def sample_goal(self):
        if self.fixed_goal is None:
            qpos, _ = self.random_reacher_position_and_velocity()
            qvel = np.zeros(2)
            return np.concatenate((qpos, qvel))
        else:
            return self.fixed_goal

    def reset_model(self):
        reacher_qpos, reacher_qvel = (
            self.random_reacher_position_and_velocity())

        self._target_position = self.sample_goal()
        self.current_goal = self._target_position

        target_qpos = self._target_position[:2]
        target_qvel = self._target_position[2:4]

        qpos = np.concatenate((reacher_qpos, target_qpos))
        qvel = np.concatenate((reacher_qvel, target_qvel))
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            theta,
            self.sim.data.qvel.flat[:2],
        ])

    def set_goal(self, goal, dtype=np.float32):
        self.fixed_goal = (
            np.array(goal, dtype=dtype)
            if goal is not None
            else None)
        self.current_goal = self.fixed_goal

        if goal is not None:
            theta = goal[:2]
            self.sim.data.qpos[2:] = theta
            self.sim.data.qvel[2:] = goal[2:4]


def angle_distance_from_positions(point1, point2, keepdims=False):
    """Given two points on a unit circle, compute their angle distance."""
    angle1 = np.arctan2(*point1) + np.pi  # [0, 2pi]
    angle2 = np.arctan2(*point2) + np.pi  # [0, 2pi]
    distance = np.linalg.norm(
        angle1 - angle2, ord=1, keepdims=keepdims, axis=1)

    distance[distance > np.pi] = 2 * np.pi - distance[distance > np.pi]

    return distance


class GoalReacherEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = TargetlessReacherEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        l2_distance_from_goal = np.linalg.norm(
            observation['observation'] - observation['desired_goal'], ord=2)
        qpos_l2_distance_from_goal = np.linalg.norm(
            observation['observation'][:2] - observation['desired_goal'][:2],
            ord=2)
        qvel_l2_distance_from_goal = np.linalg.norm(
            observation['observation'][2:] - observation['desired_goal'][2:],
            ord=2)
        goal_info = {
            'l2_distance_from_goal': l2_distance_from_goal,
            'qpos_l2_distance_from_goal': qpos_l2_distance_from_goal,
            'qvel_l2_distance_from_goal': qvel_l2_distance_from_goal,
        }
        return goal_info

    def sample_metric_goal(self):
        qpos, _ = self.env.random_reacher_position_and_velocity()
        qvel = np.zeros(2)

        metric_goal = np.concatenate([qpos, qvel])

        return metric_goal

    def set_goal(self, goal, dtype=np.float32):
        self.current_goal = goal
        self.env.set_goal(goal)


class GoalPendulumEnv(GoalEnvironment):
    ENVIRONMENT_CLASS = PendulumEnv

    def _get_goal_info(self, observation, reward, done, base_info):
        l2_distance_from_goal = np.linalg.norm(
            observation['observation'] - observation['desired_goal'], ord=2)
        goal_info = {
            'l2_distance_from_goal': l2_distance_from_goal,
        }
        return goal_info

    def sample_metric_goal(self):
        high = np.array([np.pi, 1])
        theta, thetadot = np.random.uniform(low=-high, high=high)
        metric_goal = np.array([np.cos(theta), np.sin(theta), thetadot])
        return metric_goal
