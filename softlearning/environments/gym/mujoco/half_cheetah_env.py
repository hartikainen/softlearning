import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 exclude_current_positions_from_observation=True):
        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(
            self,
            forward_reward_weight=self._forward_reward_weight,
            ctrl_cost_weight=self._ctrl_cost_weight,
            exclude_current_positions_from_observation=(
                self._exclude_current_positions_from_observation),
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        c = 0.1
        qpos = self.init_qpos + self.np_random.uniform(
            low=-c, high=c, size=self.model.nq)
        qvel = self.init_qvel + c * self.np_random.randn(self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
