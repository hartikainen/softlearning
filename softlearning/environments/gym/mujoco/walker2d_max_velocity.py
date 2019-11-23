import numpy as np
from gym import utils
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv


class Walker2dMaxVelocityEnv(Walker2dEnv):
    def __init__(self, *args, max_velocity=float('inf'), **kwargs):
        utils.EzPickle.__init__(**locals())
        self._max_velocity = max_velocity
        super(Walker2dMaxVelocityEnv, self).__init__(*args, **kwargs)

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action) * float(self.is_healthy)

        forward_reward = (
            self._forward_reward_weight
            * np.minimum(x_velocity, self._max_velocity)
        ) * float(self.is_healthy)
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'forward_reward': forward_reward,
            'healthy_reward': healthy_reward,
            'ctrl_cost': ctrl_cost,
        }

        return observation, reward, done, info
