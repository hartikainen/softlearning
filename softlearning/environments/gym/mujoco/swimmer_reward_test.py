from gym.envs.mujoco.swimmer_v3 import SwimmerEnv


class SwimmerRewardTestEnv(SwimmerEnv):
    def __init__(self, reward_offset, *args, **kwargs):
        self._reward_offset = reward_offset
        return super(SwimmerRewardTestEnv, self).__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        observation, reward, done, info = super(
            SwimmerRewardTestEnv, self).step(*args, **kwargs)

        reward += self._reward_offset

        return observation, reward, done, info
