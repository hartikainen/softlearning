from gym.envs.mujoco import hopper_v3
from gym import utils


class HopperNoTerminationEnv(hopper_v3.HopperEnv):
    def __init__(self, *args, **kwargs):
        utils.EzPickle.__init__(**locals())

        terminate_when_unhealthy = kwargs.pop('terminate_when_unhealthy', False)
        return super(HopperNoTerminationEnv, self).__init__(
            *args, terminate_when_unhealthy=terminate_when_unhealthy, **kwargs)

    def step(self, *args, **kwargs):
        observation, reward, done, info = (
            super(HopperNoTerminationEnv, self).step(*args, **kwargs))
        reward *= float(self.is_healthy)
        info.update({
            'is_healthy': self.is_healthy,
            'head_height': self.sim.data.qpos[1],
        })
        return observation, reward, done, info
