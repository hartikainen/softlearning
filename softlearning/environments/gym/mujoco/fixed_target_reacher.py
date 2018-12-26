import numpy as np
from gym.utils import EzPickle
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.reacher import ReacherEnv


# self._goal = tf.constant((
#     0.77041958, -0.03223022,  0.63753719,  0.99948047,  0.00204777,
#     0.14971093,  0.00687177, -0.00652871,  0.00217014, -0.00351536,
#     0.0))


class FixedTargetReacherEnv(ReacherEnv):
    def __init__(self, fixed_target, fixed_goal=None):
        self.fixed_target = np.array(fixed_target)
        self.fixed_goal = fixed_goal
        self.target = self.fixed_target.copy()
        EzPickle.__init__(self, fixed_target)
        super(FixedTargetReacherEnv, self).__init__()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq)
        qpos[0:2] = np.random.uniform((-np.pi, -np.pi / 2), (np.pi, np.pi / 2))
        self.target = self.fixed_target.copy()
        qpos[-2:] = self.target
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
