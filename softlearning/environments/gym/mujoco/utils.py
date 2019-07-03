from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv


LOCOMOTION_ENVS = (
    HopperEnv,
    Walker2dEnv,
    HalfCheetahEnv,
    SwimmerEnv,
    AntEnv,
    HumanoidEnv,
)


POSITION_SLICES = {
    SwimmerEnv: slice(0, 2),
    AntEnv: slice(0, 2),
    HumanoidEnv: slice(0, 2),
    HalfCheetahEnv: slice(0, 1),
    HopperEnv: slice(0, 1),
    Walker2dEnv: slice(0, 1),
}
