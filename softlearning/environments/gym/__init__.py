"""Custom Gym environments.

Every class inside this module should extend a gym.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under gym.mujoco submodule.
"""

import gym


CUSTOM_GYM_ENVIRONMENTS_PATH = __package__
MUJOCO_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.mujoco'

MUJOCO_ENVIRONMENT_SPECS = (
    {
        'id': 'Swimmer-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.swimmer_v3:SwimmerEnv'),
    },
    {
        'id': 'Hopper-NoTermination-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper_no_termination:HopperNoTerminationEnv'),
    },
    {
        'id': 'Hopper-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.hopper_v3:HopperEnv'),
    },
    {
        'id': 'Hopper-MaxVelocity-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper_max_velocity:HopperMaxVelocityEnv'),
    },
    {
        'id': 'Walker2d-NoTermination-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d_no_termination:Walker2dNoTerminationEnv'),
    },
    {
        'id': 'Walker2d-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.walker2d_v3:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-MaxVelocity-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d_max_velocity:Walker2dMaxVelocityEnv'),
    },
    {
        'id': 'HalfCheetah-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv'),
    },
    {
        'id': 'Ant-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.ant_v3:AntEnv'),
    },
    {
        'id': 'Humanoid-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.humanoid_v3:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-SimpleStand-v3',
        'entry_point': (f'gym.envs.mujoco.humanoid_v3:HumanoidEnv'),
        'kwargs': {
            'forward_reward_weight': 0.0,
            'terminate_when_unhealthy': False,
        },
    },
    {
        'id': 'Humanoid-Pothole-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_pothole:HumanoidPotholeEnv'),
    },
    {
        'id': 'Humanoid-HeightField-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.height_field_env:HumanoidHeightFieldEnv'),
    },
    {
        'id': 'Hopper-Pothole-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper_pothole:HopperPotholeEnv'),
    },
    {
        'id': 'Hopper-HeightField-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.height_field_env:HopperHeightFieldEnv'),
    },
    {
        'id': 'Walker2d-Pothole-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d_pothole:Walker2dPotholeEnv'),
    },
    {
        'id': 'Walker2d-HeightField-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.height_field_env:Walker2dHeightFieldEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImagePusher2dEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImageForkReacher2dEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:BlindForkReacher2dEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
)

MULTIWORLD_ENVIRONMENT_SPECS = (
    {
        'id': 'Point2DEnv-Default-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DEnv'
    },
    {
        'id': 'Point2DEnv-Wall-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
    {
        'id': 'Point2DEnv-Bridge-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DBridgeEnv',
        'tags': {
            'author': 'Kristian Hartikainen'
        },
        'kwargs': {
            'terminate_on_success': True,
        },
    },
    {
        'id': 'Point2DEnv-BridgeRun-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DBridgeRunEnv',
        'tags': {
            'author': 'Kristian Hartikainen'
        },
        'kwargs': {
            'terminate_on_success': False,
        },
    },
    {
        'id': 'Point2DEnv-Pond-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DPondEnv',
        'tags': {
            'author': 'Kristian Hartikainen'
        },
        'kwargs': {
            'terminate_on_success': True,
        },
    }
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


MULTIWORLD_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MULTIWORLD_ENVIRONMENT_SPECS)

GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
    *MULTIWORLD_ENVIRONMENTS,
)


def register_mujoco_environments():
    """Register softlearning mujoco environments."""
    for mujoco_environment in MUJOCO_ENVIRONMENT_SPECS:
        gym.register(**mujoco_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MUJOCO_ENVIRONMENT_SPECS)

    return gym_ids


def register_general_environments():
    """Register gym environments that don't fall under a specific category."""
    for general_environment in GENERAL_ENVIRONMENT_SPECS:
        gym.register(**general_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  GENERAL_ENVIRONMENT_SPECS)

    return gym_ids


def register_multiworld_environments():
    """Register custom environments from multiworld package."""
    for multiworld_environment in MULTIWORLD_ENVIRONMENT_SPECS:
        gym.register(**multiworld_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MULTIWORLD_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()
    registered_multiworld_environments = register_multiworld_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
        *registered_multiworld_environments,
    )
