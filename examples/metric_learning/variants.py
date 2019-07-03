from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev
from examples.development.variants import is_image_env
from examples.utils import variant_equals
from sac_envs.envs.dclaw.dclaw3_screw_v2 import LinearLossFn, NegativeLogLossFn


DEFAULT_KEY = "__DEFAULT_KEY__"
DEFAULT_LAYER_SIZE = 256


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Swimmer': {
            'v3': {
                'exclude_current_positions_from_observation': False,
            },
            'Maze-v0': {
                'exclude_current_positions_from_observation': False,
                'reset_noise_scale': 0,
            },
        },
        'GoalSwimmer': {
            'v0': {
                'exclude_current_positions_from_observation': False,
                'observation_keys': ('observation', ),
            },
        },
        'GoalAnt': {
            'v0': {
                'exclude_current_positions_from_observation': False,
                'observation_keys': ('observation', ),
                'terminate_when_unhealthy': False,
                'reset_noise_scale': tune.grid_search([0.0, 0.1])
            },
        },
        'GoalHalfCheetah': {
            'v0': {
                'observation_keys': ('observation', ),
                'forward_reward_weight': 0,
                'ctrl_cost_weight': 0,
                'exclude_current_positions_from_observation': False,
                'reset_noise_scale': tune.grid_search([0.0, 0.1])
            }
        },
        'GoalHopper': {
            'v0': {
                'exclude_current_positions_from_observation': False,
                'observation_keys': ('observation', ),
                'terminate_when_unhealthy': tune.grid_search([True, False]),
                'reset_noise_scale': tune.grid_search([0.0, 0.1])
            }
        },
        'GoalWalker': {
            'v0': {
                'exclude_current_positions_from_observation': False,
                'observation_keys': ('observation', ),
            }
        },
        'GoalHumanoid': {
            'v0': {
                'exclude_current_positions_from_observation': False,
                'observation_keys': ('observation', ),
            },
        },
        'GoalReacher': {
            'v0': {
                'observation_keys': ('observation', ),
            }
        },
        'GoalPendulum': {
            'v0': {
                'observation_keys': ('observation', ),
            }
        },
        'Hopper': {
            'Parameterizable-v3': {
                'exclude_current_positions_from_observation': False,
                'terminate_when_unhealthy': False,
                'healthy_reward': 1.0,
                'reset_noise_scale': 0,
            },
        },
        'HalfCheetah': {
            'Parameterizable-v3': {
                'exclude_current_positions_from_observation': False,
            },
        },
        'Walker': {
            'Parameterizable-v3': {
                'exclude_current_positions_from_observation': False,
                'terminate_when_unhealthy': False,
                'healthy_reward': 1.0,
                'reset_noise_scale': 0,
            },
        },
        'Ant': {
            'Parameterizable-v3': {
                'exclude_current_positions_from_observation': False,
                'terminate_when_unhealthy': False,
                'healthy_reward': 1.0,
            },
            'Maze-v0': {
                'exclude_current_positions_from_observation': False,
                'terminate_when_unhealthy': True,
                'reset_noise_scale': 0,
            },
        },
        'Humanoid': {
            'Parameterizable-v3': {
                'exclude_current_positions_from_observation': False,
                'terminate_when_unhealthy': False,
                'healthy_reward': 1.0,
                'reset_noise_scale': 0,
            },
        },
        'Pusher2d': {  # 3 DoF
            'Default-v3': {
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 1.0,
                'goal': (0, -1),
            },
            'DefaultReach-v0': {
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'ImageDefault-v0': {
                'image_shape': (32, 32, 3),
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 3.0,
            },
            'ImageReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'BlindReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            }
        },
        'Point2DEnv': {
            'Default-v0': {
                'observation_keys': ('state_observation', ),
                'goal_keys': ('state_desired_goal', ),
                'terminate_on_success': True,
                'fixed_goal': (5.0, 5.0),
                'reset_positions': ((-5.0, -5.0), ),
            },
            'Wall-v0': {
                'observation_keys': ('state_observation', ),
                'goal_keys': ('state_desired_goal', ),
                'terminate_on_success': False,
                'fixed_goal': (5.0, 4.0),
                # 'fixed_goal': (0.0, 0.0),
                # 'fixed_goal': (4.0, 0.0),
                'reset_positions': (
                    # (-5.0, -5.0),
                    (-5.0, -4.0),
                    # (-5.0, -3.0),
                ),
                # 'reset_positions': None,
                'wall_shape': tune.grid_search(['zigzag']),
                'discretize': False,
                'target_radius': 0.1,
                'reward_type': 'sparse',
            },
            'ImageWall-v0': {
                'observation_keys': ('image_observation', ),

                'image_shape': (64, 64, 3),
                'render_size': 64,
                'wall_shape': 'zigzag',
                'images_are_rgb': True,
                'render_onscreen': False,
            },
        },
        'Sawyer': {
            task_name: {
                'has_renderer': False,
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
                'reward_shaping': tune.grid_search([True, False]),
            }
            for task_name in (
                    'Lift',
                    'NutAssembly',
                    'NutAssemblyRound',
                    'NutAssemblySingle',
                    'NutAssemblySquare',
                    'PickPlace',
                    'PickPlaceBread',
                    'PickPlaceCan',
                    'PickPlaceCereal',
                    'PickPlaceMilk',
                    'PickPlaceSingle',
                    'Stack',
            )
        },
        'DClaw3': {
            'ScrewV2-v0': {
                'object_target_distance_reward_fn': NegativeLogLossFn(1e-10),
                'hand_position_cost_coeff': 0,
                'hand_velocity_cost_coeff': 0,
                'hand_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (0, 0),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (0, 0),
                'goal_keys': (
                    'desired_hand_position',
                    'desired_hand_velocity',
                    'desired_hand_acceleration',
                    'desired_object_position',
                    'desired_object_position_sin',
                    'desired_object_position_cos',
                    'desired_object_velocity',
                ),
            },
            'ImageScrewV2-v0': {
                'object_target_distance_reward_fn': NegativeLogLossFn(1e-10),
                'hand_position_cost_coeff': 0,
                'hand_velocity_cost_coeff': 0,
                'hand_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (np.pi, np.pi),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (0, 0),
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1
                    },
                },
                'observation_keys': (
                    'pixels',

                    'hand_position',
                    'hand_velocity',
                    'hand_acceleration',

                    # These are supposed to not be fed to the models,
                    # but they are here just for the reward computation
                    # when we set the goal from outside of the environment.
                    'object_position',
                    'object_position_sin',
                    'object_position_cos',
                    'object_velocity',

                    'desired_hand_position',
                    'desired_hand_velocity',
                    'desired_pixels',
                ),
                'goal_keys': (
                    'desired_hand_position',
                    'desired_hand_velocity',
                    'desired_pixels',
                ),
            },
        },
        'HardwareDClaw3': {
            'ScrewV2-v0': {
                'object_target_distance_reward_fn': NegativeLogLossFn(1e-10),
                'hand_position_cost_coeff': 0,
                'hand_velocity_cost_coeff': 0,
                'hand_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (0, 0),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (0, 0),
                'goal_keys': (
                    'desired_hand_position',
                    'desired_hand_velocity',
                    'desired_hand_acceleration',
                    'desired_object_position',
                    'desired_object_position_sin',
                    'desired_object_position_cos',
                    'desired_object_velocity',
                ),
            },
            'ImageScrewV2-v0': {
                'object_target_distance_reward_fn': NegativeLogLossFn(1e-10),
                'hand_position_cost_coeff': 0,
                'hand_velocity_cost_coeff': 0,
                'hand_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (np.pi, np.pi),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (0, 0),
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1
                    },
                },
                'observation_keys': (
                    'hand_position',
                    'hand_velocity',
                    'pixels'
                ),
            },
        },
        'DClaw': {
            'PoseStatic-v0': {},
            'PoseDynamic-v0': {},
            'TurnFixed-v0': {
                'use_dict_obs': True,
                'reward_keys': ('object_to_target_angle_dist_cost', ),
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': 1,
                        # 'camera_name': 'track',
                    },
                },
                'camera_settings': {
                    'azimuth': 0,
                    'elevation': -45,
                    'distance': 0.25,
                    'lookat': (0, 0, 1.25e-1),
                },
                # 'observation_keys': ('claw_qpos', 'last_action', 'pixels'),
                'reward_keys': ('object_to_target_angle_dist_cost', ),
            },
            'TurnRandom-v0': {},
            'TurnRandomResetSingleGoal-v0': {
                'use_dict_obs': True,
                # 'pixel_wrapper_kwargs': {
                #     'observation_key': 'pixels',
                #     'pixels_only': False,
                #     'render_kwargs': {
                #         'width': 32,
                #         'height': 32,
                #         'camera_id': 1,
                #         # 'camera_name': 'track',
                #     },
                # },
                # 'observation_keys': ('claw_qpos', 'last_action', 'pixels'),
                'reward_keys': ('object_to_target_angle_dist_cost', ),
            },
            'TurnRandomDynamics-v0': {},
            'ScrewFixed-v0': {},
            'ScrewRandom-v0': {},
            'ScrewRandomDynamics-v0': {},
            'TurnFreeValve3ResetFree-0v': {},
        },
    },
    'dm_control': {
        'ball_in_cup': {
            'catch': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'cheetah': {
            'run': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'finger': {
            'spin': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
    },
}


NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 200,
    'gym': {
        DEFAULT_KEY: 200,
        'Swimmer': {
            DEFAULT_KEY: int(1e4),
        },
        'Hopper': {
            DEFAULT_KEY: int(1e3),
        },
        'HalfCheetah': {
            DEFAULT_KEY: int(3e3),
        },
        'Walker2d': {
            DEFAULT_KEY: int(3e3),
        },
        'Ant': {
            DEFAULT_KEY: int(3e3),
        },
        'Humanoid': {
            DEFAULT_KEY: int(1e4),
        },
        'Pusher2d': {
            DEFAULT_KEY: int(2e3),
        },
        'HandManipulatePen': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateEgg': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateBlock': {
            DEFAULT_KEY: int(1e4),
        },
        'HandReach': {
            DEFAULT_KEY: int(1e4),
        },
        'Point2DEnv': {
            DEFAULT_KEY: int(50),
        },
        'Reacher': {
            DEFAULT_KEY: int(200),
        },
        'Pendulum': {
            DEFAULT_KEY: 10,
        },
        'DClaw3': {
            DEFAULT_KEY: int(300),
        },
        'HardwareDClaw3': {
            DEFAULT_KEY: int(150),
        },
        'DClaw': {
            DEFAULT_KEY: int(300),
        },
    },
    'dm_control': {
        DEFAULT_KEY: 200,
        'ball_in_cup': {
            DEFAULT_KEY: int(2e4),
        },
        'cheetah': {
            DEFAULT_KEY: int(2e4),
        },
        'finger': {
            DEFAULT_KEY: int(2e4),
        },
    },
    'robosuite': {
        DEFAULT_KEY: 200,
    }
}

MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Point2DEnv': {
            DEFAULT_KEY: 50,
        },
        'Pendulum': {
            DEFAULT_KEY: 200,
        },
        'DClaw3': {
            DEFAULT_KEY: 250,
            'ScrewV2-v0': 250,
            'ImageScrewV2-v0': 250,
        },
        'HardwareDClaw3': {
            DEFAULT_KEY: 250,
        },
        'DClaw': {
            DEFAULT_KEY: 100,
        },
    },
}


NUM_CHECKPOINTS = 10


def get_supervision_schedule_params(domain):
    DECAY_STEPS_AND_LABELS_EVERY_N_STEPS = {
        'GoalHalfCheetah': (
            (1000.0, 10),
        ),
        'GoalAnt': (
            (3000.0, 15),
        ),
        'GoalHopper': (
            (2000.0, 20),
        ),
        'Point2DEnv': ((50, 1), ),
        'DClaw3': ((100, 9), ),
        'DClaw': ((300, 1), ),
        'HardwareDClaw3': ((100, 9), ),
        'Swimmer': ((int(1e4), int(1e4)), ),
    }[domain]
    # SCHEDULER_TYPES = ('linear', 'logarithmic')
    SCHEDULER_TYPES = ('linear', )
    return tune.grid_search([
        {
            'type': scheduler_type,
            'kwargs': {
                'start_labels': 1,
                'decay_steps': decay_steps,
                'end_labels': decay_steps // labels_every_n_steps,
                **(
                    {'decay_rate': 1e-2}
                    if scheduler_type == 'logarithmic'
                    else {}
                )
            }
        }
        for scheduler_type
        in SCHEDULER_TYPES

        for decay_steps, labels_every_n_steps
        in DECAY_STEPS_AND_LABELS_EVERY_N_STEPS
    ])


def fixed_path_length(spec):
    environment_params = (
        spec.get('config', spec)['environment_params']['training'])
    domain = environment_params['domain']
    environment_kwargs = environment_params['kwargs']

    if domain == 'Point2DEnv':
        return not environment_kwargs.get('terminate_on_success', False)
    elif domain in (
            'Swimmer', 'HalfCheetah', 'CurriculumPointEnv'):
        return True
    elif domain in ('GoalReacher', 'GoalPendulum'):
        return True
    else:
        if domain in (('Ant', 'Humanoid', 'Hopper', 'Walker')
                      + ('GoalSwimmer', 'GoalHalfCheetah', 'GoalHopper')
                      + ('GoalWalker', 'GoalAnt', 'GoalHumanoid')):
            return not environment_kwargs.get('terminate_when_unhealthy', True)

    raise NotImplementedError


def replay_pool_params(spec):
    config = spec.get('config', spec)
    params = {
        'type': 'GoalReplayPool',
        'kwargs': {
            'max_size': int(1e6),
        }
    }

    return params


def distance_pool_params(spec):
    config = spec.get('config', spec)
    params = {
        'type': 'DistancePool',
        'kwargs': {
            'max_size': {
                'SupervisedMetricLearner': int(1e5),
            }.get(config['metric_learner_params']['type'], int(1e6)),
            'max_pair_distance': None,
        },
    }
    return params


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    initial_exploration_steps = 10 * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def get_environment_params(universe, domain, task):
    environment_params = (
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task
    max_path_length = get_max_path_length(universe, domain, task)

    distance_estimator_type = 'FeedforwardDistanceEstimator'
    # distance_estimator_type = 'DistributionalFeedforwardDistanceEstimator'

    def metric_learner_kwargs(spec):
        spec = spec.get('config', spec)

        shared_kwargs = {
            'distance_learning_rate': 3e-4,
            'n_train_repeat': 1,
        }

        if (spec['metric_learner_params']['type']
            == 'TemporalDifferenceMetricLearner'):
            kwargs = {
                **shared_kwargs,
                **{
                    'train_every_n_steps': 1,
                    'ground_truth_terminals': False,
                },
            }
        elif (spec['metric_learner_params']['type']
              in ['SupervisedMetricLearner', 'DistributionalSupervisedMetricLearner']):
            kwargs = {
                **shared_kwargs,
                **{
                    'train_every_n_steps': (
                        {
                            'GoalHalfCheetah': 64,
                            'GoalAnt': 64,
                            'GoalHopper': 64,
                            'Point2DEnv': 1,
                            'DClaw3': 1,
                            'DClaw': tune.grid_search([32]),
                            'HardwareDClaw3': 1,
                        }[domain]
                    ),
                },
            }
        elif spec['metric_learner_params']['type'] == 'HingeMetricLearner':
            kwargs = {
                **shared_kwargs,
                **{
                    'train_every_n_steps': 1,

                    'lambda_learning_rate': 3e-4,
                    'constraint_exp_multiplier': 0.0,
                    'objective_type': 'linear',
                    'step_constraint_coeff': 1.0,

                    'zero_constraint_threshold': 0.0,

                    'max_distance': 10 + max_path_length,
                },
            }
        else:
            raise NotImplementedError(spec['metric_learner_params']['type'])

        return kwargs

    variant_spec = {
        'prefix': '{}/{}/{}'.format(universe, domain, task),
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task),
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': tune.sample_from(lambda spec: ({
                    **spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['kwargs'],
                }))
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
                'squash': True,
                'observation_keys': ('claw_qpos', 'last_action', 'pixels'),
                'observation_preprocessors_params': {},
            },
        },
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                'observation_preprocessors_params': {}
            }
        },
        'algorithm_params': {
            'type': 'MetricLearningAlgorithm',

            'kwargs': {
                'epoch_length': 1000,
                'n_epochs': get_num_epochs(universe, domain, task),
                'n_initial_exploration_steps': tune.sample_from(
                    get_initial_exploration_steps),
                'train_every_n_steps': 1,
                'n_train_repeat': 1,
                'reparameterize': True,
                'eval_render_kwargs': {},
                # {
                #     'mode': 'rgb_array',
                #     'width': 100,
                #     'height': 100,
                #     'camera_id': -1,
                #     # 'camera_name': 'agentview', # ('frontview', 'birdview', 'agentview')
                # },
                'eval_n_episodes': 1,
                'eval_deterministic': True,

                'lr': 3e-4,
                'discount': 0.99,
                'target_update_interval': 1,
                'tau': 0.005,
                'target_entropy': 'auto',
                'reward_scale': 1.0,
                'action_prior': 'uniform',
                'save_full_state': False,

                'plot_distances': domain == 'Point2DEnv',
                'use_distance_for': tune.grid_search([
                    'reward',
                    # 'value',
                    # 'telescope_reward',
                ]),
                'final_exploration_proportion': 0.2,
            }
        },
        # 'target_proposer_params': {
        #     'type': 'UnsupervisedTargetProposer',
        #     'kwargs': {
        #         'target_proposal_rule': tune.grid_search([
        #             'farthest_estimate_from_first_observation',
        #             'random_weighted_estimate_from_first_observation',
        #             'random',
        #         ]),
        #         'random_weighted_scale': 1.0,
        #         'target_candidate_strategy': tune.grid_search([
        #             # 'all_steps',
        #             'last_steps'
        #         ]),
        #     },
        # },
        # 'target_proposer_params': {
        #     'type': 'RandomTargetProposer',
        #     'kwargs': {
        #         'target_proposal_rule': tune.grid_search([
        #             'uniform_from_environment',
        #             'uniform_from_pool'
        #         ]),
        #     }
        # },
        'target_proposer_params': {
            'type': 'SemiSupervisedTargetProposer',
            'kwargs': {
                'supervision_schedule_params': get_supervision_schedule_params(
                    domain),
                'target_candidate_strategy': 'last_steps',
                'target_candidate_window': int(1e3),
            },
        },
        'replay_pool_params': tune.sample_from(replay_pool_params),
        'distance_pool_params': tune.sample_from(distance_pool_params),
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': max_path_length,
                'min_pool_size': max_path_length,
                'batch_size': 256,
                'store_last_n_paths': tune.sample_from(lambda spec: (
                    2 * spec.get('config', spec)
                    ['replay_pool_params']
                    ['kwargs']
                    ['max_size'] // max_path_length)),
            }
        },
        'metric_learner_params': {
            'type': tune.grid_search([
                # 'TemporalDifferenceMetricLearner',
                'SupervisedMetricLearner',
                # 'DistributionalSupervisedMetricLearner',
                # 'HingeMetricLearner',
            ]),
            # 'kwargs': tune.sample_from(metric_learner_kwargs),
            'kwargs': {
                'train_every_n_steps': tune.grid_search([8, 16, 32, 64]),
                'distance_learning_rate': 3e-4,
                'n_train_repeat': 1,
            }
        },
        'distance_estimator_params': {
            'type': distance_estimator_type,
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                **({'n_bins': 51}
                    if (distance_estimator_type
                        == 'DistributionalFeedforwardDistanceEstimator')
                    else {}),
                'observation_preprocessors_params': {},
                'hidden_layer_sizes': (256, 256),
                'activation': 'relu',
                'output_activation': 'linear',
                'condition_with_action': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['metric_learner_params']
                    ['type']
                    == 'TemporalDifferenceMetricLearner')),
                'target_input_type': tune.grid_search([
                    # 'full',
                    'xy_coordinates',
                    # 'xy_velocities',
                ]),
            }
        },
        'lambda_estimator_params': {
            'type': 'FeedforwardLambdaEstimator',
            'kwargs': {
                'observation_keys': variant_equals(
                    'distance_estimator_params',
                    'kwargs',
                    'observation_keys'),
                'observation_preprocessors_params': variant_equals(
                    'distance_estimator_params',
                    'kwargs',
                    'observation_preprocessors_params'),
                'hidden_layer_sizes': variant_equals(
                    'distance_estimator_params',
                    'kwargs',
                    'hidden_layer_sizes'),
                'activation': 'relu',
                'output_activation': 'softplus',
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': (
                get_num_epochs(universe, domain, task) // NUM_CHECKPOINTS),
            'checkpoint_replay_pool': False,
        },
    }

    if is_image_env(universe, domain, task, variant_spec):
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': 'layer',
                'downsampling_type': 'conv',
            },
        }

        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                ))))
        variant_spec['distance_estimator_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                ))))

    return variant_spec
