from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev
from examples.development.variants import is_image_env
from examples.utils import variant_equals
from sac_envs.envs.dclaw.dclaw3_screw_v2 import LinearLossFn, NegativeLogLossFn


DEFAULT_LAYER_SIZE = 256

ENVIRONMENT_PARAMS = {
    'Swimmer': {
        'Parameterizable-v3': {
            'exclude_current_positions_from_observation': False,
            'reset_noise_scale': 0,
        },
        'Maze-v0': {
            'exclude_current_positions_from_observation': False,
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
    'HalfCheetah': {
        'Parameterizable-v3': {
            'exclude_current_positions_from_observation': False,
        },
    },
    'Hopper': {
        'Parameterizable-v3': {
            'exclude_current_positions_from_observation': False,
            'terminate_when_unhealthy': False,
            'healthy_reward': 1.0,
            'reset_noise_scale': 0,
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
    'Humanoid': {
        'Parameterizable-v3': {
            'exclude_current_positions_from_observation': False,
            'terminate_when_unhealthy': False,
            'healthy_reward': 1.0,
            'reset_noise_scale': 0,
        },
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('state_observation', 'state_desired_goal'),
            'goal_key_map': {
                'state_desired_goal': 'state_observation',
            },
            'terminate_on_success': True,
            'fixed_goal': (5.0, 5.0),
            'reset_positions': ((-5.0, -5.0), ),
        },
        'Wall-v0': {
            'observation_keys': ('state_observation', 'state_desired_goal'),
            'goal_key_map': {
                'state_desired_goal': 'state_observation',
            },
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
            'goal_key_map': {
                'desired_hand_position': 'hand_position',
                'desired_hand_velocity': 'hand_velocity',
                'desired_hand_acceleration': 'hand_acceleration',
                'desired_object_position': 'object_position',
                'desired_object_position_sin': 'object_position_sin',
                'desired_object_position_cos': 'object_position_cos',
                'desired_object_velocity': 'object_velocity',
            },
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
            'goal_key_map': {
                f'desired_{key}': key
                for key in (
                    'hand_position',
                    'hand_velocity',
                    'pixels',
                )
            },
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
            'goal_key_map': {
                'desired_hand_position': 'hand_position',
                'desired_hand_velocity': 'hand_velocity',
                'desired_hand_acceleration': 'hand_acceleration',
                'desired_object_position': 'object_position',
                'desired_object_position_sin': 'object_position_sin',
                'desired_object_position_cos': 'object_position_cos',
                'desired_object_velocity': 'object_velocity',
            },
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
            'observation_keys': ('hand_position', 'hand_velocity', 'pixels'),
        },
    },
}

DEFAULT_NUM_EPOCHS = 200
NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e3 + 1),
    'GoalSwimmer': int(3e3 + 1),
    'Hopper': int(5e3 + 1),
    'HalfCheetah': int(1e4 + 1),
    'Walker': int(1e4 + 1),
    'Ant': int(1e4 + 1),
    'Humanoid': int(3e4 + 1),
    'Pusher2d': int(2e3 + 1),
    'HandManipulatePen': int(1e4 + 1),
    'HandManipulateEgg': int(1e4 + 1),
    'HandManipulateBlock': int(1e4 + 1),
    'HandReach': int(1e4 + 1),
    'DClaw3': int(150),
    'HardwareDClaw3': int(150),
    'Point2DEnv': int(50 + 1)
}


DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'GoalReacher': 200,
    'DClaw3': 250,
    'HardwareDClaw3': 250,
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
        'DClaw3': ((200, 10), ),
        'HardwareDClaw3': ((200, 10), ),
    }[domain]
    # SCHEDULER_TYPES = ('linear', 'logarithmic')
    SCHEDULER_TYPES = ('linear', )
    return tune.grid_search([
        {
            'type': scheduler_type,
            'kwargs': {
                'start_labels': 1,
                'decay_steps': decay_steps,
                'end_labels': decay_steps / labels_every_n_steps,
                **(
                    {'decay_rate': 0.25}
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


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task
    max_path_length = MAX_PATH_LENGTH_PER_DOMAIN.get(
        domain, DEFAULT_MAX_PATH_LENGTH)

    def metric_learner_kwargs(spec):
        spec = spec.get('config', spec)

        shared_kwargs = {
            'distance_learning_rate': 3e-4,
            'n_train_repeat': 1,

            'distance_input_type': tune.grid_search([
                'full',
                # 'xy_coordinates',
                # 'xy_velocities',
            ]),
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
              == 'SupervisedMetricLearner'):
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
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
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
                    **(
                        {
                            'fixed_goal': (5.0, 4.0),
                            'terminate_on_success': True,
                        }
                        if 'point' in domain
                        else {
                                'target_initial_position_range': (np.pi, np.pi),
                        }
                    )
                }))
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
                'squash': True,
                # 'observation_keys': ('state_observation', ),
                'observation_keys': (
                    'hand_position',
                    'hand_velocity',
                    'object_position',
                    'object_position_sin',
                    'object_position_cos',
                    'object_velocity',
                ),
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
                # 'observation_keys': ('state_observation', ),
                'observation_keys': (
                    'hand_position',
                    'hand_velocity',
                    'object_position',
                    'object_position_sin',
                    'object_position_cos',
                    'object_velocity',
                ),
                'observation_preprocessors_params': {}
            }
        },
        'algorithm_params': {
            'type': 'MetricLearningAlgorithm',

            'kwargs': {
                'epoch_length': 1000,
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'train_every_n_steps': 1,
                'n_train_repeat': 1,
                'n_initial_exploration_steps': int(1e3),
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

                'plot_distances': False,
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
                # 'HingeMetricLearner',
            ]),
            'kwargs': tune.sample_from(metric_learner_kwargs),
        },
        'distance_estimator_params': {
            'type': 'FeedforwardDistanceEstimator',
            'kwargs': {
                'observation_keys': (
                    'hand_position',
                    'hand_velocity',
                    'object_position',
                    'object_position_sin',
                    'object_position_cos',
                    'object_velocity',
                ),
                # 'observation_keys': ('state_observation', ),
                'observation_preprocessors_params': {},
                'hidden_layer_sizes': (256, 256),
                'activation': 'relu',
                'output_activation': 'linear',
                'condition_with_action': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['metric_learner_params']
                    ['type']
                    == 'TemporalDifferenceMetricLearner')),
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
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    if is_image_env(domain, task, variant_spec):
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
