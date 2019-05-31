from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev
from examples.utils import variant_equals


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
            'observation_keys': ('observation', ),
            'terminate_on_success': True,
            'fixed_goal': (5.0, 5.0),
            'reset_positions': ((-5.0, -5.0), ),
        },
        'Wall-v0': {
            'observation_keys': ('observation', ),
            'terminate_on_success': False,
            # 'fixed_goal': (5.0, 4.0),
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
    'DClaw3': int(5e2 + 1),
    'ImageDClaw3': int(5e3 + 1),
    'Point2DEnv': int(50 + 1)
}


DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'GoalReacher': 200,
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
        'Point2DEnv': ((50, 1), )
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


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task
    max_path_length = MAX_PATH_LENGTH_PER_DOMAIN.get(
        domain, DEFAULT_MAX_PATH_LENGTH)

    def metric_learner_kwargs(spec):
        spec = spec.get('config', spec)

        shared_kwargs = {
            'distance_learning_rate': 3e-4,
            'n_train_repeat': 1,

            'condition_with_action': (
                spec['metric_learner_params']['type']
                == 'TemporalDifferenceMetricLearner'),
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
                    'ground_truth_terminals': True,
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
                    # 'fixed_goal': (5.0, 4.0),
                    # 'terminate_on_success': True,
                }))
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
                'squash': True,
            },
        },
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {},
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
            }
        },
        'preprocessor_params': {},
        'algorithm_params': {
            'type': 'MetricLearningAlgorithm',

            'kwargs': {
                'epoch_length': 1000,
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'train_every_n_steps': 1,
                'n_train_repeat': 1,
                'n_initial_exploration_steps': int(1e4),
                'reparameterize': True,
                'eval_render_mode': None,
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
                'final_exploration_proportion': 0.1,
            }
        },
        'target_proposer_params': {
            'type': 'UnsupervisedTargetProposer',
            'kwargs': {
                'target_proposal_rule': tune.grid_search([
                    'farthest_estimate_from_first_observation',
                    'random_weighted_estimate_from_first_observation',
                    'random',
                ]),
                'random_weighted_scale': 1.0,
                'target_candidate_strategy': tune.grid_search([
                    'all_steps', 'last_steps'
                ]),
            },
        },
        # 'target_proposer_params': {
        #     'type': 'RandomTargetProposer',
        #     'kwargs': {
        #         'target_proposal_rule': tune.grid_search([
        #             'uniform_from_environment',
        #             'uniform_from_pool'
        #         ]),
        #     }
        # },
        # 'target_proposer_params': {
        #     'type': 'SemiSupervisedTargetProposer',
        #     'kwargs': {
        #         'supervision_schedule_params': get_supervision_schedule_params(
        #             domain),
        #     },
        # },
        'replay_pool_params': {
            'type': 'DistancePool',
            'kwargs': {
                'max_size': int(1e6),
                'on_policy_window': tune.sample_from(lambda spec: (
                    {
                        'SupervisedMetricLearner': 1000,
                    }.get(spec.get('config', spec)
                          ['metric_learner_params']
                          ['type'],
                          None)
                )),
                'max_pair_distance': None,
                'path_length': max_path_length,
                'fixed_path_length': tune.sample_from(fixed_path_length),
                'terminal_epsilon': 0.0,
            }
        },
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
                'TemporalDifferenceMetricLearner',
                # 'SupervisedMetricLearner',
                # 'HingeMetricLearner',
            ]),
            'kwargs': tune.sample_from(metric_learner_kwargs),
        },
        'distance_estimator_params': {
            'type': 'FeedforwardDistanceEstimator',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
                'activation': 'relu',
                'output_activation': 'linear',
            }
        },
        'lambda_estimator_params': {
            'type': 'FeedforwardLambdaEstimator',
            'kwargs': {
                'hidden_layer_sizes': variant_equals(
                    'distance_estimator_params',
                    'kwargs',
                    'hidden_layer_sizes'
                ),
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

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': (
                    variant_spec
                    ['environment_params']
                    ['training']
                    ['kwargs']
                    ['image_shape']),
                'output_size': None,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['preprocessor_params']
            )))
        variant_spec['distance_estimator_params']['kwargs'][
            'preprocessor_params'] = (
            tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['preprocessor_params']
            )))

    return variant_spec
