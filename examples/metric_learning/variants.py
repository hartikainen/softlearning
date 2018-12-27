from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev
from examples.utils import variant_equals


DEFAULT_LAYER_SIZE = 256

ENV_PARAMS = {
    'Swimmer': {
        'CustomDefault': {
            'exclude_current_positions_from_observation': False,
        },
    },
    'Ant': {
        'CustomDefault': {
            'exclude_current_positions_from_observation': False,
        },
    },
    'Humanoid': {
        'CustomDefault': {
            'exclude_current_positions_from_observation': False,
        },
    },
    'Point2DEnv': {
        'Default': {
            'observation_keys': ('observation', ),
            'fixed_goal': (5.0, 5.0),
            'reset_positions': ((-5.0, -5.0), ),
        },
        'Wall': {
            'observation_keys': ('observation', ),
            'fixed_goal': (5.0, 5.0),
            # 'fixed_goal': (0.0, 0.0),
            # 'fixed_goal': (4.0, 0.0),
            'reset_positions': (
                # (-5.0, -5.0),
                (-5.0, -4.0),
                # (-5.0, -3.0),
            ),
            'wall_shape': tune.grid_search(['zigzag']),
            'discretize': False,
        }
    }
}

DEFAULT_NUM_EPOCHS = 200
NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e3 + 1),
    'Hopper': int(3e3 + 1),
    'HalfCheetah': int(3e3 + 1),
    'Walker': int(3e3 + 1),
    'Ant': int(3e3 + 1),
    'Humanoid': int(1e4 + 1),
    'Pusher2d': int(2e3 + 1),
    'HandManipulatePen': int(1e4 + 1),
    'HandManipulateEgg': int(1e4 + 1),
    'HandManipulateBlock': int(1e4 + 1),
    'HandReach': int(1e4 + 1),
    'DClaw3': int(5e2 + 1),
    'ImageDClaw3': int(5e3 + 1),
    'Point2DEnv': int(100 + 1)
}


DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50
}

NUM_CHECKPOINTS = 10


def get_variant_spec(universe, domain, task, policy):
    variant_spec = {
        'prefix': '{}/{}/{}'.format(universe, domain, task),
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
                'squash': True,
            },
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
                'n_initial_exploration_steps': int(1e3),
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

                'plot_distances': True,
                'temporary_goal_update_rule': tune.grid_search([
                    # 'closest_l2_from_goal',
                    'farthest_estimate_from_first_observation',
                    # 'operator_query_last_step',
                    # 'random',
                ]),
                'use_distance_for': tune.grid_search([
                    'reward',
                    # 'value',
                ]),
            }
        },
        'replay_pool_params': {
            'type': 'DistancePool',
            'kwargs': {
                'max_size': int(1e6),
                'on_policy_window': tune.grid_search([
                    2000, 4000, 8000,
                ]),
                # lambda spec: (
                #     2 * spec.get('config', spec)
                #     ['sampler_params']
                #     ['kwargs']
                #     ['max_path_length']
                #     if (spec.get('config', spec)
                #         ['metric_learner_params']
                #         ['type']
                #         == 'OnPolicyMetricLearner')
                #     else None),
                'max_pair_distance': None,
                'path_length': variant_equals(
                    'sampler_params',
                    'kwargs',
                    'max_path_length')
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 512,
            }
        },
        'metric_learner_params': {
            'type': tune.grid_search([
                'OnPolicyMetricLearner',
                # 'MetricLearner',
            ]),
            'kwargs': {
                'distance_learning_rate': 3e-4,
                'lambda_learning_rate': 3e-4,
                'train_every_n_steps': tune.grid_search([16, 32, 64, 128]),
                # 'train_every_n_steps': lambda spec: (
                #     {
                #         'OnPolicyMetricLearner': 128,
                #         'MetricLearner': 1,
                #     }[spec.get('config', spec)
                #       ['metric_learner_params']
                #       ['type']]),
                'n_train_repeat': 1,

                'constraint_exp_multiplier': 0.3,
                'objective_type': 'squared',
                'step_constraint_coeff': 1.0,

                'zero_constraint_threshold': 0.0,

                'max_distance': lambda spec: (
                    10 + spec.get('config', spec)
                    ['sampler_params']
                    ['kwargs']
                    ['max_path_length']),

                'condition_with_action': tune.grid_search([
                    # True,
                    False
                ]),
                'distance_input_type': tune.grid_search([
                    'full',
                    'xy_coordinates',
                    # 'xy_velocities',
                ]),
            },
        },
        'distance_estimator_params': {
            'type': 'FeedforwardDistanceEstimator',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
                'hidden_activation': 'relu',
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
                'hidden_activation': 'relu',
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

    return variant_spec
