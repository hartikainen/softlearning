from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev
from examples.utils import variant_equals


DEFAULT_LAYER_SIZE = 256

ENV_PARAMS = {
    'Point2DEnv': {
        'default': {
            'observation_keys': ('observation', ),
        },
        'wall': {
            'observation_keys': ('observation', ),
        },
    }
}

NUM_EPOCHS_PER_DOMAIN = {
    'swimmer': int(5e2 + 1),
    'hopper': int(3e3 + 1),
    'half-cheetah': int(3e3 + 1),
    'walker': int(3e3 + 1),
    'ant': int(3e3 + 1),
    'humanoid': int(1e4 + 1),
    'pusher-2d': int(2e3 + 1),
    'sawyer-torque': int(1e3 + 1),
    'HandManipulatePen': int(1e4 + 1),
    'HandManipulateEgg': int(1e4 + 1),
    'HandManipulateBlock': int(1e4 + 1),
    'HandReach': int(1e4 + 1),
    'DClaw3': int(5e2 + 1),
    'ImageDClaw3': int(5e3 + 1),
}


DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50
}


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
            'reg': 1e-3,
            'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
            'reparameterize': True,
            'squash': True,
        },
        'V_params': {
            'type': 'metric_V_function',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
            }
        },
        'Q_params': {
            'type': 'double_metric_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
            }
        },
        'preprocessor_params': None,
        'algorithm_params': {
            'type': 'MetricLearningAlgorithmOne',

            'epoch_length': 1000,
            'n_epochs': NUM_EPOCHS_PER_DOMAIN[domain],
            'train_every_n_steps': 1,
            'n_train_repeat': 1,
            'n_initial_exploration_steps': int(1e3),
            'eval_render': False,
            'eval_n_episodes': 1,
            'eval_deterministic': True,

            'lr': 3e-4,
            'discount': tune.grid_search([0.99]),
            'target_update_interval': 1,
            'tau': 0.005,
            'target_entropy': 'auto',
            'reward_scale': 1.0,
            'action_prior': 'uniform',
            'save_full_state': False,
        },
        'replay_pool_params': {
            'type': 'MetricLearningPool',
            'max_size': 1e6,
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.grid_search(
                np.random.randint(0, 1000, 3).tolist()),
            'snapshot_mode': 'gap',
            'snapshot_gap': lambda spec: (
                spec.get('config', spec)
                ['algorithm_params']
                ['n_epochs'] // 10),
            'sync_pkl': True,
        },
    }

    return variant_spec
