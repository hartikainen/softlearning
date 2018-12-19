from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev
from examples.utils import variant_equals


DEFAULT_LAYER_SIZE = 256

ENV_PARAMS = {
    'Point2DEnv': {
        'Default': {
            'observation_keys': ('observation', ),
            'fixed_goal': (5.0, 5.0),
            'reset_positions': ((-5.0, -5.0), ),
        },
        'Wall': {
            'observation_keys': ('observation', ),
            'fixed_goal': (5.0, 5.0),
            # 'fixed_goal': (4.0, 0.0),
            'reset_positions': ((-5.0, -5.0), ),
            'wall_shape': tune.grid_search(['zigzag']),
            'discretize': False,
        }
    }
}

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(5e2 + 1),
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
    'Point2DEnv': int(200 + 1)
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
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
                'squash': True,
            },
        },
        'Q_params': {
            'type': 'double_metric_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (DEFAULT_LAYER_SIZE, ) * 2,
            }
        },
        'preprocessor_params': {},
        'algorithm_params': {
            'type': 'MetricLearningSoftActorCritic',

            'kwargs': {
                'epoch_length': 1000,
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
            }
        },
        'replay_pool_params': {
            'type': 'MetricLearningPool',
            'kwargs': {
                'max_size': 1e6,
            }
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
            'seed': lambda spec: np.random.randint(0, 10000),
            'checkpoint_at_end': False,
            'checkpoint_frequency': 0,
        },
    }

    return variant_spec
