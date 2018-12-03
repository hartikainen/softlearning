from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})


POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'DClaw3': 200,
    'ImageDClaw3': 200,
    'HardwareDClaw3': 200,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': int(1e3),
        'reparameterize': REPARAMETERIZE,
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'lr': 3e-4,
        'discount': 0.99,
        'target_update_interval': 1,
        'tau': 0.005,
        'target_entropy': 'auto',
        'reward_scale': 1.0,
        'store_extra_policy_info': False,
        'action_prior': 'uniform',
        'save_full_state': False,
    }
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'swimmer': int(3e2 + 1),
    'hopper': int(1e3 + 1),
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
    'DClaw3': int(2e2 + 1),
    'ImageDClaw3': int(5e3 + 1),
    'Point2DEnv': int(200 + 1),
    'Reacher': int(200 + 1),
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}


class NegativeLogLossFn(object):
    def __init__(self, eps):
        self._eps = eps

    def __call__(self, object_target_distance):
        return -np.log(object_target_distance + self._eps)

    def __str__(self):
        return str(f'eps={self._eps:e}')


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
    },
    'hopper': {  # 3 DoF
    },
    'half-cheetah': {  # 6 DoF
    },
    'walker': {  # 6 DoF
    },
    'ant': {  # 8 DoF
        'custom-default': {
            'survive_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'humanoid': {  # 17 DoF
        'custom-default': {
            'survive_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'pusher-2d': {  # 3 DoF
        'default': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'default-reach': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'image-default': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'image-reach': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'blind-reach': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'sawyer-torque': {

    },
    'DClaw3': {
        'ScrewV2': {
            'object_target_distance_reward_fn': tune.grid_search([
                *[
                    NegativeLogLossFn(eps)
                    for eps in [1e-1, 1e-2, 1e-4, 1e-6]
                ],
            ]),
            'pose_difference_cost_coeff': tune.grid_search([
                1e-4, 1e-3, 1e-2, 1e-1
            ]),
            'joint_velocity_cost_coeff': tune.grid_search([
                1e-4, 1e-3, 1e-2, 1e-1
            ]),
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, -np.pi),
        }
    },
    'ImageDClaw3': {
        'Screw': {
            'image_shape': (32, 32, 3),
            'object_target_distance_cost_coeff': 2.0,
            'pose_difference_cost_coeff': 0.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        }
    },
    'Point2DEnv': {
        'default': {
            'observation_keys': ('observation', ),
        },
        'wall': {
            'observation_keys': ('observation', ),
        },
    }
}

NUM_CHECKPOINTS = 5


def get_variant_spec(universe, domain, task, policy):
    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
        ),
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
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
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS
        },
    }

    return variant_spec


def get_variant_spec_image(universe, domain, task, policy, *args, **kwargs):
    variant_spec = get_variant_spec(
        universe, domain, task, policy, *args, **kwargs)

    if 'image' in task or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': variant_spec['env_params']['image_shape'],
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

    return variant_spec
