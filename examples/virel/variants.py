from ray import tune
import tensorflow as tf

from softlearning.utils.dict import deep_update

from examples.development import (
    get_variant_spec as get_development_variant_spec)
from examples.development.variants import (
    ALGORITHM_PARAMS_BASE,
    get_algorithm_params)


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_development_variant_spec(args)

    variant_spec['algorithm_params']['type'] = 'VIREL'
    virel_algorithm_kwargs = {
        'type': 'VIREL',
        'kwargs': {
            'policy_lr': 3e-4,
            'Q_lr': 3e-4,
            'tau': 5e-3,
            'beta_scale': tune.grid_search([1.0]),
            'beta_update_type': tune.grid_search([
                # None,
                # 'MSBE',
                # 'MSBBE',
                'uncertainty',
            ]),
            'target_update_interval': 1,

            'n_initial_exploration_steps': int(1e3),

            'reward_scale': 1.0,
            'discount': 0.99,
            'Q_update_type': tune.grid_search([
                # 'MSBBE',
                'MSBE',
            ]),
        },
    }

    variant_spec['algorithm_params'] = deep_update(
        ALGORITHM_PARAMS_BASE,
        virel_algorithm_kwargs,
        get_algorithm_params(universe, domain, task),
    )

    variant_spec['uncertainty_estimator_params'] = {
        'type': 'FeedforwardEnsemble',
        'kwargs': tune.sample_from(lambda spec: {
            'ensemble_size': (spec.get('config', spec)
                              ['Q_params']
                              ['kwargs']
                              ['ensemble_size']),
            'output_size': 10,
            'hidden_layer_sizes': (spec.get('config', spec)
                                   ['Q_params']
                                   ['kwargs']
                                   ['hidden_layer_sizes']),
            'prior_kernel_initializer': (spec.get('config', spec)
                                         ['Q_params']
                                         ['kwargs']
                                         ['prior_kernel_initializer'])
        }),
    }

    return variant_spec
