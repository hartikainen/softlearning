from ray import tune

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
            'beta_scale': tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
            'beta_update_type': tune.grid_search([
                # None,
                'MSBE',
                # 'MSBBE',
                # 'uncertainty',
            ]),
            'target_update_interval': 1,

            'n_initial_exploration_steps': int(1e3),

            'reward_scale': 1.0,
            'discount': 0.99,
            'TD_target_model_update_interval': 100,
            'Q_update_type': tune.grid_search([
                # 'MSBBE',
                'MSBE',
            ]),
            'diagonal_noise_scale': tune.grid_search([1e-3]),
            'uncertainty_model_type': tune.grid_search(['online']),
        },
    }

    variant_spec['algorithm_params'] = deep_update(
        ALGORITHM_PARAMS_BASE,
        virel_algorithm_kwargs,
        get_algorithm_params(universe, domain, task),
    )

    return variant_spec
