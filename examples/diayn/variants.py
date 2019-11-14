from ray import tune
import numpy as np

from softlearning.utils.dict import deep_update
from examples.development.variants import get_variant_spec_image


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    base_variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm)

    variant_spec = deep_update(base_variant_spec, {
        'discriminator_params': {
            'kwargs': {
                'num_skills': tune.grid_search([10, 20]),
            },
        },
        'algorithm_params': {
            'type': 'DIAYN',
            'kwargs': {
                'target_entropy': None,
                'reward_scale': 1.0,
                'entropy_scale': 1/10,
            },
        },
        'environment_params': {
            'training': {
                'kwargs': {
                    'diayn_skill_wrapper_kwargs': {
                        'num_skills': tune.sample_from(
                            lambda spec: (
                                spec.get('config', spec)
                                ['discriminator_params']
                                ['kwargs']
                                ['num_skills']
                            )),
                    },
                }
            }
        }
    })

    return variant_spec
