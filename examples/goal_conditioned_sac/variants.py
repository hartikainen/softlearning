from ray import tune

from examples.metric_learning import (
    get_variant_spec as get_metric_learning_variant_spec)
from examples.development import (
    get_variant_spec as get_development_variant_spec)


def get_variant_spec(args):
    metric_learning_variant_spec = get_metric_learning_variant_spec(args)

    variant_spec = get_development_variant_spec(args)
    variant_spec['Q_params']['type'] = (
        'double_goal_conditioned_feedforward_Q_function')
    variant_spec['algorithm_params']['type'] = (
        'HERSAC')
    variant_spec['exploration_policy_params']['type'] = (
        'GoalConditionedUniformPolicy')
    variant_spec['policy_params']['type'] = 'GoalConditionedGaussianPolicy'
    variant_spec['sampler_params']['type'] = 'GoalSampler'

    variant_spec['env_params'] = metric_learning_variant_spec['env_params']
    variant_spec['replay_pool_params'] = metric_learning_variant_spec[
        'replay_pool_params']
    variant_spec['replay_pool_params']['kwargs'].update({
        'on_policy_window': None,
        'use_distances': False,
        'her_strategy': {
            'type': tune.grid_search(['episode', 'final', 'future', 'random']),
            'resampling_probability': tune.grid_search([
                0.0, 0.5, 0.8, 1.0])
        }
    })

    return variant_spec
