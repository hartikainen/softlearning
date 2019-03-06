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
    variant_spec['algorithm_params']['type'] = 'HERSAC'
    variant_spec['algorithm_params']['kwargs'] = {
        key: value
        for key, value in
        metric_learning_variant_spec['algorithm_params']['kwargs'].items()
        if key not in ('save_full_state', 'use_distance_for')
    }
    variant_spec['exploration_policy_params']['type'] = (
        'GoalConditionedUniformPolicy')
    variant_spec['policy_params']['type'] = 'GoalConditionedGaussianPolicy'
    variant_spec['sampler_params'] = (
        metric_learning_variant_spec['sampler_params'].copy())
    variant_spec['sampler_params']['type'] = 'GoalSampler'

    variant_spec['target_proposer_params'] = metric_learning_variant_spec[
        'target_proposer_params'].copy()

    variant_spec['env_params'] = (
        metric_learning_variant_spec['env_params'].copy())

    variant_spec['replay_pool_params'] = metric_learning_variant_spec[
        'replay_pool_params'].copy()
    variant_spec['replay_pool_params']['kwargs'].update({
        'on_policy_window': None,
        'use_distances': False,
        'her_strategy': {
            'type': 'future',
            'resampling_probability': tune.grid_search([0.5, 0.8]),
        }
    })

    return variant_spec
