from examples.metric_learning import (
    get_variant_spec as get_metric_learning_variant_spec)


def get_variant_spec(args):
    variant_spec = get_metric_learning_variant_spec(args)
    variant_spec['Q_params']['type'] = (
        'double_goal_conditioned_feedforward_Q_function')
    variant_spec['algorithm_params']['type'] = (
        'GoalConditionedMetricLearningAlgorithm')
    variant_spec['exploration_policy_params']['type'] = (
        'GoalConditionedUniformPolicy')
    variant_spec['policy_params']['type'] = 'GoalConditionedGaussianPolicy'
    variant_spec['sampler_params']['type'] = 'GoalSampler'
    variant_spec['replay_pool_params']['kwargs'].update({
        'her_strategy': {
            'type': 'future',
            'resampling_probability': 0.8
        },
    })

    return variant_spec
