from copy import deepcopy

from .metric_learner import MetricLearner, OnPolicyMetricLearner
from .lambda_estimator import get_lambda_estimator_from_variant
from .distance_estimator import get_distance_estimator_from_variant


def get_metric_learner_from_variant(variant, env):
    lambda_estimators = {
        lambda_name: get_lambda_estimator_from_variant(variant)
        for lambda_name in
        ['step', 'zero', 'max_distance', 'triangle_inequality']
    }

    distance_estimator = get_distance_estimator_from_variant(variant)

    metric_learner_params = variant['metric_learner_params']
    metric_learner_type = metric_learner_params['type']
    metric_learner_kwargs = deepcopy(metric_learner_params['kwargs'])

    metric_learner_kwargs.update({
        'env': env,
        'observation_shape': env.active_observation_shape,
        'action_shape': env.action_space.shape,
        'lambda_estimators': lambda_estimators,
        'distance_estimator': distance_estimator,
    })

    metric_learner_type = metric_learner_params['type']
    if metric_learner_type == 'OnPolicyMetricLearner':
        metric_learner = OnPolicyMetricLearner(**metric_learner_kwargs)
    elif metric_learner_type == 'MetricLearner':
        metric_learner = MetricLearner(**metric_learner_kwargs)

    return metric_learner
