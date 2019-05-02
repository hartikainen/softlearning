from copy import deepcopy

from .metric_learner import (
    HingeMetricLearner,
    OnPolicyMetricLearner,
    TemporalDifferenceMetricLearner)
from .lambda_estimator import get_lambda_estimator_from_variant
from .distance_estimator import get_distance_estimator_from_variant


def get_metric_learner_from_variant(variant, env, policy):
    distance_estimator = get_distance_estimator_from_variant(variant, env)

    metric_learner_params = variant['metric_learner_params']
    metric_learner_type = metric_learner_params['type']
    metric_learner_kwargs = deepcopy(metric_learner_params['kwargs'])

    metric_learner_kwargs.update({
        'env': env,
        'policy': policy,
        'observation_shape': env.active_observation_shape,
        'action_shape': env.action_space.shape,
        'distance_estimator': distance_estimator,
    })

    metric_learner_type = metric_learner_params['type']
    if metric_learner_type == 'OnPolicyMetricLearner':
        metric_learner = OnPolicyMetricLearner(**metric_learner_kwargs)
    if metric_learner_type == 'TemporalDifferenceMetricLearner':
        metric_learner = TemporalDifferenceMetricLearner(
            **metric_learner_kwargs)
    elif metric_learner_type == 'HingeMetricLearner':
        metric_learner_kwargs['lambda_estimators'] = {
            lambda_name: get_lambda_estimator_from_variant(variant)
            for lambda_name in
            ['step', 'zero', 'max_distance', 'triangle_inequality']
        }
        metric_learner = HingeMetricLearner(**metric_learner_kwargs)

    return metric_learner


def get_target_proposer_from_variant(variant, *args, **kwargs):
    from . import target_proposer as target_proposer_lib

    target_proposer_params = variant['target_proposer_params']
    target_proposer_type = target_proposer_params['type']
    target_proposer_kwargs = deepcopy(target_proposer_params['kwargs'])

    target_proposer_class = getattr(target_proposer_lib, target_proposer_type)

    if target_proposer_type == 'SemiSupervisedTargetProposer':
        target_proposer_kwargs.update({
            'epoch_length': (
                variant['algorithm_params']['kwargs']['epoch_length']),
            'max_path_length': (
                variant['sampler_params']['kwargs']['max_path_length']),
        })

    target_proposer = target_proposer_class(
        *args, **target_proposer_kwargs, **kwargs)

    return target_proposer
