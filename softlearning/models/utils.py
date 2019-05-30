from copy import deepcopy

import tensorflow as tf
from flatten_dict import flatten


from .lambda_estimator import get_lambda_estimator_from_variant
from .distance_estimator import get_distance_estimator_from_variant


def get_metric_learner_from_variant(variant, env, policy):
    from .metric_learner import (
        HingeMetricLearner,
        OnPolicyMetricLearner,
        TemporalDifferenceMetricLearner)
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


def get_inputs_for_nested_shapes(input_shapes, name=None):
    if isinstance(input_shapes, dict):
        return type(input_shapes)([
            (name, get_inputs_for_nested_shapes(value, name))
            for name, value in input_shapes.items()
        ])
    elif isinstance(input_shapes, (tuple, list)):
        if all(isinstance(x, int) for x in input_shapes):
            return tf.keras.layers.Input(shape=input_shapes, name=name)
        else:
            return type(input_shapes)((
                get_inputs_for_nested_shapes(input_shape, name=None)
                for input_shape in input_shapes
            ))
    elif isinstance(input_shapes, tf.TensorShape):
        return tf.keras.layers.Input(shape=input_shapes, name=name)

    raise NotImplementedError(input_shapes)


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


def flatten_input_structure(inputs):
    if isinstance(inputs, dict):
        inputs_flat_dict = flatten(inputs)
        inputs_flat = [
            inputs_flat_dict[key]
            for key in sorted(inputs_flat_dict.keys())
        ]
    elif isinstance(inputs, list):
        inputs_flat = list(inputs)
    elif isinstance(inputs, tuple):
        if all (isinstance(x, int) for x in inputs):
            inputs_flat = [inputs]
        else:
            inputs_flat = list(inputs)

    return inputs_flat


def create_inputs(input_shapes):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs_flat: a tuple of `tf.keras.layers.Input`s.

    TODO(hartikainen): Need to figure out a better way for handling the dtypes.
    """
    if isinstance(input_shapes, dict):
        inputs_flat_dict = flatten(input_shapes)
        inputs_flat = [
            tf.keras.layers.Input(
                shape=inputs_flat_dict[key],
                name=key[-1],
                dtype=(tf.uint8 # Image observation
                       if len(inputs_flat_dict[key]) == 3
                       else tf.float32) # Non-image
            )
            for key in sorted(inputs_flat_dict.keys())
        ]
    elif isinstance(input_shapes, list):
        inputs_flat = [
            tf.keras.layers.Input(shape=shape)
            for shape in input_shapes
        ]
    elif isinstance(input_shapes, tuple):
        if all (isinstance(x, int) for x in input_shapes):
            inputs_flat = [tf.keras.layers.Input(shape=input_shapes)]
        else:
            inputs_flat = [
                tf.keras.layers.Input(shape=shape)
                for shape in input_shapes
            ]

    return inputs_flat
