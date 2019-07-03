import types

from collections import OrderedDict

import numpy as np

import tensorflow as tf

from softlearning.preprocessors.utils import get_preprocessor_from_params
from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel
from softlearning.environments.gym.mujoco.utils import (
    LOCOMOTION_ENVS, POSITION_SLICES)


class DistributionalPicklableModel(PicklableModel):
    """Only return the expected distance when called. This keeps the API the
    same for distributional and non-distributional distance estimator. """

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)[0]

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)[0]

    def compute_all_outputs(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def feedforward_distance_estimator(input_shapes,
                                   *args,
                                   preprocessors=None,
                                   observation_keys=None,
                                   name='feedforward_distance_estimator',
                                   **kwargs):
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    model = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    model = PicklableModel(inputs_flat, model(preprocessed_inputs))
    model.observation_keys = observation_keys

    return model


def distributional_feedforward_distance_estimator(
        input_shapes,
        n_bins=10,
        max_distance=100,
        *args,
        preprocessors=None,
        observation_keys=None,
        name='feedforward_distance_estimator',
        **kwargs):
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    ff_model = feedforward_model(
        *args,
        output_size=n_bins,
        name=name,
        **kwargs
    )

    logits = ff_model(preprocessed_inputs)
    softmax = tf.keras.backend.softmax(logits)

    bin_values = tf.keras.backend.constant(
        np.linspace(0, max_distance, n_bins, dtype=np.float32).reshape(1, -1)
    )

    expectation = tf.keras.backend.sum(
        softmax * bin_values,
        axis=1, keepdims=True
    )

    model = DistributionalPicklableModel(
        inputs_flat, [expectation, logits]
    )
    model.observation_keys = observation_keys
    model.n_bins = n_bins
    model.max_distance = max_distance

    return model


DISTANCE_ESTIMATORS = {
    'FeedforwardDistanceEstimator': feedforward_distance_estimator,
    'DistributionalFeedforwardDistanceEstimator': distributional_feedforward_distance_estimator,
}


def get_distance_estimator_from_variant(variant, env, *args, **kwargs):
    distance_estimator_params = variant['distance_estimator_params']
    distance_estimator_type = distance_estimator_params['type']
    distance_estimator_kwargs = distance_estimator_params.get('kwargs', {})
    target_input_type = distance_estimator_kwargs.pop(
        'target_input_type', 'full')

    observation_preprocessors_params = distance_estimator_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = distance_estimator_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    # These shapes need to match with
    # `MetrciLearner._distance_estimator_inputs`
    observation1_shapes = OrderedDict((
        (name, shape)
        for name, shape in env.observation_shape.items()
        if name in observation_keys
    ))

    if target_input_type == 'full':
        observation2_shapes = OrderedDict((
            (name, shape)
            for name, shape in env.observation_shape.items()
            if name in observation_keys
        ))
    elif target_input_type == 'xy_coordinates':
        assert isinstance(env.unwrapped, LOCOMOTION_ENVS)
        assert set(env.observation_space.spaces.keys()) == {'observations'}, (
            set(env.observation_space.spaces.keys()))
        assert not env.unwrapped._exclude_current_positions_from_observation

        position_slice = POSITION_SLICES[type(env.unwrapped)]
        position_size = position_slice.stop - position_slice.start
        observation2_shapes = OrderedDict((
            ('positions', tf.TensorShape(position_size, )),
        ))
    elif target_input_type == 'xy_velocities':
        raise NotImplementedError(target_input_type)
    else:
        raise NotImplementedError(target_input_type)

    action_shape = env.action_shape
    input_shapes = {
        'observations1': observation1_shapes,
        'observations2': observation2_shapes
    }

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation1_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue
        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    preprocessors = {
        'observations1': observation_preprocessors,
        'observations2': observation_preprocessors,
    }

    condition_with_action = distance_estimator_kwargs.pop('condition_with_action')
    if condition_with_action:
        input_shapes['actions'] = action_shape
        preprocessors['actions'] = None

    distance_estimator = DISTANCE_ESTIMATORS[distance_estimator_type](
        input_shapes,
        *args,
        observation_keys=observation_keys,
        preprocessors=preprocessors,
        **distance_estimator_kwargs,
        **kwargs)
    distance_estimator.condition_with_action = condition_with_action
    distance_estimator.target_input_type = target_input_type

    return distance_estimator
