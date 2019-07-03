from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.keras.engine import training_utils

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def create_feedforward_Q_function(env,
                                  input_shapes,
                                  *args,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
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

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function = PicklableModel(inputs_flat, Q_function(preprocessed_inputs))
    Q_function.observation_keys = observation_keys

    return Q_function


def create_goal_conditioned_feedforward_Q_function(*args,
                                                   env=None,
                                                   input_shapes=None,
                                                   preprocessors=None,
                                                   observation_keys=None,
                                                   goal_keys=None,
                                                   name='feedforward_Q',
                                                   **kwargs):
    goal_keys = goal_keys or env.goal_keys
    goal_shapes = OrderedDict((
        (key, value)
        for key, value in env.observation_shape.items()
        if key in goal_keys
    ))

    goal_preprocessors = OrderedDict((
        (key, preprocessors['observations'].get(key, None))
        for key in goal_keys
    ))

    input_shapes = {**input_shapes, 'goals': goal_shapes}
    preprocessors = {**preprocessors, 'goals': goal_preprocessors}

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

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

    Q_function = PicklableModel(inputs_flat, Q_function(preprocessed_inputs))
    Q_function.observation_keys = observation_keys
    Q_function.goal_keys = goal_keys

    return Q_function
