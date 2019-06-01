from collections import OrderedDict

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


def create_goal_conditioned_feedforward_Q_function(observation_shape,
                                                   action_shape,
                                                   *args,
                                                   observation_preprocessor=None,
                                                   name='feedforward_Q',
                                                   **kwargs):
    input_shapes = (observation_shape, action_shape, observation_shape)
    preprocessors = (observation_preprocessor, None, observation_preprocessor)
    return feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)
