import tensorflow as tf
from tensorflow.python.keras.engine import training_utils

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel
from softlearning.utils.tensorflow import nest


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  layer_normalize_inputs=False,
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

    def cast_and_concat(x):
        x = nest.map_structure(
            lambda element: tf.cast(element, tf.float32), x)
        x = nest.flatten(x)
        x = tf.concat(x, axis=-1)
        return x

    conditions = tf.keras.layers.Lambda(
        cast_and_concat
    )(preprocessed_inputs)

    if layer_normalize_inputs:
        conditions = tf.keras.layers.LayerNormalization()(conditions)

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function = PicklableModel(inputs_flat, Q_function(conditions))
    Q_function.observation_keys = observation_keys

    return Q_function
