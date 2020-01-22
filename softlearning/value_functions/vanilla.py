import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import nest, apply_preprocessors
from softlearning.utils.keras import PicklableModel
from .base_value_function import StateActionValueFunction


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    inputs = create_inputs(input_shapes)
    if preprocessors is None:
        preprocessors = nest.map_structure(lambda x: None, inputs)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    Q_model_body = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs
    )

    Q_model = tf.keras.Model(inputs, Q_model_body(preprocessed_inputs))

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys)

    return Q_function


def create_pretrained_feature_Q_function(input_shapes,
                                         *args,
                                         hidden_layer_sizes=(32, 32),
                                         preprocessors=None,
                                         observation_keys=None,
                                         name='pretrained_features_Q',
                                         **kwargs):
    inputs = create_inputs(input_shapes)

    def assert_none(preprocessor):
        assert preprocessor is None, preprocessor

    if preprocessors is None:
        nest.map_structure(assert_none, inputs)

    Q_model_body = feedforward_model(
        *args,
        hidden_layer_sizes=hidden_layer_sizes[:-1],
        output_size=hidden_layer_sizes[-1],
        activation='relu',
        output_activation='relu',
        name="pretrained_features_body",
        trainable=False,
    )

    Q_model_body.trainable = False

    Q_model_head = feedforward_model(
        *args,
        hidden_layer_sizes=(),
        output_size=1,
        activation='linear',
        output_activation='linear',
        name="linear_head",
    )

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    Q_model = tf.keras.Model(
        inputs, Q_model_head(Q_model_body(preprocessed_inputs)))

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys)

    return Q_function
