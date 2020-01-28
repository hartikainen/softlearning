import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import nest, apply_preprocessors
from softlearning.models.bae.linear import LinearizedModel
from softlearning.models.bae.student_t import (
    create_n_degree_polynomial_form_observations_actions_v4,
)
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
        **kwargs
    )

    Q_model = tf.keras.Model(
        inputs,
        Q_model_body(preprocessed_inputs),
        name=name)

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys)

    return Q_function


def create_linear_polynomial_Q_function(input_shapes,
                                        *args,
                                        preprocessors=None,
                                        observation_keys=None,
                                        degree=3,
                                        name='linear_Q',
                                        **kwargs):
    inputs = create_inputs(input_shapes)
    if preprocessors is None:
        preprocessors = nest.map_structure(lambda x: None, inputs)

    Q_model_body = feedforward_model(
        *args,
        output_size=1,
        hidden_layer_sizes=(),
        activation='linear',
        name=name,
        **kwargs
    )

    D = sum([input_.shape[-1] for input_ in nest.flatten(inputs)])
    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)
    fn = create_n_degree_polynomial_form_observations_actions_v4(D, degree)
    preprocessed_inputs = tf.keras.layers.Lambda(
        # n_degree_polynomial_form_observations_actions.python_function
        fn.python_function
    )(preprocessed_inputs)

    preprocessed_inputs = tf.keras.layers.Reshape(
        {
            # (D + D ** 2, )
            # (D * (D + 3) // 2, )
            6: (209, ),
            5: (125, ),
            4: (69, ),
            3: (34, ),
            2: (14, ),
        }[degree]
    )(preprocessed_inputs)

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


def linearized_feedforward_Q_function(input_shapes,
                                      *args,
                                      preprocessors=None,
                                      observation_keys=None,
                                      activation='tanh',
                                      name='linearized_feedforward_Q',
                                      **kwargs):

    assert activation == 'tanh', (
        "This is not guaranteed to work with non-smooth non-linearities.")
    inputs = create_inputs(input_shapes)
    if preprocessors is None:
        preprocessors = nest.map_structure(lambda x: None, inputs)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    non_linear_model = feedforward_model(
        *args,
        output_size=1,
        activation=activation,
        name='non-linear',
        **kwargs
    )

    linearized_model = LinearizedModel(
        non_linear_model, name='linearized_model')

    out = linearized_model(preprocessed_inputs)

    Q_model = tf.keras.Model(
        inputs,
        out,
        name=name)

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys)

    return Q_function
