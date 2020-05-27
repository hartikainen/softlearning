import tensorflow as tf
import tree

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import apply_preprocessors
from softlearning import preprocessors as preprocessors_lib
from softlearning.utils.tensorflow import cast_and_concat

from .base_value_function import StateActionValueFunction


def create_ensemble_value_function(N, value_fn, *args, **kwargs):
    # TODO(hartikainen): The ensemble Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(N))
    return value_fns


def double_feedforward_Q_function(*args, **kwargs):
    return create_ensemble_value_function(
        2, feedforward_Q_function, *args, **kwargs)


def ensemble_feedforward_Q_function(N, *args, **kwargs):
    return create_ensemble_value_function(
        N, feedforward_Q_function, *args, **kwargs)


def random_prior_ensemble_feedforward_Q_function(N, *args, **kwargs):
    return create_ensemble_value_function(
        N, random_prior_feedforward_Q_function, *args, **kwargs)


class RandomPriorModel(tf.keras.Model):
    def __init__(self, model):
        super(RandomPriorModel, self).__init__()
        self.model = model

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def random_prior_feedforward_Q_function(input_shapes,
                                        *args,
                                        preprocessors=None,
                                        observation_keys=None,
                                        prior_scale=1.0,
                                        name='feedforward_Q',
                                        **kwargs):
    inputs = create_inputs(input_shapes)

    if preprocessors is None:
        preprocessors = tree.map_structure(lambda _: None, inputs)

    preprocessors = tree.map_structure_up_to(
        inputs, preprocessors_lib.deserialize, preprocessors)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    # NOTE(hartikainen): `feedforward_model` would do the `cast_and_concat`
    # step for us, but tf2.2 broke the sequential multi-input handling: See:
    # https://github.com/tensorflow/tensorflow/issues/37061.
    model_inputs = tf.keras.layers.Lambda(
        cast_and_concat
    )(preprocessed_inputs)
    Q_predictor_model_out = feedforward_model(
        *args,
        output_shape=[1],
        name=f'{name}-predictor',
        **kwargs,
    )(model_inputs)

    assert 'trainable' not in kwargs or not kwargs['trainable']
    assert 'kernel_initializer' not in kwargs
    assert 'bias_initializer' not in kwargs

    Q_prior_kwargs = {
        **kwargs,
        'kernel_regularizer': None,
        'bias_regularizer': None,
    }

    Q_prior_model_out = feedforward_model(
        *args,
        output_shape=[1],
        name=f'{name}-prior',
        trainable=False,
        # kernel_initializer={
        #     'class_name': 'VarianceScaling',
        #     'config': {
        #         'scale': prior_scale,
        #         # 'scale': 1.0,
        #         'mode': 'fan_avg',
        #         'distribution': 'uniform',
        #     },
        # },
        **Q_prior_kwargs,
        )(model_inputs)

    Q_prior_model_out = tf.keras.layers.Lambda(
        lambda x: x * prior_scale,
    )(Q_prior_model_out)

    Q_model_out = tf.keras.layers.Add()(
        (Q_predictor_model_out, Q_prior_model_out))

    Q_model = tf.keras.Model(inputs, Q_model_out, name=name)

    assert not Q_model.get_layer('feedforward_Q-prior').losses

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys, name=name)

    return Q_function


def feedforward_Q_function(input_shapes,
                           *args,
                           preprocessors=None,
                           observation_keys=None,
                           name='feedforward_Q',
                           **kwargs):
    inputs = create_inputs(input_shapes)

    if preprocessors is None:
        preprocessors = tree.map_structure(lambda _: None, inputs)

    preprocessors = tree.map_structure_up_to(
        inputs, preprocessors_lib.deserialize, preprocessors)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    # NOTE(hartikainen): `feedforward_model` would do the `cast_and_concat`
    # step for us, but tf2.2 broke the sequential multi-input handling: See:
    # https://github.com/tensorflow/tensorflow/issues/37061.
    out = tf.keras.layers.Lambda(cast_and_concat)(preprocessed_inputs)
    Q_model_body = feedforward_model(
        *args,
        output_shape=[1],
        name=name,
        **kwargs
    )

    Q_model = tf.keras.Model(inputs, Q_model_body(out), name=name)

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys, name=name)

    return Q_function
