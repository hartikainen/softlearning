import numpy as np
import tensorflow as tf


def feedforward_distance_estimator(input_shape,
                                   hidden_layer_sizes,
                                   hidden_activation='relu',
                                   output_activation='linear'):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape)
    ] + [
        tf.keras.layers.Dense(layer_size, activation=hidden_activation)
        for layer_size in hidden_layer_sizes
    ] + [
        tf.keras.layers.Dense(1, activation=output_activation)
    ])
    return model


DISTANCE_ESTIMATORS = {
    'FeedforwardDistanceEstimator': feedforward_distance_estimator,
}


def get_distance_estimator_from_variant(variant, env, *args, **kwargs):
    observation_shape = env.active_observation_shape
    action_shape = env.action_space.shape

    distance_input_type = (
        variant['metric_learner_params']['kwargs']['distance_input_type'])

    if distance_input_type == 'full':
        input_shapes = (observation_shape, observation_shape)
    elif distance_input_type in ('xy_coordinates', 'xy_velocities'):
        input_shapes = (observation_shape, (2, ))
    else:
        raise NotImplementedError(distance_input_type)

    if variant['metric_learner_params']['kwargs']['condition_with_action']:
        input_shapes = (input_shapes[0], action_shape, input_shapes[1])
    input_shape = tuple(
        sum(np.prod(shape, keepdims=True) for shape in input_shapes))

    distance_estimator_params = variant['distance_estimator_params']
    distance_estimator_type = distance_estimator_params['type']
    distance_estimator_kwargs = distance_estimator_params.get('kwargs', {})

    return DISTANCE_ESTIMATORS[distance_estimator_type](
        input_shape,
        *args,
        **distance_estimator_kwargs,
        **kwargs)
