import numpy as np
import tensorflow as tf


class LambdaEstimatorBase(tf.keras.Model):
    def call(self, inputs):
        observations, values, target_values = inputs

        result = self.lambda_estimator(tf.concat((
            observations,
            values[:, None],
            target_values[:, None],
        ), axis=-1))

        return result

    def predict(self, inputs):
        observations, values, target_values = inputs

        result = self.lambda_estimator.predict(np.concatenate((
            observations,
            values[:, None],
            target_values[:, None],
        ), axis=-1))

        return result

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 1))


class ConstantLambdaEstimator(LambdaEstimatorBase):
    def __init__(self, constant):
        super(ConstantLambdaEstimator, self).__init__(name='lambda_estimator')
        self.lambda_estimator = tf.keras.Sequential((
            tf.keras.layers.Dense(1, activation='linear'),
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, constant, constant)),
        ))


class FeedforwardLambdaEstimator(LambdaEstimatorBase):
    def __init__(self,
                 hidden_activation,
                 output_activation,
                 hidden_layer_sizes):
        super(FeedforwardLambdaEstimator, self).__init__(
            name='feedforward_lambda_estimator')

        if output_activation == 'abs':
            output_activation = tf.abs

        layers = []
        for layer_size in hidden_layer_sizes:
            layers += [
                tf.keras.layers.Dense(
                    layer_size, activation=hidden_activation),
                # tf.keras.layers.Dense(layer_size),
                # tf.keras.layers.BatchNormalization(),
                # tf.keras.layers.Activation(activation=hidden_activation),
            ]

        self.lambda_estimator = tf.keras.Sequential((
            *layers,
            tf.keras.layers.Dense(1, activation=output_activation),
        ))


LAMBDA_ESTIMATORS = {
    'ConstantLambdaEstimator': ConstantLambdaEstimator,
    'FeedforwardLambdaEstimator': FeedforwardLambdaEstimator,
}


def get_lambda_estimator_from_variant(variant):
    lambda_estimator_params = variant['lambda_estimator_params']
    lambda_estimator_type = lambda_estimator_params['type']
    lambda_estimator_kwargs = lambda_estimator_params.get('kwargs', {})

    return LAMBDA_ESTIMATORS[lambda_estimator_type](**lambda_estimator_kwargs)
