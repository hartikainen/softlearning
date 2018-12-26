import tensorflow as tf


class FeedforwardDistanceEstimator(tf.keras.Model):
    def __init__(self,
                 hidden_layer_sizes,
                 hidden_activation,
                 output_activation):
        super(FeedforwardDistanceEstimator, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def build(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape[1:])

        out = input_
        for layer_size in self.hidden_layer_sizes:
            out = tf.keras.layers.Dense(
                layer_size, activation=self.hidden_activation
            )(out)
            # out = tf.keras.layers.Dense(layer_size)(out)
            # out = tf.keras.layers.BatchNormalization()(out)
            # out = tf.keras.layers.Activation(
            #     activation=self.hidden_activation
            # )(out)

        out = tf.keras.layers.Dense(1, activation=self.output_activation)(out)
        self.distance_model = tf.keras.Model(input_, out)

    def call(self, inputs, training=True):
        return self.distance_model(inputs)

    def predict(self, inputs):
        return self.distance_model.predict(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 1))


DISTANCE_ESTIMATORS = {
    'FeedforwardDistanceEstimator': FeedforwardDistanceEstimator,
}


def get_distance_estimator_from_variant(variant, *args, **kwargs):
    distance_estimator_params = variant['distance_estimator_params']
    distance_estimator_type = distance_estimator_params['type']
    distance_estimator_kwargs = distance_estimator_params.get('kwargs', {})

    return DISTANCE_ESTIMATORS[distance_estimator_type](
        *args, **distance_estimator_kwargs, **kwargs)
