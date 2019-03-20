import tensorflow as tf


from softlearning.utils.keras import PicklableKerasModel


class ConstantOffsetLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.offset = self.add_weight(name='kernel',
                                      shape=(),
                                      initializer='zeros',
                                      trainable=True)
        super(ConstantOffsetLayer, self).build(input_shape)

    def call(self, x):
        return x + self.offset

    def compute_output_shape(self, input_shape):
        return input_shape


def feedforward_model(input_shapes,
                      output_size,
                      hidden_layer_sizes,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      learn_offset=False,
                      *args,
                      **kwargs):
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    if preprocessors is None:
        preprocessors = (None, ) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    concatenated = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(preprocessed_inputs)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs
    )(out)

    if learn_offset:
        out = ConstantOffsetLayer()(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model
