from copy import deepcopy

import tensorflow as tf


def get_convnet_preprocessor(name='convnet_preprocessor', **kwargs):
    from softlearning.models.convnet import convnet_model

    preprocessor = convnet_model(name=name, **kwargs)

    return preprocessor


def get_feedforward_preprocessor(name='feedforward_preprocessor', **kwargs):
    from softlearning.models.feedforward import feedforward_model

    preprocessor = feedforward_model(name=name, **kwargs)

    return preprocessor


def get_normalize_preprocessor(input_low,
                               input_high,
                               output_low,
                               output_high,
                               name='normalize_preprocessor'):
    input_low = tf.convert_to_tensor(input_low)
    input_high = tf.convert_to_tensor(input_high)
    output_low = tf.convert_to_tensor(output_low)
    output_high = tf.convert_to_tensor(output_high)

    def normalize(inputs):

        tf.debugging.assert_shapes((
            (input_low, ('D', )),
            (input_high, ('D', )),
            (output_low, ('D', )),
            (output_high, ('D', )),
            (inputs, ('B', 'D')),
        ))

        tf.debugging.assert_less_equal(input_low, inputs)
        tf.debugging.assert_less_equal(inputs, input_high)

        outputs = (
            output_low
            + (output_high - output_low)
            * ((inputs - input_low) / (input_high - input_low)))

        # action = np.clip(action, output_low, output_high)

        tf.debugging.assert_less_equal(output_low, outputs)
        tf.debugging.assert_less_equal(outputs, output_high)

        tf.debugging.assert_equal(tf.shape(outputs), tf.shape(inputs))

        return outputs

    normalize_layer = tf.keras.layers.Lambda(normalize, name=name)

    return normalize_layer


PREPROCESSOR_FUNCTIONS = {
    'convnet_preprocessor': get_convnet_preprocessor,
    'feedforward_preprocessor': get_feedforward_preprocessor,
    'normalize_preprocessor': get_normalize_preprocessor,
    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_params(env, preprocessor_params, *args, **kwargs):
    if preprocessor_params is None:
        return None

    preprocessor_type = preprocessor_params.get('type', None)
    preprocessor_kwargs = deepcopy(preprocessor_params.get('kwargs', {}))

    if preprocessor_type is None:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_type](
            *args,
            **preprocessor_kwargs,
            **kwargs)

    return preprocessor


def get_preprocessor_from_variant(variant, env, *args, **kwargs):
    preprocessor_params = variant['preprocessor_params']
    return get_preprocessor_from_params(
        env, preprocessor_params, *args, **kwargs)
