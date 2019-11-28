from collections import OrderedDict

import numpy as np
from scipy.stats import mode
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.engine import training_utils

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs

from .base_policy import BasePolicy


class CategoricalPolicy(BasePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 action_range,
                 *args,
                 squash=True,
                 preprocessors=None,
                 name=None,
                 **kwargs):
        assert np.prod(output_shape) == 1, output_shape
        assert (action_range[0] == 0 and 0 < action_range[1]), action_range
        self._Serializable__initialize(locals())

        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._action_range = action_range
        self._squash = squash
        self._name = name

        num_actions = action_range[1]

        super(CategoricalPolicy, self).__init__(*args, **kwargs)

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

        float_inputs = tf.keras.layers.Lambda(
            lambda inputs: training_utils.cast_if_floating_dtype(inputs)
        )(preprocessed_inputs)

        conditions = tf.keras.layers.Lambda(
            lambda inputs: tf.concat(inputs, axis=-1)
        )(float_inputs)

        self.condition_inputs = inputs_flat

        action_logits = self._logits_model(
            output_size=num_actions,
        )(conditions)

        deterministic_actions = tf.keras.layers.Lambda(
            lambda logits: tf.argmax(logits, axis=-1)[..., tf.newaxis]
        )(action_logits)

        self.deterministic_actions_model = tf.keras.Model(
            self.condition_inputs, deterministic_actions)

        action_distribution = tfp.layers.OneHotCategorical(
            num_actions,
            sample_dtype=tf.int32,
        )(action_logits)
        actions_one_hot = action_distribution.sample()
        actions = tf.keras.layers.Lambda(
            lambda logits: tf.argmax(logits, axis=-1)[..., tf.newaxis]
        )(actions_one_hot)

        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        self.actions_input = tf.keras.layers.Input(
            shape=output_shape, name='actions')

        log_pis = action_distribution.log_prob(actions)[..., tf.newaxis]
        self.log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.actions_input),
            log_pis)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs, (action_logits, log_pis, actions))

    def _logits_model(self, input_shapes, output_shape):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    def actions(self, observations):
        if self._deterministic:
            return self.deterministic_actions_model(observations)
        return self.actions_model(observations)

    def log_pis(self, observations, actions):
        assert not self._deterministic, self._deterministic
        return self.log_pis_model([*observations, actions])

    def actions_np(self, observations):
        if self._deterministic:
            return self.deterministic_actions_model.predict(observations)
        return self.actions_model.predict(observations)

    def log_pis_np(self, observations, actions):
        assert not self._deterministic, self._deterministic
        return self.log_pis_model.predict([*observations, actions])

    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (action_logits_np,
         log_pis_np,
         actions_np) = self.diagnostics_model.predict(inputs)

        return OrderedDict((
            ('action_logits-mean', np.mean(action_logits_np)),
            ('action_logits-std', np.std(action_logits_np)),
            ('action_logits-min', np.min(action_logits_np)),
            ('action_logits-max', np.max(action_logits_np)),

            ('entropy-mean', np.mean(-log_pis_np)),
            ('entropy-std', np.std(-log_pis_np)),

            ('actions-mean', np.mean(actions_np)),
            ('actions-mode', np.squeeze(mode(actions_np).mode)),
            ('actions-min', np.min(actions_np)),
            ('actions-max', np.max(actions_np)),
        ))


class FeedforwardCategoricalPolicy(CategoricalPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args,
                 **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(FeedforwardCategoricalPolicy, self).__init__(*args, **kwargs)

    def _logits_model(self, output_size):
        logits_model = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return logits_model
