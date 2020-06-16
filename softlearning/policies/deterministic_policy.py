"""DeterministicPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.tensorflow import nest

from .base_policy import BasePolicy


class DeterministicPolicy(BasePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 action_range,
                 *args,
                 squash=True,
                 preprocessors=None,
                 name=None,
                 **kwargs):

        assert (np.all(action_range == np.array([[-1], [1]]))), (
            "The action space should be scaled to (-1, 1)."
            " TODO(hartikainen): We should support non-scaled actions spaces.")

        self._Serializable__initialize(locals())

        self._action_range = action_range
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name

        super(DeterministicPolicy, self).__init__(*args, **kwargs)

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

        self.condition_inputs = inputs_flat

        actions = self._action_net(
            output_size=np.prod(output_shape),
        )(conditions)

        self.actions_model = tf.keras.Model(self.condition_inputs, actions)
        self.diagnostics_model = self.actions_model

    def _action_net(self, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    def actions(self, observations):
        return self.actions_model(observations)

    def actions_np(self, observations):
        return self.actions_model.predict(observations)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        actions_np = self.diagnostics_model.predict(inputs)

        return OrderedDict((

            ('actions-mean', np.mean(actions_np)),
            ('actions-std', np.std(actions_np)),
            ('actions-min', np.min(actions_np)),
            ('actions-max', np.max(actions_np)),
        ))


class FeedforwardDeterministicPolicy(DeterministicPolicy):
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
        super(FeedforwardDeterministicPolicy, self).__init__(*args, **kwargs)

    def _action_net(self, output_size):
        action_net = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return action_net
