from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .base_policy import BasePolicy
from softlearning.models.utils import create_inputs


class UniformPolicy(BasePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 action_range,
                 *args,
                 preprocessors=None,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._output_shape = output_shape
        self._action_range = action_range

        super(UniformPolicy, self).__init__(*args, **kwargs)

        inputs_flat = create_inputs(input_shapes)

        self.inputs = inputs_flat

        x = self.inputs

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x)[0]
        )(inputs_flat[0])

        actions = tf.keras.layers.Lambda(
            self._actions_fn
        )(batch_size)

        self.actions_model = tf.keras.Model(self.inputs, actions)

        self.actions_input = tf.keras.Input(shape=output_shape, name='actions')

        log_pis = tf.keras.layers.Lambda(
            self._log_pis_fn
        )(self.actions_input)

        self.log_pis_model = tf.keras.Model(
            (*self.inputs, self.actions_input), log_pis)

    def get_weights(self):
        return []

    def set_weights(self, *args, **kwargs):
        return

    @property
    def trainable_variables(self):
        return []

    def reset(self):
        pass

    def get_diagnostics(self, observations):
        return OrderedDict({})

    def actions(self, observations):
        return self.actions_model(observations)

    def actions_np(self, observations):
        return self.actions_model.predict(observations)


class ContinuousUniformPolicy(UniformPolicy):
    def __init__(self,
                 *args,
                 action_range=np.array(((-1.0, ), (1.0, ))),
                 **kwargs):
        self._Serializable__initialize(locals())

        return super(ContinuousUniformPolicy, self).__init__(
            *args, action_range=action_range, **kwargs)

    def _actions_fn(self, batch_size):
        actions = tf.random.uniform(
            (batch_size, np.prod(self._output_shape)),
            *self._action_range)
        return actions

    def _log_pis_fn(self, actions):
        log_pis = tf.tile(tf.math.log(
            (self._action_range[1] - self._action_range[0]) / 2.0
        )[None], (tf.shape(input=actions)[0], 1))
        return log_pis


class DiscreteUniformPolicy(UniformPolicy):
    def __init__(self,
                 *args,
                 output_shape=None,
                 action_range=None,
                 **kwargs):
        assert (
            output_shape is not None
            and np.prod(output_shape) == 1
        ), output_shape
        assert (
            action_range is not None
            and action_range[0] == 0
            and 0 < action_range[1]
        ), action_range

        self._Serializable__initialize(locals())

        return super(DiscreteUniformPolicy, self).__init__(
            *args,
            output_shape=output_shape,
            action_range=action_range,
            **kwargs)

    def _actions_fn(self, batch_size):
        actions = tf.random.uniform(
            (batch_size, np.prod(self._output_shape)),
            minval=self._action_range[0],
            maxval=self._action_range[1],
            dtype=tf.int32,
        )
        return actions

    def _log_pis_fn(self, actions):
        log_pis = tf.fill(
            (tf.shape(input=actions)[0], 1),
            1.0 / self._action_range[1])
        return log_pis
