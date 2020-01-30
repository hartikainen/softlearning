"""GaussianPolicy."""

from collections import OrderedDict

import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.models.bae.student_t import (
    create_n_degree_polynomial_form_observations_actions_v4,
)
from softlearning.models.bae.linear import LinearizedModel, JacobianModel
from softlearning.utils.numpy import custom_combinations
from softlearning.utils.tensorflow import (
    nest,
    apply_preprocessors,
    cast_and_concat)

from .base_policy import LatentSpacePolicy


class GaussianPolicy(LatentSpacePolicy):
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
        self._preprocessors = preprocessors

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        inputs = create_inputs(input_shapes)
        preprocessed_inputs = self.preprocess_inputs(inputs)

        conditions = tf.keras.layers.Lambda(
            cast_and_concat
        )(preprocessed_inputs)

        self.condition_inputs = inputs

        shift_and_log_scale_diag = self._shift_and_log_scale_diag_net(
            output_size=np.prod(output_shape) * 2,
        )(conditions)

        shift, log_scale_diag = tf.keras.layers.Lambda(
            lambda shift_and_log_scale_diag: tf.split(
                shift_and_log_scale_diag,
                num_or_size_splits=2,
                axis=-1)
        )(shift_and_log_scale_diag)

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(input=x)[0])(conditions)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(output_shape),
            scale_diag=tf.ones(output_shape))

        latents = tf.keras.layers.Lambda(
            lambda batch_size: base_distribution.sample(batch_size)
        )(batch_size)

        self.latents_model = tf.keras.Model(self.condition_inputs, latents)
        self.latents_input = tf.keras.layers.Input(
            shape=output_shape, name='latents')

        def raw_actions_fn(inputs):
            shift, log_scale_diag, latents = inputs
            bijector = tfp.bijectors.Affine(
                shift=shift,
                scale_diag=tf.exp(log_scale_diag))
            actions = bijector(latents)
            return actions

        raw_actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, latents))

        raw_actions_for_fixed_latents = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, self.latents_input))

        squash_bijector = (
            tfp.bijectors.Tanh()
            if self._squash
            else tfp.bijectors.Identity())

        actions = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector(raw_actions)
        )(raw_actions)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        actions_for_fixed_latents = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector(raw_actions)
        )(raw_actions_for_fixed_latents)
        self.actions_model_for_fixed_latents = tf.keras.Model(
            (self.condition_inputs, self.latents_input),
            actions_for_fixed_latents)

        deterministic_actions = tf.keras.layers.Lambda(
            lambda shift: squash_bijector(shift)
        )(shift)

        self.deterministic_actions_model = tf.keras.Model(
            self.condition_inputs, deterministic_actions)

        def log_pis_fn(inputs):
            shift, log_scale_diag, actions = inputs
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(output_shape),
                scale_diag=tf.ones(output_shape))
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=shift,
                    scale_diag=tf.exp(log_scale_diag)),
            ))
            distribution = bijector(base_distribution)
            log_pis = distribution.log_prob(actions)[:, None]
            return log_pis

        self.actions_input = tf.keras.layers.Input(
            shape=output_shape, name='actions')

        log_pis = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, actions])

        log_pis_for_action_input = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, self.actions_input])

        self.log_pis_model = tf.keras.Model(
            (self.condition_inputs, self.actions_input),
            log_pis_for_action_input)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs,
            (shift, log_scale_diag, log_pis, raw_actions, actions))

    def preprocess_inputs(self, inputs):
        preprocessors = self._preprocessors
        if preprocessors is None:
            preprocessors = nest.map_structure(lambda x: None, inputs)

        preprocessed_inputs = apply_preprocessors(preprocessors, inputs)
        return preprocessed_inputs

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (shifts_np,
         log_scale_diags_np,
         log_pis_np,
         raw_actions_np,
         actions_np) = self.diagnostics_model(inputs)

        return OrderedDict((
            ('shifts-mean', np.mean(shifts_np)),
            ('shifts-std', np.std(shifts_np)),

            ('log_scale_diags-mean', np.mean(log_scale_diags_np)),
            ('log_scale_diags-std', np.std(log_scale_diags_np)),

            ('entropy-mean', np.mean(-log_pis_np)),
            ('entropy-std', np.std(-log_pis_np)),

            ('raw-actions-mean', np.mean(raw_actions_np)),
            ('raw-actions-std', np.std(raw_actions_np)),

            ('actions-mean', np.mean(actions_np)),
            ('actions-std', np.std(actions_np)),
            ('actions-min', np.min(actions_np)),
            ('actions-max', np.max(actions_np)),
        ))


class FeedforwardGaussianPolicy(GaussianPolicy):
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
        super(FeedforwardGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, output_size):
        shift_and_log_scale_diag_net = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shift_and_log_scale_diag_net


class PretrainedFeatureGaussianPolicy(GaussianPolicy):
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
        super(PretrainedFeatureGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, output_size):
        shift_and_log_scale_diag_body = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes[:-1],
            output_size=self._hidden_layer_sizes[-1],
            activation='relu',
            output_activation='relu',
            name="pretrained_features_body",
            trainable=False,
        )

        shift_and_log_scale_diag_body.trainable = False

        shift_and_log_scale_diag_head = feedforward_model(
            hidden_layer_sizes=(),
            output_size=output_size,
            activation='linear',
            output_activation='linear',
            name="linear_head",
        )

        shift_and_log_scale_diag_net = tf.keras.Sequential((
            shift_and_log_scale_diag_body,
            shift_and_log_scale_diag_head,
        ))
        return shift_and_log_scale_diag_net


class LinearPolynomialGaussianPolicy(GaussianPolicy):
    def __init__(self,
                 activation='linear',
                 output_activation='linear',
                 degree=3,
                 *args,
                 **kwargs):
        self._activation = activation
        self._output_activation = output_activation
        self._degree = degree

        self._Serializable__initialize(locals())
        super(LinearPolynomialGaussianPolicy, self).__init__(*args, **kwargs)

    def preprocess_inputs(self, inputs):
        D = sum([input_.shape[-1] for input_ in nest.flatten(inputs)])
        preprocessed_inputs = (
            super(LinearPolynomialGaussianPolicy, self)
            .preprocess_inputs(inputs))
        polynomial_features_fn = preprocessing.PolynomialFeatures(self._degree)
        preprocessed_inputs = tf.keras.layers.Lambda(
            cast_and_concat
        )(preprocessed_inputs)
        preprocessed_inputs = tf.keras.layers.Lambda(
            lambda inputs: tf.numpy_function(
                polynomial_features_fn.fit_transform,
                [inputs],
                inputs.dtype
            )
        )(preprocessed_inputs)

        feature_size = len(custom_combinations(D, self._degree))
        preprocessed_inputs = tf.keras.layers.Reshape(
            (feature_size + 1, )
        )(preprocessed_inputs)

        return preprocessed_inputs

    def _shift_and_log_scale_diag_net(self, output_size):
        shift_and_log_scale_diag_net = feedforward_model(
            hidden_layer_sizes=(),
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shift_and_log_scale_diag_net


class LinearizedFeedforwardGaussianPolicy(GaussianPolicy):
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
        super(LinearizedFeedforwardGaussianPolicy, self).__init__(
            *args, **kwargs)

    def _shift_and_log_scale_diag_net(self, output_size):
        non_linear_model = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation,
            name='non-linear',
        )

        shift_and_log_scale_diag_net = LinearizedModel(
            non_linear_model, name='linearized_model')

        return shift_and_log_scale_diag_net


class LinearizedFeedforwardGaussianPolicyV2(
        LinearizedFeedforwardGaussianPolicy):
    def _shift_and_log_scale_diag_net(self, output_size):
        non_linear_model = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation,
            name='non-linear',
        )

        jacobian_model = JacobianModel(non_linear_model, name='jacobian_model')

        linear_model = feedforward_model(
            hidden_layer_sizes=(),
            output_size=output_size,
            activation=None,
            output_activation=self._output_activation,
            name='linear',
        )

        shift_and_log_scale_diag_net = tf.keras.Sequential((
            jacobian_model,
            tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2)),
            linear_model,
        ))

        return shift_and_log_scale_diag_net
