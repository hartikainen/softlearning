from collections import OrderedDict

import tensorflow as tf

from softlearning.utils.tensorflow import (
    cast_and_concat,
    batch_quadratic_form,
    nest)


class LinearGaussianModel(tf.keras.Model):
    def build(self, input_shapes):
        D = 1 + sum(
            input_shape[-1] for input_shape in nest.flatten(input_shapes))
        self.phi_omega_N = self.add_weight('phi_omega_N', shape=(D, 1))
        self.Sigma_N = self.add_weight('Sigma_N', shape=(D, D))
        self.beta = self.add_weight('beta', shape=(), dtype=tf.float32)
        self.N = self.add_weight('N', shape=(), dtype=tf.int32)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs = tf.concat((tf.ones(tf.shape(inputs)[:-1])[..., None], inputs), axis=-1)
        loc = tf.matmul(inputs, self.phi_omega_N)

        aleatoric_uncertainty = 1.0 / self.beta
        epistemic_uncertainty = batch_quadratic_form(self.Sigma_N, inputs)

        scale = aleatoric_uncertainty + epistemic_uncertainty

        return loc, scale, aleatoric_uncertainty, epistemic_uncertainty

    @tf.function(experimental_relax_shapes=True)
    def update(self, B, Y, diagonal_noise_scale):
        B = tf.concat((tf.ones(tf.shape(B)[:-1])[..., None], B), axis=-1)
        diagonal_noise_scale = tf.cast(diagonal_noise_scale, tf.float64)
        N = tf.shape(B)[0]

        beta = 1.0 / tf.math.reduce_variance(Y)
        eye = tf.eye(tf.shape(B)[-1], dtype=tf.float64)
        Sigma_N_inv = (
            tf.cast(beta, tf.float64)
            * tf.matmul(
                tf.cast(B, tf.float64),
                tf.cast(B, tf.float64),
                transpose_a=True)
            # diagonal_noise_scale is a small constant
            # to guarantee that Sigma_N_inv is invertible.
            + eye
            * diagonal_noise_scale)

        cholesky = tf.linalg.cholesky(Sigma_N_inv)
        Sigma_N = tf.cast(tf.linalg.cholesky_solve(cholesky, eye), tf.float32)

        tf.debugging.assert_equal(Sigma_N, tf.transpose(Sigma_N))

        phi_omega_N = tf.matmul(
            Sigma_N, beta * tf.matmul(B, Y, transpose_a=True))

        self.phi_omega_N.assign(phi_omega_N)
        self.Sigma_N.assign(Sigma_N)
        self.beta.assign(beta)
        self.N.assign(N)

    def get_diagnostics(self):
        diagnostics = OrderedDict((
            ('N', self.N.numpy()),
            ('beta', self.beta.numpy()),
        ))
        return diagnostics


class LinearStudentTModel(tf.keras.Model):
    def build(self, input_shapes):
        D = 1 + sum(
            input_shape[-1] for input_shape in nest.flatten(input_shapes))
        self.v_N = self.add_weight('v_N', shape=())
        self.nu_N = self.add_weight('nu_N', shape=())
        self.phi_omega_N = self.add_weight('phi_omega_N', shape=(D, 1))
        self.Sigma_N = self.add_weight('Sigma_N', shape=(D, D))

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs = tf.concat((tf.ones(tf.shape(inputs)[:-1])[..., None], inputs), axis=-1)
        loc = tf.tensordot(inputs, self.phi_omega_N, 1)
        aleatoric_uncertainty = self.nu_N / self.v_N
        epistemic_uncertainty = (
            self.nu_N / self.v_N * batch_quadratic_form(self.Sigma_N, inputs))
        scale = aleatoric_uncertainty + epistemic_uncertainty
        df = 2 * self.v_N

        return loc, scale, df, aleatoric_uncertainty, epistemic_uncertainty

    @tf.function(experimental_relax_shapes=True)
    def update(self, B, Y, diagonal_noise_scale):
        B = tf.concat((tf.ones(tf.shape(B)[:-1])[..., None], B), axis=-1)
        diagonal_noise_scale = tf.cast(diagonal_noise_scale, tf.float64)
        N = tf.shape(B)[0]

        eye = tf.eye(tf.shape(B)[-1], dtype=tf.float64)
        Sigma_N_inv = (
            tf.matmul(
                tf.cast(B, tf.float64),
                tf.cast(B, tf.float64),
                transpose_a=True)
            + eye
            * diagonal_noise_scale)

        cholesky = tf.linalg.cholesky(Sigma_N_inv)
        Sigma_N = tf.cast(tf.linalg.cholesky_solve(cholesky, eye), tf.float32)

        phi_omega_N = tf.matmul(
            Sigma_N, tf.matmul(B, Y, transpose_a=True))
        v_N = tf.cast(N, tf.float32) / 2.0
        x = Y - tf.matmul(B, phi_omega_N)
        nu_N = tf.squeeze(0.5 * tf.matmul(x, x, transpose_a=True))

        self.v_N.assign(v_N)
        self.nu_N.assign(nu_N)
        self.phi_omega_N.assign(phi_omega_N)
        self.Sigma_N.assign(Sigma_N)

    def get_diagnostics(self):
        diagnostics = OrderedDict((
            ('N', tf.cast(self.v_N * 2.0, tf.int32).numpy()),
            ('v_N', self.v_N.numpy()),
            ('nu_N', self.nu_N.numpy()),
            ('nu_N / self.v_N', (self.nu_N / self.v_N).numpy()),
        ))
        return diagnostics


class JacobianModel(tf.keras.Model):
    def __init__(self, non_linear_model, *args, **kwargs):
        super(JacobianModel, self).__init__(
            *args, **kwargs)
        self.non_linear_model = non_linear_model

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            non_linear_values = self.non_linear_model(inputs)
            # non_linear_values = tf.reduce_sum(non_linear_values, axis=-1)

        batch_shape = tf.shape(non_linear_values)
        jacobians = tape.jacobian(
            non_linear_values,
            self.non_linear_model.trainable_variables,
            experimental_use_pfor=True,
        )
        del tape

        final_shape = tf.concat((batch_shape, [-1]), axis=0)
        features = tf.concat([
            tf.reshape(j, final_shape) for j in jacobians], axis=-1)

        # We need to explicitly reshape the output features since otherwise
        # the last dimension has size of `None`, which causes e.g. keras models
        # to fail on build.
        feature_size = tf.reduce_sum([
            tf.reduce_prod(tf.shape(x))
            for x in self.non_linear_model.trainable_variables
        ])

        tf.debugging.assert_equal(tf.shape(features)[-1], feature_size)
        tf.debugging.assert_equal(batch_shape, tf.shape(features)[:-1])

        final_shape = tf.concat((batch_shape, [feature_size]), axis=0)
        features = tf.reshape(features, final_shape)

        return features

    def get_config(self, *args, **kwargs):
        base_config = {}

        model_config = {
            'model': {
                'config': self.non_linear_model.get_config(),
                'class_name': self.non_linear_model.__class__.__name__,
            },
        }
        config = {**base_config, **model_config}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
        model = deserialize_layer(
            config.pop('model'), custom_objects=custom_objects)
        return cls(model, **config)


class LinearizedModel(JacobianModel):
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        features = super(LinearizedModel, self).call(inputs)
        weights = tf.concat([
            tf.reshape(weight, (1, -1))
            for weight in self.non_linear_model.trainable_weights
        ], axis=-1)
        features = tf.reduce_sum((features * weights), axis=-1)

        return features
