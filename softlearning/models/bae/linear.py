from collections import OrderedDict

import tensorflow as tf

from softlearning.utils.tensorflow import (
    cast_and_concat,
    batch_quadratic_form,
    nest)


class OnlineUncertaintyModel(tf.keras.Model):
    def build(self, input_shapes):
        D = 1 + sum(
            input_shape[-1] for input_shape in nest.flatten(input_shapes))
        self.mu_hat = self.add_weight(
            'mu_hat', shape=(1, D), initializer='zeros')
        self.Sigma_hat = self.add_weight(
            'Sigma_hat', shape=(D, D), initializer='identity')
        self.Sigma_N = self.add_weight(
            'Sigma_N', shape=(D, D), initializer='identity')
        self.Delta_N = self.add_weight(
            'Delta_N', shape=(D, D), initializer='zeros')
        self.N = self.add_weight(
            'N', shape=(), dtype=tf.int32, initializer='zeros')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        mu_hat = self.mu_hat
        Sigma_N = self.Sigma_N
        Sigma_hat = self.Sigma_hat

        mu_hat_mu_hat_T = tf.matmul(
            mu_hat[..., None], mu_hat[..., None], transpose_b=True
        )[0]
        epistemic_uncertainty = tf.linalg.trace(
            tf.matmul(Sigma_hat - mu_hat_mu_hat_T, Sigma_N)
        ) + tf.squeeze(batch_quadratic_form(Sigma_N, mu_hat))

        return epistemic_uncertainty

    @tf.function(experimental_relax_shapes=True)
    def online_update(self, inputs):
        b_N, b_hat, b_N_not, gamma = inputs
        N = tf.shape(b_N)[0]
        tf.debugging.assert_equal(N, 1)
        b_N = tf.concat((tf.ones(tf.shape(b_N)[:-1])[..., None], b_N), axis=-1)
        b_hat = tf.concat((
            tf.ones(tf.shape(b_hat)[:-1])[..., None], b_hat), axis=-1)
        b_N_not = tf.concat((
            tf.ones(tf.shape(b_N_not)[:-1])[..., None], b_N_not), axis=-1)

        variance = 1.0
        b_b_T = tf.matmul(b_N, b_N, transpose_a=True)

        def initialize_Sigma_N():
            Sigma_N_inv = tf.linalg.inv(self.Sigma_N) + b_b_T / variance
            cholesky = tf.linalg.cholesky(tf.cast(Sigma_N_inv, tf.float64))
            Sigma_N = tf.linalg.cholesky_solve(
                cholesky,
                tf.eye(tf.shape(self.Sigma_N)[-1], dtype=tf.float64))
            return tf.cast(Sigma_N, tf.float32)

        def update_Sigma_N():
            Sigma_N_delta = - (
                tf.matmul(tf.matmul(self.Sigma_N, b_b_T), self.Sigma_N)
                / (variance + tf.reduce_mean(batch_quadratic_form(self.Sigma_N, b_N))))
            Sigma_N = self.Sigma_N + Sigma_N_delta
            return Sigma_N

        Sigma_N = tf.cond(
            tf.equal(self.N, 0),
            initialize_Sigma_N,
            update_Sigma_N)

        self.Sigma_N.assign(Sigma_N)

        Delta_N_delta = tf.matmul(
            (gamma * b_N_not - b_N), b_N, transpose_a=True)
        self.Delta_N.assign_add(Delta_N_delta)

        mu_hat = (
            tf.cast(self.N / (self.N + N), b_hat.dtype)
            * self.mu_hat
        ) + (
            tf.cast(N / (self.N + N), b_hat.dtype)
            * tf.reduce_mean(b_hat, axis=0))

        self.mu_hat.assign(mu_hat)

        b_hat_b_hat_T = tf.matmul(
            b_hat[..., None], b_hat[..., None], transpose_b=True)

        Sigma_hat = (
            tf.cast(self.N / (self.N + N), b_hat_b_hat_T.dtype)
            * self.Sigma_hat
        ) + (
            tf.cast(N / (self.N + N), b_hat_b_hat_T.dtype)
            * tf.reduce_mean(b_hat_b_hat_T, axis=0))

        self.Sigma_hat.assign(Sigma_hat)

        self.N.assign_add(N)

        return True

    def get_diagnostics(self):
        diagnostics = OrderedDict((
            ('N', self.N.numpy()),
            ('epistemic_uncertainty', self(True).numpy()),
        ))
        return diagnostics


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
