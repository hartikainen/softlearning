import pytest
import tensorflow as tf
import numpy as np

import unittest

from softlearning.models.bae.linear import (
    LinearStudentTModel,
    LinearGaussianModel,
    OnlineUncertaintyModel,
    JacobianModel,
    LinearizedModel)


tf.config.experimental_run_functions_eagerly(True)


def load_dataset(n=150, n_tst=150):
    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]

    # np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = y[..., np.newaxis]
    x = x[..., np.newaxis]
    x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
    x_tst = x_tst[..., np.newaxis]

    y = y.astype(np.float32)
    x = x.astype(np.float32)
    x_tst = x_tst.astype(np.float32)
    return y, x, x_tst


class OnlineUncertaintyModelTest(unittest.TestCase):
    @pytest.mark.skip(reason="Not implemented properly.")
    def test_online_updates_batch(self):
        y, x, x_test = load_dataset()

        uncertainty_model = OnlineUncertaintyModel()
        _ = uncertainty_model(x)

        batch_size = 25
        for batch_start in range(0, x.shape[0], batch_size):
            b = x[batch_start:batch_start+batch_size, ...]
            b_hat = x[batch_start:batch_start+batch_size, ...]
            uncertainty_model.online_update((b, b_hat))
            epistemic_uncertainty = uncertainty_model(x_test[0])
            print(f"batch_start: {batch_start}, epistemic_uncertainty: {epistemic_uncertainty}")

            xx_ = tf.concat((tf.ones(tf.shape(b)[:-1])[..., None], b), axis=-1)
            expected_mu_hat = tf.reduce_mean(xx_, axis=0)
            tf.debugging.assert_near(uncertainty_model.mu_hat, expected_mu_hat)

            diagnostics = uncertainty_model.get_diagnostics()

            tf.debugging.assert_equal(uncertainty_model.N, batch_start + batch_size)
            assert ('N', ) == tuple(diagnostics.keys()), diagnostics.keys()
            tf.debugging.assert_equal(diagnostics['N'], batch_start + batch_size)

            # xx = tf.concat((tf.ones(tf.shape(x)[:-1])[..., None], x), axis=-1)

            variance = 1.0
            expected_Sigma_N = tf.linalg.inv(
                tf.eye(xx_.shape[-1], dtype=tf.float32)
                + tf.matmul(xx_, xx_, transpose_a=True) / variance)
            try:
                tf.debugging.assert_near(uncertainty_model.Sigma_N, expected_Sigma_N)
            except Exception as e:
                breakpoint()
                pass

    def test_online_updates_single(self):
        y, x, x_test = load_dataset()
        hat_x = x + 5.0 + np.random.uniform(-1, 1, x.shape)
        next_x = x - 5.0 + np.random.uniform(-1, 1, x.shape)

        uncertainty_model = OnlineUncertaintyModel()
        _ = uncertainty_model(x)

        # single sample mode
        for i, (x_, hat_x_, next_x_, y_) in enumerate(zip(x, hat_x, next_x, y), 1):
            b = x_[None, ...]
            b_hat = hat_x_[None, ...]
            b_not = next_x_[None, ...]
            gamma = 0.99
            uncertainty_model.online_update((b, b_hat, b_not, gamma))
            epistemic_uncertainty = uncertainty_model(x_test[0])
            print(f"i: {i}, epistemic_uncertainty: {epistemic_uncertainty}")

            xx_ = tf.concat((
                tf.ones(tf.shape(x[:i])[:-1])[..., None],
                x[:i],
            ), axis=-1)
            next_xx_ = tf.concat((
                tf.ones(tf.shape(next_x[:i])[:-1])[..., None],
                next_x[:i],
            ), axis=-1)
            hat_xx_ = tf.concat((
                tf.ones(tf.shape(hat_x[:i])[:-1])[..., None],
                hat_x[:i],
            ), axis=-1)

            expected_mu_hat = tf.reduce_mean(hat_xx_, axis=0)
            tf.debugging.assert_near(uncertainty_model.mu_hat, expected_mu_hat)

            diagnostics = uncertainty_model.get_diagnostics()

            tf.debugging.assert_equal(uncertainty_model.N, i)
            assert ('N', ) == tuple(diagnostics.keys()), diagnostics.keys()
            tf.debugging.assert_equal(diagnostics['N'], i)

            # xx = tf.concat((tf.ones(tf.shape(x)[:-1])[..., None], x), axis=-1)

            expected_Delta_N = tf.matmul(
                (gamma * next_xx_ - xx_), xx_, transpose_a=True)
            expected_Delta_N_2 = tf.reduce_sum(
                tf.matmul(
                    (gamma * next_xx_ - xx_)[:, None, :],
                    xx_[:, None, :],
                    transpose_a=True
                ), axis=0)
            expected_Delta_N_3 = tf.reduce_sum([
                tf.matmul(
                    (gamma * next_xx_[i] - xx_[i])[None, ...],
                    xx_[i][None, ...],
                    transpose_a=True)
                for i in range(xx_.shape[0])
            ], axis=0)
            tf.debugging.assert_near(
                uncertainty_model.Delta_N,
                expected_Delta_N,
                rtol=1e-6,
            )
            tf.debugging.assert_near(
                uncertainty_model.Delta_N,
                expected_Delta_N_2,
                rtol=1e-6,
            )
            tf.debugging.assert_near(
                uncertainty_model.Delta_N,
                expected_Delta_N_3,
                rtol=1e-6,
            )

            variance = 1.0
            expected_Sigma_N = tf.linalg.inv(
                tf.eye(xx_.shape[-1], dtype=tf.float32)
                + tf.matmul(xx_, xx_, transpose_a=True) / variance)
            tf.debugging.assert_near(uncertainty_model.Sigma_N, expected_Sigma_N)

            expected_Sigma_hat = tf.reduce_sum(
                tf.matmul(hat_xx_[:, None, :], hat_xx_[:, None, :], transpose_a=True),
                axis=0
            ) / i

            tf.debugging.assert_near(uncertainty_model.Sigma_hat, expected_Sigma_hat)

        expected_epistemic_uncertainty = tf.reduce_mean(
            tf.matmul(
                tf.matmul(xx_[:, None, :], uncertainty_model.Sigma_N),
                xx_[:, None, :],
                transpose_b=True))

        expected_epistemic_uncertainty_2 = tf.reduce_mean([
            tf.matmul(
                tf.matmul(xx_[i][None, ...], uncertainty_model.Sigma_N),
                xx_[i][..., None])
            for i in range(xx_.shape[0])
        ], axis=0)

        tf.debugging.assert_near(
            expected_epistemic_uncertainty,
            expected_epistemic_uncertainty_2)
        tf.debugging.assert_near(
            epistemic_uncertainty,
            expected_epistemic_uncertainty)


if __name__ == '__main__':
    unittest.main()
