from collections import OrderedDict

import tensorflow as tf
import tree

from .virel import VIREL, compute_Q_targets


class BBOVIREL(VIREL):
    """Bayesian Bellman Operator VIREL.

    References
    ----------
    [1]
    """

    def __init__(self,
                 *args,
                 fs,
                 f_lr=1e-3,
                 prior_mean=0.0,
                 prior_stddev=0.1,
                 **kwargs):
        """
        Args:
            fs: TODO(hartikainen)
            f_lr: TODO(hartikainen)
            prior_mean: TODO(hartikainen)
            prior_stddev: TODO(hartikainen)
        """
        self._f_lr = f_lr
        self._fs = fs
        self._f_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._f_lr,
                name=f'f_{i}_optimizer'
            ) for i, f in enumerate(self._fs))

        self._prior_mean = prior_mean
        self._prior_stddev = prior_stddev

        return super(BBOVIREL, self).__init__(*args, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        reward_scale = self._reward_scale
        discount = self._discount

        next_actions = self._policy.actions(next_observations)
        next_Qs_values = tuple(
            Q.values(next_observations, next_actions) for Q in self._Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            rewards,
            terminals,
            discount,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, batch):
        """Update the Q-function."""
        observations = batch['observations']
        actions = tf.random.uniform(
            tf.shape(batch['actions']),
            minval=self._training_environment.action_space.low,
            maxval=self._training_environment.action_space.high)
        tf.debugging.assert_shapes((
            (actions, ('B', 'dA')), (batch['actions'], ('B', 'dA'))))

        fs_values = tuple(
            f.values(observations, actions) for f in self._fs)
        f_values = tf.reduce_min(fs_values, axis=0)

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q.values(observations, actions)
                Q_losses = (
                    0.5 * tf.losses.MSE(y_true=f_values, y_pred=Q_values))

            gradients = tape.gradient(Q_losses, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_f(self, batch):
        Q_targets = self._compute_Q_targets(batch)
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        fs_values = []
        fs_losses = []
        for f, optimizer in zip(self._fs, self._f_optimizers):
            with tf.GradientTape() as tape:
                f_values = f.values(observations, actions)
                prior_losses = tf.reduce_sum(tree.map_structure(
                    lambda phi: tf.reduce_mean(
                        tf.losses.MSE(phi, tf.random.normal(
                            tf.shape(phi),
                            mean=self._prior_mean,
                            stddev=self._prior_stddev))),
                    tree.flatten(f.trainable_variables)))

                f_losses = 0.5 * (
                    tf.losses.MSE(y_true=Q_targets, y_pred=f_values)
                    + prior_losses)

            gradients = tape.gradient(f_losses, f.trainable_variables)
            optimizer.apply_gradients(zip(gradients, f.trainable_variables))
            fs_losses.append(f_losses)
            fs_values.append(f_values)

        return fs_values, fs_losses

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch, *args, **kwargs):
        # TODO(hartikainen): Separate batch for f?
        fs_values, fs_losses = self._update_f(batch)
        diagnostics = super(BBOVIREL, self)._do_updates(batch, *args, **kwargs)

        diagnostics.update((
            ('f_value-mean', tf.reduce_mean(fs_values)),
            ('f_loss-mean', tf.reduce_mean(fs_losses)),
        ))
        return diagnostics

    def _init_training(self):
        self._update_target(tau=tf.constant(1.0))

    @property
    def tf_saveables(self, *args, **kwargs):
        saveables = super(BBOVIREL, self).tf_saveables(*args, **kwargs)
        saveables.update({
            f'f_optimizer_{i}': optimizer
            for i, optimizer in enumerate(self._f_optimizers)
        })

        return saveables
