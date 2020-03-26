from copy import deepcopy
from collections import OrderedDict
import numbers

import tensorflow as tf

from .rl_algorithm import RLAlgorithm

from .sac import td_targets, heuristic_target_entropy


@tf.function(experimental_relax_shapes=True)
def compute_Q_targets(next_Q_values,
                      rewards,
                      terminals,
                      discount,
                      reward_scale):
    next_values = next_Q_values
    terminals = tf.cast(terminals, next_values.dtype)

    Q_targets = td_targets(
        rewards=reward_scale * rewards,
        discounts=discount,
        next_values=(1.0 - terminals) * next_values)

    return Q_targets


class VIREL(RLAlgorithm):
    """A Variational Inference Framework for Reinforcement Learning (VIREL).

    References
    ----------
    [1] Matthew Fellows*, Anuj Mahajan*, Tim G. J. Rudner, and Shimon Whiteson.
        VIREL: A Variational Inference Framework for Reinforcement Learning.
        arXiv preprint arXiv:1811.01132. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            plotter=None,

            policy_lr=3e-4,
            Q_lr=3e-4,
            reward_scale=1.0,
            discount=0.99,
            tau=5e-3,
            beta_scale=4e-4,
            beta_batch_size=4096,
            beta_update_type='MSBE',
            beta_lr=1e-3,
            target_entropy='auto',
            target_update_interval=1,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
        """

        super(VIREL, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(deepcopy(Q) for Q in Qs)
        self._update_target(tau=tf.constant(1.0))

        self._plotter = plotter

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr

        self._reward_scale = reward_scale
        self._discount = discount
        self._tau = tau
        self._beta_scale = beta_scale
        self._beta_batch_size = beta_batch_size
        self._target_update_interval = target_update_interval
        self._beta_update_type = beta_update_type

        if beta_update_type == 'learn':
            self._target_entropy = (
                heuristic_target_entropy(self._training_environment.action_space)
                if target_entropy == 'auto'
                else target_entropy)
            self._beta_lr = beta_lr
            self._beta_optimizer = tf.optimizers.Adam(
                self._beta_lr, name='beta_optimizer')
        else:
            self._target_entropy = None

        self._save_full_state = save_full_state

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        self._beta = tf.Variable(self._beta_scale, name='beta', dtype=tf.float32)

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
        Q_targets = self._compute_Q_targets(batch)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q.values(observations, actions)
                Q_losses = (
                    0.5 * tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))

            gradients = tape.gradient(Q_losses, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self, batch):
        """Update the policy."""
        observations = batch['observations']

        with tf.GradientTape() as tape:
            actions, log_pis = self._policy.actions_and_log_probs(observations)

            Qs_log_targets = tuple(
                Q.values(observations, actions) for Q in self._Qs)
            Q_log_targets = tf.reduce_min(Qs_log_targets, axis=0)

            policy_losses = self._beta * log_pis - Q_log_targets

        tf.debugging.assert_shapes((
            (actions, ('B', 'nA')),
            (log_pis, ('B', 1)),
            (policy_losses, ('B', 1)),
        ))

        policy_gradients = tape.gradient(
            policy_losses, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(
            policy_gradients, self._policy.trainable_variables))

        return policy_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_beta_MSBE(self, batch):
        Q_targets = self._compute_Q_targets(batch)

        observations = batch['observations']
        actions = batch['actions']

        Qs_losses = [
            0.5 * tf.losses.MSE(
                y_true=Q_targets,
                y_pred=Q.values(observations, actions),
            )
            for Q in self._Qs
        ]

        Qs_loss = tf.nn.compute_average_loss(Qs_losses)

        self._beta.assign(self._beta_scale * Qs_loss)

        return Qs_loss

    @tf.function(experimental_relax_shapes=True)
    def _update_beta_learn(self, batch):
        if not isinstance(self._target_entropy, numbers.Number):
            return 0.0

        observations = batch['observations']

        actions, log_pis = self._policy.actions_and_log_probs(observations)

        with tf.GradientTape() as tape:
            beta_losses = -1.0 * (
                self._beta * tf.stop_gradient(log_pis + self._target_entropy))
            # NOTE(hartikainen): It's important that we take the average here,
            # otherwise we end up effectively having `batch_size` times too
            # large learning rate.
            beta_loss = tf.nn.compute_average_loss(beta_losses)

        beta_gradient = tape.gradient(beta_loss, self._beta)
        self._beta_optimizer.apply_gradients([[beta_gradient, self._beta]])

        return beta_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_beta(self, *args, **kwargs):
        if self._beta_update_type is None:
            return 0.0
        elif self._beta_update_type == 'MSBE':
            return self._update_beta_MSBE(*args, **kwargs)
        elif self._beta_update_type == 'learn':
            return self._update_beta_learn(*args, **kwargs)

        raise NotImplementedError(self._beta_update_type)

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch, beta_batch):
        Qs_values, Qs_losses = self._update_critic(batch)

        policy_losses = self._update_actor(batch)
        beta_losses = self._update_beta(beta_batch)

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('beta', self._beta),
            ('beta_loss-mean', tf.reduce_mean(beta_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        beta_batch_size = min(self._beta_batch_size, self.pool.size)
        beta_batch = self._training_batch(beta_batch_size)
        training_diagnostics = self._do_updates(batch, beta_batch)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

        return training_diagnostics

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as an ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        diagnostics = OrderedDict((
            ('beta', self._beta.numpy()),
            ('policy', self._policy.get_diagnostics_np(batch['observations'])),
        ))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_beta': self._beta,
        }

        return saveables
