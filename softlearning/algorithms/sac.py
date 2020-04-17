from copy import deepcopy
from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from softlearning.utils.gym import is_continuous_space, is_discrete_space
from softlearning.rl_ops import retrace_ops
from .rl_algorithm import RLAlgorithm


@tf.function(experimental_relax_shapes=True)
def td_targets(rewards, discounts, next_values):
    return rewards + discounts * next_values


@tf.function(experimental_relax_shapes=True)
def compute_Q_targets(next_Q_values,
                      next_log_pis,
                      rewards,
                      terminals,
                      discount,
                      entropy_scale,
                      reward_scale):
    next_values = next_Q_values - entropy_scale * next_log_pis
    terminals = tf.cast(terminals, next_values.dtype)

    Q_targets = td_targets(
        rewards=reward_scale * rewards,
        discounts=discount,
        next_values=(1.0 - terminals) * next_values)

    return Q_targets


def heuristic_target_entropy(action_space):
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(action_space.shape)
    elif is_discrete_space(action_space):
        raise NotImplementedError(
            "TODO(hartikainen): implement for discrete spaces.")
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
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
            alpha_lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,

            save_full_state=False,
            target_type='TD(0)',
            retrace_n_step=1,
            retrace_lambda=1.0,
            sequence_type='random',
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(deepcopy(Q) for Q in Qs)
        self._update_target(tau=tf.constant(1.0))

        self._plotter = plotter

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr
        self._alpha_lr = alpha_lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            heuristic_target_entropy(self._training_environment.action_space)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval

        self._save_full_state = save_full_state

        self._target_type = target_type
        self._retrace_n_step = retrace_n_step
        self._retrace_lambda = retrace_lambda

        self._sequence_type = sequence_type

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        self._alpha = tf.Variable(tf.exp(0.0), name='alpha')

        self._alpha_optimizer = tf.optimizers.Adam(
            self._alpha_lr, name='alpha_optimizer')

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets_retrace(self, batch):
        entropy_scale = self._alpha
        reward_scale = self._reward_scale
        discount = self._discount

        mask_t_0 = ~batch['mask'][..., 1:]
        float_mask_t_0 = tf.cast(mask_t_0[..., None], tf.float32)

        (observations_t_0,
         actions_t_0,
         raw_actions_t_0,
         observations_t_1,
         rewards_t_0,
         terminals_t_0,
         p_old_a_t_0) = tree.map_structure(
             lambda x: tf.where(
                 mask_t_0[..., None], x[:, 1:, ...], tf.zeros(1, dtype=x.dtype)),
             (batch['observations'],
              batch['actions'],
              batch['raw_actions'],
              batch['next_observations'],
              batch['rewards'],
              batch['terminals'],
              batch['p_a_t']))

        p_a_t_0 = self._policy.probs_for_raw_actions(
            observations_t_0, raw_actions_t_0)

        safe_denominator = 1.0
        safe_trace_denominators = tf.where(
            mask_t_0[..., None], p_old_a_t_0, safe_denominator)
        trace_numerators = p_a_t_0
        traces_t_0 = tf.where(
            mask_t_0[..., None],
            tf.minimum(trace_numerators / safe_trace_denominators, 1.0),
            1.0)
        traces_t_0 = self._retrace_lambda * traces_t_0
        traces_t_1 = tf.concat((
            traces_t_0[:, 1:, ...], tf.zeros(tf.shape(traces_t_0[:, :1, ...]))
        ), axis=1)

        actions_t_1_sampled, log_p_a_t_1_sampled = (
            self._policy.actions_and_log_probs(observations_t_1))

        expected_target_Q_values_t_1 = tuple(
            Q.values(observations_t_1, actions_t_1_sampled)
            for Q in self._Q_targets)
        expected_target_Q_values_t_1 = tf.reduce_min(
            expected_target_Q_values_t_1, axis=0)

        actions_t_1 = tf.concat((
            actions_t_0[:, 1:],
            tf.zeros(tf.shape(actions_t_0[:, -1:, ...]))
        ), axis=1)
        target_Qs_values_t_1 = tuple(
            Q.values(observations_t_1, actions_t_1) for Q in self._Q_targets)
        target_Q_values_t_1 = tf.reduce_min(target_Qs_values_t_1, axis=0)

        raw_actions_t_1 = tf.concat((
            raw_actions_t_0[:, 1:],
            tf.zeros(tf.shape(raw_actions_t_0[:, -1:, ...])),
        ), axis=1)
        log_p_a_t_1 = self._policy.log_probs_for_raw_actions(
            observations_t_1, raw_actions_t_1)
        retrace_inputs = (
            # rewards and continuation_probs should be of t_0
            reward_scale * rewards_t_0,
            discount * (1.0 - tf.cast(terminals_t_0, rewards_t_0.dtype)),
            # traces and q values should be of t_1
            traces_t_1,
            expected_target_Q_values_t_1 - entropy_scale * log_p_a_t_1_sampled,
            target_Q_values_t_1 - entropy_scale * log_p_a_t_1)

        # NOTE(hartikainen): We need to swap the batch_shape and
        # sequence_length axes because the retrace implementation uses
        # `tf.scan` which does not support custom axes.
        retrace_inputs = tree.map_structure(
            lambda x: tf.transpose(
                x, tf.concat(((1, 0), tf.range(2, tf.rank(x))), axis=0)),
            retrace_inputs)

        Q_targets_retrace = retrace_ops.off_policy_corrected_multistep_target(
            *retrace_inputs, back_prop=False)

        # NOTE(hartikainen): Swap the axes back. See note above.
        Q_targets_retrace = tf.transpose(
            Q_targets_retrace,
            tf.concat((
                (1, 0), tf.range(2, tf.rank(Q_targets_retrace))
            ), axis=0))

        Q_targets_retrace = tf.where(
            mask_t_0[..., None], Q_targets_retrace, 0.0)

        return tf.stop_gradient(Q_targets_retrace)

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets_td(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        entropy_scale = self._alpha
        reward_scale = self._reward_scale
        discount = self._discount

        next_actions, next_log_pis = self._policy.actions_and_log_probs(
            next_observations)
        next_Qs_values = tuple(
            Q.values(next_observations, next_actions) for Q in self._Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            next_log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, batch):
        if self._target_type == 'TD(0)':
            return self._compute_Q_targets_td(batch)
        elif self._target_type == 'retrace':
            return self._compute_Q_targets_retrace(batch)

        raise NotImplementedError(self._target_type)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, batch):
        """Update the Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._compute_Q_targets(batch)

        if self._target_type == 'retrace':
            mask = ~batch['mask'][..., 1:]
            observations, actions = tree.map_structure(
                lambda x: tf.where(
                    mask[..., None], x[:, 1:, ...], tf.zeros(1, dtype=x.dtype)),
                (batch['observations'], batch['actions']))
            tf.debugging.assert_shapes((
                (Q_targets, ('B', 'S', 1)), (actions, ('B', 'S', 'dA'))))
        else:
            mask = tf.ones(tf.shape(batch['rewards'])[:1], dtype=bool)
            observations = batch['observations']
            actions = batch['actions']
            tf.debugging.assert_shapes((
                (Q_targets, ('B', 1)), (actions, ('B', 'dA'))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q.values(observations, actions)
                Q_losses = tf.where(
                    mask,
                    0.5 * tf.losses.MSE(y_true=Q_targets, y_pred=Q_values),
                    0.0)

            gradients = tape.gradient(Q_losses, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self, batch):
        """Update the policy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations = batch['observations']

        with tf.GradientTape() as tape:
            actions, log_pis = self._policy.actions_and_log_probs(observations)

            Qs_log_targets = tuple(
                Q.values(observations, actions) for Q in self._Qs)
            Q_log_targets = tf.reduce_min(Qs_log_targets, axis=0)

            policy_losses = self._alpha * log_pis - Q_log_targets

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
    def _update_alpha(self, batch):
        if not isinstance(self._target_entropy, Number):
            return 0.0

        observations = batch['observations']

        actions, log_pis = self._policy.actions_and_log_probs(observations)

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
            # NOTE(hartikainen): It's important that we take the average here,
            # otherwise we end up effectively having `batch_size` times too
            # large learning rate.
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._alpha])
        self._alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [self._alpha]))

        return alpha_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch):
        """Runs the update operations for policy, Q, and alpha."""
        if self._target_type == 'retrace':
            batch_size = self._batch_size

            if self._sequence_type == 'fill_whole_sequence':
                last_step_indices = np.flatnonzero(
                    self.pool.data['episode_index_backwards'][:self.pool.size] == 0)
                batch_indices = np.random.choice(last_step_indices, batch_size)
                # NOTE(hartikainen): Need to cast because the indices are unsigned
                # ints and we're subtracting!
                num_steps_before = np.int32(self.pool.data[
                    'episode_index_forwards'
                ][:self.pool.size][batch_indices].flatten())
                sampling_window_widths = np.maximum(
                    # +1 to include the current sample to the window
                    1 + num_steps_before - self._retrace_n_step, 0)
                batch_indices_offset = -1 * np.random.randint(
                    low=np.zeros_like(sampling_window_widths),
                    # +1 to account the excluded right boundary
                    high=sampling_window_widths + 1)
                batch_indices = batch_indices + batch_indices_offset
                # offset_batch_indices = batch_indices
            elif self._sequence_type == 'last_steps':
                last_step_indices = np.flatnonzero(
                    self.pool.data['episode_index_backwards'][:self.pool.size] == 0)
                batch_indices = np.random.choice(last_step_indices, batch_size)
            elif self._sequence_type == 'random':
                batch_indices = self.pool.random_indices(batch_size)
            else:
                raise NotImplementedError(self._sequence_type)

            retrace_batch = self.pool.sequence_batch_by_indices(
                # NOTE(hartikainen): Add 1 to the sequence length in order to
                # account for the shifted states and actions
                # (i.e. (s_{t}, a_{t-1})) in q_z_model inputs.
                batch_indices, self._retrace_n_step + 1)
            Qs_values, Qs_losses = self._update_critic(retrace_batch)
        else:
            Qs_values, Qs_losses = self._update_critic(batch)
        policy_losses = self._update_actor(batch)
        alpha_losses = self._update_alpha(batch)

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('alpha', self._alpha),
            ('alpha_loss-mean', tf.reduce_mean(alpha_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        training_diagnostics = self._do_updates(batch)

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
            ('alpha', self._alpha.numpy()),
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
            '_alpha': self._alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
