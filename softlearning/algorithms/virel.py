from copy import deepcopy
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .rl_algorithm import RLAlgorithm

from .sac import td_targets


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
            pool,
            plotter=None,

            policy_lr=3e-4,
            Q_lr=3e-4,
            reward_scale=1.0,
            discount=0.99,
            tau=5e-3,
            beta_scale=4e-4,
            beta_batch_size=4096,
            beta_update_type='MSBE',
            target_update_interval=1,
            Q_update_type='MSBE',
            uncertainty_estimator=None,
            sample_actions_from='random',

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

        self._pool = pool
        self._plotter = plotter

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr

        self._reward_scale = reward_scale
        self._discount = discount
        self._tau = tau
        self._beta_scale = beta_scale
        self._beta_batch_size = beta_batch_size
        self._beta_update_type = beta_update_type
        self._target_update_interval = target_update_interval
        self._Q_update_type = Q_update_type

        self._save_full_state = save_full_state
        self.last_training_step = -1

        self._epistemic_uncertainty = tf.Variable(float('inf'))
        self._uncertainty_estimator = uncertainty_estimator
        self._sample_actions_from = sample_actions_from

        self._uncertainty_optimizer = tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'uncertainty_optimizer')

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
    def _compute_Q_targets(self, next_observations, rewards, terminals, Qs=None):
        Qs = Qs or self._Q_targets
        reward_scale = self._reward_scale
        discount = self._discount

        next_actions = self._policy.actions(next_observations)
        next_Qs_values = tuple(
            Q.values(next_observations, next_actions) for Q in Qs)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            rewards,
            terminals,
            discount,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic_MSBBE(self,
                             observations,
                             actions,
                             next_observations,
                             rewards,
                             terminals):
        """Update the Q-function."""
        raise NotImplementedError(
            "TODO(hartikainen): Adapt for RPF ensembles.")
        Q_targets = self._compute_Q_targets(
            next_observations, rewards, terminals,
            Qs=self.linearized_Q_targets)

        Qs_values = []
        Qs_losses = []
        for Q, uncertainty_model, feature_fn, optimizer in zip(
                self._Qs,
                self.uncertainty_models,
                self.Q_jacobian_features,
                self._Q_optimizers):
            b = feature_fn((observations, actions))

            Delta_N = uncertainty_model.Delta_N
            Sigma_N = uncertainty_model.Sigma_N
            Sigma_hat = uncertainty_model.Sigma_hat

            Q_values = Q.values(observations, actions)
            deltas = Q_targets - Q_values

            gradients = tf.reduce_mean(
                tf.einsum(
                    'ij,jk,kl,lm,bm,bX->bi',
                    Delta_N,
                    Sigma_N,
                    Sigma_hat,
                    Sigma_N,
                    b,
                    deltas
                ), axis=0)

            variable_shapes = [tf.shape(x) for x in Q.trainable_variables]
            variable_sizes = [tf.size(x) for x in Q.trainable_variables]
            gradient_splits = tf.split(gradients, variable_sizes)
            reshaped_gradients = [
                tf.reshape(gradient, shape)
                for gradient, shape
                in zip(gradient_splits, variable_shapes)
            ]
            optimizer.apply_gradients(
                zip(reshaped_gradients, Q.trainable_variables))

            Qs_losses.append(deltas)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_critic_MSBE(self,
                            observations,
                            actions,
                            next_observations,
                            rewards,
                            terminals):
        """Update the Q-function."""
        Q_targets = self._compute_Q_targets(
            next_observations, rewards, terminals)

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
    def _update_critic(self, *args, **kwargs):
        if self._Q_update_type == 'MSBE':
            return self._update_critic_MSBE(*args, **kwargs)
        elif self._Q_update_type == 'MSBBE':
            return self._update_critic_MSBBE(*args, **kwargs)

        raise NotImplementedError(self._Q_update_type)

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self, observations):
        """Update the policy."""

        with tf.GradientTape() as tape:
            actions = self._policy.actions(observations)
            log_pis = self._policy.log_pis(observations, actions)

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
    def _update_beta_MSBE(self,
                          observations,
                          actions,
                          next_observations,
                          rewards,
                          terminals):
        Q_targets = self._compute_Q_targets(
            next_observations, rewards, terminals)

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

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
    def _update_beta_MSBBE(self,
                           observations,
                           actions,
                           next_observations,
                           rewards,
                           terminals):
        b = self.Q_jacobian_features((observations, actions))
        model_output = self.uncertainty_model(b)
        loc = model_output[0]

        Q_targets = self._compute_Q_targets(
            next_observations, rewards, terminals)

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        MSBBEs = 0.5 * tf.losses.MSE(y_true=Q_targets, y_pred=loc)
        MSBBE = tf.nn.compute_average_loss(MSBBEs)
        self._beta.assign(self._beta_scale * MSBBE)

        return MSBBE

    @tf.function(experimental_relax_shapes=True)
    def _update_beta_uncertainty(self,
                                 observations,
                                 actions,
                                 next_observations,
                                 *args,
                                 **kwargs):
        beta_losses = self._epistemic_uncertainty
        self._beta.assign(self._beta_scale * self._epistemic_uncertainty)
        return beta_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_beta(self, *args, **kwargs):
        if self._beta_update_type is None:
            return 0.0
        elif self._beta_update_type == 'MSBE':
            return self._update_beta_MSBE(*args, **kwargs)
        elif self._beta_update_type == 'MSBBE':
            return self._update_beta_MSBBE(*args, **kwargs)
        elif self._beta_update_type == 'uncertainty':
            return self._update_beta_uncertainty(*args, **kwargs)

        raise NotImplementedError(self._beta_update_type)

    @tf.function(experimental_relax_shapes=True)
    def _update_uncertainty_model(self,
                                  observations,
                                  actions,
                                  next_observations):
        with tf.GradientTape() as tape:
            predictions = self._uncertainty_estimator((observations, actions))
            losses = 0.5 * tf.losses.MSE(y_true=0.0, y_pred=predictions)

        model_and_prior_predictions_before = [
            model.model_and_prior((observations, actions))
            for model in self._uncertainty_estimator.models
        ]
        gradients = tape.gradient(
            losses, self._uncertainty_estimator.trainable_variables)
        self._uncertainty_optimizer.apply_gradients(zip(
            gradients, self._uncertainty_estimator.trainable_variables))
        model_and_prior_predictions_after = [
            model.model_and_prior((observations, actions))
            for model in self._uncertainty_estimator.models
        ]

        for (model_before, prior_before), (model_after, prior_after) in zip(
                model_and_prior_predictions_before,
                model_and_prior_predictions_after):
            tf.debugging.Assert(
                tf.reduce_any(model_before != model_after),
                (model_before, model_after))
            tf.debugging.assert_equal(prior_before, prior_after)

        return losses

    @tf.function(experimental_relax_shapes=True)
    def _update_epistemic_uncertainty(self,
                                      observations,
                                      actions,
                                      next_actions):
        if self._sample_actions_from == 'random':
            random_actions = tf.random.uniform(
                actions.shape,
                minval=self._training_environment.action_space.low,
                maxval=self._training_environment.action_space.high)
        elif self._sample_actions_from == 'pool':
            random_actions = actions

        predictions = self._uncertainty_estimator((observations, random_actions))
        epistemic_uncertainties = tf.reduce_mean(predictions ** 2, axis=(-1, -2))
        epistemic_uncertainty = tf.reduce_mean(epistemic_uncertainties)
        self._epistemic_uncertainty.assign(epistemic_uncertainty)
        return epistemic_uncertainties

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch, uncertainty_batch):
        uncertainty_losses = self._update_uncertainty_model(
            uncertainty_batch['observations'],
            uncertainty_batch['actions'],
            uncertainty_batch['next_observations'])
        epistemic_uncertainties = self._update_epistemic_uncertainty(
            uncertainty_batch['observations'],
            uncertainty_batch['actions'],
            uncertainty_batch['next_observations'])

        beta_losses = self._update_beta(
            uncertainty_batch['observations'],
            uncertainty_batch['actions'],
            uncertainty_batch['next_observations'],
            uncertainty_batch['rewards'],
            uncertainty_batch['terminals'])
        epistemic_uncertainty = self._epistemic_uncertainty

        Qs_values, Qs_losses = self._update_critic(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            batch['rewards'],
            batch['terminals'])

        policy_losses = self._update_actor(batch['observations'])

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('beta', self._beta),
            ('epistemic_uncertainty', epistemic_uncertainty),
            ('batch_epistemic_uncertainty-mean', tf.reduce_mean(
                epistemic_uncertainties)),
            ('beta_loss-mean', tf.reduce_mean(beta_losses)),
            ('uncertainty_loss-mean', tf.reduce_mean(uncertainty_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        # steps_since_last_training = (
        #     self._total_timestep - self.last_training_step)
        # TODO(hartikainen): For not, assume that training update interval is 1
        # assert steps_since_last_training == 1, steps_since_last_training
        # self.last_training_step = self._total_timestep
        # uncertainty_batch = self._pool.last_n_batch(steps_since_last_training)
        uncertainty_batch = self._pool.random_batch(1024)
        training_diagnostics = self._do_updates(batch, uncertainty_batch)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

        return training_diagnostics

    def _init_training(self):
        self._update_target(tau=tf.constant(1.0))

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
            ('policy', self._policy.get_diagnostics(batch['observations'])),
            ('epistemic_uncertainty',
             tf.reduce_mean(self._epistemic_uncertainty).numpy())
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
