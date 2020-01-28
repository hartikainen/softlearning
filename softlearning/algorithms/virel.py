from copy import deepcopy
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .rl_algorithm import RLAlgorithm

from softlearning.utils.tensorflow import nest
from softlearning.models.bae.linear import (
    LinearStudentTModel,
    JacobianModel)
from softlearning.utils.tensorflow import cast_and_concat, nest
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
            TD_target_model_update_interval=100,
            Q_update_type='MSBE',
            features_from=None,

            save_full_state=False,
            diagonal_noise_scale=1e-4,
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
        self._TD_target_model_update_interval = TD_target_model_update_interval
        self._Q_update_type = Q_update_type
        self._features_from = features_from
        self._diagonal_noise_scale = diagonal_noise_scale

        self._save_full_state = save_full_state

        D = (np.prod(self._training_environment.action_space.shape)
             + np.sum(
                 np.prod(space.shape)
                 for space in self._training_environment.observation_space.spaces.values()
             ))

        class wrapped_Q:
            def __init__(self, models):
                self._models = models

            @property
            def trainable_variables(self):
                return [
                    variable
                    for model in self._models
                    for variable in model.trainable_variables
                ]

            @tf.function(experimental_relax_shapes=True)
            def __call__(self, inputs):
                outputs = tuple(
                    model(inputs) for model in self._models)
                outputs = tf.reduce_min(outputs, axis=0)
                return outputs

        if self._Qs[0].model.name == 'linearized_feedforward_Q':
            features_from = {
                'Q_targets': self._Q_targets,
                'Qs': self._Qs,
            }[self._features_from]

            features_from[0].model.summary()
            for Q in features_from:
                linearized_model = features_from[0].model.layers[-2]
                assert isinstance(
                    linearized_model, JacobianModel), (
                        linearized_model)
            feature_fns = [
                tf.keras.Model(
                    Q.model.inputs,
                    Q.model.layers[-2].output,
                    name=f'linearized_feature_model_{i}',
                    trainable=False)
                for i, Q in enumerate(features_from)
            ]
            self.feature_fn = wrapped_Q(feature_fns)
        elif self._Qs[0].model.name == 'feedforward_Q':
            features_from = {
                'Q_targets': self._Q_targets,
                'Qs': self._Qs,
            }[self._features_from]

            features_from[0].model.summary()
            feature_fns = [
                JacobianModel(
                    Q.model,
                    name=f'linearized_feature_model_{i}')
                for i, Q in enumerate(features_from)
            ]
            self.feature_fn = wrapped_Q(feature_fns)
        else:
            raise NotImplementedError(self._Q_targets[0].model.name)

        self.uncertainty_model = LinearStudentTModel()

        self._epistemic_uncertainty = tf.Variable(float('inf'))

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
    def _compute_Q_targets(self, next_observations, rewards, terminals):
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
    def _update_critic_MSBBE(self,
                             observations,
                             actions,
                             next_observations,
                             rewards,
                             terminals):
        """Update the Q-function."""
        b = self.feature_fn((observations, actions))
        loc = self.uncertainty_model(b)[0]
        Q_targets = loc

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
    def _update_beta_uncertainty(self, *args, **kwargs):
        self._beta.assign(self._beta_scale * self._epistemic_uncertainty)
        return self._epistemic_uncertainty

    @tf.function(experimental_relax_shapes=True)
    def _update_beta(self, *args, **kwargs):
        if self._beta_update_type is None:
            return 0.0
        elif self._beta_update_type == 'MSBE':
            return self._update_beta_MSBE(*args, **kwargs)
        elif self._beta_update_type == 'uncertainty':
            return self._update_beta_uncertainty(*args, **kwargs)

        raise NotImplementedError(self._beta_update_type)

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _update_uncertainty_model(self, B, Y):
        # Y = self._compute_Q_targets(
        #     data['next_observations'],
        #     data['rewards'],
        #     data['terminals'])

        # B = self.feature_fn((data['observations'], data['actions']))

        diagonal_noise_scale = tf.constant(self._diagonal_noise_scale)
        self.uncertainty_model.update(B, Y, diagonal_noise_scale)

        return tf.constant(True)

    @tf.function(experimental_relax_shapes=True)
    def _update_uncertainties(self, observations, actions):
        b = self.feature_fn((observations, actions))
        epistemic_uncertainties = self.uncertainty_model(b)[-1]

        self._epistemic_uncertainty.assign(
            tf.reduce_mean(epistemic_uncertainties))

        return epistemic_uncertainties

    # @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch, beta_batch):
        Qs_values, Qs_losses = self._update_critic(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            batch['rewards'],
            batch['terminals'])

        policy_losses = self._update_actor(batch['observations'])
        epistemic_uncertainties = self._update_uncertainties(
            beta_batch['observations'], beta_batch['actions'])
        beta_losses = self._update_beta(
            beta_batch['observations'],
            beta_batch['actions'],
            beta_batch['next_observations'],
            beta_batch['rewards'],
            beta_batch['terminals'])

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('beta', self._beta),
            ('epistemic_uncertainty-mean', tf.reduce_mean(
                epistemic_uncertainties)),
            ('beta_loss-mean', tf.reduce_mean(beta_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        beta_batch_size = min(self._beta_batch_size, self._pool.size)
        beta_batch = self._training_batch(beta_batch_size)

        if iteration == 0:
            initialize_model_batch = self._training_batch(1)
            _ = self.uncertainty_model((
                self.feature_fn((
                    initialize_model_batch['observations'],
                    initialize_model_batch['actions']))))

        if iteration % self._TD_target_model_update_interval == 0:
            target_model_prior_data = self._pool.last_n_batch(1e5)
            N = target_model_prior_data['rewards'].shape[0]
            target_model_batch_size = 256
            Y_parts, B_parts = [], []
            for i in range(0, N, target_model_batch_size):
                B = self.feature_fn(
                    nest.map_structure(
                        lambda x: x[i:i+target_model_batch_size, ...],
                        (target_model_prior_data['observations'],
                         target_model_prior_data['actions']))
                    )
                Y = self._compute_Q_targets(
                    *nest.map_structure(
                        lambda x: x[i:i+target_model_batch_size, ...],
                        (target_model_prior_data['next_observations'],
                         target_model_prior_data['rewards'],
                         target_model_prior_data['terminals'])
                    ))

                B_parts.append(B)
                Y_parts.append(Y)

            B = tf.concat(B_parts, axis=0)
            Y = tf.concat(Y_parts, axis=0)
            self._update_uncertainty_model(B, Y)

            del target_model_prior_data
            del B
            del Y

        training_diagnostics = self._do_updates(batch, beta_batch)

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

        diagnostics['uncertainty_model'] = (
            self.uncertainty_model.get_diagnostics())

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
