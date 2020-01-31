from copy import deepcopy
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .rl_algorithm import RLAlgorithm

from softlearning.utils.tensorflow import nest
from softlearning.models.bae.linear import (
    LinearStudentTModel,
    LinearGaussianModel,
    OnlineUncertaintyModel,
    JacobianModel,
    LinearizedModel)
from softlearning.utils.tensorflow import nest
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
            learn_beta=True,
            beta_batch_size=4096,
            target_update_interval=1,
            Q_update_type='MSBE',

            save_full_state=False,
            diagonal_noise_scale=1e-4,
            uncertainty_model_type='online',
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
        self._learn_beta = learn_beta
        self._beta_batch_size = beta_batch_size
        self._target_update_interval = target_update_interval
        self._Q_update_type = Q_update_type
        self._diagonal_noise_scale = diagonal_noise_scale
        self._uncertainty_model_type = uncertainty_model_type

        self._save_full_state = save_full_state
        self.last_training_step = -1

        D = (np.prod(self._training_environment.action_space.shape)
             + np.sum(
                 np.prod(space.shape)
                 for space in self._training_environment.observation_space.spaces.values()
             ))

        self._Qs[0].model.summary()

        if self._Qs[0].model.name == 'linearized_feedforward_Q':

            self.linearized_Q_targets = self._Q_targets

            def create_jacobian_feature_model(Qs):
                Q = Qs[0]
                linearized_model = Q.model.layers[-1]
                assert isinstance(
                    linearized_model, LinearizedModel), linearized_model

                out = JacobianModel(
                    linearized_model.non_linear_model,
                )(linearized_model.inputs)
                out = tf.keras.layers.Lambda(lambda x: (
                    tf.reduce_sum(x, axis=-2)
                ))(out)
                jacobian_feature_model = tf.keras.Model(
                    Q.model.inputs,
                    out,
                    name=f'jacobian_feature_model',
                    trainable=False
                )

                return jacobian_feature_model

            self.Q_jacobian_features = create_jacobian_feature_model(self._Qs)
            self.Q_target_jacobian_features = create_jacobian_feature_model(
                self._Q_targets)

        elif self._Qs[0].model.name == 'linearized_feedforward_Q_v2':

            self.linearized_Q_targets = self._Q_targets

            def create_jacobian_feature_model(Qs):
                Q = Qs[0]
                # jacobian_model = Q.model.layers[-3]
                jacobian_model = Q.model.get_layer('jacobian_model')
                assert isinstance(
                    jacobian_model, JacobianModel), jacobian_model

                out = jacobian_model(jacobian_model.inputs)
                out = tf.keras.layers.Lambda(lambda x: (
                    tf.reduce_sum(x, axis=-2)
                ))(out)
                jacobian_feature_model = tf.keras.Model(
                    Q.model.inputs,
                    out,
                    name=f'jacobian_feature_model',
                    trainable=False
                )

                return jacobian_feature_model

            self.Q_jacobian_features = create_jacobian_feature_model(self._Qs)
            self.Q_target_jacobian_features = create_jacobian_feature_model(
                self._Q_targets)

        elif self._Qs[0].model.name == 'feedforward_Q':
            self.linearized_Q_targets = [
                type(Q)(
                    model=LinearizedModel(
                        Q.model, name=f'linearized_Q_target_{i}'),
                    observation_keys=Q.observation_keys)
                for i, Q in enumerate(self._Q_targets)
            ]

            def create_jacobian_feature_model(Qs):
                Q = Qs[0]
                out = JacobianModel(
                    Q.model, name=f'jacobian_feature_model',
                )(Q.model.inputs)
                out = tf.keras.layers.Lambda(lambda x: (
                    tf.reduce_sum(x, axis=-2)
                ))(out)
                jacobian_feature_model = tf.keras.Model(
                    Q.model.inputs,
                    out,
                    name=f'jacobian_feature_model',
                    trainable=False
                )
                return jacobian_feature_model

            self.Q_jacobian_features = create_jacobian_feature_model(self._Qs)
            self.Q_target_jacobian_features = create_jacobian_feature_model(
                self._Q_targets)
        else:
            raise NotImplementedError(self._Q_targets[0].model.name)

        if uncertainty_model_type == 'online':
            self.uncertainty_model = OnlineUncertaintyModel()
        else:
            raise NotImplementedError(self._uncertainty_model_type)

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
        b = self.Q_jacobian_features((observations, actions))
        breakpoint()
        raise NotImplementedError("TODO(hartikainen): Implement.")
        loc = self.uncertainty_model(b)[0]
        Q_targets = tf.stop_gradient(loc)

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
    def _update_estimators_and_covariance_matrix(self,
                                                 observations,
                                                 actions,
                                                 next_observations):
        b = self.Q_jacobian_features((observations, actions))
        random_actions = tf.random.uniform(
            actions.shape,
            minval=self._training_environment.action_space.low,
            maxval=self._training_environment.action_space.high)
        b_hat = self.Q_jacobian_features((observations, random_actions))
        next_actions = self._policy.actions(next_observations)
        b_not = self.Q_jacobian_features((next_observations, next_actions))
        self.uncertainty_model.online_update((b, b_hat, b_not, self._discount))
        return 0.0

    @tf.function(experimental_relax_shapes=True)
    def _update_beta(self,
                     observations,
                     actions,
                     next_observations,
                     rewards,
                     terminals):
        self._update_estimators_and_covariance_matrix(
            observations, actions, next_observations)

        dummy_input = True
        epistemic_uncertainty = self.uncertainty_model(dummy_input)
        beta_losses = epistemic_uncertainty
        self._epistemic_uncertainty.assign(epistemic_uncertainty)
        self._beta.assign(self._beta_scale * epistemic_uncertainty)
        return beta_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch, uncertainty_batch):
        beta_losses = self._update_beta(
            uncertainty_batch['observations'],
            uncertainty_batch['actions'],
            uncertainty_batch['next_observations'],
            uncertainty_batch['rewards'],
            uncertainty_batch['terminals'])
        epistemic_uncertainty = beta_losses

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
            ('beta_loss-mean', tf.reduce_mean(beta_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        if iteration == 0:
            initialize_model_batch = self._training_batch(1)
            _ = self.uncertainty_model((
                self.Q_jacobian_features((
                    initialize_model_batch['observations'],
                    initialize_model_batch['actions']))))

        steps_since_last_training = (
            self._total_timestep - self.last_training_step)
        # TODO(hartikainen): For not, assume that training update interval is 1
        assert steps_since_last_training == 1, steps_since_last_training
        self.last_training_step = self._total_timestep
        uncertainty_batch = self._pool.last_n_batch(steps_since_last_training)
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
