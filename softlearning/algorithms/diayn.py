from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .sac import SAC, td_target


class DIAYN(SAC):
    """Diversity Is All You Need (DIAYN)

    References
    ----------
    [1]
    """

    def __init__(self,
                 discriminator,
                 discriminator_lr=3e-4,
                 *args,
                 add_prior_to_reward=True,
                 **kwargs):
        """
        Args:
            discriminator: TODO(hartikainen)
        """
        self._discriminator = discriminator
        self._discriminator_lr = discriminator_lr
        self._add_prior_to_reward = add_prior_to_reward
        return super(DIAYN, self).__init__(*args, **kwargs)

    def _build(self):
        super(SAC, self)._build()

        self._init_actor_update()
        self._init_critic_update()
        self._init_discriminator_update()
        self._init_diagnostics_ops()

    def discriminator_inputs(self, observations, actions):
        return (observations, )

    def _get_Q_target(self):
        policy_inputs = flatten_input_structure({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        })
        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        discriminator_inputs = self.discriminator_inputs(
            self._placeholders['observations'],
            self._placeholders['actions'])
        discriminator_logits = self._discriminator(discriminator_inputs)

        diayn_reward = -1.0 * tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self._placeholders['observations']['active_skill'],
            logits=discriminator_logits
        )[..., None]
        diayn_reward = tf.check_numerics(diayn_reward, "diayn_reward 1")

        num_skills = self._training_environment._num_skills
        p_z = tf.reduce_sum(
            np.full((1, num_skills), 1.0/num_skills)
            * self._placeholders['observations']['active_skill'],
            axis=1,
            keepdims=True)

        EPS = 1e-8
        log_p_z = tf.log(p_z + EPS)

        if self._add_prior_to_reward:
            diayn_reward -= log_p_z
            diayn_reward = tf.check_numerics(diayn_reward, "diayn_reward 2")

        Q_target = td_target(
            reward=self._reward_scale * diayn_reward,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': self._placeholders['actions']})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=np.log(self._entropy_scale, dtype=np.float32))
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        policy_kl_losses = (
            alpha * log_pis
            - min_Q_log_target
            - policy_prior_log_probs)

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_discriminator_update(self):
        discriminator_inputs = self.discriminator_inputs(
            self._placeholders['observations'],
            self._placeholders['actions'])
        discriminator_logits = self._discriminator(discriminator_inputs)
        self._discriminator_losses = (
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self._placeholders['observations']['active_skill'],
                logits=discriminator_logits))

        self._discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._discriminator_lr,
            name="discriminator_optimizer")

        discriminator_loss = tf.reduce_mean(self._discriminator_losses)
        discriminator_train_op = self._discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=self._discriminator.trainable_variables)

        self._training_ops.update({
            'discriminator_train_op': discriminator_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('discriminator_loss', self._discriminator_losses),
            ('alpha', self._alpha)
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        # TODO(hartikainen): log for diayn:
        # - p(z)
        # - discriminator reward
        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(flatten_input_structure({
                name: batch['observations'][name]
                for name in self._policy.observation_keys
            })).items()
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        # TODO(hartikainen): discriminator
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
            '_discriminator': self._discriminator,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
