from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class DDPG(RLAlgorithm):
    """Deep Deterministic Policy Gradient (DDPG)

    References
    ----------
    [1]
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            Q_target_action_noise=2e-1,

            policy_train_every_n_steps=1,

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

        super(DDPG, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._policy_train_every_n_steps = policy_train_every_n_steps
        self._Q_target_action_noise = Q_target_action_noise

        self._save_full_state = save_full_state

        self._build()

    def _build(self):
        super(DDPG, self)._build()

        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _get_Q_target(self):
        policy_inputs = flatten_input_structure({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        })

        next_actions = self._policy.actions(policy_inputs)
        action_space = self._training_environment.action_space
        assert not self._policy._squash, "DDPG policy should no be squashed."
        noise_mean = 0.0
        noise_stddev = self._Q_target_action_noise * (
            (action_space.high - action_space.low) / 2)
        noise = tf.random.normal(
            tf.shape(next_actions), mean=noise_mean, stddev=noise_stddev)
        noise_clip = 2.5 * noise_stddev
        noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
        next_actions = tf.clip_by_value(
            next_actions + noise, action_space.low, action_space.high)
        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._placeholders['rewards'],
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
        with self._policy.set_deterministic(True):
            actions = self._policy.actions(policy_inputs)

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        policy_kl_losses = - min_Q_log_target

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

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
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

    def _do_training(self, iteration, batch, ops):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        if timestep % self._policy_train_every_n_steps > 0:
            # don't train policy
            train_ops = type(self._training_ops)((
                (key, op)
                for key, op in self._training_ops.items()
                if 'policy' not in key
            ))
        else:
            train_ops = self._training_ops

        for i in range(self._n_train_repeat):
            self._do_training(
                iteration=timestep,
                batch=self._training_batch(),
                ops=train_ops)

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

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
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
        }

        return saveables
