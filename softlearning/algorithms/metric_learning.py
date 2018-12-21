from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .rl_algorithm import RLAlgorithm
from .sac import td_target

from softlearning.environments.adapters.gym_adapter import (
    Point2DEnv,
    Point2DWallEnv,
    CustomSwimmerEnv,
    CustomAntEnv,
    CustomHumanoidEnv)

from gym.envs.mujoco.swimmer import SwimmerEnv as GymSwimmerEnv
from gym.envs.mujoco.ant import AntEnv as GymAntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as GymHalfCheetahEnv
from gym.envs.mujoco.humanoid import HumanoidEnv as GymHumanoidEnv

from softlearning.visualization import point_2d_plotter


class MetricLearningSoftActorCritic(RLAlgorithm):

    def __init__(
            self,
            env,
            policy,
            initial_exploration_policy,
            Qs,
            pool,
            plotter=None,
            tf_summaries=False,

            lr=3e-3,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=1e-2,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            plot_distances=False,
            save_full_state=False,
            **kwargs):

        super(MetricLearningSoftActorCritic, self).__init__(**kwargs)

        self._goal = getattr(env.unwrapped, 'fixed_goal', None)
        self._temporary_goal = None
        self._first_observation = None
        # self._temporary_goal_update_rule = (
        #     'farthest_estimate_from_first_observation')
        self._temporary_goal_update_rule = 'operator_query_last_step'

        self._env = env
        self._policy = policy
        self._initial_exploration_policy = initial_exploration_policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._env.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior
        self._reparameterize = reparameterize

        self._plot_distances = plot_distances
        self._save_full_state = save_full_state

        observation_shape = self._env.active_observation_shape
        action_shape = self._env.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""

        return self._train(
            self._env,
            self._policy,
            self._pool,
            initial_exploration_policy=self._initial_exploration_policy,
            *args,
            **kwargs)

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observations',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observations',
        )

        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='goals')

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        self._temporary_goal_ph = tf.placeholder(
            tf.float32,
            shape=self._env.observation_space.shape,
            name='goals',
        )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        # next_log_pis = self._policy.log_pis(
        #     [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q_target([
                self._next_observations_ph,
                self._goals_ph,
                next_actions
            ])
            for Q_target in self._Q_targets
        )

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q
        # next_value = min_next_Q - self._alpha * next_log_pis

        terminals = tf.to_float(
            tf.norm(
                self._next_observations_ph - self._goals_ph,
                ord=2,
                axis=1,
                keepdims=True)
            < 0.1)

        Q_target = td_target(
            reward=-1.0,
            discount=self._discount,
            next_value=(1.0 - terminals) * next_value
        )  # N

        return Q_target

    def _init_critic_update(self):
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._goals_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        goals = tf.tile(
            self._temporary_goal_ph[None, :],
            # self._env.unwrapped.fixed_goal[None, :],
            (tf.shape(self._observations_ph)[0], 1))
        Q_log_targets = tuple(
            Q([self._observations_ph, goals, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target()

    def _update_target(self):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                (1 - self._tau) * target + self._tau * source
                for target, source in zip(target_params, source_params)
            ])

    def _update_temporary_goal(self, training_paths):
        if self._temporary_goal_update_rule == 'closest_l2_from_goal':
            new_observations = np.concatenate([
                path['observations'] for path in training_paths], axis=0)
            new_distances = np.linalg.norm(
                new_observations - self._goal, axis=1)

            min_distance_idx = np.argmin(new_distances)
            min_distance = new_distances[min_distance_idx]

            current_distance = np.linalg.norm(
                self._temporary_goal - self._goal)
            if min_distance < current_distance:
                self._temporary_goal = new_observations[min_distance_idx]
        elif (self._temporary_goal_update_rule
              == 'farthest_l2_from_first_observation'):
            new_observations = np.concatenate([
                path['observations'] for path in training_paths], axis=0)
            new_distances = np.linalg.norm(
                new_observations - self._first_observation, axis=1)

            max_distance_idx = np.argmax(new_distances)
            max_distance = new_distances[max_distance_idx]

            current_distance = np.linalg.norm(
                self._temporary_goal - self._first_observation)
            if max_distance > current_distance:
                self._temporary_goal = new_observations[max_distance_idx]
        elif (self._temporary_goal_update_rule
              == 'farthest_estimate_from_first_observation'):

            new_observations = getattr(
                self._pool, 'observations.observation')[:self._pool.size]
            first_observations = np.tile(
                self._first_observation[None, :],
                (new_observations.shape[0], 1))
            actions = self._policy.actions_np([first_observations])
            new_distances = tuple(
                Q.predict([
                    np.tile(self._first_observation[None, :],
                            (new_observations.shape[0], 1)),
                    new_observations,
                    actions,
                ])
                for Q in self._Qs)

            new_distances = - np.min(new_distances, axis=0)
            max_distance_idx = np.argmax(new_distances)
            max_distance = new_distances[max_distance_idx]

            current_distances = tuple(
                Q.predict([
                    self._first_observation[None, :],
                    self._temporary_goal[None, :],
                    self._policy.actions_np([self._first_observation[None, :]]),
                ])
                for Q in self._Qs)
            current_distance = - np.min(current_distances)
            if max_distance > current_distance:
                self._temporary_goal = new_observations[max_distance_idx]

        elif (self._temporary_goal_update_rule == 'operator_query_last_step'):
            new_observations = np.concatenate([
                path['observations'] for path in training_paths], axis=0)
            path_last_observations = new_observations[
                -1::-self.sampler._max_path_length]
            if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
                last_observations_distances = (
                    self._env.unwrapped.get_approximate_shortest_paths(
                        np.round(path_last_observations),
                        np.tile(np.round(self._goal),
                                (path_last_observations.shape[0], 1))
                    ))

                min_distance_idx = np.argmin(last_observations_distances)
                min_distance = last_observations_distances[min_distance_idx]

                current_distance = (
                    self._env.unwrapped.get_approximate_shortest_paths(
                        np.round(self._temporary_goal[None, :]),
                        np.round(self._goal[None, :])
                    ))

                if min_distance < current_distance:
                    self._temporary_goal = path_last_observations[
                        min_distance_idx]
            elif isinstance(self._env.unwrapped,
                            (GymSwimmerEnv,
                             GymAntEnv,
                             GymHalfCheetahEnv,
                             GymHumanoidEnv)):
                velocity_indices = {
                    GymSwimmerEnv: slice(
                        self._env.unwrapped.sim.data.qpos.size - 2,
                        self._env.unwrapped.sim.data.qpos.size),
                    GymAntEnv: slice(
                        self._env.unwrapped.sim.data.qpos.size - 2,
                        self._env.unwrapped.sim.data.qpos.size),
                    GymHalfCheetahEnv: slice(
                        self._env.unwrapped.sim.data.qpos.size - 1,
                        self._env.unwrapped.sim.data.qpos.size),
                    GymHumanoidEnv: slice(
                        self._env.unwrapped.sim.data.qpos.size - 2,
                        self._env.unwrapped.sim.data.qpos.size),
                }[type(self._env.unwrapped)]
                new_velocities = new_observations[
                    :, velocity_indices]
                new_velocities = np.linalg.norm(new_velocities, ord=2, axis=1)

                max_velocity_idx = np.argmax(new_velocities)
                max_velocity = new_velocities[max_velocity_idx]

                current_velocity = np.linalg.norm(
                    self._temporary_goal[velocity_indices], ord=2)
                if max_velocity > current_velocity:
                    self._temporary_goal = new_observations[
                        max_velocity_idx]
            elif isinstance(self._env.unwrapped,
                            (CustomSwimmerEnv,
                             CustomAntEnv,
                             CustomHumanoidEnv)):
                if self._env.unwrapped._exclude_current_positions_from_observation:
                    raise NotImplementedError

                position_idx = slice(0, 2)
                last_observations_positions = path_last_observations[
                    :, position_idx]
                last_observations_distances = np.linalg.norm(
                    last_observations_positions, ord=2, axis=1)

                max_distance_idx = np.argmax(last_observations_distances)
                max_distance = last_observations_distances[max_distance_idx]

                current_distance = np.linalg.norm(
                    self._temporary_goal[position_idx], ord=2)
                if max_distance > current_distance:
                    self._temporary_goal = path_last_observations[
                        max_distance_idx]
            else:
                raise NotImplementedError

    def _epoch_after_hook(self, training_paths):
        self._previous_training_paths = training_paths
        self._update_temporary_goal(training_paths)

    def _timestep_before_hook(self, *args, **kwargs):
        if ((self._timestep % self.sampler._max_path_length)
            >= self.sampler._max_path_length * 0.8):
            self.sampler.initialize(
                self._env, self._initial_exploration_policy, self._pool)
            # self.sampler.initialize(
            #     self._env, self._env.unwrapped.optimal_policy, self._pool)
        else:
            # self.sampler.initialize(
            #     self._env, self._initial_exploration_policy, self._pool)
            self.sampler.initialize(self._env, self._policy, self._pool)
            # self.sampler.initialize(
            #     self._env, self._env.unwrapped.optimal_policy, self._pool)
            if self.sampler.policy is not self._policy:
                assert isinstance(self._env.unwrapped, Point2DEnv)

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)

        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._goals_ph: batch['goals'],
            # self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._temporary_goal is None:
            self._temporary_goal = batch['observations'][0]
            self._first_observation = batch['observations'][0]

        feed_dict.update({self._temporary_goal_ph: self._temporary_goal})

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        if self._plot_distances:
            if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
                point_2d_plotter.point_2d_plotter(
                    self, iteration, training_paths, evaluation_paths)

        return diagnostics

    @property
    def tf_saveables(self):
        return {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
            '_alpha_optimizer': self._alpha_optimizer,
        }
