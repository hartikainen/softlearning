import os
import pickle

import numpy as np
import tensorflow as tf

from gym.envs.mujoco import ReacherEnv

from rllab.core.serializable import Serializable

from .sac import SAC, td_target

from softlearning.environments.adapters.rllab_adapter import (
    SwimmerEnv as RllabSwimmerEnv)
from softlearning.environments.adapters.gym_adapter import (
    FixedGoalReacherEnv)
from softlearning.visualization import fixed_goal_reacher_plotter
from softlearning.misc.nn import MLPFunction
from softlearning.misc import tf_utils


class GoalEstimator(MLPFunction):
    def __init__(self,
                 observation_shape,
                 hidden_layer_sizes,
                 output_nonlinearity=None,
                 name='goal_estimator'):
        Serializable.quick_init(self, locals())

        assert len(observation_shape) == 1, observation_shape
        self._Do = observation_shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32, shape=(None, self._Do), name='observations')

        layer_sizes = (*hidden_layer_sizes, self._Do)
        super(GoalEstimator, self).__init__(
            (self._observations_ph, ),
            name=name,
            layer_sizes=layer_sizes,
            output_nonlinearity=output_nonlinearity)


class GoalEstimator2(object):
    def __init__(self,
                 observation_shape,
                 name='goal_estimator',
                 *args,
                 **kwargs):
        assert len(observation_shape) == 1, observation_shape
        self._Do = observation_shape[0]

        self._goal_estimate = tf.get_variable(
            name,
            shape=observation_shape,
            initializer=tf.zeros_initializer())

        # # Reacher
        # self._goal = tf.constant((
        #     0.77041958, -0.03223022,  0.63753719,  0.99948047,  0.00204777,
        #     0.14971093,  0.00687177, -0.00652871,  0.00217014, -0.00351536,
        #     0.0))

        # Swimmer random rollout last
        self._output = self._goal_estimate

    def output_for(self, *inputs, reuse=False):
        return tf.tile(
            self._goal_estimate[None], (tf.shape(inputs[0])[0], 1))

    def eval(self, *inputs):
        goal_estimate = tf_utils.get_default_session().run(self._goal_estimate)
        return np.tile(goal_estimate[None], (inputs[0].shape[0], 1))

    def get_params_internal(self, *args, **kwargs):
        return self._goal_estimate


class GoalLearner(SAC):
    """GoalLearner."""

    def __init__(self,
                 env,
                 preference_vf,
                 preference_rf,
                 preference_pool,
                 segment_pool,
                 operator_query_scheduler,
                 optimal_label_scheduler,
                 *args,
                 preference_query_lag=1,
                 goal_learning_rate=3e-4,
                 preference_vf_loss_type='cross_entropy',
                 optimal_policy=None,
                 n_optimal_paths=0,
                 add_optimal_path_to_base_pool=False,
                 preference_q_loss_weight=None,
                 q_target_type='VICE-any-query',
                 zero_constraint_threshold=0.1,
                 goal_estimator_kwargs=None,
                 **kwargs):
        Serializable.quick_init(self, locals())

        assert preference_q_loss_weight is not None
        self._preference_vf = preference_vf
        self._preference_rf = preference_rf
        self._preference_pool = preference_pool
        self._segment_pool = segment_pool
        self._preference_q_loss_weight = preference_q_loss_weight
        self._preference_vf_loss_type = preference_vf_loss_type
        self.optimal_policy = optimal_policy
        self.n_optimal_paths = n_optimal_paths
        self.add_optimal_path_to_base_pool = add_optimal_path_to_base_pool
        self._operator_query_scheduler = operator_query_scheduler
        self._optimal_label_scheduler = optimal_label_scheduler
        self._q_target_type = q_target_type
        self._preference_query_lag = preference_query_lag
        self._zero_constraint_threshold = zero_constraint_threshold

        self._goal_learning_rate = goal_learning_rate

        assert len(env.observation_space.shape) == 1, (
            env.observation_space.shape)

        goal_estimator_kwargs = goal_estimator_kwargs or {}
        self._goal_estimator_kwargs = goal_estimator_kwargs
        self._goal_estimator = GoalEstimator(
            observation_shape=(env.observation_space.shape[0], ),
            **goal_estimator_kwargs)

        super(GoalLearner, self).__init__(*args, env=env, **kwargs)

    def _build(self, *args, **kwargs):
        super(GoalLearner, self)._build(*args, **kwargs)
        self._init_goal_update()

    def goal_estimates(self, observations):
        # tf.tile(self._goal[None], (tf.shape(observations)[0], 1))
        # return tf.tile(self._goal[None], (tf.shape(observations)[0], 1))
        return self._goal_estimator.output_for(observations, reuse=True)

    def goal_estimate(self, observation):
        return self.goal_estimates(observation[None])[0]

    def _init_placeholders(self):
        """Create input placeholders for the GoalLearner algorithm."""

        super(GoalLearner, self)._init_placeholders()

        preference_fields = self._preference_pool.fields

        self._preference_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *preference_fields['observations']['shape']),
            name='preference_observations')

        self._preference_preferences_ph = tf.placeholder(
            tf.int32,
            shape=(None, *preference_fields['preferences']['shape']),
            name='preference_preferences')

    def _initial_exploration_hook(self, initial_exploration_policy):
        paths = [
            {
                'observations': np.array([
                    (5, 2),
                    (4, 1),
                    (3, 0),
                    (3, -1),
                    (3, -2),
                    (3, -3),
                    (2, -3),
                    (1, -3),
                    (1, -2),
                    (1, -1),
                    (0, 0),
                ]),
                'actions': np.array([
                    (-1, -1),
                    (-1, -1),
                    (0, -1),
                    (0, -1),
                    (0, -1),
                    (-1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, 1),
                    (-1, 1),
                ]),
                'rewards': np.array([0.0] * 10),
                'terminals': np.array([0] * 10),
                'agent_infos': [{}] * 10,
                'env_infos': [{}] * 10,
            },
            {
                'observations': np.array([
                    (-5, 2),
                    (-4, 1),
                    (-3, 0),
                    (-3, -1),
                    (-3, -2),
                    (-3, -3),
                    (-2, -3),
                    (-1, -3),
                    (-1, -2),
                    (-1, -1),
                    (0, 0),
                ]),
                'actions': np.array([
                    (1, -1),
                    (1, -1),
                    (0, -1),
                    (0, -1),
                    (0, -1),
                    (1, 0),
                    (1, 0),
                    (0, 1),
                    (0, 1),
                    (1, 1),
                ]),
                'rewards': np.array([0.0] * 10),
                'terminals': np.array([0] * 10),
                'agent_infos': [{}] * 10,
                'env_infos': [{}] * 10,
            },
        ]

        for path in paths:
            path['next_observations'] = path['observations'][1:]
            path['observations'] = path['observations'][:-1]
            self._pool.add_samples({
                k: v for k, v in path.items()
                if k in self._pool.fields_attrs
            })

        super(GoalLearner, self)._initial_exploration_hook(
            initial_exploration_policy)

    def _get_q_target(self):
        with tf.variable_scope('target'):
            vf_next_target = self._vf.output_for(
                self._next_observations_ph, reuse=False)
            self._vf_target_params = self._vf.get_params_internal()

        # goals = self._observations_ph[:, 2:4]
        # goal_observations = tf.tile(goals, (1, 2))
        # goal_position = tf.constant(
        #     self._env.unwrapped._target_goal, dtype=tf.float32)
        # goal_observations = tf.tile(
        #     goal_position[None], (tf.shape(self._observations_ph)[0], 1))
        goal_observations = self.goal_estimates(self._observations_ph)
        distances = self.distance_estimator(tf.concat((
            self._observations_ph, goal_observations,
        ), axis=-1))[:, 0]
        return td_target(
            reward=-distances,
            discount=self._discount,
            next_value=vf_next_target)

    def _init_goal_update(self):
        """Create minimization operations for goal estimator."""

        observations = self._preference_observations_ph
        preferences = self._preference_preferences_ph

        goal_distance_estimates = tf.map_fn(
            lambda observation: self.distance_estimator(tf.concat(
                (observation, self.goal_estimates(observation)),
                axis=-1))[:, 0],
            tf.transpose(observations, (1, 0, 2)))
        logits = tf.transpose(-goal_distance_estimates, (1, 0))

        self.goal_distance_estimates = goal_distance_estimates
        self.goal_logits = logits

        if self._preference_vf_loss_type == 'hinge':
            loss = self.goal_loss = tf.losses.hinge_loss(
                logits=(logits - tf.reduce_mean(logits, axis=1, keepdims=True)),
                labels=tf.one_hot(preferences, 2))
        elif self._preference_vf_loss_type == 'softmax':
            loss = self.goal_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=preferences))

        goal_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._goal_learning_rate)
        gradients = self.goal_gradients = goal_optimizer.compute_gradients(
            loss, var_list=self._goal_estimator.get_params_internal())
        goal_train_op = goal_optimizer.apply_gradients(gradients)

        self._goal_train_ops = (goal_train_op, )

    def _epoch_after_hook(self, epoch, paths):
        """Method called at the end of each epoch."""

        self._segment_pool.add_paths(paths)

        if epoch % self._preference_query_lag > 0:
            return

        self._previous_training_paths = paths

        expected_num_operator_queries = (
            self._operator_query_scheduler.num_labels(time_step=epoch))
        expected_num_optimal_preferences = (
            self._optimal_label_scheduler.num_labels(time_step=epoch))

        queries_available = (
            expected_num_operator_queries
            - self._segment_pool.queries_used)
        optimal_preferences_available = (
            expected_num_optimal_preferences
            - self._segment_pool.optimal_preferences_used)

        observation_pairs, action_pairs, preferences = (
            self._segment_pool.generate_preference_pairs(
                queries_available, optimal_preferences_available))

        preference_pool_samples = {
            'observations': observation_pairs,
            'actions': action_pairs,
            'preferences': preferences
        }

        self._preference_pool.add_samples(preference_pool_samples)

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        batch = self.sampler.random_batch(batch_size)
        # last_n_preferences = self._preference_pool.size
        last_n_preferences = 5
        preference_batch = self._preference_pool.batch_by_indices(
            np.arange(self._preference_pool.size-last_n_preferences,
                      self._preference_pool.size))

        batch.update({
            'preference_' + k: v for k, v in preference_batch.items()
        })

        return batch

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = super(GoalLearner, self)._get_feed_dict(iteration, batch)

        feed_dict.update({
            self._preference_observations_ph: batch['preference_observations'],
            self._preference_preferences_ph: batch['preference_preferences'],
        })

        return feed_dict

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        super(GoalLearner, self)._do_training(iteration, batch)

        for i in range(1):
            batch = self._training_batch()
            feed_dict = self._get_feed_dict(iteration, batch)
            self._sess.run(self._goal_train_ops, feed_dict)

    def _epoch_before_hook(self, epoch):
        """Method called at the beginning of each epoch."""
        pass

    def _evaluate(self, policy, evaluation_env, epoch):
        return super(GoalLearner, self)._evaluate(
            policy, evaluation_env, epoch)

    def get_diagnostics(self, iteration, batch, paths, *args, **kwargs):
        diagnostics = super(GoalLearner, self).get_diagnostics(
            iteration, batch, paths, *args, **kwargs)

        feed_dict = self._get_feed_dict(iteration, batch)

        goal_loss = self._sess.run(self.goal_loss, feed_dict)

        diagnostics = diagnostics.update({
            'preference-pool-size', self._preference_pool.size,
            'segment-pool-size', self._segment_pool.size,
            'segment-pool-queries-used', self._segment_pool.queries_used,
            'goal_loss-mean', np.mean(goal_loss),
            'goal_loss-std', np.std(goal_loss),
        })

        if hasattr(self._env.unwrapped, '_target_goal'):
            estimated_goals = self._goal_estimator.eval(
                batch['observations'])
            true_goal = self._env.unwrapped._target_goal
            goal_errors = np.linalg.norm(
                estimated_goals - true_goal, ord=2, axis=1)

            diagnostics.update({
                'goal_errors-mean': np.mean(goal_errors),
                'goal_errors-std': np.std(goal_errors),
            })

        try:
            if isinstance(self._env.unwrapped, (ReacherEnv, FixedGoalReacherEnv)):
                # fixed_goal_reacher_plotter(self, iteration, paths)

                true_targets = batch['observations'][:, -7:-5]
                estimated_targets = self._goal_estimator.eval(
                    batch['observations'])[:, -7:-5]

                target_errors = np.linalg.norm(
                    estimated_targets - true_targets, ord=2, axis=1)

                diagnostics.update({
                    'target_error-mean', np.mean(target_errors),
                    'target_error-std', np.std(target_errors),
                })

            elif isinstance(self._env.unwrapped, RllabSwimmerEnv):
                estimated_targets = self._goal_estimator.eval(
                    batch['observations'])
                estimated_target = estimated_targets[0]
                for i, value in enumerate(estimated_target):
                    diagnostics[f'estimated_target[{i}]'] = value

        except TypeError as e:
            print(f"{self.__name__}.get_diagnostics failed: ", e)

        return diagnostics
