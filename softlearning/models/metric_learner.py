from collections import OrderedDict

import numpy as np
import tensorflow as tf


from softlearning.algorithms.sac import td_target

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv


class MetricLearner(object):
    """Base class for metric learners.

    MetricLearner provides the base functionality for training metric learner.
    It does not specify how the metric learning estimator is trained and thus
    the training logic should be implemented in the child class.
    """

    def __init__(self,
                 env,
                 policy,
                 observation_shape,
                 action_shape,
                 distance_learning_rate=3e-4,
                 train_every_n_steps=1,
                 n_train_repeat=1,
                 distance_estimator=None,
                 distance_input_type='full'):
        self._env = env
        self._policy = policy
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._distance_learning_rate = distance_learning_rate

        self.distance_estimator = distance_estimator
        self._distance_input_type = distance_input_type

        self._train_every_n_steps = train_every_n_steps
        self._n_train_repeat = n_train_repeat

        self._session = tf.keras.backend.get_session()

        self._build()

    def _build(self, *args, **kwargs):
        self._init_placeholders()
        self._init_distance_update()

    def _init_placeholders(self):
        """Create input placeholders for the MetricLearner algorithm."""
        self.distance_pairs_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2, *self._observation_shape),
            name='distance_pairs_observations')

        self.distance_pairs_goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='distance_pairs_goals')

        self.distance_pairs_actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2, *self._action_shape),
            name='distance_pairs_actions')

        self.distance_pairs_distances_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='distance_pairs_distances')

    def _distance_estimator_inputs(self,
                                   observations1,
                                   observations2,
                                   actions):
        inputs = [observations1]

        if self.distance_estimator.condition_with_action:
            inputs.append(actions)

        if self._distance_input_type == 'full':
            inputs.append(observations2)

        elif self._distance_input_type == 'xy_coordinates':
            if isinstance(self._env.unwrapped,
                          (SwimmerEnv,
                           AntEnv,
                           HumanoidEnv,
                           HalfCheetahEnv,
                           HopperEnv,
                           Walker2dEnv)):
                if (self._env.unwrapped
                    ._exclude_current_positions_from_observation):
                    raise NotImplementedError
                inputs.append(observations2[:, :2])
            else:
                raise NotImplementedError(self._env.unwrapped)

        elif self._distance_input_type == 'xy_velocities':
            if isinstance(self._env.unwrapped,
                          (SwimmerEnv,
                           AntEnv,
                           HumanoidEnv,
                           HalfCheetahEnv,
                           HopperEnv,
                           Walker2dEnv)):
                if (self._env.unwrapped
                    ._exclude_current_positions_from_observation):
                    raise NotImplementedError

                qvel_start_idx = self._env.unwrapped.sim.data.qpos.size
                qvel_end_idx = qvel_start_idx + 2

                if self._env.unwrapped._exclude_current_positions_from_observation:
                    qvel_start_idx -= 2
                    qvel_end_idx -= 2

                xy_velocities = observations2[:, qvel_start_idx:qvel_end_idx]
                inputs.append(xy_velocities)
            else:
                raise NotImplementedError(self._env.unwrapped)

        elif self._distance_input_type == 'reward_sum':
            raise NotImplementedError(self._distance_input_type)

        else:
            raise NotImplementedError(self._distance_input_type)

        return inputs

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = {
            self.distance_pairs_observations_ph: (
                batch['distance_pairs_observations']),
            self.distance_pairs_actions_ph: (
                batch['distance_pairs_actions']),
            self.distance_pairs_distances_ph: (
                batch['distance_pairs_distances']),
            self.distance_pairs_goals_ph: (
                batch['distance_pairs_goals']),
        }

        return feed_dict

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        if iteration % self._train_every_n_steps > 0: return
        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._distance_train_ops, feed_dict)

    def _evaluate(self, observations, actions, y):
        inputs = self._distance_estimator_inputs(
            observations[:, 0], observations[:, 1], actions)
        distance_predictions = self.distance_estimator.predict(inputs)[:, 0]
        errors = distance_predictions - y
        mean_abs_error = np.mean(np.abs(errors))

        evaluation_results = {
            'distance-mean_absolute_error': mean_abs_error,
        }

        return evaluation_results

    @property
    def tf_saveables(self):
        raise NotImplementedError

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        pass


class HingeMetricLearner(MetricLearner):
    """MetricLearner."""

    def __init__(self,
                 lambda_learning_rate=1e-3,
                 lambda_estimators=None,
                 constraint_exp_multiplier=1.0,
                 step_constraint_coeff=1.0,
                 objective_type='linear',
                 zero_constraint_threshold=0.1,
                 max_distance=None,
                 **kwargs):
        self._lambda_learning_rate = lambda_learning_rate

        self.lambda_estimators = lambda_estimators

        self._constraint_exp_multiplier = constraint_exp_multiplier
        self._step_constraint_coeff = step_constraint_coeff
        self._objective_type = objective_type
        self._zero_constraint_threshold = zero_constraint_threshold

        self._max_distance = max_distance

        super(HingeMetricLearner, self).__init__(**kwargs)

    def _init_placeholders(self):
        """Create input placeholders for the HingeMetricLearner algorithm."""

        super(HingeMetricLearner, self)._init_placeholders()

        self.distance_triples_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, 3, *self._observation_shape),
            name='distance_triples_observations')

        self.distance_triples_actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, 3, *self._action_shape),
            name='distance_triples_actions')

        self.distance_objectives_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2, *self._observation_shape),
            name='distance_objectives_observations')

        self.distance_objectives_actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2, *self._action_shape),
            name='distance_objectives_actions')

    def _get_objectives(self):
        observations = tf.unstack(
            self.distance_objectives_observations_ph, 2, axis=1)
        actions = tf.unstack(
            self.distance_objectives_actions_ph, 2, axis=1)[0]

        inputs = self._distance_estimator_inputs(*observations, actions)
        distance_predictions = self.distance_estimator(inputs)[:, 0]

        if self._objective_type == 'huber':
            objectives = tf.losses.huber_loss(
                labels=tf.zeros_like(distance_predictions),
                predictions=distance_predictions)
        elif self._objective_type == 'linear':
            objectives = distance_predictions
        elif self._objective_type == 'squared':
            objectives = tf.square(distance_predictions)

        return objectives

    def _compute_constraints(self,
                             lambda_estimator,
                             inputs,
                             values,
                             target_values):
        constraints = values - target_values
        lambdas = lambda_estimator((
            inputs,
            tf.stop_gradient(values),
            tf.stop_gradient(target_values)
        ))[:, 0]

        multiplier = tf.exp(
            self._constraint_exp_multiplier * tf.stop_gradient(constraints))

        constraints = multiplier * lambdas * constraints

        return constraints, lambdas

    def _get_max_distance_constraints(self):
        observations = tf.unstack(
            self.distance_triples_observations_ph, 3, axis=1)[:2]
        actions = tf.unstack(self.distance_triples_actions_ph, 3, axis=1)[0]

        inputs = self._distance_estimator_inputs(*observations, actions)
        distance_predictions = self.distance_estimator(inputs)[:, 0]

        target_values = tf.ones_like(distance_predictions) * self._max_distance
        max_distance_constraints, _ = self._compute_constraints(
            self.lambda_estimators['max_distance'],
            inputs,
            distance_predictions,
            target_values)

        return max_distance_constraints

    def _get_step_constraints(self):
        observations = tf.unstack(
            self.distance_pairs_observations_ph, 2, axis=1)
        actions = tf.unstack(self.distance_pairs_actions_ph, 2, axis=1)[0]
        distances = self.distance_pairs_distances_ph

        inputs = self._distance_estimator_inputs(*observations, actions)
        distance_predictions = self.distance_estimator(inputs)[:, 0]

        target_values = distances
        step_constraints, _ = self._compute_constraints(
            self.lambda_estimators['step'],
            inputs,
            distance_predictions,
            target_values)

        return step_constraints

    def _get_zero_constraints(self):
        zero_observations = tf.tile(tf.reshape(
            self.distance_triples_observations_ph,
            (tf.reduce_prod(
                tf.shape(self.distance_triples_observations_ph)[0:2]),
             1,
             *self.distance_triples_observations_ph.shape[2:])
        ), (1, 2, 1))
        observations = tf.unstack(zero_observations, 2, axis=1)

        if self.distance_estimator.condition_with_action:
            raise ValueError(self._distance_input_type)

        with tf.control_dependencies([tf.assert_equal(*observations)]):
            inputs = self._distance_estimator_inputs(*observations, None)
            distance_predictions = self.distance_estimator(inputs)[:, 0]

        target_values = tf.fill(
            tf.shape(distance_predictions), self._zero_constraint_threshold)
        zero_constraints, _ = self._compute_constraints(
            self.lambda_estimators['zero'],
            inputs,
            distance_predictions,
            target_values)

        return zero_constraints

    def _get_triangle_inequality_constraints(self):
        observations = tf.unstack(
            self.distance_triples_observations_ph, 3, axis=1)
        actions = tf.unstack(
            self.distance_triples_actions_ph, 3, axis=1)

        inputs = self._distance_estimator_inputs(
            observations[0], observations[2], actions[0])
        distance_predictions = self.distance_estimator(inputs)[:, 0]

        d01, d12 = [
            self.distance_estimator(
                self._distance_estimator_inputs(
                    observations[i], observations[j], actions[i])
            )[:, 0]
            for i, j in ((0, 1), (1, 2))
        ]

        lambda_inputs = tf.concat((inputs, observations[1]), axis=-1)
        target_values = d01 + d12
        triangle_inequality_constraints, lambdas = self._compute_constraints(
             self.lambda_estimators['triangle_inequality'],
             lambda_inputs,
             distance_predictions,
             target_values)

        return triangle_inequality_constraints

    def _init_distance_update(self):
        """Create minimization operations for distance estimator."""

        objectives = self._get_objectives()
        step_constraints = self._get_step_constraints()
        zero_constraints = (
            self._get_zero_constraints()
            if not self.distance_estimator.condition_with_action
            else tf.constant(0.0))
        triangle_inequality_constraints = (
            self._get_triangle_inequality_constraints())
        max_distance_constraints = self._get_max_distance_constraints()

        self.objectives = objectives
        self.step_constraints = step_constraints
        self.zero_constraints = zero_constraints
        self.triangle_inequality_constraints = triangle_inequality_constraints
        self.max_distance_constraints = max_distance_constraints

        objective = tf.reduce_mean(objectives)
        step_constraint = tf.reduce_mean(step_constraints)
        zero_constraint = tf.reduce_mean(zero_constraints)
        triangle_inequality_constraint = tf.reduce_mean(
            triangle_inequality_constraints)
        max_distance_constraint = tf.reduce_mean(max_distance_constraints)

        lagrangian_loss = self.lagrangian_loss = (
            - objective
            + step_constraint * self._step_constraint_coeff
            + zero_constraint
            + triangle_inequality_constraint
            + max_distance_constraint)

        lagrangian_optimizer = self._lagrangian_optimizer = (
            tf.train.AdamOptimizer(learning_rate=self._distance_learning_rate))

        lagrangian_grads_and_vars = lagrangian_optimizer.compute_gradients(
            loss=lagrangian_loss,
            var_list=self.distance_estimator.trainable_variables)

        lambda_losses = self.lambda_losses = {
            'step': step_constraint,
            'zero': zero_constraint,
            'max_distance': max_distance_constraint,
            'triangle_inequality': triangle_inequality_constraint,
        }

        assert set(self.lambda_estimators.keys()) == set(lambda_losses.keys())

        lambda_optimizer = self._lambda_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._lambda_learning_rate)
        lambda_grads_and_vars = {}

        for lambda_name in lambda_losses.keys():
            lambda_loss = -lambda_losses[lambda_name]

            grads_and_vars = lambda_optimizer.compute_gradients(
                loss=lambda_loss,
                var_list=self.lambda_estimators[
                    lambda_name].trainable_variables)

            lambda_grads_and_vars[lambda_name] = grads_and_vars

        all_gradients = [
            gradient for gradient, _ in
            lagrangian_grads_and_vars
            + sum(lambda_grads_and_vars.values(), [])
            if gradient is not None
        ]

        with tf.control_dependencies(all_gradients):
            lagrangian_train_op = lagrangian_optimizer.apply_gradients(
                lagrangian_grads_and_vars)

            lambda_train_ops = {
                # lambda_key: lambda_optimizer[lambda_key].apply_gradients(
                lambda_key: lambda_optimizer.apply_gradients(
                    lambda_grads_and_vars[lambda_key])
                for lambda_key in lambda_losses.keys()
            }

        self._distance_train_ops = (
            lagrangian_train_op, *lambda_train_ops.values())

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(HingeMetricLearner, self)._get_feed_dict(
            iteration, batch)

        feed_dict.update({
            self.distance_triples_observations_ph: (
                batch['distance_triples_observations']),
            self.distance_triples_actions_ph: (
                batch['distance_triples_actions']),

            self.distance_objectives_observations_ph: (
                batch['distance_objectives_observations']),
            self.distance_objectives_actions_ph: (
                batch['distance_objectives_actions']),
        })

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        feed_dict = self._get_feed_dict(iteration, batch)

        (objectives,
         step_constraints,
         zero_constraints,
         triangle_inequality_constraints,
         max_distance_constraints,
         lagrangian_loss,
         lambda_losses) = self._session.run((
             self.objectives,
             self.step_constraints,
             self.zero_constraints,
             self.triangle_inequality_constraints,
             self.max_distance_constraints,
             self.lagrangian_loss,
             self.lambda_losses,
         ), feed_dict)

        result = OrderedDict((
            ('lagrangian_loss-mean', np.mean(lagrangian_loss)),

            *[
                (f'lambda_losses[{lambda_name}]', np.mean(lambda_loss))
                for lambda_name, lambda_loss in lambda_losses.items()
            ],

            ('objectives-mean', np.mean(objectives)),
            ('objectives-std', np.std(objectives)),

            ('step_constraints-mean', np.mean(step_constraints)),
            ('step_constraints-std', np.std(step_constraints)),

            ('zero_constraints-mean', np.mean(zero_constraints)),
            ('zero_constraints-std', np.std(zero_constraints)),

            ('triangle_inequality_constraints-mean', np.mean(
                triangle_inequality_constraints)),
            ('triangle_inequality_constraints-std', np.std(
                triangle_inequality_constraints)),

            ('max_distance_constraints-mean', np.mean(
                max_distance_constraints)),
            ('max_distance_constraints-std', np.std(
                max_distance_constraints)),
        ))

        return result

    @property
    def tf_saveables(self):
        return {
            '_lagrangian_optimizer': self._lagrangian_optimizer,
            '_lambda_optimizer': self._lambda_optimizer,
        }


class SupervisedMetricLearner(MetricLearner):
    def _init_distance_update(self):
        """Create minimization operations for distance estimator."""
        observations = tf.unstack(
            self.distance_pairs_observations_ph, 2, axis=1)
        actions = tf.unstack(self.distance_pairs_actions_ph, 2, axis=1)[0]
        inputs = self._distance_estimator_inputs(*observations, actions)
        distance_predictions = self.distance_estimator(inputs)

        distance_loss = self.distance_loss = tf.losses.mean_squared_error(
            labels=self.distance_pairs_distances_ph,
            predictions=distance_predictions,
            weights=0.5)

        distance_optimizer = self._distance_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._distance_learning_rate)
        distance_grads_and_vars = distance_optimizer.compute_gradients(
            loss=distance_loss,
            var_list=self.distance_estimator.trainable_variables)

        distance_train_op = distance_optimizer.apply_gradients(
            distance_grads_and_vars)

        self._distance_train_ops = (distance_train_op, )

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        feed_dict = self._get_feed_dict(iteration, batch)
        distance_loss = self._session.run(self.distance_loss, feed_dict)
        return OrderedDict((
            ('distance_loss-mean', np.mean(distance_loss)),
        ))

    @property
    def tf_saveables(self):
        return {
            '_distance_optimizer': self._distance_optimizer,
            'distance_estimator': self.distance_estimator
        }


class TemporalDifferenceMetricLearner(MetricLearner):
    def __init__(self,
                 distance_estimator,
                 *args,
                 ground_truth_terminals=False,
                 **kwargs):
        self._ground_truth_terminals = ground_truth_terminals
        self.distance_estimator_target = tf.keras.models.clone_model(
            distance_estimator)
        result = super(TemporalDifferenceMetricLearner, self).__init__(
            *args,
            distance_estimator=distance_estimator,
            **kwargs)
        assert self.distance_estimator.condition_with_action
        return result

    def _init_placeholders(self):
        """Create input placeholders for the MetricLearner algorithm."""
        super(TemporalDifferenceMetricLearner, self)._init_placeholders()

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='goals',
        )

    def _init_distance_update(self):
        """Create minimization operations for distance estimator."""

        observations = self._observations_ph
        actions = self._actions_ph
        next_observations = self._next_observations_ph
        goals = self._goals_ph

        inputs_1 = self._distance_estimator_inputs(
            observations, goals, actions)
        distance_predictions_1 = self.distance_estimator(inputs_1)

        next_actions = self._policy.actions([next_observations, goals])
        inputs_2 = self._distance_estimator_inputs(
            next_observations, goals, next_actions)
        distance_predictions_2 = self.distance_estimator_target(inputs_2)

        goal_successes = tf.cast(tf.reduce_all(tf.equal(
            self._next_observations_ph, self._goals_ph
        ), axis=1, keepdims=True), tf.float32)

        assert (goal_successes.shape.as_list()
                == distance_predictions_2.shape.as_list())
        next_values = (1 - goal_successes) * distance_predictions_2

        distance_targets = td_target(
            reward=1.0,
            discount=1.0,
            next_value=next_values)

        # raise ValueError(
        #     "TODO(hartikainen): This should work as well as the above."
        #     " For some reason it doesn't. Probably max_pair_distance.")

        # observations = tf.unstack(
        #     self.distance_pairs_observations_ph, 2, axis=1)
        # goals = self.distance_pairs_goals_ph

        # # actions_1, actions_2 = tf.unstack(
        # #     self.distance_pairs_actions_ph, 2, axis=1)
        # actions_1 = tf.unstack(self.distance_pairs_actions_ph, 2, axis=1)[0]
        # inputs_1 = self._distance_estimator_inputs(
        #     observations[0], goals, actions_1)
        # distance_predictions_1 = self.distance_estimator(inputs_1)

        # actions_2 = self._policy.actions([observations[1], goals])
        # inputs_2 = self._distance_estimator_inputs(
        #     observations[1], goals, actions_2)
        # distance_predictions_2 = self.distance_estimator_target(inputs_2)

        # # tf.to_float(tf.reduce_all(tf.equal(observations[1], goals), axis=1, keepdims=True))
        # terminals = tf.to_float(tf.norm(
        #     observations[1] - goals,
        #     ord=2,
        #     axis=1,
        #     keepdims=True,
        # ) < 0.1)

        # next_values = (
        #     (1 - terminals) * distance_predictions_2
        #     if self._ground_truth_terminals
        #     else distance_predictions_2)

        # distance_targets = td_target(
        #     reward=self.distance_pairs_distances_ph,
        #     discount=1.0 ** self.distance_pairs_distances_ph,
        #     next_value=next_values)

        distance_loss = self.distance_loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(distance_targets),
            predictions=distance_predictions_1,
            weights=0.5)

        distance_optimizer = self._distance_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._distance_learning_rate)
        distance_grads_and_vars = distance_optimizer.compute_gradients(
            loss=distance_loss,
            var_list=self.distance_estimator.trainable_variables)

        distance_train_op = distance_optimizer.apply_gradients(
            distance_grads_and_vars)

        self._distance_train_ops = (distance_train_op, )

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = (
            super(TemporalDifferenceMetricLearner, self)._get_feed_dict(
                iteration, batch))

        feed_dict.update({
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._goals_ph: batch['goals'],
        })

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        feed_dict = self._get_feed_dict(iteration, batch)
        distance_loss = self._session.run(self.distance_loss, feed_dict)
        return OrderedDict((
            ('distance_loss-mean', np.mean(distance_loss)),
        ))

    @property
    def tf_saveables(self):
        return {
            '_distance_optimizer': self._distance_optimizer,
            'distance_estimator': self.distance_estimator
        }

    def _update_target(self, tau=None):
        source_params = self.distance_estimator.get_weights()
        target_params = self.distance_estimator_target.get_weights()
        self.distance_estimator_target.set_weights([
            tau * source + (1.0 - tau) * target
            for source, target in zip(source_params, target_params)
        ])
