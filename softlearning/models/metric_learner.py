from collections import OrderedDict

import numpy as np
import tensorflow as tf
from flatten_dict import flatten

from softlearning.algorithms.sac import td_target
from softlearning.models.utils import flatten_input_structure
from softlearning.environments.gym.mujoco.utils import (
    LOCOMOTION_ENVS, POSITION_SLICES)


class MetricLearner(object):
    """Base class for metric learners.

    MetricLearner provides the base functionality for training metric learner.
    It does not specify how the metric learning estimator is trained and thus
    the training logic should be implemented in the child class.
    """

    def __init__(self,
                 env,
                 policy,
                 pool,
                 observation_shape,
                 action_shape,
                 distance_learning_rate=3e-4,
                 train_every_n_steps=1,
                 max_train_repeat_per_timestep=1,
                 n_train_repeat=1,
                 distance_estimator=None):
        self._env = env
        self._policy = policy
        self._pool = pool
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._distance_learning_rate = distance_learning_rate

        self.distance_estimator = distance_estimator

        self._train_every_n_steps = train_every_n_steps
        self._n_train_repeat = n_train_repeat
        self._max_train_repeat_per_timestep = max_train_repeat_per_timestep

        self._session = tf.keras.backend.get_session()
        self._num_train_steps = 0

        self._build()

    def _build(self, *args, **kwargs):
        self._init_placeholders()
        self._init_distance_update()

    def _init_placeholders(self):
        """Create input placeholders for the MetricLearner algorithm."""
        self._placeholders = {
            'observations1': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f'observations1/{name}')
                for name, observation_space
                in self._env.observation_space.spaces.items()
            },
            'next_observations1': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f'next_observations1/{name}')
                for name, observation_space
                in self._env.observation_space.spaces.items()
            },
            'observations2': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f'observations2/{name}')
                for name, observation_space
                in self._env.observation_space.spaces.items()
            },
            'actions1': tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(None, *self._env.action_space.shape),
                name='actions',
            ),
            'distances': tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='distances'
            ),
        }

    def _distance_estimator_inputs(self,
                                   observations1,
                                   observations2,
                                   actions):
        inputs1 = {
            name: observations1[name]
            for name in self.distance_estimator.observation_keys
        }
        target_input_type = self.distance_estimator.target_input_type

        if target_input_type == 'full':
            inputs2 = {
                name: observations2[name]
                for name in self.distance_estimator.observation_keys
            }

        elif target_input_type == 'xy_coordinates':
            # This only works for gym locomotion environments
            environment = self._env.unwrapped
            assert set(observations2.keys()) == {'observations'}, observations2
            assert isinstance(environment, LOCOMOTION_ENVS)
            assert not environment._exclude_current_positions_from_observation
            position_slice = POSITION_SLICES[type(environment)]

            positions = observations2['observations'][:, position_slice]
            inputs2 = {
                'positions': positions
            }

        elif target_input_type == 'xy_velocities':
            raise NotImplementedError("TODO(hartikainen)")

        elif target_input_type == 'reward_sum':
            raise NotImplementedError(target_input_type)

        else:
            raise NotImplementedError(target_input_type)

        inputs = {'observations1': inputs1, 'observations2': inputs2}

        if self.distance_estimator.condition_with_action:
            assert actions is not None
            inputs['actions'] = actions

        inputs_flat = flatten_input_structure(inputs)
        return inputs_flat

    def _training_batch(self, batch_size=256):
        batch = self._pool.random_batch(batch_size)
        return batch

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    def _epoch_before_hook(self, *args, **kwargs):
        self._train_steps_this_epoch = 0

    def _do_training_repeats(self, timestep, total_samples):
        trained_enough = (
            self._num_train_steps
            > self._max_train_repeat_per_timestep * total_samples)

        if trained_enough:
            return

        for i in range(self._n_train_repeat):
            self._do_training(
                iteration=timestep,
                batch=self._training_batch())

        self._num_train_steps += self._n_train_repeat

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._distance_train_ops, feed_dict)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        return feed_dict

    def _evaluate(self, observations, actions, y):
        raise NotImplementedError("TODO(hartikainen)")
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
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        diagnostics = OrderedDict((
            ('num-train-steps', self._num_train_steps),
        ))

        return diagnostics


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
            raise ValueError(self.distance_estimator.condition_with_action)

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
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        batch = self._evaluation_batch()
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
        distances = tf.cast(self._placeholders['distances'], tf.float32)
        inputs = self._distance_estimator_inputs(
            self._placeholders['observations1'],
            self._placeholders['observations2'],
            self._placeholders['actions1'])
        distance_predictions = self.distance_estimator(inputs)

        distance_loss = tf.losses.mean_squared_error(
            labels=distances,
            predictions=distance_predictions,
            weights=0.5)

        self.distance_mean_squared_error = distance_loss
        self.distance_absolute_error = tf.losses.absolute_difference(
            labels=distances,
            predictions=distance_predictions)

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
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        batch = self._evaluation_batch()
        feed_dict = self._get_feed_dict(iteration, batch)
        (distance_mean_squared_error,
         distance_absolute_error) = self._session.run((
             self.distance_mean_squared_error,
             self.distance_absolute_error
         ), feed_dict)
        return OrderedDict((
            ('distance_mean_squared_error-mean',
             np.mean(distance_mean_squared_error)),
            ('distance_absolute_error-mean',
             np.mean(distance_absolute_error)),
        ))

    @property
    def tf_saveables(self):
        return {
            '_distance_optimizer': self._distance_optimizer,
            'distance_estimator': self.distance_estimator
        }


class DistributionalSupervisedMetricLearner(SupervisedMetricLearner):
    def _init_distance_update(self):
        """Create minimization operations for distance estimator."""
        distances = tf.cast(self._placeholders['distances'], tf.float32)
        inputs = self._distance_estimator_inputs(
            self._placeholders['observations1'],
            self._placeholders['observations2'],
            self._placeholders['actions1'])

        distance_predictions, distance_logits = (
            self.distance_estimator.compute_all_outputs(inputs)
        )

        bin_size = float(self.distance_estimator.max_distance
                         / (self.distance_estimator.n_bins - 1))

        labels = tf.cast(
            tf.clip_by_value(
                tf.round(distances / bin_size),
                0.0, float(self.distance_estimator.n_bins - 1)
            ),
            tf.int64
        )

        distance_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=distance_logits
        )

        self.distance_cross_entropy_loss = distance_loss
        self.distance_absolute_error = tf.losses.absolute_difference(
            labels=distances,
            predictions=distance_predictions)

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
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        batch = self._evaluation_batch()
        feed_dict = self._get_feed_dict(iteration, batch)
        (distance_cross_entropy_loss,
         distance_absolute_error) = self._session.run((
             self.distance_cross_entropy_loss,
             self.distance_absolute_error
         ), feed_dict)
        return OrderedDict((
            ('distance_cross_entropy_loss-mean',
             np.mean(distance_cross_entropy_loss)),
            ('distance_absolute_error-mean',
             np.mean(distance_absolute_error)),
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
        if ground_truth_terminals:
            raise NotImplementedError("TODO(hartikainen)")
        self._ground_truth_terminals = ground_truth_terminals
        self.distance_estimator_target = tf.keras.models.clone_model(
            distance_estimator)
        result = super(TemporalDifferenceMetricLearner, self).__init__(
            *args,
            distance_estimator=distance_estimator,
            **kwargs)
        assert self.distance_estimator.condition_with_action
        return result

    def _init_distance_update(self):
        """Create minimization operations for distance estimator."""

        observations1_ph = self._placeholders['observations1']
        actions1_ph = self._placeholders['actions1']
        next_observations1_ph = self._placeholders['next_observations1']
        observations2_ph = self._placeholders['observations2']

        inputs_1 = self._distance_estimator_inputs(
            observations1_ph, observations2_ph, actions1_ph)
        distance_predictions_1 = self.distance_estimator(inputs_1)

        next_actions = self._policy.actions(flatten_input_structure({
            'observations': {
                name: values
                for name, values in next_observations1_ph.items()
                if name in self._policy.observation_keys
            },
            'goals': {
                name: values
                for name, values in observations2_ph.items()
                if name in self._policy.goal_keys
            },
        }))
        inputs_2 = self._distance_estimator_inputs(
            next_observations1_ph, observations2_ph, next_actions)
        distance_predictions_2 = self.distance_estimator_target(inputs_2)

        goal_successes = tf.cast(tf.reduce_all(tf.equal(
            tf.concat(flatten_input_structure(next_observations1_ph), axis=-1),
            tf.concat(flatten_input_structure(observations2_ph), axis=-1),
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

    def get_diagnostics(self,
                        iteration,
                        training_paths,
                        evaluation_paths,
                        *args,
                        **kwargs):
        batch = self._evaluation_batch()
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
