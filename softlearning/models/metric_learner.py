from collections import OrderedDict

import numpy as np
import tensorflow as tf

from serializable import Serializable

from softlearning.environments.adapters.gym_adapter import (
    CustomSwimmerEnv,
    CustomAntEnv,
    CustomHumanoidEnv)


class FloatLookupLayer(tf.keras.layers.Layer):
    def __init__(self, all_observation_pairs, **kwargs):
        self.all_observation_pairs = all_observation_pairs
        self.precision = 5
        self.separator = '|'

        x = np.reshape(
            self.all_observation_pairs,
            (self.all_observation_pairs.shape[0],
             np.prod(self.all_observation_pairs.shape[1:3])))

        x = self.to_np_string(x)
        x = np.unique(x)
        self.keys = x.astype(bytes)
        self.values = np.arange(x.shape[0])
        self.num_observation_pairs = all_observation_pairs.shape[0]

        super(FloatLookupLayer, self).__init__(**kwargs)

    def to_np_string(self, inputs):
        strs = np.vectorize(
            lambda e: f'{e:.0{self.precision}f}')(inputs)
        outputs = np.array([self.separator.join(row) for row in strs])
        return outputs

    def to_tf_string(self, inputs):
        outputs = tf.reduce_join(
            tf.as_string(inputs, precision=self.precision),
            separator=self.separator,
            axis=-1)
        return outputs

    def build(self, input_shape):
        self.table = tf.contrib.lookup.index_table_from_tensor(
            mapping=self.keys, num_oov_buckets=0, default_value=-1)

        self.distance_estimates = self.add_weight(
            name='distance_estimates',
            shape=(self.num_observation_pairs, 1),
            initializer='uniform',
            trainable=True)

    def call(self, inputs):
        x = self.to_tf_string(inputs)
        variable_idx = self.table.lookup(x)
        out = tf.gather(self.distance_estimates, variable_idx)
        return out

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 1))


FloatLookupModel = lambda *args, **kwargs: tf.keras.Sequential((
    FloatLookupLayer(*args, **kwargs),
))


class MetricLearner(Serializable):
    """MetricLearner."""

    def __init__(self,
                 env,
                 observation_shape,
                 action_shape,
                 distance_learning_rate=3e-4,
                 lambda_learning_rate=1e-3,
                 train_every_n_steps=1,
                 n_train_repeat=1,
                 distance_estimator_kwargs=None,
                 lambda_estimator=None,
                 te_lambda_estimator=None,
                 distance_estimator=None,
                 condition_with_action=False,
                 distance_input_type='full',
                 constraint_exp_multiplier=1.0,
                 objective_type='linear',
                 zero_constraint_threshold=0.1,
                 max_distance=None):
        self._Serializable__initialize(locals())

        self._env = env
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._distance_learning_rate = distance_learning_rate
        self._lambda_learning_rate = lambda_learning_rate

        self.distance_estimator = distance_estimator
        self._condition_with_action = condition_with_action
        self._distance_input_type = distance_input_type

        self.lambda_estimator = lambda_estimator
        self.te_lambda_estimator = te_lambda_estimator

        self._constraint_exp_multiplier = constraint_exp_multiplier
        self._objective_type = objective_type
        self._zero_constraint_threshold = zero_constraint_threshold

        self._max_distance = max_distance

        self._train_every_n_steps = train_every_n_steps
        self.__n_train_repeat = n_train_repeat

        self._session = tf.keras.backend.get_session()

        self._build()

    def _distance_estimator_inputs(self,
                                   observations1,
                                   observations2,
                                   actions):
        concat_fn = (
            tf.concat
            if isinstance(observations1, tf.Tensor)
            else np.concatenate)

        to_concat = [observations1]

        if self._condition_with_action:
            to_concat.append(actions)

        if self._distance_input_type == 'full':
            to_concat.append(observations2)

        elif self._distance_input_type == 'xy_coordinates':
            if isinstance(self._env.unwrapped,
                          (CustomSwimmerEnv, CustomAntEnv, CustomHumanoidEnv)):
                if (self._env.unwrapped
                    ._exclude_current_positions_from_observation):
                    raise NotImplementedError
                to_concat.append(observations2[:, :2])
            else:
                raise NotImplementedError(self._env)

        elif self._distance_input_type == 'xy_velocities':
            if isinstance(self._env.unwrapped,
                          (CustomSwimmerEnv, CustomAntEnv, CustomHumanoidEnv)):
                qvel_start_idx = self._env.unwrapped.sim.data.qpos.size
                qvel_end_idx = qvel_start_idx + 2

                if self._env.unwrapped._exclude_current_positions_from_observation:
                    qvel_start_idx -= 2
                    qvel_end_idx -= 2

                xy_velocities = observations2[:, qvel_start_idx:qvel_end_idx]
                to_concat.append(xy_velocities)
            else:
                raise NotImplementedError(self._env)

        elif self._distance_input_type == 'reward_sum':
            raise NotImplementedError(self._distance_input_type)

        else:
            raise NotImplementedError(self._distance_input_type)

        inputs = concat_fn(to_concat, axis=-1)
        return inputs

    def _build(self, *args, **kwargs):
        self._init_placeholders()
        self._init_distance_update()
        self._init_quiver_ops()

    @property
    def trainable_variables(self):
        trainable_variables = (
            self.distance_estimator.trainable_variables
            + self.lambda_estimator.trainable_variables
            + self.te_lambda_estimator.trainable_variables)
        return trainable_variables

    def _init_placeholders(self):
        """Create input placeholders for the MetricLearner algorithm."""
        self.distance_pairs_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2, *self._observation_shape),
            name='distance_pairs_observations')

        self.distance_pairs_actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2, *self._action_shape),
            name='distance_pairs_actions')

        self.distance_pairs_distances_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='distance_pairs_distances')

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

    def _init_quiver_ops(self):
        input_shape = self._distance_estimator_inputs(
            self.distance_pairs_observations_ph[:, 0, ...],
            self.distance_pairs_observations_ph[:, 1, ...],
            self.distance_pairs_actions_ph[:, 0, ...],).shape

        inputs = tf.keras.layers.Input(shape=input_shape[1:].as_list())

        if self.distance_estimator.layers[0].name != 'float_lookup_layer':
            distance_predictions = self.distance_estimator(inputs)[:, 0]
            quiver_gradients = tf.keras.backend.gradients(
                distance_predictions, [inputs])
            gradients_fn = tf.keras.backend.function(
                [inputs], quiver_gradients)
        else:
            n_actions = 9
            possible_actions = tf.reshape(tf.stack(tf.meshgrid(
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
            ), axis=-1), (n_actions, 2))

            s1 = inputs[:, :2]
            s2 = inputs[:, 2:]

            s1_new = tf.tile(s1[None], (n_actions, 1, 1))
            s2_new = tf.tile(s2[None], (n_actions, 1, 1))

            s1_new = s1_new + possible_actions[:, None, :]

            old_shape = tf.shape(s1_new)

            # s1_new = tf.reshape(
            #     s1_new,
            #     (old_shape[0] * old_shape[1],
            #      old_shape[2]))
            # s2_new = tf.reshape(
            #     s2_new,
            #     (old_shape[0] * old_shape[1],
            #      old_shape[2]))

            # new_inputs = tf.concat((s1_new, s2_new), axis=-1)

            # distance_predictions = self.distance_estimator(new_inputs)[:, 0]
            # distance_predictions = tf.reshape(
            #     distance_predictions,
            #     (old_shape[0] * old_shape[1]))
            # new_inputs = tf.reshape(
            #     new_inputs,
            #     (old_shape[0] * old_shape[1], old_shape[2] * 2))

            # max_indices = tf.argmax(distance_predictions, axis=0)

            gradients_fn = None

        self.quiver_gradients = gradients_fn

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
            self.lambda_estimator,
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
            self.lambda_estimator,
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

        if self._condition_with_action:
            raise ValueError(self._distance_input_type)

        with tf.control_dependencies([tf.assert_equal(*observations)]):
            inputs = self._distance_estimator_inputs(*observations, None)
            distance_predictions = self.distance_estimator(inputs)[:, 0]

        target_values = tf.fill(
            tf.shape(distance_predictions), self._zero_constraint_threshold)
        zero_constraints, _ = self._compute_constraints(
            self.lambda_estimator,
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
             self.te_lambda_estimator,
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
            if not self._condition_with_action
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
            + step_constraint
            + zero_constraint
            + triangle_inequality_constraint
            + max_distance_constraint)

        lambda_loss = self.lambda_loss = -(
            step_constraint
            + zero_constraint
            + max_distance_constraint)

        te_lambda_loss = self.te_lambda_loss = -triangle_inequality_constraint

        lagrangian_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._distance_learning_rate)

        lagrangian_grads_and_vars = lagrangian_optimizer.compute_gradients(
            loss=lagrangian_loss,
            var_list=self.distance_estimator.trainable_variables)

        lambda_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._lambda_learning_rate)

        lambda_grads_and_vars = lambda_optimizer.compute_gradients(
            loss=lambda_loss,
            var_list=self.lambda_estimator.trainable_variables)

        te_lambda_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._lambda_learning_rate)

        te_lambda_grads_and_vars = te_lambda_optimizer.compute_gradients(
            loss=te_lambda_loss,
            var_list=self.te_lambda_estimator.trainable_variables)

        all_gradients = [
            gradient for gradient, _ in
            lagrangian_grads_and_vars
            + lambda_grads_and_vars
            + te_lambda_grads_and_vars
            if gradient is not None
        ]

        with tf.control_dependencies(all_gradients):
            lagrangian_train_op = lagrangian_optimizer.apply_gradients(
                lagrangian_grads_and_vars)
            lambda_train_op = (
                lambda_optimizer.apply_gradients(lambda_grads_and_vars))
            te_lambda_train_op = (
                te_lambda_optimizer.apply_gradients(te_lambda_grads_and_vars))

        self._distance_train_ops = (
            lagrangian_train_op,
            lambda_train_op,
            te_lambda_train_op,
        )

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = {
            self.distance_pairs_observations_ph: (
                batch['distance_pairs_observations']),
            self.distance_pairs_actions_ph: (
                batch['distance_pairs_actions']),
            self.distance_pairs_distances_ph: (
                batch['distance_pairs_distances']),

            self.distance_triples_observations_ph: (
                batch['distance_triples_observations']),
            self.distance_triples_actions_ph: (
                batch['distance_triples_actions']),

            self.distance_objectives_observations_ph: (
                batch['distance_objectives_observations']),
            self.distance_objectives_actions_ph: (
                batch['distance_objectives_actions']),
        }

        return feed_dict

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        if iteration % self._train_every_n_steps > 0: return

        for i in range(self.__n_train_repeat):
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

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths,
                        *args, **kwargs):
        feed_dict = self._get_feed_dict(iteration, batch)

        (objectives,
         step_constraints,
         zero_constraints,
         triangle_inequality_constraints,
         max_distance_constraints,
         lagrangian_loss,
         lambda_loss,
         te_lambda_loss) = self._session.run((
             self.objectives,
             self.step_constraints,
             self.zero_constraints,
             self.triangle_inequality_constraints,
             self.max_distance_constraints,
             self.lagrangian_loss,
             self.lambda_loss,
             self.te_lambda_loss
         ), feed_dict)

        result = OrderedDict((
            ('lagrangian_loss-mean', np.mean(lagrangian_loss)),
            ('lambda_loss-mean', np.mean(lambda_loss)),
            ('te_lambda_loss-mean', np.mean(te_lambda_loss)),

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


class OnPolicyMetricLearner(MetricLearner):
    def _init_distance_update(self):
        """Create minimization operations for distance estimator."""

        observations = tf.unstack(
            self.distance_pairs_observations_ph, 2, axis=1)
        actions = tf.unstack(self.distance_pairs_actions_ph, 2, axis=1)[0]
        inputs = self._distance_estimator_inputs(*observations, actions)
        distance_predictions = self.distance_estimator(inputs)[:, 0]

        distance_loss = self.distance_loss = tf.losses.mean_squared_error(
            labels=self.distance_pairs_distances_ph[:, None],
            predictions=distance_predictions[:, None],
            weights=0.5)

        distance_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._distance_learning_rate)
        distance_grads_and_vars = distance_optimizer.compute_gradients(
            loss=distance_loss,
            var_list=self.distance_estimator.trainable_variables)

        distance_train_op = distance_optimizer.apply_gradients(
            distance_grads_and_vars)

        self._distance_train_ops = (distance_train_op, )

    def get_diagnostics(self, iteration, batch, paths, *args, **kwargs):
        feed_dict = self._get_feed_dict(iteration, batch)
        distance_loss = self._session.run(self.distance_loss, feed_dict)
        return OrderedDict((
            ('distance_loss-mean', np.mean(distance_loss)),
        ))
