import numpy as np
import tensorflow as tf


from .metric_learning_algorithm import MetricLearningAlgorithm


class GoalConditionedMetricLearningAlgorithm(MetricLearningAlgorithm):
    def _init_placeholders(self):
        super(
            GoalConditionedMetricLearningAlgorithm, self
        )._init_placeholders()

        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='goals',
        )

    def _action_inputs(self, observations):
        return [observations, self._goals_ph]

    def _Q_inputs(self, observations, actions):
        return [observations, actions, self._goals_ph]

    def _policy_diagnostics(self, iteration, batch):
        policy_diagnostics = self._policy.get_diagnostics([
            batch['observations'], batch['goals']])
        return policy_diagnostics

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = super(
            GoalConditionedMetricLearningAlgorithm, self
        )._get_feed_dict(iteration, batch)

        feed_dict[self._goals_ph] = batch['goals']

        return feed_dict

    def diagnostics_distances_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations, goals])
        inputs = self._metric_learner._distance_estimator_inputs(
            observations, goals, actions)
        distances = self._metric_learner.distance_estimator.predict(inputs)
        return distances

    def diagnostics_Q_values_fn(self, observations, goals, actions):
        # TODO(hartikainen): in point 2d plotter, make sure that
        # the observations and goals work correctly.
        inputs = [observations, actions, goals]
        Qs = tuple(Q.predict(inputs) for Q in self._Qs)
        Qs = np.min(Qs, axis=0)
        return Qs

    def diagnostics_V_values_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations, goals])
        V_values = self.diagnostics_Q_values_fn(observations, goals, actions)
        return V_values

    def diagnostics_quiver_gradients_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations, goals])
        inputs = (
            self._metric_learner._distance_estimator_inputs(
                observations, goals, actions))
        return self._metric_learner.quiver_gradients([inputs])
