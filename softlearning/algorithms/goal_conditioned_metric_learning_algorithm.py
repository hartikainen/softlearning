import numpy as np
import tensorflow as tf


from softlearning.models.utils import flatten_input_structure
from .metric_learning_algorithm import MetricLearningAlgorithm


class GoalConditionedMetricLearningAlgorithm(MetricLearningAlgorithm):
    def _policy_inputs(self, observations, goals=None):
        goals = goals or self._placeholders['goals']
        policy_observations = {
            name: observations[name]
            for name in self._policy.observation_keys
        }
        policy_goals = {
            name: goals[name]
            for name in self._policy.goal_keys
        }
        policy_inputs = flatten_input_structure({
            'observations': policy_observations,
            'goals': policy_goals,
        })
        return policy_inputs

    def _Q_inputs(self, observations, actions, goals=None):
        goals = goals or self._placeholders['goals']
        Q_observations = {
            name: observations[name]
            for name in self._Qs[0].observation_keys
        }
        Q_goals = {
            name: goals[name]
            for name in self._Qs[0].goal_keys
        }
        Q_inputs = flatten_input_structure({
            'observations': Q_observations,
            'actions': actions,
            'goals': Q_goals
        })
        return Q_inputs

    def _policy_diagnostics(self, iteration, batch):
        policy_inputs = self._policy_inputs(
            batch['observations'], batch['goals'])
        policy_diagnostics = self._policy.get_diagnostics(policy_inputs)
        return policy_diagnostics

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = super(
            GoalConditionedMetricLearningAlgorithm, self
        )._get_feed_dict(iteration, batch)

        for key, placeholder in self._placeholders['goals'].items():
            feed_dict[placeholder] = batch['goals'][key]

        return feed_dict

    def diagnostics_distances_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            policy_inputs = self._policy_inputs(observations, goals)
            actions = self._policy.actions_np(policy_inputs)

        inputs = self._metric_learner._distance_estimator_inputs(
            observations, goals, actions)
        distances = self._metric_learner.distance_estimator.predict(inputs)
        return distances

    def diagnostics_Q_values_fn(self, observations, goals, actions):
        # TODO(hartikainen): in point 2d plotter, make sure that
        # the observations and goals work correctly.
        Q_inputs = self._Q_inputs(observations, actions, goals)
        Qs = tuple(Q.predict(Q_inputs) for Q in self._Qs)
        Qs = np.min(Qs, axis=0)
        return Qs

    def diagnostics_V_values_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            policy_inputs = self._policy_inputs(observations, goals)
            actions = self._policy.actions_np(policy_inputs)
        V_values = self.diagnostics_Q_values_fn(observations, goals, actions)
        return V_values
