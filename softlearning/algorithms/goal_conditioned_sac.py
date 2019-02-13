import tensorflow as tf

from .sac import SAC


class GoalConditionedSAC(SAC):
    def _init_placeholders(self):
        super(GoalConditionedSAC, self)._init_placeholders()
        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='goals',
        )

    def _action_inputs(self, observations):
        return [observations, self._goals_ph]

    def _Q_inputs(self, observations, actions):
        return [observations, actions, self._goals_ph]

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""
        feed_dict = super(GoalConditionedSAC, self)._get_feed_dict(
            iteration, batch)

        feed_dict[self._goals_ph] = batch['goals']

        return feed_dict

    def _policy_diagnostics(self, iteration, batch):
        policy_diagnostics = self._policy.get_diagnostics([
            batch['observations'], batch['goals']])
        return policy_diagnostics
