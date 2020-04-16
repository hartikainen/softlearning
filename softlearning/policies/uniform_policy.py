import tensorflow as tf
import tensorflow_probability as tfp
import tree

from .base_policy import ContinuousPolicy


class UniformPolicyMixin:
    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        actions = self.distribution.sample(batch_shape)

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        log_probs = self.distribution.log_prob(actions)[..., tf.newaxis]
        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def probs(self, observations, actions):
        probs = self.distribution.prob(actions)[..., tf.newaxis]
        return probs


class ContinuousUniformPolicy(UniformPolicyMixin, ContinuousPolicy):
    def __init__(self, *args, **kwargs):
        super(ContinuousUniformPolicy, self).__init__(*args, **kwargs)
        low, high = self._action_range
        self.distribution = tfp.distributions.Independent(
            tfp.distributions.Uniform(low=low, high=high),
            reinterpreted_batch_ndims=1)

    @tf.function(experimental_relax_shapes=True)
    def actions_raw_actions_and_log_probs(self, *args, **kwargs):
        """Compute actions, raw_actions, and log probabilities together."""
        actions = self.actions(*args, **kwargs)
        raw_actions = actions
        log_probs = self.log_probs(*args, **kwargs, actions=actions)
        return actions, raw_actions, log_probs

    @tf.function(experimental_relax_shapes=True)
    def action_raw_action_and_log_prob(self, *args, **kwargs):
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        result = self.actions_raw_actions_and_log_probs(*args_, **kwargs_)
        result = tree.map_structure(lambda x: x[0], result)
        return result

    @tf.function(experimental_relax_shapes=True)
    def actions_raw_actions_and_probs(self, *args, **kwargs):
        """Compute actions, raw_actions, and probabilities together."""
        actions = self.actions(*args, **kwargs)
        raw_actions = actions
        probs = self.probs(*args, **kwargs, actions=actions)
        return actions, raw_actions, probs

    @tf.function(experimental_relax_shapes=True)
    def action_raw_action_and_prob(self, *args, **kwargs):
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        result = self.actions_raw_actions_and_probs(*args_, **kwargs_)
        result = tree.map_structure(lambda x: x[0], result)
        return result
