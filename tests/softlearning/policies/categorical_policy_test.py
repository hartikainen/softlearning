import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.models.utils import flatten_input_structure
from softlearning.policies.categorical_policy import (
    FeedforwardCategoricalPolicy)
from softlearning.environments.utils import get_environment


class CategoricalPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'CartPole', 'v0', {})
        self.hidden_layer_sizes = (16, 16)

        self.policy = FeedforwardCategoricalPolicy(
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_shape,
            action_range=(
                0,
                self.env.action_space.n
            ),
            hidden_layer_sizes=self.hidden_layer_sizes,
            observation_keys=self.env.observation_keys)

    def test_actions_and_log_pis_symbolic(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)

        observations_np = flatten_input_structure(observations_np)
        observations_tf = [tf.constant(x, dtype=tf.float32)
                           for x in observations_np]

        actions = self.policy.actions(observations_tf)
        log_pis = self.policy.log_pis(observations_tf, actions)

        self.assertEqual(actions.shape, (2, *self.env.action_shape))
        self.assertEqual(log_pis.shape, (2, 1))

        self.evaluate(tf.compat.v1.global_variables_initializer())

        actions_np = self.evaluate(actions)
        log_pis_np = self.evaluate(log_pis)

        self.assertEqual(actions_np.shape, (2, *self.env.action_shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_actions_and_log_pis_numeric(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
            observations_np = flatten_input_structure(observations_np)

        actions_np = self.policy.actions_np(observations_np)
        action_logits_np = self.policy.diagnostics_model.predict(observations_np)[0]
        log_pis_np = self.policy.log_pis_np(observations_np, actions_np)

        session = tf.keras.backend.get_session()
        expected_log_pis_np = np.array(session.run((
            tfp.distributions.Categorical(action_logits_np[0]).log_prob(
                actions_np[0]),
            tfp.distributions.Categorical(action_logits_np[1]).log_prob(
                actions_np[1]),
        )))

        np.testing.assert_array_equal(expected_log_pis_np, log_pis_np)
        self.assertEqual(actions_np.shape, (2, *self.env.action_shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_env_step_with_actions(self):
        observation_np = self.env.reset()
        observations_np = flatten_input_structure({
            key: value[None, :] for key, value in observation_np.items()
        })
        action = self.policy.actions_np(observations_np)[0, ...]
        self.env.step(action)

    def test_trainable_variables(self):
        self.assertEqual(
            len(self.policy.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

    def test_get_diagnostics(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
        observations_np = flatten_input_structure(observations_np)

        diagnostics = self.policy.get_diagnostics(observations_np)

        self.assertTrue(isinstance(diagnostics, OrderedDict))
        self.assertEqual(
            tuple(diagnostics.keys()),
            ('action_logits-mean',
             'action_logits-std',
             'action_logits-min',
             'action_logits-max',
             'entropy-mean',
             'entropy-std',
             'actions-mean',
             'actions-mode',
             'actions-min',
             'actions-max',))

        for value in diagnostics.values():
            self.assertTrue(np.isscalar(value))

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
        observations_np = flatten_input_structure(observations_np)

        weights = self.policy.get_weights()
        actions_np = self.policy.actions_np(observations_np)
        log_pis_np = self.policy.log_pis_np(observations_np, actions_np)

        serialized = pickle.dumps(self.policy)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()
        log_pis_np_2 = deserialized.log_pis_np(observations_np, actions_np)

        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight, weight_2)

        np.testing.assert_array_equal(log_pis_np, log_pis_np_2)
        np.testing.assert_equal(
            actions_np.shape, deserialized.actions_np(observations_np).shape)


if __name__ == '__main__':
    tf.test.main()
