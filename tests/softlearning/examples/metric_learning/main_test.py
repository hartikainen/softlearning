import numpy as np
import tensorflow as tf
import unittest

from examples.metric_learning.main import MetricExperimentRunner

CONFIG = {
    'Q_params': {
        'kwargs': {
            'hidden_layer_sizes': (10, 10),
        },
        'type': 'double_feedforward_Q_function'
    },
    'algorithm_params': {
        'kwargs': {
            'action_prior': 'uniform',
            'discount': 0.99,
            'epoch_length': 20,
            'eval_deterministic': True,
            'eval_n_episodes': 1,
            'eval_render_mode': None,
            'lr': 0.0003,
            'n_epochs': 3001,
            'n_initial_exploration_steps': 10,
            'n_train_repeat': 1,
            'plot_distances': True,
            'reparameterize': True,
            'reward_scale': 1.0,
            'save_full_state': False,
            'target_entropy': 'auto',
            'target_update_interval': 1,
            'tau': 0.005,
            'temporary_goal_update_rule': 'farthest_estimate_from_first_observation',
            'train_every_n_steps': 1,
            'use_distance_for': 'reward'},
        'type': 'MetricLearningAlgorithm'
    },
    'distance_estimator_params': {
        'kwargs': {
            'hidden_activation': 'relu',
            'hidden_layer_sizes': (256, 256),
            'output_activation': 'linear'
        },
        'type': 'FeedforwardDistanceEstimator'
    },
    'environment_params': {
        'training': {
            'universe': 'gym',
            'domain': 'Swimmer',
            'task': 'v3',
            'kwargs': {
                'exclude_current_positions_from_observation': False,
                'reset_noise_scale': 0
            },
        },
    },
    'git_sha':
    'fb03db4b0ffafc61d8ea6d550e7fdebeecb34d15 '
    'refactor/pick-utils-changes',
    'mode':
    'local',
    'lambda_estimator_params': {
        'kwargs': {
            'hidden_activation': 'relu',
            'hidden_layer_sizes': (256, 256),
            'output_activation': 'softplus'
        },
        'type': 'FeedforwardLambdaEstimator'
    },
    'metric_learner_params': {
        'kwargs': {
            'condition_with_action': False,
            'constraint_exp_multiplier': 0.0,
            'distance_input_type': 'full',
            'distance_learning_rate': 0.0003,
            'lambda_learning_rate': 0.0003,
            'max_distance': 1010,
            'n_train_repeat': 1,
            'objective_type': 'linear',
            'step_constraint_coeff': 0.1,
            'train_every_n_steps': 128,
            'zero_constraint_threshold': 0.0},
        'type': 'OnPolicyMetricLearner'
    },

    'policy_params': {
        'kwargs': {
            'hidden_layer_sizes': (10, 10),
            'squash': True
        },
        'type': 'GaussianPolicy'
    },
    'replay_pool_params': {
        'kwargs': {
            'max_pair_distance': None,
            'max_size': 1000,
            'on_policy_window': 100,
            'path_length': 10,
        },
        'type': 'DistancePool'
    },
    'run_params': {
        'checkpoint_at_end': True,
        'checkpoint_frequency': 60,
        'seed': 5666
    },
    'sampler_params': {
        'kwargs': {
            'batch_size': 256,
            'max_path_length': 10,
            'min_pool_size': 15
        },
        'type': 'SimpleSampler'
    },
}


def assert_weights_not_equal(weights1, weights2):
    for weight1, weight2 in zip(weights1, weights2):
        assert not np.all(np.equal(weight1, weight2))


class TestMetricExperimentRunner(tf.test.TestCase):

    def test_checkpoint_dict(self):
        tf.reset_default_graph()
        tf.keras.backend.clear_session()
        self.assertFalse(tf.trainable_variables())

        config = CONFIG.copy()

        experiment_runner = MetricExperimentRunner(config=config)

        session = experiment_runner._session
        experiment_runner._build()

        self.assertEqual(experiment_runner.algorithm._epoch, 0)
        self.assertEqual(experiment_runner.algorithm._timestep, 0)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 0)
        self.assertFalse(experiment_runner.algorithm._training_started)

        self.assertEqual(experiment_runner.replay_pool.size, 0)
        self.assertEqual(session.run(experiment_runner.algorithm._alpha), 1.0)

        initial_policy_weights = experiment_runner.policy.get_weights()
        initial_Qs_weights = [Q.get_weights() for Q in experiment_runner.Qs]

        for i in range(10):
            experiment_runner.train()

        self.assertEqual(experiment_runner.algorithm._epoch, 9)
        self.assertEqual(experiment_runner.algorithm._timestep, 20)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 200)
        self.assertTrue(experiment_runner.algorithm._training_started)
        self.assertNotEqual(
            session.run(experiment_runner.algorithm._alpha), 1.0)

        self.assertEqual(experiment_runner.replay_pool.size, 210)

        policy_weights = experiment_runner.policy.get_weights()
        Qs_weights = [Q.get_weights() for Q in experiment_runner.Qs]

        # Make sure that the training changed all the weights
        assert_weights_not_equal(initial_policy_weights, policy_weights)

        for initial_Q_weights, Q_weights in zip(initial_Qs_weights, Qs_weights):
            assert_weights_not_equal(initial_Q_weights, Q_weights)

        expected_alpha_value = 5.0
        session.run(
            tf.assign(experiment_runner.algorithm._log_alpha,
                      np.log(expected_alpha_value)))
        self.assertEqual(
            session.run(experiment_runner.algorithm._alpha),
            expected_alpha_value)

        trainable_variables_1 = {
            'policy': experiment_runner.policy.trainable_variables,
            'Q0': experiment_runner.Qs[0].trainable_variables,
            'Q1': experiment_runner.Qs[1].trainable_variables,
            'target_Q0': (
                experiment_runner.algorithm._Q_targets[0].trainable_variables),
            'target_Q1': (
                experiment_runner.algorithm._Q_targets[1].trainable_variables),
            'log_alpha': [experiment_runner.algorithm._log_alpha],
            'distance_estimator': (experiment_runner
                                   .algorithm
                                   ._metric_learner
                                   .distance_estimator
                                   .trainable_variables)
        }
        trainable_variables_1_np = session.run(trainable_variables_1)

        expected_variables = set(
            variable
            for _, variables in trainable_variables_1.items()
            for variable in variables)
        actual_variables = set(
            variable for variable in tf.trainable_variables()
            if 'save_counter' not in variable.name)

        self.assertEqual(expected_variables, actual_variables)

        optimizer_variables_1 = {
            'Q_optimizer_1': (
                experiment_runner.algorithm._Q_optimizers[0].variables()),
            'Q_optimizer_2': (
                experiment_runner.algorithm._Q_optimizers[1].variables()),
            'policy_optimizer': (
                experiment_runner.algorithm._policy_optimizer.variables()),
            'alpha_optimizer': (
                experiment_runner.algorithm._alpha_optimizer.variables()),
            'distance_optimizer': (experiment_runner
                                   .algorithm
                                   ._metric_learner
                                   ._distance_optimizer
                                   .variables()),
        }
        optimizer_variables_1_np = session.run(optimizer_variables_1)

        checkpoint = experiment_runner.save()

        tf.reset_default_graph()
        tf.keras.backend.clear_session()
        self.assertFalse(tf.trainable_variables())

        experiment_runner_2 = MetricExperimentRunner(config=config)
        session = experiment_runner_2._session
        self.assertFalse(experiment_runner_2._built)

        experiment_runner_2.restore(checkpoint)

        trainable_variables_2 = {
            'policy': experiment_runner_2.policy.trainable_variables,
            'Q0': experiment_runner_2.Qs[0].trainable_variables,
            'Q1': experiment_runner_2.Qs[1].trainable_variables,
            'target_Q0': (
                experiment_runner_2.algorithm._Q_targets[0].trainable_variables
            ),
            'target_Q1': (
                experiment_runner_2.algorithm._Q_targets[1].trainable_variables
            ),
            'log_alpha': [experiment_runner_2.algorithm._log_alpha],
            'distance_estimator': (experiment_runner_2
                                   .algorithm
                                   ._metric_learner
                                   .distance_estimator
                                   .trainable_variables)
        }
        trainable_variables_2_np = session.run(trainable_variables_2)

        expected_variables = set(
            variable
            for _, variables in trainable_variables_2.items()
            for variable in variables)
        actual_variables = set(
            variable for variable in tf.trainable_variables()
            if 'save_counter' not in variable.name)

        optimizer_variables_2 = {
            'Q_optimizer_1': (
                experiment_runner_2.algorithm._Q_optimizers[0].variables()),
            'Q_optimizer_2': (
                experiment_runner_2.algorithm._Q_optimizers[1].variables()),
            'policy_optimizer': (
                experiment_runner_2.algorithm._policy_optimizer.variables()),
            'alpha_optimizer': (
                experiment_runner_2.algorithm._alpha_optimizer.variables()),
            'distance_optimizer': (experiment_runner_2
                                   .algorithm
                                   ._metric_learner
                                   ._distance_optimizer
                                   .variables()),
        }
        optimizer_variables_2_np = session.run(optimizer_variables_2)

        for i, (key, variables_1_np) in enumerate(trainable_variables_1_np.items()):
            print()
            variables_1_tf = trainable_variables_1[key]
            variables_2_tf = trainable_variables_2[key]
            variables_2_np = trainable_variables_2_np[key]
            for j, (variable_1_np, variable_2_np,
                    variable_1_tf, variable_2_tf) in enumerate(
                        zip(variables_1_np, variables_2_np,
                            variables_1_tf, variables_2_tf)):
                allclose = np.allclose(variable_1_np, variable_2_np)
                variable_1_name = variable_1_tf.name
                variable_2_name = variable_2_tf.name

                print(f"i: {i}; j: {j}; {key};"
                      f" {allclose}; {variable_1_name}; {variable_2_name}")

                if 'target_Q' in key:
                    pass
                else:
                    np.testing.assert_allclose(variable_1_np, variable_2_np)

        for i in (0, 1):
            Q_variables_tf = trainable_variables_1[f'Q{i}']
            Q_variables_np = trainable_variables_1_np[f'Q{i}']
            target_Q_variables_tf = trainable_variables_2[f'target_Q{i}']
            target_Q_variables_np = trainable_variables_2_np[f'target_Q{i}']

            for j, (Q_np, target_Q_np, Q_tf, target_Q_tf) in enumerate(
                    zip(Q_variables_np, target_Q_variables_np,
                        Q_variables_tf, target_Q_variables_tf)):
                allclose = np.allclose(Q_np, target_Q_np)
                Q_name = Q_tf.name
                target_Q_name = target_Q_tf.name

                # print(f"i: {i}; {allclose}; {Q_name}; {target_Q_name}")

        self.assertEqual(experiment_runner_2.algorithm._epoch, 10)
        self.assertEqual(experiment_runner_2.algorithm._timestep, 0)
        self.assertEqual(
            session.run(experiment_runner_2.algorithm._alpha),
            expected_alpha_value)

        for i in range(10):
            experiment_runner_2.train()

        self.assertEqual(experiment_runner_2.algorithm._epoch, 19)
        self.assertEqual(experiment_runner_2.algorithm._timestep, 20)
        self.assertEqual(experiment_runner_2.algorithm._total_timestep, 400)
        self.assertTrue(experiment_runner_2.algorithm._training_started)


if __name__ == "__main__":
    tf.test.main()
