import numpy as np
import tensorflow as tf

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from softlearning.algorithms.sac import SAC, td_target
from softlearning.environments.gym.mujoco.goal_environment import (
    GoalEnvironment)
from softlearning.environments.utils import is_point_2d_env


class MetricLearningAlgorithm(SAC):

    def __init__(self,
                 metric_learner,
                 target_proposer,
                 *args,
                 use_distance_for='reward',
                 plot_distances=False,
                 supervision_schedule_params=None,
                 final_exploration_proportion=0.25,
                 **kwargs):
        self._use_distance_for = use_distance_for
        self._plot_distances = plot_distances

        self._metric_learner = metric_learner
        self._target_proposer = target_proposer
        self._final_exploration_proportion = final_exploration_proportion
        super(MetricLearningAlgorithm, self).__init__(*args, **kwargs)

    def _init_placeholders(self):
        super(MetricLearningAlgorithm, self)._init_placeholders()
        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='goals')

    def _get_Q_target(self):
        goal_successes = tf.cast(tf.reduce_all(tf.equal(
            self._next_observations_ph, self._goals_ph
        ), axis=1, keepdims=True), tf.float32)

        if self._use_distance_for == 'reward':
            policy_inputs = self._action_inputs(
                observations=self._next_observations_ph)
            next_actions = self._policy.actions(policy_inputs)
            next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

            Q_inputs = self._Q_inputs(
                observations=self._next_observations_ph, actions=next_actions)
            next_Qs_values = tuple(Q(Q_inputs) for Q in self._Q_targets)

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
            next_value = min_next_Q - self._alpha * next_log_pis

            inputs = self._metric_learner._distance_estimator_inputs(
                self._observations_ph, self._goals_ph, self._actions_ph)
            distances = self._metric_learner.distance_estimator(inputs)

            # Add constant reward to prevent the agent of intentionally
            # terminating the episodes. Only give this bonus when alive.
            constant_reward = 0.0 * (
                self.sampler._max_path_length * (1 - goal_successes))

            rewards = -1.0 * distances + constant_reward
            values = next_value

        elif self._use_distance_for == 'value':
            if self._metric_learner._condition_with_action:
                policy_inputs = self._action_inputs(
                    observations=self._next_observations_ph)
                next_actions = self._policy.actions(policy_inputs)

                inputs = self._metric_learner._distance_estimator_inputs(
                    self._next_observations_ph, self._goals_ph, next_actions)
            else:
                inputs = self._metric_learner._distance_estimator_inputs(
                    self._next_observations_ph, self._goals_ph, None)

            distances = self._metric_learner.distance_estimator(inputs)
            rewards = 0.0
            values = -1.0 * distances

        elif self._use_distance_for == 'telescope_reward':
            inputs1 = self._metric_learner._distance_estimator_inputs(
                self._observations_ph, self._goals_ph, self._actions_ph)
            distances1 = self._metric_learner.distance_estimator(inputs1)

            policy_inputs = self._action_inputs(
                observations=self._next_observations_ph)
            next_actions = self._policy.actions(policy_inputs)
            inputs2 = self._metric_learner._distance_estimator_inputs(
                self._next_observations_ph, self._goals_ph, next_actions)
            distances2 = self._metric_learner.distance_estimator(inputs2)

            rewards = -1.0 * (distances2 - distances1)

            Q_inputs = self._Q_inputs(
                observations=self._next_observations_ph, actions=next_actions)
            next_Qs_values = tuple(Q(Q_inputs) for Q in self._Q_targets)

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)

            next_log_pis = self._policy.log_pis(policy_inputs, next_actions)
            next_value = min_next_Q - self._alpha * next_log_pis

            values = next_value

        else:
            raise NotImplementedError(self._use_distance_for)

        values = (1 - goal_successes) * values

        Q_target = td_target(
            reward=rewards,
            discount=self._discount,
            next_value=values)  # N

        return Q_target

    def _update_goal(self, training_paths):
        new_goal = self._target_proposer.propose_target(
            training_paths, epoch=self._epoch)

        try:
            self._training_environment._env.env.set_goal(new_goal)
        except Exception as e:
            self._training_environment.unwrapped.set_goal(new_goal)

        if is_point_2d_env(self._training_environment.unwrapped):
            self._training_environment.unwrapped.optimal_policy.set_goal(
                new_goal)

    def _epoch_after_hook(self, training_paths):
        self._previous_training_paths = training_paths

    def _timestep_before_hook(self, *args, **kwargs):
        if self.sampler._path_length == 0:
            self._update_goal(self.sampler.get_last_n_paths())

        random_explore_after = (
            self.sampler._max_path_length
            * (1.0 - self._final_exploration_proportion))

        if isinstance(self._training_environment.unwrapped,
                      (SwimmerEnv,
                       AntEnv,
                       HumanoidEnv,
                       HalfCheetahEnv,
                       HopperEnv,
                       Walker2dEnv)):
            succeeded_this_episode = (
                self._training_environment._env.env.succeeded_this_episode)
        else:
            succeeded_this_episode = getattr(
                self._training_environment.unwrapped,
                'succeeded_this_episode',
                False)

        if (self.sampler._path_length >= random_explore_after
            or succeeded_this_episode):
            self.sampler.initialize(
                self._training_environment,
                self._initial_exploration_policy,
                self._pool)
            # self.sampler.initialize(
            #     self._training_environment,
            #     self._training_environment.unwrapped.optimal_policy,
            #     self._pool)
        else:
            # self.sampler.initialize(
            #     self._training_environment,
            #     self._initial_exploration_policy,
            #     self._pool)
            self.sampler.initialize(
                self._training_environment,
                self._policy,
                self._pool)
            # self.sampler.initialize(
            #     self._training_environment,
            #     self._training_environment.unwrapped.optimal_policy,
            #     self._pool)
            if self.sampler.policy is not self._policy:
                assert is_point_2d_env(self._training_environment.unwrapped)

    def _timestep_after_hook(self, *args, **kwargs):
        if hasattr(self._metric_learner, '_update_target'):
            self._metric_learner._update_target(tau=self._tau)

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)

        if trained_enough:
            raise ValueError("Should not be here")

        n_mutual_train_repeat = min(
            self._n_train_repeat, self._metric_learner._n_train_repeat)
        for i in range(n_mutual_train_repeat):
            batch = self._training_batch()
            self._do_training(
                iteration=timestep,
                batch=batch)
            self._metric_learner._do_training(
                iteration=timestep,
                batch=batch)

        for i in range(self._n_train_repeat - n_mutual_train_repeat):
            self._do_training(
                iteration=timestep,
                batch=self._training_batch())

        for i in range(self._metric_learner._n_train_repeat
                       - n_mutual_train_repeat):
            self._metric_learner._do_training(
                iteration=timestep,
                batch=self._training_batch())

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(MetricLearningAlgorithm, self)._get_feed_dict(
            iteration, batch)
        feed_dict[self._goals_ph] = batch['goals']

        del feed_dict[self._rewards_ph]

        return feed_dict

    def _evaluate_rollouts(self, paths, env):
        result = super(MetricLearningAlgorithm, self)._evaluate_rollouts(
            paths, env)

        if is_point_2d_env(self._training_environment.unwrapped):
            observations = (
                self._training_environment.unwrapped.all_pairs_observations)
            distances = (
                self._training_environment.unwrapped.all_pairs_shortest_distances)

            boundaries = np.arange(0, np.max(distances) + 5, 5)

            zero_distance_idx = np.where(distances == 0)

            results = self._metric_learner._evaluate(
                observations=observations[zero_distance_idx],
                actions=np.zeros((observations[zero_distance_idx].shape[0], 2)),
                y=distances[zero_distance_idx])

            for key, value in results.items():
                result[f"d==0-{key}"] = value

            for low, high in list(zip(boundaries[:-1], boundaries[1:])):
                within_boundary_idx = np.where(
                    np.logical_and(low < distances, distances <= high))

                results = self._metric_learner._evaluate(
                    observations=observations[within_boundary_idx],
                    actions=np.zeros(
                        (observations[within_boundary_idx].shape[0], 2)),
                    y=distances[within_boundary_idx])

                for key, value in results.items():
                    result[f"{low}<d<={high}-{key}"] = value

            full_results = self._metric_learner._evaluate(
                observations=observations,
                actions=np.zeros((observations.shape[0], 2)),
                y=distances)

            for key, value in full_results.items():
                result[key] = value

            all_observations = np.concatenate(
                [path['observations.observation'] for path in paths], axis=0)
            all_xy_positions = all_observations[:, :2]
            all_xy_distances = np.linalg.norm(all_xy_positions, ord=2, axis=1)
            max_xy_distance = np.max(all_xy_distances)
            result['max_xy_distance'] = max_xy_distance

        elif isinstance(self._training_environment.unwrapped,
                        (SwimmerEnv,
                         AntEnv,
                         HumanoidEnv,
                         HalfCheetahEnv,
                         HopperEnv,
                         Walker2dEnv)):
            if self._training_environment.unwrapped._exclude_current_positions_from_observation:
                raise NotImplementedError

            if isinstance(self._training_environment._env.env, GoalEnvironment):
                return result

            all_observations = np.concatenate(
                [path['observations'] for path in paths], axis=0)
            all_actions = np.concatenate(
                [path['actions'] for path in paths], axis=0)
            all_xy_positions = all_observations[:, :2]
            all_xy_distances = np.linalg.norm(all_xy_positions, ord=2, axis=1)
            max_xy_distance = np.max(all_xy_distances)

            temporary_goals = np.tile(self._temporary_goal[None, :],
                                      (all_observations.shape[0], 1))
            distances_from_goal = (
                self._metric_learner.distance_estimator.predict(
                    self._metric_learner._distance_estimator_inputs(
                        all_observations, temporary_goals, all_actions))[:, 0])
            l2_distances_from_goal = np.linalg.norm(
                all_xy_positions - temporary_goals[:, :2], ord=2, axis=1)
            goal_l2_distance_from_origin = np.linalg.norm(
                self._temporary_goal[:2], ord=2)
            goal_estimated_distance_from_origin = (
                self._metric_learner.distance_estimator.predict(
                    self._metric_learner._distance_estimator_inputs(
                        self._first_observation[None, :],
                        self._temporary_goal[None, :],
                        np.zeros((1, *all_actions.shape[1:]))))[0, 0])

            result['max_xy_distance'] = max_xy_distance
            result['min_distance_from_goal'] = np.min(distances_from_goal)
            result['min_l2_distance_from_goal'] = np.min(
                l2_distances_from_goal)
            result['goal_l2_distance_from_origin'] = (
                goal_l2_distance_from_origin)
            result['goal_estimated_distance_from_origin'] = (
                goal_estimated_distance_from_origin)
            result['goal_x'] = self._temporary_goal[0]
            result['goal_y'] = self._temporary_goal[1]
            result['full_goal'] = np.array2string(
                self._temporary_goal, max_line_width=float('inf'))

        return result

    def diagnostics_distances_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations])
        inputs = self._metric_learner._distance_estimator_inputs(
            observations, goals, actions)
        distances = (
            self._metric_learner.distance_estimator.predict(
                inputs))
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

    def _evaluation_paths(self, policy, evaluation_env):
        try:
            goal = self._evaluation_environment._env.env.sample_metric_goal()
            evaluation_env._env.env.set_goal(goal)
        except Exception as e:
            goal = self._evaluation_environment.unwrapped.sample_metric_goal()
            evaluation_env.unwrapped.set_goal(goal)

        if is_point_2d_env(evaluation_env.unwrapped):
            evaluation_env.unwrapped.optimal_policy.set_goal(goal)

        return super(MetricLearningAlgorithm, self)._evaluation_paths(
            policy, evaluation_env)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(MetricLearningAlgorithm, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        metric_learner_diagnostics = self._metric_learner.get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)
        diagnostics.update([
            (f'metric-learner/{key}', value)
            for key, value in
            metric_learner_diagnostics.items()
        ])

        if hasattr(self._target_proposer, '_supervision_labels_used'):
            diagnostics['supervision_labels_used'] = (
                self._target_proposer._supervision_labels_used)

        if self._plot_distances:
            env = self._training_environment.unwrapped
            if is_point_2d_env(self._training_environment.unwrapped):
                from softlearning.visualization import point_2d_plotter
                point_2d_plotter.point_2d_plotter(
                    self,
                    iteration,
                    training_paths=training_paths,
                    evaluation_paths=evaluation_paths,
                    get_distances_fn=self.diagnostics_distances_fn,
                    get_quiver_gradients_fn=None,  # self.diagnostics_quiver_gradients_fn,
                    get_Q_values_fn=None,  # self.diagnostics_Q_values_fn,
                    get_V_values_fn=self.diagnostics_V_values_fn)
            else:
                raise NotImplementedError(self._training_environment.unwrapped)

        return diagnostics

    @property
    def tf_saveables(self):
        tf_saveables = super(MetricLearningAlgorithm, self).tf_saveables

        tf_saveables.update({
            f'_metric_learner_{key}': value
            for key, value in self._metric_learner.tf_saveables.items()
        })

        return tf_saveables
