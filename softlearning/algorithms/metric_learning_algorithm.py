import numpy as np
import tensorflow as tf

from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv

from softlearning.algorithms.sac import SAC, td_target

from softlearning.environments.gym.mujoco.swimmer import (
    SwimmerEnv as CustomSwimmerEnv)
from softlearning.environments.gym.mujoco.ant import (
    AntEnv as CustomAntEnv)
from softlearning.environments.gym.mujoco.humanoid import (
    HumanoidEnv as CustomHumanoidEnv)
from softlearning.environments.gym.mujoco.half_cheetah import (
    HalfCheetahEnv as CustomHalfCheetahEnv)
from softlearning.environments.gym.mujoco.hopper import (
    HopperEnv as CustomHopperEnv)
from softlearning.environments.gym.mujoco.walker2d import (
    Walker2dEnv as CustomWalker2dEnv)


class MetricLearningAlgorithm(SAC):

    def __init__(self,
                 metric_learner,
                 env,
                 target_proposer,
                 *args,
                 use_distance_for='reward',
                 plot_distances=False,
                 final_exploration_proportion=0.25,
                 **kwargs):
        self._use_distance_for = use_distance_for
        self._plot_distances = plot_distances
        self._metric_learner = metric_learner
        self._target_proposer = target_proposer
        self._final_exploration_proportion = final_exploration_proportion
        super(MetricLearningAlgorithm, self).__init__(*args, env=env, **kwargs)

    def _init_placeholders(self):
        super(MetricLearningAlgorithm, self)._init_placeholders()
        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=self._env.observation_space.shape,
            name='goals')

    def _get_Q_target(self):
        if self._use_distance_for == 'reward':
            action_inputs = self._action_inputs(
                observations=self._next_observations_ph)
            next_actions = self._policy.actions(action_inputs)
            next_log_pis = self._policy.log_pis(action_inputs, next_actions)

            Q_inputs = self._Q_inputs(
                observations=self._next_observations_ph, actions=next_actions)
            next_Qs_values = tuple(Q(Q_inputs) for Q in self._Q_targets)

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
            next_value = min_next_Q - self._alpha * next_log_pis

            inputs = self._metric_learner._distance_estimator_inputs(
                self._observations_ph, self._goals_ph, self._actions_ph)
            distances = self._metric_learner.distance_estimator(inputs)
            rewards = -1.0 * distances
            values = (1 - self._terminals_ph) * next_value

        elif self._use_distance_for == 'value':
            if self._metric_learner._condition_with_action:
                inputs = self._metric_learner._distance_estimator_inputs(
                    self._observations_ph, self._goals_ph, self._actions_ph)
            else:
                inputs = self._metric_learner._distance_estimator_inputs(
                    self._next_observations_ph, self._goals_ph, None)

            distances = self._metric_learner.distance_estimator(inputs)
            rewards = 0.0
            values = -1.0 * distances

        Q_target = td_target(
            reward=rewards, discount=self._discount, next_value=values)  # N

        return Q_target

    def _update_goal(self, training_paths):
        new_goal = self._target_proposer.propose_target(training_paths)

        self._env.unwrapped.set_goal(new_goal)
        if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
            self._env.unwrapped.optimal_policy.set_goal(new_goal)

    def _epoch_after_hook(self, training_paths):
        self._previous_training_paths = training_paths

    def _timestep_before_hook(self, *args, **kwargs):
        if self.sampler._path_length == 0:
           self._update_goal(self.sampler.get_last_n_paths())

        random_explore_after = (
            self.sampler._max_path_length
            * (1.0 - self._final_exploration_proportion))
        if self.sampler._path_length >= random_explore_after:
            self.sampler.initialize(
                self._env, self._initial_exploration_policy, self._pool)
            # self.sampler.initialize(
            #     self._env, self._env.unwrapped.optimal_policy, self._pool)
        else:
            # self.sampler.initialize(
            #     self._env, self._initial_exploration_policy, self._pool)
            self.sampler.initialize(self._env, self._policy, self._pool)
            # self.sampler.initialize(
            #     self._env, self._env.unwrapped.optimal_policy, self._pool)
            if self.sampler.policy is not self._policy:
                assert isinstance(self._env.unwrapped, Point2DEnv)

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

        if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
            observations = self._env.unwrapped.all_pairs_observations
            distances = self._env.unwrapped.all_pairs_shortest_distances

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

        elif isinstance(self._env.unwrapped,
                        (CustomSwimmerEnv,
                         CustomAntEnv,
                         CustomHumanoidEnv,
                         CustomHalfCheetahEnv,
                         CustomHopperEnv,
                         CustomWalker2dEnv)):
            if self._env.unwrapped._exclude_current_positions_from_observation:
                raise NotImplementedError
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

    def diagnostics_quiver_gradients_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations, goals])
        inputs = (
            self._metric_learner._distance_estimator_inputs(
                observations, goals, actions))
        return self._metric_learner.quiver_gradients([inputs])

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

        if self._plot_distances:
            if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
                from softlearning.visualization import point_2d_plotter
                point_2d_plotter.point_2d_plotter(
                    self,
                    iteration,
                    training_paths=training_paths,
                    evaluation_paths=evaluation_paths,
                    get_distances_fn=self.diagnostics_distances_fn,
                    get_quiver_gradients_fn=self.diagnostics_quiver_gradients_fn,
                    get_Q_values_fn=self.diagnostics_Q_values_fn,
                    get_V_values_fn=self.diagnostics_V_values_fn)

            elif isinstance(self._env.unwrapped, FixedTargetReacherEnv):
                from softlearning.visualization import fixed_target_reacher_plotter
                fixed_target_reacher_plotter.fixed_target_reacher_plotter(
                    self, iteration, training_paths, evaluation_paths)

        return diagnostics

    @property
    def tf_saveables(self):
        tf_saveables = super(MetricLearningAlgorithm, self).tf_saveables

        tf_saveables.update({
            f'_metric_learner_{key}': value
            for key, value in self._metric_learner.tf_saveables.items()
        })

        return tf_saveables
