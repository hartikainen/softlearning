import numpy as np
import tensorflow as tf

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as GymHalfCheetahEnv
from gym.envs.mujoco.ant import AntEnv as GymAntEnv
from gym.envs.mujoco.humanoid import HumanoidEnv as GymHumanoidEnv
from gym.envs.mujoco.inverted_double_pendulum import (
    InvertedDoublePendulumEnv as GymInvertedDoublePendulumEnv)
from gym.envs.mujoco.inverted_pendulum import (InvertedPendulumEnv as
                                               GymInvertedPendulumEnv)
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv


from softlearning.algorithms.sac import SAC, td_target
from softlearning.utils.numpy import softmax

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
                 *args,
                 use_distance_for='reward',
                 temporary_goal_update_rule='closest_l2_from_goal',
                 plot_distances=False,
                 final_exploration_proportion=0.25,
                 **kwargs):
        self._use_distance_for = use_distance_for
        self._plot_distances = plot_distances
        self._metric_learner = metric_learner
        self._goal = getattr(env.unwrapped, 'fixed_goal', None)
        self._temporary_goal = None
        self._first_observation = None
        self._temporary_goal_update_rule = temporary_goal_update_rule
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

    def _update_temporary_goal(self, training_paths):
        if self._temporary_goal_update_rule == 'closest_l2_from_goal':
            new_observations = np.concatenate(
                [path['observations'] for path in training_paths], axis=0)
            new_distances = np.linalg.norm(
                new_observations - self._goal, axis=1)

            min_distance_idx = np.argmin(new_distances)
            min_distance = new_distances[min_distance_idx]

            current_distance = np.linalg.norm(self._temporary_goal - self._goal)
            if min_distance < current_distance:
                temporary_goal = new_observations[min_distance_idx]
        elif (self._temporary_goal_update_rule ==
              'farthest_l2_from_first_observation'):
            new_observations = np.concatenate(
                [path['observations'] for path in training_paths], axis=0)
            new_distances = np.linalg.norm(
                new_observations - self._first_observation, axis=1)

            max_distance_idx = np.argmax(new_distances)
            max_distance = new_distances[max_distance_idx]

            current_distance = np.linalg.norm(self._temporary_goal -
                                              self._first_observation)
            if max_distance >= current_distance:
                temporary_goal = new_observations[max_distance_idx]

        elif (self._temporary_goal_update_rule in
              ('farthest_estimate_from_first_observation',
               'random_weighted_estimate_from_first_observation')):
            new_observations = self._pool.last_n_batch(
                min(self._pool.size, int(1e5)),
                field_name_filter='observations',
                observation_keys=getattr(self._env, 'observation_keys', None),
            )['observations']
            new_distances = self._metric_learner.distance_estimator.predict(
                self._metric_learner._distance_estimator_inputs(
                    np.tile(self._first_observation[None, :],
                            (new_observations.shape[0], 1)),
                    new_observations,
                    np.zeros((new_observations.shape[0], *self._action_shape)),
                ))[:, 0]

            if (self._temporary_goal_update_rule
                == 'farthest_estimate_from_first_observation'):
                max_distance_idx = np.argmax(new_distances)
                max_distance = new_distances[max_distance_idx]

                temporary_goal = new_observations[max_distance_idx]
            elif (self._temporary_goal_update_rule
                  == 'random_weighted_estimate_from_first_observation'):
                raise NotImplementedError("TODO: check this")
                temporary_goal = new_observations[np.random.choice(
                    new_distances.size, p=softmax(new_distances))]

        elif (self._temporary_goal_update_rule == 'operator_query_last_step'):
            new_observations = np.concatenate(
                [path['observations'] for path in training_paths], axis=0)
            path_last_observations = new_observations[-1::-self.sampler.
                                                      _max_path_length]
            if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
                goals = np.tile(
                    self._goal, (path_last_observations.shape[0], 1))
                last_observations_distances = (
                    self._env.unwrapped.get_optimal_paths(
                        path_last_observations, goals))

                min_distance_idx = np.argmin(last_observations_distances)
                temporary_goal = path_last_observations[min_distance_idx]

            elif isinstance(self._env.unwrapped,
                            (GymAntEnv, GymHalfCheetahEnv, GymHumanoidEnv)):
                velocity_indices = {
                    GymAntEnv:
                    slice(self._env.unwrapped.sim.data.qpos.size - 2,
                          self._env.unwrapped.sim.data.qpos.size),
                    GymHalfCheetahEnv:
                    slice(self._env.unwrapped.sim.data.qpos.size - 1,
                          self._env.unwrapped.sim.data.qpos.size),
                    GymHumanoidEnv:
                    slice(self._env.unwrapped.sim.data.qpos.size - 2,
                          self._env.unwrapped.sim.data.qpos.size),
                }[type(self._env.unwrapped)]
                new_velocities = new_observations[:, velocity_indices]
                new_velocities = np.linalg.norm(new_velocities, ord=2, axis=1)

                max_velocity_idx = np.argmax(new_velocities)
                temporary_goal = new_observations[max_velocity_idx]

            elif isinstance(
                    self._env.unwrapped,
                    (CustomSwimmerEnv,
                     CustomAntEnv,
                     CustomHumanoidEnv,
                     CustomHalfCheetahEnv,
                     CustomHopperEnv,
                     CustomWalker2dEnv)):
                if self._env.unwrapped._exclude_current_positions_from_observation:
                    raise NotImplementedError
                position_idx = slice(0, 2)
                last_observations_positions = path_last_observations[
                    :, position_idx]
                last_observations_distances = np.linalg.norm(
                    last_observations_positions, ord=2, axis=1)

                max_distance_idx = np.argmax(last_observations_distances)
                temporary_goal = path_last_observations[max_distance_idx]

            elif isinstance(
                    self._env.unwrapped,
                    (GymInvertedPendulumEnv, GymInvertedDoublePendulumEnv)):
                raise NotImplementedError
            elif isinstance(self._env.unwrapped, FixedTargetReacherEnv):
                last_distances_from_target = np.linalg.norm(
                    path_last_observations[:, -3:], ord=2, axis=1)

                min_distance_idx = np.argmin(last_distances_from_target)
                temporary_goal = path_last_observations[min_distance_idx]

            elif isinstance(self._env.unwrapped, SawyerPushAndReachXYZEnv):
                temporary_goal = self._env.unwrapped._state_goal.copy()
        elif (self._temporary_goal_update_rule == 'random'):
            temporary_goal = self._env.unwrapped.sample_metric_goal()
        else:
            raise NotImplementedError

        self._env.unwrapped.set_goal(temporary_goal)

        if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
            self._env.unwrapped.optimal_policy.set_goal(temporary_goal)

    def _epoch_after_hook(self, training_paths):
        self._previous_training_paths = training_paths

    def _timestep_before_hook(self, *args, **kwargs):
        if self.sampler._path_length == 0:
            if self._first_observation is None:
                first_sample = self._pool.batch_by_indices(0)
                self._first_observation = first_sample.get(
                    'observations.observation', first_sample['observations'])
                self._temporary_goal = self._first_observation.copy()

            self._update_temporary_goal(self.sampler.get_last_n_paths(1))

        random_explore_after = (
            self.sampler._max_path_length
            * (1.0 - self._final_exploration_proportion))
        if (self.sampler._path_length >= random_explore_after):
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
        inputs = [observations, actions]
        Qs = tuple(Q.predict(inputs) for Q in self._Qs)
        Qs = np.min(Qs, axis=0)
        return Qs

    def diagnostics_V_values_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations])
        V_values = self.get_Q_values_fn(observations, goals, actions)
        return V_values

    def diagnostics_quiver_gradients_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            actions = self._policy.actions_np([observations])
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
