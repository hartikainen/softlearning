import numpy as np
import tensorflow as tf

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as GymHalfCheetahEnv
from gym.envs.mujoco.ant import AntEnv as GymAntEnv
from gym.envs.mujoco.humanoid import HumanoidEnv as GymHumanoidEnv
from gym.envs.mujoco.inverted_double_pendulum import (
    InvertedDoublePendulumEnv as GymInvertedDoublePendulumEnv)
from gym.envs.mujoco.inverted_pendulum import (InvertedPendulumEnv as
                                               GymInvertedPendulumEnv)

from softlearning.algorithms.sac import SAC, td_target
from softlearning.environments.adapters.gym_adapter import (
    Point2DEnv,
    Point2DWallEnv,
    CustomSwimmerEnv,
    CustomAntEnv,
    CustomHumanoidEnv,
    CustomHalfCheetahEnv,
    CustomHopperEnv,
    CustomWalker2dEnv,
    FixedTargetReacherEnv)


class LogarithmicLabelScheduler(object):
    def __init__(self,
                 decay_steps,
                 end_labels,
                 start_labels=None,
                 start_labels_frac=None,
                 decay_rate=1e-2):
        assert (start_labels is None) or (start_labels_frac is None)
        self._start_labels = int(
            start_labels
            if start_labels is not None
            else start_labels_frac * end_labels)
        self._decay_steps = decay_steps
        self._end_labels = end_labels
        self._decay_rate = decay_rate

    @property
    def num_pretrain_labels(self):
        return self._start_labels

    def num_labels(self, time_step):
        """Return the number of labels desired at this point in time."""
        decayed_labels = (
            (self._start_labels - self._end_labels)
            * self._decay_rate ** (time_step / self._decay_steps)
            + self._end_labels)

        return int(decayed_labels)


class LinearLabelScheduler(object):
    def __init__(self,
                 decay_steps,
                 end_labels,
                 start_labels=None,
                 start_labels_frac=None,
                 power=1.0):
        assert (start_labels is None) or (start_labels_frac is None)
        self._start_labels = int(
            start_labels
            if start_labels is not None
            else start_labels_frac * end_labels)
        self._decay_steps = decay_steps
        self._end_labels = end_labels
        self._power = power

    @property
    def num_pretrain_labels(self):
        return self._start_labels

    def num_labels(self, time_step):
        time_step = min(time_step, self._decay_steps)
        decayed_labels = (
            (self._start_labels - self._end_labels)
            * (1 - time_step / self._decay_steps) ** (self._power)
            + self._end_labels)

        return int(decayed_labels)


PREFERENCE_SCHEDULERS = {
    'linear': LinearLabelScheduler,
    'logarithmic': LogarithmicLabelScheduler,
}


class MetricLearningAlgorithm(SAC):

    def __init__(self,
                 metric_learner,
                 env,
                 *args,
                 use_distance_for='reward',
                 temporary_goal_update_rule='closest_l2_from_goal',
                 plot_distances=False,
                 supervision_schedule_params=None,
                 **kwargs):
        self._use_distance_for = use_distance_for
        self._plot_distances = plot_distances

        assert supervision_schedule_params is not None
        self._supervision_schedule_params = supervision_schedule_params
        self._supervision_scheduler = PREFERENCE_SCHEDULERS[
            supervision_schedule_params['type']](
                **supervision_schedule_params['kwargs'])
        self._supervision_labels_used = 0
        self._last_supervision_epoch = -1

        self._metric_learner = metric_learner
        self._goal = getattr(env.unwrapped, 'fixed_goal', None)
        self._temporary_goal = None
        self._first_observation = None
        self._temporary_goal_update_rule = temporary_goal_update_rule
        super(MetricLearningAlgorithm, self).__init__(*args, env=env, **kwargs)

    def _init_placeholders(self):
        super(MetricLearningAlgorithm, self)._init_placeholders()
        self._temporary_goal_ph = tf.placeholder(
            tf.float32, shape=self._env.observation_space.shape, name='goal')

    def _get_Q_target(self):
        goals = tf.tile(self._temporary_goal_ph[None, :],
                        (tf.shape(self._observations_ph)[0], 1))

        if self._use_distance_for == 'reward':
            next_actions = self._policy.actions([self._next_observations_ph])
            next_log_pis = self._policy.log_pis([self._next_observations_ph],
                                                next_actions)

            next_Qs_values = tuple(
                Q([self._next_observations_ph, next_actions])
                for Q in self._Q_targets)

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
            next_value = min_next_Q - self._alpha * next_log_pis

            inputs = self._metric_learner._distance_estimator_inputs(
                self._observations_ph, goals, self._actions_ph)
            distances = self._metric_learner.distance_estimator(inputs)
            rewards = -1.0 * distances
            values = (1 - self._terminals_ph) * next_value

        elif self._use_distance_for == 'value':
            if self._metric_learner._condition_with_action:
                inputs = self._metric_learner._distance_estimator_inputs(
                    self._observations_ph, goals, self._actions_ph)
            else:
                inputs = self._metric_learner._distance_estimator_inputs(
                    self._next_observations_ph, goals, None)

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
                self._temporary_goal = new_observations[min_distance_idx]
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
                self._temporary_goal = new_observations[max_distance_idx]
        elif (self._temporary_goal_update_rule ==
              'farthest_estimate_from_first_observation'):
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

            max_distance_idx = np.argmax(new_distances)
            max_distance = new_distances[max_distance_idx]

            current_distance = self._metric_learner.distance_estimator.predict(
                self._metric_learner._distance_estimator_inputs(
                    self._first_observation[None, :],
                    self._temporary_goal[None, :],
                    np.zeros((1, *self._action_shape)),
                ))[0, 0]
            if max_distance >= current_distance:
                self._temporary_goal = new_observations[max_distance_idx]
        elif (self._temporary_goal_update_rule == 'operator_query_last_step'):
            expected_num_supervision_labels = (
                self._supervision_scheduler.num_labels(self._epoch))
            should_supervise = (
                expected_num_supervision_labels > self._supervision_labels_used)

            if not should_supervise:
                return

            self._supervision_labels_used += 1
            num_epochs_since_last_supervision = (
                self._epoch - self._last_supervision_epoch)
            self._last_supervision_epoch = self._epoch

            use_last_n_paths = (
                num_epochs_since_last_supervision // self.sampler._max_path_length)

            new_observations = np.concatenate(
                [path.get('observations.observation', path.get('observations'))
                 for path in training_paths[-use_last_n_paths:]], axis=0)
            last_observations = new_observations[-1::-self.sampler.
                                                 _max_path_length]
            if isinstance(
                    self._env.unwrapped,
                    (CustomSwimmerEnv,
                     CustomAntEnv,
                     CustomHumanoidEnv,
                     CustomHalfCheetahEnv,
                     CustomHopperEnv,
                     CustomWalker2dEnv)):
                if self._env.unwrapped._exclude_current_positions_from_observation:
                    raise NotImplementedError
                last_observations_x_positions = last_observations[:, 0:1]
                last_observations_distances = last_observations_x_positions

                max_distance_idx = np.argmax(last_observations_distances)
                max_distance = last_observations_distances[max_distance_idx]

                current_distance = self._temporary_goal[0]
                if max_distance > current_distance:
                    self._temporary_goal = last_observations[max_distance_idx]
            else:
                raise NotImplementedError

        elif (self._temporary_goal_update_rule == 'random'):
            self._temporary_goal = self._env.unwrapped.sample_goal(
            )['desired_goal']
        else:
            raise NotImplementedError

        if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
            self._env.unwrapped.optimal_policy.set_goal(self._temporary_goal)

    def _epoch_after_hook(self, training_paths):
        self._previous_training_paths = training_paths
        self._update_temporary_goal(training_paths)

    def _timestep_before_hook(self, *args, **kwargs):
        if ((self._timestep % self.sampler._max_path_length)
            >= self.sampler._max_path_length * 0.75):
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

    def _do_training(self, iteration, batch):
        super(MetricLearningAlgorithm, self)._do_training(iteration, batch)

        for i in range(self._metric_learner._MetricLearner__n_train_repeat):
            self._metric_learner._do_training(iteration, batch)
            batch = self._training_batch()

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(MetricLearningAlgorithm, self)._get_feed_dict(
            iteration, batch)

        del feed_dict[self._rewards_ph]

        if self._temporary_goal is None:
            self._temporary_goal = batch['observations'][0]
            self._first_observation = batch['observations'][0]

        feed_dict.update({self._temporary_goal_ph: self._temporary_goal})

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
            l2_distances_from_goal_xy = np.linalg.norm(
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
            result['min_goal_distance'] = np.min(distances_from_goal)
            result['min_l2_distance_to_goal_xy'] = np.min(
                l2_distances_from_goal_xy)
            result['goal_l2_distance_from_origin'] = (
                goal_l2_distance_from_origin)
            result['goal_estimated_distance_from_origin'] = (
                goal_estimated_distance_from_origin)

        return result

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

        diagnostics['supervision_labels_used'] = self._supervision_labels_used

        if self._plot_distances:
            if isinstance(self._env.unwrapped, (Point2DEnv, Point2DWallEnv)):
                def get_distances_fn(observations, goals):
                    with self._policy.set_deterministic(True):
                        actions = self._policy.actions_np([observations])
                    inputs = self._metric_learner._distance_estimator_inputs(
                        observations, goals, actions)
                    distances = (
                        self._metric_learner.distance_estimator.predict(
                            inputs))
                    return distances

                def get_Q_values_fn(observations, _, actions):
                    inputs = [observations, actions]
                    Qs = tuple(Q.predict(inputs) for Q in self._Qs)
                    Qs = np.min(Qs, axis=0)
                    return Qs

                def get_V_values_fn(observations, _):
                    with self._policy.set_deterministic(True):
                        actions = self._policy.actions_np([observations])
                    V_values = get_Q_values_fn(observations, _, actions)
                    return V_values

                def get_quiver_gradients_fn(observations, goals):
                    with self._policy.set_deterministic(True):
                        actions = self._policy.actions_np([observations])
                        inputs = (
                            self._metric_learner._distance_estimator_inputs(
                                observations, goals, actions))
                        return self._metric_learner.quiver_gradients([inputs])

                from softlearning.visualization import point_2d_plotter
                point_2d_plotter.point_2d_plotter(
                    self,
                    iteration,
                    training_paths=training_paths,
                    evaluation_paths=evaluation_paths,
                    get_distances_fn=get_distances_fn,
                    get_quiver_gradients_fn=get_quiver_gradients_fn,
                    get_Q_values_fn=get_Q_values_fn,
                    get_V_values_fn=get_V_values_fn)

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
