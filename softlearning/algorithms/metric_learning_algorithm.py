import numpy as np
import tensorflow as tf

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from softlearning.algorithms.sac import SAC, td_target
from softlearning.environments.utils import is_point_2d_env


GYM_LOCOMOTION_ENVS = (
    HopperEnv,
    Walker2dEnv,
    HalfCheetahEnv,
    SwimmerEnv,
    AntEnv,
    HumanoidEnv,
)


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
        self._current_distance_goal = None
        super(MetricLearningAlgorithm, self).__init__(*args, **kwargs)

    def _init_placeholders(self):
        super(MetricLearningAlgorithm, self)._init_placeholders()
        self._placeholders['distance_goals'] = {
            name: tf.compat.v1.placeholder(
                dtype=(
                    np.float32
                    if np.issubdtype(observation_space.dtype, np.floating)
                    else observation_space.dtype
                ),
                shape=(None, *observation_space.shape),
                name=f'goals/{name}')
            for name, observation_space
            in self._training_environment.observation_space.spaces.items()
            # if name in self._metric_learner.distance_estimator.observation_keys
        }

    def _get_Q_target(self):
        observations_ph = self._placeholders['observations']
        actions_ph = self._placeholders['actions']
        next_observations_ph = self._placeholders['next_observations']

        # goals_ph is possibly a single element. Broadcast it to batch size.
        distance_goals = {
            name: placeholder + tf.zeros(
                (tf.shape(actions_ph)[0], *placeholder.shape[1:]),
                dtype=placeholder.dtype,
            )
            for name, placeholder in
            self._placeholders['distance_goals'].items()
        }

        # equal_shapes = (
        #     set(next_observations_ph.keys()) == set(goals_ph.keys())
        #     and all(next_observations_ph[key].shape[1:]
        #             == goals_ph[key].shape[1:]
        #             for key in next_observations_ph.keys()))

        # if equal_shapes:
        #     goal_successes = (
        #         tf.cast(tf.reduce_all(tf.equal(
        #             tf.concat(self._policy_inputs(
        #                 observations=next_observations_ph), axis=-1),
        #             tf.concat(self._policy_inputs(
        #                 observations=goals_ph), axis=-1),
        #         ), axis=1, keepdims=True), tf.float32))
        # else:
        #     goal_successes = tf.zeros_like(self._placeholders['terminals'])

        goal_successes = tf.zeros_like(
            self._placeholders['terminals'], dtype=tf.float32)

        if self._use_distance_for == 'reward':
            policy_inputs = self._policy_inputs(
                observations=next_observations_ph)
            next_actions = self._policy.actions(policy_inputs)
            next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

            Q_inputs = self._Q_inputs(
                observations=next_observations_ph,
                actions=next_actions)
            next_Qs_values = tuple(Q(Q_inputs) for Q in self._Q_targets)

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
            next_value = min_next_Q - self._alpha * next_log_pis

            inputs = self._metric_learner._distance_estimator_inputs(
                observations_ph, distance_goals, actions_ph)
            distances = self._metric_learner.distance_estimator(inputs)

            # Add constant reward to prevent the agent of intentionally
            # terminating the episodes. Only give this bonus when alive.
            constant_reward = 0.0 * (
                self.sampler._max_path_length * (1 - goal_successes))

            rewards = -1.0 * distances + constant_reward
            values = next_value

        elif self._use_distance_for == 'value':
            if self._metric_learner.distance_estimator.condition_with_action:
                policy_inputs = self._policy_inputs(
                    observations=next_observations_ph)
                next_actions = self._policy.actions(policy_inputs)

                inputs = self._metric_learner._distance_estimator_inputs(
                    next_observations_ph, distance_goals, next_actions)
            else:
                inputs = self._metric_learner._distance_estimator_inputs(
                    next_observations_ph, distance_goals, None)

            distances = self._metric_learner.distance_estimator(inputs)
            rewards = 0.0
            values = -1.0 * distances

        elif self._use_distance_for == 'telescope_reward':
            inputs1 = self._metric_learner._distance_estimator_inputs(
                observations_ph, distance_goals, actions_ph)
            distances1 = self._metric_learner.distance_estimator(inputs1)

            policy_inputs = self._policy_inputs(
                observations=next_observations_ph)
            next_actions = self._policy.actions(policy_inputs)
            inputs2 = self._metric_learner._distance_estimator_inputs(
                next_observations_ph, distance_goals, next_actions)
            distances2 = self._metric_learner.distance_estimator(inputs2)

            rewards = -1.0 * (distances2 - distances1)

            Q_inputs = self._Q_inputs(
                observations=next_observations_ph, actions=next_actions)
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

    def _update_goal(self):
        new_goal = self._target_proposer.propose_target(epoch=self._epoch)
        self._current_distance_goal = new_goal.copy()

        # self._training_environment.set_goal(new_goal)

        # try:
        #     self._training_environment._env.env.set_fixed_goal(new_goal)
        # except AttributeError:
        #     self._training_environment.unwrapped.set_fixed_goal(new_goal)

    def _epoch_after_hook(self, training_paths):
        self._previous_training_paths = training_paths

    def _timestep_before_hook(self, *args, **kwargs):
        if self.sampler._path_length == 0:
            self._update_goal()

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
        self._metric_learner._do_training_repeats(timestep)

        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)

        if trained_enough:
            raise ValueError("Should not be here")

        for i in range(self._n_train_repeat):
            self._do_training(
                iteration=timestep,
                batch=self._training_batch())

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(MetricLearningAlgorithm, self)._get_feed_dict(
            iteration, batch)

        if self._current_distance_goal is None:
            first_observations = self._pool.last_n_batch(1)['observations']
            self._current_distance_goal = type(first_observations)((
                (key, values[0]) for key, values in first_observations.items()
            ))

        for name, value in self._current_distance_goal.items():
            feed_dict[self._placeholders['distance_goals'][name]] = (
                value[None, ...])

        return feed_dict

    def diagnostics_distances_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            policy_inputs = self._policy_inputs(observations)
            actions = self._policy.actions_np(policy_inputs)
        inputs = self._metric_learner._distance_estimator_inputs(
            observations, goals, actions)
        distances = (
            self._metric_learner.distance_estimator.predict(
                inputs))
        return distances

    def diagnostics_Q_values_fn(self, observations, goals, actions):
        # TODO(hartikainen): in point 2d plotter, make sure that
        # the observations and goals work correctly.
        Q_inputs = self._Q_inputs(observations, actions)
        Qs = tuple(Q.predict(Q_inputs) for Q in self._Qs)
        Qs = np.min(Qs, axis=0)
        return Qs

    def diagnostics_V_values_fn(self, observations, goals):
        with self._policy.set_deterministic(True):
            policy_inputs = self._policy_inputs(observations)
            actions = self._policy.actions_np(policy_inputs)
        V_values = self.diagnostics_Q_values_fn(observations, goals, actions)
        return V_values

    def _evaluation_paths(self, policy, evaluation_env):
        # try:
        #     goal = self._evaluation_environment._env.env.sample_metric_goal()
        #     evaluation_env._env.env.set_goal(goal)
        # except AttributeError:
        #     goal = self._evaluation_environment.unwrapped.sample_metric_goal()
        #     evaluation_env.unwrapped.set_goal(goal)

        # if is_point_2d_env(evaluation_env.unwrapped):
        #     evaluation_env.unwrapped.optimal_policy.set_goal(goal)

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
            iteration, training_paths, evaluation_paths)
        diagnostics.update([
            (f'metric-learner/{key}', value)
            for key, value in
            metric_learner_diagnostics.items()
        ])

        target_proposer_diagnostics = self._target_proposer.get_diagnostics()
        diagnostics.update((
            (f'target_proposer/{key}', value)
            for key, value in target_proposer_diagnostics.items()
        ))

        environment = self._training_environment.unwrapped
        environment_class_name = type(environment).__name__

        if environment_class_name == 'DClawTurnFixed-v0':
            diagnostics['_current_distance_goal/object_angle'] = np.arctan2(
                self._current_distance_goal['object_angle_sin'],
                self._current_distance_goal['object_angle_cos']
            ).item()
        elif isinstance(environment, GYM_LOCOMOTION_ENVS):
            assert not environment._exclude_current_positions_from_observation
            position_slice = slice(*{
                SwimmerEnv: (0, 2),
                AntEnv: (0, 2),
                HumanoidEnv: (0, 2),
                HalfCheetahEnv: (0, 1),
                HopperEnv: (0, 1),
                Walker2dEnv: (0, 1),
            }[type(environment)])

            qpos_size = environment.sim.data.qpos.size
            velocity_slice = slice(
                position_slice.start + qpos_size,
                position_slice.stop + qpos_size,
                position_slice.step)

            positions = (
                self._current_distance_goal['observations'][position_slice])
            velocities = (
                self._current_distance_goal['observations'][velocity_slice])

            diagnostics[f'_current_distance_goal/position_l2'] = (
                np.linalg.norm(positions, ord=2))
            diagnostics.update({
                f'_current_distance_goal/position[{i}]': value.item()
                for i, value in enumerate(positions)
            })
            diagnostics[f'_current_distance_goal/velocity_l2'] = (
                np.linalg.norm(velocities, ord=2))
            diagnostics.update({
                f'_current_distance_goal/velocity[{i}]': value.item()
                for i, value in enumerate(velocities)
            })
        elif environment_class_name == 'Reacher-v2':
            fingertip_to_target_xy = self._current_distance_goal[
                'observations'][-3:-1]
            diagnostics['_current_distance_goal/fingertip_to_target_x'] = (
                fingertip_to_target_xy[0])
            diagnostics['_current_distance_goal/fingertip_to_target_y'] = (
                fingertip_to_target_xy[1])

            reacher_joint_velocities = self._current_distance_goal[
                'observations'][-5:-3]
            for i, joint_velocity in enumerate(reacher_joint_velocities):
                diagnostics[
                    f'_current_distance_goal/reacher_joint_velocity[{i}]'
                ] = joint_velocity
            diagnostics[
                    f'_current_distance_goal/reacher_joint_velocity-mean'
                ] = np.mean(np.abs(reacher_joint_velocities))
        elif environment_class_name == 'InvertedDoublePendulum-v2':
            raise NotImplementedError(environment_class_name)
        elif environment_class_name == 'Pendulum-v0':
            raise NotImplementedError(environment_class_name)

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
