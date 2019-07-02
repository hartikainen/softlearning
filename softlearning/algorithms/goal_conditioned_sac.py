import numpy as np
import tensorflow as tf

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from softlearning.environments.utils import is_point_2d_env
from .sac import SAC, td_target


class GoalConditionedSAC(SAC):
    def __init__(self,
                 target_proposer,
                 *args,
                 plot_distances=False,
                 final_exploration_proportion=0.25,
                 **kwargs):
        self._target_proposer = target_proposer
        self._plot_distances = plot_distances
        self._final_exploration_proportion = final_exploration_proportion

        super(GoalConditionedSAC, self).__init__(*args, **kwargs)

    def _init_placeholders(self):
        super(GoalConditionedSAC, self)._init_placeholders()
        self._goals_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='goals',
        )

    def _policy_inputs(self, observations):
        return [observations, self._goals_ph]

    def _Q_inputs(self, observations, actions):
        return [observations, actions, self._goals_ph]

    def _update_goal(self):
        new_goal = self._target_proposer.propose_target(epoch=self._epoch)

        # try:
        #     self._training_environment._env.env.set_goal(new_goal)
        # except AttributeError:
        #     self._training_environment.unwrapped.set_goal(new_goal)

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
                self._training_environment, self._initial_exploration_policy, self._pool)
            # self.sampler.initialize(
            #     self._training_environment, self._training_environment.unwrapped.optimal_policy, self._pool)
        else:
            # self.sampler.initialize(
            #     self._training_environment, self._initial_exploration_policy, self._pool)
            self.sampler.initialize(self._training_environment, self._policy, self._pool)
            # self.sampler.initialize(
            #     self._training_environment, self._training_environment.unwrapped.optimal_policy, self._pool)
            if self.sampler.policy is not self._policy:
                assert is_point_2d_env(self._training_environment.unwrapped)

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

    def diagnostics_distances_fn(self, observations, goals):
        distances = -self.diagnostics_V_values_fn(observations, goals)
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
        # try:
        #     goal = self._evaluation_environment._env.env.sample_metric_goal()
        #     evaluation_env._env.env.set_goal(goal)
        # except Exception as e:
        #     goal = self._evaluation_environment.unwrapped.sample_metric_goal()
        #     evaluation_env.unwrapped.set_goal(goal)

        # if is_point_2d_env(evaluation_env.unwrapped):
        #     evaluation_env.unwrapped.optimal_policy.set_goal(goal)

        return super(GoalConditionedSAC, self)._evaluation_paths(
            policy, evaluation_env)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(GoalConditionedSAC, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        if self._plot_distances:
            if is_point_2d_env(self._training_environment.unwrapped):
                from softlearning.visualization import point_2d_plotter
                point_2d_plotter.point_2d_plotter(
                    self,
                    iteration,
                    training_paths=training_paths,
                    evaluation_paths=evaluation_paths,
                    get_distances_fn=self.diagnostics_distances_fn,
                    get_quiver_gradients_fn=None,
                    get_Q_values_fn=self.diagnostics_Q_values_fn,
                    get_V_values_fn=self.diagnostics_V_values_fn)

        return diagnostics


class HERSAC(GoalConditionedSAC):
    def __init__(self, ground_truth_terminals, *args, **kwargs):
        self._ground_truth_terminals = ground_truth_terminals
        super(HERSAC, self).__init__(*args, **kwargs)

    def _get_Q_target(self):
        action_inputs = self._policy_inputs(
            observations=self._next_observations_ph)
        next_actions = self._policy.actions(action_inputs)

        Q_inputs = self._Q_inputs(
            observations=self._next_observations_ph, actions=next_actions)
        next_Qs_values = tuple(Q(Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q

        rewards = -1.0
        terminals = (
            self._terminals_ph
            if self._ground_truth_terminals
            else 0.0 # tf.cast(next_value > -0.5, tf.float32)
        )

        values = (1 - terminals) * next_value

        Q_target = td_target(
            reward=rewards,
            discount=self._discount,
            next_value=values)

        return Q_target
