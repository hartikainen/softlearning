"""
Adapted from https://github.com/haarnoja/softqlearning/blob/master/softqlearning/environments/multigoal.py
"""
import os
import itertools
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches

import gym
from gym import spaces


GOAL_ORDERS = np.array(sorted(itertools.permutations(range(4), 4)))


class PointMassSequentialEnv(gym.Env):

    def __init__(self,
                 goal_reward=150,
                 actuation_cost_coeff=0.1,
                 distance_cost_coeff=1,
                 init_sigma=0.1,
                 mode='train'):

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )

        assert self.goal_positions.shape[0] == GOAL_ORDERS.shape[1], (
            self.goal_positions.shape, GOAL_ORDERS.shape)

        number_of_goal_orders = GOAL_ORDERS.shape[0]
        if mode == 'train':
            # self.goal_orders = GOAL_ORDERS[:int(0.75 * number_of_goal_orders)]
            # self.goal_orders = GOAL_ORDERS[::6][:2]
            self.goal_orders = GOAL_ORDERS[::6][:1]
        elif mode == 'evaluation':
            self.goal_orders = GOAL_ORDERS[int(0.75 * number_of_goal_orders):]
        else:
            raise ValueError(mode)

        self.goal_threshold = 1.
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        # variables to be set in self.reset()
        self.goal_counter = None

        self._ax = None
        self._env_lines = []
        self.goal_changed = False  # for plotting
        self.fixed_plots = None
        self.dynamic_plots = []

        self.action_space = spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(OrderedDict((
            ('position', spaces.Box(
                low=np.array((self.xlim[0], self.ylim[0])),
                high=np.array((self.xlim[1], self.ylim[1])),
                shape=None,
                dtype=np.float32,
            )),
            ('task_id', spaces.Box(
                low=np.array([0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1]),
                shape=None,
                dtype=np.float32,
            ))
        )))

    @property
    def task_id_shape(self):
        return self.goal_positions.shape[:1]

    def _get_observation(self):
        goal_index = self.current_goal_order[self.goal_counter]
        # goal_position = self.goal_positions[goal_index]
        one_hot_task_id = np.roll(
            np.eye(1, self.goal_positions.shape[0])[0], goal_index)
        observation = OrderedDict((
            ('position', self.position),
            ('task_id', one_hot_task_id),
        ))
        return observation

    def step(self, action):
        action = action.ravel()

        action = np.clip(
            action, self.action_space.low, self.action_space.high
        ).ravel()

        next_position = self.dynamics.forward(self.position, action)
        next_position = np.clip(
            next_position,
            self.observation_space['position'].low,
            self.observation_space['position'].high)

        self.position = next_position.copy()

        observation = self._get_observation()

        reward = self.compute_reward(observation, action)

        goal_index = self.current_goal_order[self.goal_counter]
        goal_position = self.goal_positions[goal_index]
        distance_to_goal = np.linalg.norm(next_position - goal_position)
        goal_done = distance_to_goal < self.goal_threshold
        if goal_done:
            self.goal_counter += 1
            self.goal_changed = True  # for plotting
            reward += self.goal_reward * self.goal_counter

        done = self.current_goal_order.size <= self.goal_counter
        one_hot_task_id = np.roll(
            np.eye(1, self.goal_positions.shape[0])[0], goal_index)

        info = {
            'position': next_position,
            'task_id': one_hot_task_id,
            'goal_position': goal_position,
            'goal_order': self.current_goal_order,
        }
        return observation, reward, done, info

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        action_cost = np.linalg.norm(action, ord=2) * self.action_cost_coeff

        # penalize squared dist to goal
        current_position = observation['position']
        goal_index = self.current_goal_order[self.goal_counter]
        goal_position = self.goal_positions[goal_index]
        goal_cost = self.distance_cost_coeff * np.sum(
            np.abs(current_position - goal_position))

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def reset(self):
        unclipped_position = np.random.normal(
            loc=self.init_mu, scale=self.init_sigma, size=self.dynamics.s_dim)
        self.position = np.clip(
            unclipped_position,
            self.observation_space['position'].low,
            self.observation_space['position'].high)
        self.current_goal_order = self.goal_orders[
            np.random.choice(self.goal_orders.shape[0])]
        self.goal_counter = 0
        observation = self._get_observation()
        return observation

    def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self.xx, self.yy = [], []
        self.path = None
        self._plot_position_cost(self._ax)

    def render(self, mode='human', *args, **kwargs):
        if mode != 'human':
            raise ValueError(
                f"PointMass only dupports render mode='human'"
                f", got mode={mode}.")

        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        self.xx.append(self.position[0])
        self.yy.append(self.position[1])
        # remove current lines for performance:
        if self.path is not None:
            [line.remove() for line in self.path]
        if self.goal_changed:
            for artist in plt.gca().lines + plt.gca().collections:
                artist.remove()
            self._plot_position_cost(self._ax)
            self.goal_changed = False
        self.path = self._ax.plot(self.xx, self.yy, 'b')
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))
        plt.draw()
        plt.pause(0.01)

    def render_multi(self, paths):
        """For plotting multiple paths. Might be helpful for showing
        variation in VOD solutions"""
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []

        for path in paths:
            positions = np.stack([info['position'] for info in path['env_infos']])
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.draw()
        plt.pause(0.01)

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_index = self.current_goal_order[self.goal_counter]
        goal_x, goal_y = self.goal_positions[goal_index]
        goal_costs = -((X - goal_x) ** 2 + (Y - goal_y) ** 2)

        contours = ax.contour(X, Y, goal_costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return contours, goal

    def _plot_paths(self, paths, axis):
        axis.set_xlim(np.array(self.xlim) + (-1, 1))
        axis.set_ylim(np.array(self.ylim) + (-1, 1))

        goal_colors = [
            [*mpl.colors.to_rgba(x)[:3], 0.3]
            for x in mpl.colors.TABLEAU_COLORS.values()
        ]
        for j, (goal_position, goal_color) in enumerate(
                zip(self.goal_positions, goal_colors)):
            goal_patch = patches.Circle(
                goal_position,
                radius=self.goal_threshold,
                transform=axis.transData,
                edgecolor='black',
                facecolor=goal_color)
            axis.add_patch(goal_patch)
            axis.text(
                *goal_position,
                str(j),
                fontsize='xx-large',
                horizontalalignment='center',
                verticalalignment='center')

        color_map = plt.cm.get_cmap('tab10', len(paths))
        for i, path in enumerate(paths):
            positions = np.concatenate((
                path['observations']['position'],
                path['next_observations']['position'][[-1]],
            ), axis=0)

            task_ids = np.argmax(path['infos']['task_id'], axis=-1)
            # task_change_indices = np.concatenate(
            #     ([0], np.flatnonzero(np.diff(task_ids)), positions.shape[:1]))
            task_change_indices = np.concatenate(
                ([-1], np.flatnonzero(np.diff(task_ids))))

            color = color_map(i)
            axis.plot(
                positions[..., 0],
                positions[..., 1],
                color=color,
                linestyle=':',
                linewidth=1.0,
                label='evaluation_paths' if i == 0 else None,
            )

            for task_change_i in task_change_indices:
                start_goal = str(
                    '\\O' if task_change_i < 0 else task_ids[task_change_i])
                end_goal = str(task_ids[task_change_i + 1])

                axis.text(
                    positions[task_change_i + 1, 0],
                    positions[task_change_i + 1, 1],
                    f"${start_goal}\\rightarrow{end_goal}$",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize='xx-small',
                    bbox={
                        'edgecolor': 'black',
                        'facecolor': color,
                    }
                )

            axis.scatter(
                *positions[0],
                edgecolors='black',
                c=[color],
                marker='o',
                s=75.0,
                # linewidths=2.0,
            )
            axis.scatter(
                *positions[-1],
                edgecolors='black',
                c=[color],
                marker='X',
                s=90.0,
                # linewidths=2.0,
            )

        axis.grid(True, linestyle='-', linewidth=0.2)

    def _plot_latents(self, paths, axes):
        for i, (path, latent_axis) in enumerate(zip(paths, axes)):
            task_ids = np.argmax(path['infos']['task_id'], axis=-1)[..., None]
            # q_zs = np.array(path['infos']['q_z'])
            q_z_probs = np.array(path['infos']['q_z_probs'])
            goal_order = np.array(path['infos']['goal_order'][0])
            assert np.all(goal_order[None] == path['infos']['goal_order'])

            latent_axis.stackplot(
                np.arange(q_z_probs.shape[0]),
                *q_z_probs.T,
            )

            latent_axis.grid(True, linestyle='-', linewidth=0.2)
            latent_axis.set_ylim(
                1 - latent_axis.get_ylim()[1], latent_axis.get_ylim()[1])

            num_tasks = self.task_id_shape[0]
            task_axis = latent_axis.twinx()

            task_axis.plot(
                np.arange(q_z_probs.shape[0]),
                task_ids,
                color='black',
                linestyle=':',
                linewidth=2.0,
            )

            goal_order_label = AnchoredText(
                " -> ".join(goal_order.astype(str)), loc='upper right')
            task_axis.add_artist(goal_order_label)

            task_axis_y_margin = (num_tasks - 1) * 0.05
            task_axis.set_ylim((
                - task_axis_y_margin, (num_tasks - 1) + task_axis_y_margin))
            task_axis.set_yticks(np.linspace(0, num_tasks - 1, num_tasks))

    def _save_figures(self, paths, iteration, evaluation_type=None):
        # n_paths = len(paths)
        # to_show_paths = min(n_paths, 10)

        default_figsize = plt.rcParams.get('figure.figsize')
        figsize = np.array((2, 2)) * np.max(default_figsize[0])
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        gridspec = figure.add_gridspec(2, 2)

        axis_1 = figure.add_subplot(gridspec[0:2, 0:2])
        self._plot_paths(paths, axis_1)

        # TODO(hartikainen): The figures should be logged somewhere outside
        # this method, ideally through ray logger. Currently, `os.getcwd()`
        # is fine, since Ray and our main script (`main.py` inside `runners`)
        # guarantees that it points to the trial logdir.
        figure_dir = os.path.join(os.getcwd(), 'figures', 'pointmass')
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(
            figure_dir,
            '-'.join((
                evaluation_type or '',
                'iteration',
                f'{iteration:05}.png',
            )))

        plt.savefig(figure_path)
        figure.clf()
        plt.close(figure)

    def get_path_infos(self, paths, iteration, evaluation_type=None):
        path_infos = {}

        self._save_figures(paths, iteration, evaluation_type=evaluation_type)

        return path_infos


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """

    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = (
            mu_next + self.sigma * np.random.normal(size=self.s_dim))
        return state_next
