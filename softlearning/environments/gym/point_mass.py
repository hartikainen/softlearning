"""Sequential (massless) 2d point environment.

Adapted from https://github.com/haarnoja/softqlearning/blob/master/softqlearning/environments/multigoal.py.

TODO(hartikainen): The environment and plotting bounds should be
automatically handled based on the goals and reset positions.
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


DEFAULT_GOAL_ORDERS = np.array(sorted(itertools.permutations(range(4), 4)))


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


class PointMassSequentialEnv(gym.Env):

    def __init__(self,
                 goal_reward=150.0,
                 actuation_cost_coeff=1.0,
                 distance_cost_coeff=1.0,
                 init_sigma=0.1,
                 goal_orders=None,
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

        if goal_orders is None:
            number_of_goal_orders = DEFAULT_GOAL_ORDERS.shape[0]
            assert (
                self.goal_positions.shape[0]
                == DEFAULT_GOAL_ORDERS.shape[1]), (
                    self.goal_positions.shape, DEFAULT_GOAL_ORDERS.shape)
            if mode == 'train':
                self.goal_orders = DEFAULT_GOAL_ORDERS[
                    :int(0.75 * number_of_goal_orders)]
            elif mode == 'evaluation':
                self.goal_orders = DEFAULT_GOAL_ORDERS[
                    int(0.75 * number_of_goal_orders):]
            else:
                raise ValueError(mode)
        else:
            self.goal_orders = goal_orders

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
                low=np.zeros(self.task_id_shape),
                high=np.ones(self.task_id_shape),
                shape=None,
                dtype=np.float32,
            )),
        )))

    @property
    def task_id_shape(self):
        return self.goal_positions.shape[:1]

    @property
    def current_task_id(self):
        goal_index = self.current_goal_order[self.goal_counter]
        one_hot_task_id = np.roll(
            np.eye(1, self.task_id_shape[0])[0], goal_index)
        return one_hot_task_id

    def _get_observation(self):
        observation = OrderedDict((
            ('position', self.position),
            ('task_id', self.current_task_id),
        ))
        return observation

    @property
    def current_goal_position(self):
        current_goal_index = self.current_goal_order[self.goal_counter]
        current_goal_position = self.goal_positions[current_goal_index].copy()
        return current_goal_position

    def step(self, action):
        action_t_0 = np.clip(
            action, self.action_space.low, self.action_space.high
        ).ravel()
        goal_t_0 = self.current_goal_position

        position_t_1 = self.dynamics.forward(self.position, action)
        position_t_1 = np.clip(
            position_t_1,
            self.observation_space['position'].low,
            self.observation_space['position'].high)

        self.position = position_t_1.copy()

        distance_to_goal = np.linalg.norm(position_t_1 - goal_t_0, ord=2)

        # Penalize the L2 norm of acceleration
        action_cost = self.action_cost_coeff * np.linalg.norm(
            action_t_0, ord=2)
        goal_cost = self.distance_cost_coeff * distance_to_goal

        reward = -1.0 * (action_cost + goal_cost)

        goal_done = distance_to_goal < self.goal_threshold
        episode_done = self.current_goal_order.size <= self.goal_counter

        if goal_done:
            self.goal_counter = min(
                self.goal_counter + 1, self.current_goal_order.size - 1)
            self.goal_changed = True  # for plotting
            reward += self.goal_reward

        next_goal_index = self.current_goal_order[self.goal_counter]
        next_goal_position = self.goal_positions[next_goal_index]

        observation = self._get_observation()

        info = {
            'goal_position': next_goal_position,
            'goal_order': self.current_goal_order,
            'goal_index': next_goal_index,
        }

        return observation, reward, episode_done, info

    def reset(self):
        unclipped_position = np.random.normal(
            loc=self.init_mu, scale=self.init_sigma, size=self.dynamics.s_dim)
        self.position = np.clip(
            unclipped_position,
            self.observation_space['position'].low,
            self.observation_space['position'].high)
        self.current_goal_order_index = np.random.choice(
            self.goal_orders.shape[0])
        self.current_goal_order = self.goal_orders[
            self.current_goal_order_index]
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
        axis.set_xlim(np.array(self.xlim))
        axis.set_ylim(np.array(self.ylim))

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

        path_lines = []
        color_map = plt.cm.get_cmap('tab10', len(paths))
        for i, path in enumerate(paths):
            positions = np.concatenate((
                path['observations']['position'],
                path['next_observations']['position'][[-1]],
            ), axis=0)

            task_ids = np.argmax(path['observations']['task_id'], axis=-1)
            task_change_indices = np.concatenate(
                ([-1], np.flatnonzero(np.diff(task_ids))))

            color = color_map(i)
            path_line = axis.plot(
                positions[..., 0],
                positions[..., 1],
                color=color,
                linestyle=':',
                linewidth=2.0,
                label='evaluation_paths' if i == 0 else None,
            )
            path_lines.append(path_line)

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
        return path_lines

    def _plot_latents(self, paths, path_lines, axes):
        for i, (path, path_line, latent_axis) in enumerate(
                zip(paths, path_lines, axes)):
            goal_ids = path['infos']['goal_index']
            q_z_probs = np.array(path['infos']['q_z_probs'])
            goal_order = np.array(path['infos']['goal_order'][0])
            assert np.all(goal_order[None] == path['infos']['goal_order'])

            latent_axis.stackplot(np.arange(q_z_probs.shape[0]), *q_z_probs.T)

            latent_axis.grid(True, linestyle='-', linewidth=0.2)
            latent_axis.set_ylim(
                1 - latent_axis.get_ylim()[1], latent_axis.get_ylim()[1])

            num_goals = self.goal_positions.shape[0]
            goal_axis = latent_axis.twinx()

            goal_axis.plot(
                np.arange(q_z_probs.shape[0]),
                goal_ids,
                color='black',
                linestyle=':',
                linewidth=2.0,
            )

            label = '\\rightarrow'.join(goal_order.astype(str))
            goal_order_label = AnchoredText(
                f"${label}$",
                loc='upper right',
                prop={'backgroundcolor': path_line[0].get_color()})
            goal_axis.add_artist(goal_order_label)

            goal_axis_y_margin = (num_goals - 1) * 0.05
            goal_axis.set_ylim((
                - goal_axis_y_margin, (num_goals - 1) + goal_axis_y_margin))
            goal_axis.set_yticks(np.linspace(0, num_goals - 1, num_goals))

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

        num_goals_reached = np.array([
            np.unique(path['infos']['goal_index']).size - 1
            for path in paths
        ])

        path_infos.update({
            'num_goals_reached-mean': np.mean(num_goals_reached),
            'num_goals_reached-min': np.min(num_goals_reached),
            'num_goals_reached-max': np.max(num_goals_reached),
        })

        self._save_figures(paths, iteration, evaluation_type=evaluation_type)

        return path_infos


class PointMassSequentialEnvV2(PointMassSequentialEnv):

    def __init__(self, *args, **kwargs):
        return super(PointMassSequentialEnvV2, self).__init__(*args, **kwargs)

    @property
    def task_id_shape(self):
        num_goal_orders = self.goal_orders.shape[0]
        return (num_goal_orders, )

    @property
    def current_task_id(self):
        goal_order_index = self.current_goal_order_index
        one_hot_task_id = np.roll(
            np.eye(1, self.task_id_shape[0])[0], goal_order_index)
        return one_hot_task_id
