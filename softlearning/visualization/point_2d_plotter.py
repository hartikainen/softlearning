import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.use('TkAgg')


RESOLUTION_MULTIPLIER = 2.0


def plot_walls(ax, walls):
    wall_rectangles = []

    for wall in walls:
        top_right = wall.endpoint1
        bottom_right = wall.endpoint2
        bottom_left = wall.endpoint3
        top_left = wall.endpoint4

        width = top_right[0] - top_left[0]
        height = top_right[1] - bottom_right[1]
        wall_rectangle = Rectangle(
            bottom_left,
            width,
            height,
            fill=True)

        wall_rectangles.append(wall_rectangle)

    wall_patch_collection = PatchCollection(
        wall_rectangles,
        facecolor='black',
        edgecolor='black')

    ax.add_collection(wall_patch_collection)

    return wall_patch_collection, wall_rectangles


def get_debugging_distances(algorithm):
    path_length = algorithm.sampler._max_path_length
    paths = np.split(
        algorithm._pool.observations[:path_length*2],
        (path_length*2) // path_length)

    d_x_to_y_multi_dict = {}
    d_x_to_y_multi_true_dict = {}
    d_x_to_y_multi_lambda_dict = {}
    for path in paths:
        assert path.shape[0] == path_length
        index = np.stack(np.meshgrid(
            np.arange(path_length),
            np.arange(path_length),
            indexing='ij'
        ), axis=-1)[np.triu_indices(path_length, 1)]
        all_observations = np.stack([
            [path[i], path[j]] for i, j in index
        ], axis=0)
        all_distances = index[:, 1] - index[:, 0]
        actions = np.zeros((all_observations.shape[0], 2))
        inputs = algorithm._metric_learner._distance_estimator_inputs(
            all_observations[:, 0, ...],
            all_observations[:, 1, ...],
            actions)
        distance_predictions = (
            algorithm._metric_learner.distance_estimator.predict(inputs)[:, 0])

        for (start, end), true_distance, distance_prediction in zip(
                all_observations, all_distances, distance_predictions):
            key = (tuple(start), tuple(end))
            d_x_to_y_multi_true_dict[key] = np.min((
                d_x_to_y_multi_true_dict.get(key, float('inf')),
                true_distance))
            d_x_to_y_multi_dict[key] = np.min((
                d_x_to_y_multi_dict.get(key, float('inf')),
                distance_prediction))
            inputs = algorithm._metric_learner._distance_estimator_inputs(
                np.array(key[0])[None],
                np.array(key[1])[None],
                np.zeros([1, 2]))
            d_x_to_y_multi_lambda_dict[key] = (
                algorithm._metric_learner.lambda_estimator.predict(inputs)
            )[0, 0]

    d_x_to_y_multi_keys = []
    d_x_to_y_multi = []
    d_x_to_y_multi_true = []
    d_x_to_y_multi_lambda = []

    for key in sorted(d_x_to_y_multi_dict.keys()):
        d_x_to_y_multi_keys.append(sum(key, ()))
        d_x_to_y_multi.append(d_x_to_y_multi_dict[key])
        d_x_to_y_multi_true.append(d_x_to_y_multi_true_dict[key])
        d_x_to_y_multi_lambda.append(d_x_to_y_multi_lambda_dict[key])

    df = pd.DataFrame(
        data={
            'd_x_to_y_multi': d_x_to_y_multi,
            'd_x_to_y_multi_true': d_x_to_y_multi_true,
            'd_x_to_y_lambda': d_x_to_y_multi_lambda
        },
        index=d_x_to_y_multi_keys
    )
    df['d_x_to_y_diff'] = df['d_x_to_y_multi'] - df['d_x_to_y_multi_true']

    return df


def get_observations(positions,
                     goal_positions,
                     nx,
                     ny,
                     velocities=None,
                     observation_type=None):
    if velocities is None:
        velocities = np.zeros_like(positions)
    if velocities.ndim == 1:
        velocities = np.repeat(velocities[None, :], nx * ny, axis=0)
    if positions.ndim == 1:
        positions = np.repeat(positions[None, :], nx * ny, axis=0)
    if goal_positions.ndim == 1:
        goal_positions = np.repeat(goal_positions[None, :], nx * ny, axis=0)

    if observation_type == 'reacher':
        observations = np.concatenate([
            positions,
            # Duplicate position to make consistent with reacher observation
            positions,
            goal_positions,
            velocities,
            positions - goal_positions,
            # Add z-axis to make consistent with reacher observation
            np.zeros((positions.shape[0], 1))
        ], axis=1)
    elif observation_type is None:
        observations = np.concatenate((
            positions,
            # goal_positions
        ), axis=1)

    return observations


def plot_distances(figure,
                   grid_spec,
                   algorithm,
                   observations_xy,
                   goals_xy,
                   num_heatmaps=16):
    min_x, max_x = algorithm._env.unwrapped.observation_x_bounds
    min_y, max_y = algorithm._env.unwrapped.observation_y_bounds

    nx = ny = int(np.sqrt(observations_xy.shape[0]))
    subplots_per_side = int(np.sqrt(num_heatmaps))
    gridspec_0 = mpl.gridspec.GridSpecFromSubplotSpec(
        subplots_per_side, subplots_per_side, subplot_spec=grid_spec)

    goal_estimate = algorithm._temporary_goal
    reset_position = algorithm._env._env.unwrapped.get_reset_position()

    algorithm._previous_goals = getattr(algorithm, '_previous_goals', np.zeros((0, 2)))
    algorithm._previous_goals = np.append(
        algorithm._previous_goals, goal_estimate[None], axis=0)

    # gridspec = mpl.gridspec.GridSpec(subplots_per_side*2, subplots_per_side)

    # fig, subplots = plt.subplots(
    #     subplots_per_side,
    #     subplots_per_side,
    #     sharex=True,
    #     sharey=True,
    #     figsize=(8.4, 8.4))

    V_star_axes = []
    V_star_axes_dict = {}
    for row in range(subplots_per_side-1, -1, -1):
        V_star_axes_dict[row] = {}
        for column in range(subplots_per_side):
            i = row * subplots_per_side + column
            ax = figure.add_subplot(
                gridspec_0[row, column],
                sharex=(
                    None if row == subplots_per_side - 1
                    else V_star_axes_dict[subplots_per_side-1][column]
                ),
                sharey=(
                    None if column == 0 else V_star_axes_dict[row][0]
                )
            )

            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

            V_star_axes_dict[row][column] = ax
            V_star_axes.append(ax)

            goal_position = goals_xy[i, :]
            observations_1 = get_observations(
                goal_position,
                observations_xy,
                nx,
                ny,
                observation_type=None)
            observations_2 = get_observations(
                observations_xy,
                observations_xy,
                nx,
                ny,
                velocities=None,
                observation_type=None)
            actions = algorithm._policy.actions_np([observations_1])
            # actions = np.zeros((observations_1.shape[0], 2))
            # inputs = algorithm._metric_learner._distance_estimator_inputs(
            #     observations_1, observations_2, actions)
            inputs = [observations_1, observations_2, actions]
            X = np.reshape(observations_xy[:, 0], (nx, ny))
            Y = np.reshape(observations_xy[:, 1], (nx, ny))

            Z = - algorithm._Qs[0].predict(inputs).reshape(nx, ny)

            filled_contour = ax.contourf(
                X, Y, Z,
                # levels=np.arange(algorithm._metric_learner._max_distance),
                # levels=np.arange(25),
                levels=25,
                extend='both',
                cmap='PuBuGn')

            contour = ax.contour(
                filled_contour,
                levels=filled_contour.levels,
                linewidths=0.2,
                linestyles='dotted',
                colors='black')

            ax.clabel(contour, contour.levels[2::2],
                      inline=1, fmt='%d', fontsize=8)

            wall_collection, wall_rectangles = plot_walls(
                ax, algorithm._env.unwrapped.walls)

            ax.scatter(*goal_position,
                       s=(6 * RESOLUTION_MULTIPLIER) ** 2,
                       color='blue',
                       marker='o',
                       label='s1')

            # if i == num_heatmaps - 1:
            #     ax.plot(*reset_position, 'go')

    colorbar_ax, kw = mpl.colorbar.make_axes(
        V_star_axes,
        location='bottom',)
    figure.colorbar(filled_contour, cax=colorbar_ax, **kw)

    rightmost_rectangle = max(
        wall_rectangles, key=lambda x: x.xy[0] + x.get_width())
    handles, labels = V_star_axes[-1].get_legend_handles_labels()
    V_star_axes[-1].legend(
        handles=handles,
        labels=labels,
        loc='center',
        bbox_to_anchor=rightmost_rectangle.get_bbox(),
        bbox_transform=ax.transData)


def plot_Q(figure,
           grid_spec,
           algorithm,
           observations_xy,
           goals_xy,
           num_heatmaps=16):
    nx = ny = int(np.sqrt(observations_xy.shape[0]))
    subplots_per_side = int(np.sqrt(num_heatmaps))
    gridspec_01 = mpl.gridspec.GridSpecFromSubplotSpec(
        subplots_per_side, subplots_per_side, subplot_spec=grid_spec)

    actions_min_x, actions_min_y = -1.0, -1.0
    actions_max_x, actions_max_y = 1.0, 1.0
    actions_x = np.linspace(actions_min_x, actions_max_x, nx)
    actions_y = np.linspace(actions_min_y, actions_max_y, ny)
    actions_X, actions_Y = np.meshgrid(actions_x, actions_y)
    actions_xy = np.stack([actions_X, actions_Y], axis=-1).reshape(-1, 2)

    Q_axes = []
    # for row in range(subplots_per_side-1, -1, -1):
    for row in range(subplots_per_side):
        for column in range(subplots_per_side):
            i = row * subplots_per_side + column
            ax = figure.add_subplot(gridspec_01[row, column])
            # ax = subplots[row][column]
            Q_axes.append(ax)

            # raise ValueError('nope')

            position = observations_xy[i, :]
            goal = goals_xy[i, :]
            observations_1 = get_observations(
                position,
                goal,
                nx,
                ny,
                velocities=None,
                observation_type=None)
            goal_position = goals_xy[i, :]
            observations_2 = get_observations(
                goal_position,
                observations_xy,
                nx,
                ny,
                observation_type=None)
            actions = actions_xy
            # Z = algorithm._q_functions[0].eval(observations, actions).reshape(nx, ny)
            Z = - algorithm._Qs[0].predict([
                observations_1, observations_2, actions
            ]).reshape(nx, ny)
            contour = ax.contourf(
                actions_X,
                actions_Y,
                Z,
                20,
                linestyles='dashed',
                extend='both',
                cmap='PuBuGn')
            # ax.plot(*state, 'go')

    colorbar_ax, kw = mpl.colorbar.make_axes(
        Q_axes,
        location='bottom',)
    figure.colorbar(contour, cax=colorbar_ax, **kw)


def plot_V(figure,
           grid_spec,
           algorithm,
           observations_xy,
           goals_xy,
           training_paths,
           evaluation_paths,
           num_heatmaps=16):
    min_x, max_x = algorithm._env.unwrapped.observation_x_bounds
    min_y, max_y = algorithm._env.unwrapped.observation_y_bounds

    nx = ny = int(np.sqrt(observations_xy.shape[0]))
    # V_pi
    ax = figure.add_subplot(grid_spec)
    # figure = plt.figure(figsize=(8.4, 8.4))
    # ax = figure.gca()

    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    if hasattr(algorithm, '_V'):
        observations = get_observations(
            observations_xy,
            goals_xy[-1, :],
            nx,
            ny,
            velocities=None,
            observation_type=None)
        X = np.reshape(observations_xy[:, 0], (nx, ny))
        Y = np.reshape(observations_xy[:, 1], (nx, ny))
        Z = np.reshape(algorithm._V.predict(
            [observations_1, goals]), (nx, ny))
        contour = ax.contourf(
            X,
            Y,
            Z,
            15,
            linestyles='dashed',
            cmap='PuBuGn')

        colorbar_ax, kw = mpl.colorbar.make_axes(ax, location='bottom',)
        figure.colorbar(contour, cax=colorbar_ax, **kw)

    if getattr(algorithm._env.unwrapped, 'fixed_goal', None) is not None:
        fixed_goal = algorithm._env._env.unwrapped.fixed_goal[:2]
        goals_xy[-1, :] = fixed_goal
    else:
        fixed_goal = None

    ax.scatter(*goals_xy[-1],
               s=(10 * RESOLUTION_MULTIPLIER) ** 2,
               color='blue',
               marker='*',
               label='final goal')

    goal_estimate = algorithm._temporary_goal
    ax.scatter(*goal_estimate,
               s=(10 * RESOLUTION_MULTIPLIER) ** 2,
               color=(1.0, 0.0, 0.0, 1.0),
               label='temporary goal',
               marker='*')
    # ax.arrow(
    #     *goal_estimate,
    #     # *goal_gradient[0],
    #     *np.mean(algorithm.gradients, axis=0),
    #     color='r')
    # algorithm.gradients = np.zeros((0, 2))
    ax.scatter(algorithm._previous_goals[:-1, 0],
               algorithm._previous_goals[:-1, 1],
               s=(10 * RESOLUTION_MULTIPLIER) ** 2,
               color=(1.0, 0.0, 0.0, 0.2),
               marker='*',
               label="past temporary goals")

    for i, training_path in enumerate(training_paths):
        observations = training_path['observations']

        if observations.shape[1] == 11:
            # Reacher
            assert np.all(observations[:, :2] == observations[:, 2:4])
            positions = observations[:, :2]
            target_positions = observations[:, 4:6]
            observation_target_distances = observations[:, -3:-1]
            assert np.allclose(
                observation_target_distances,
                positions - target_positions, rtol=1e-01)
        elif observations.shape[1] == 2:
            positions = observations

        ax.plot(positions[:, 0],
                positions[:, 1],
                'y:',
                linewidth=1.0,
                label="training paths" if i == 0 else None)
        ax.scatter(*positions[0], color='y', marker='o')
        ax.scatter(*positions[-1], color='y', marker='x')

    for path in evaluation_paths:
        observations = path['observations']
        assert observations.shape[1] == 2, observations.shape
        positions = observations

        ax.plot(positions[:, 0], positions[:, 1],
                'k:',
                linewidth=1.0 * RESOLUTION_MULTIPLIER,
                label="evaluation paths")
        ax.scatter(*positions[0],
                   color='black',
                   marker='o',
                   s=15 * RESOLUTION_MULTIPLIER)
        ax.scatter(*positions[-1],
                   color='black',
                   marker='x',
                   s=15 * RESOLUTION_MULTIPLIER)

    wall_collection, wall_rectangles = plot_walls(
        ax, algorithm._env.unwrapped.walls)

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    rightmost_rectangle = max(
        wall_rectangles, key=lambda x: x.xy[0] + x.get_width())

    ax.legend(
        handles=by_label.values(),
        labels=by_label.keys(),
        loc='center',
        bbox_to_anchor=rightmost_rectangle.get_bbox(),
        bbox_transform=ax.transData)


def plot_distance_quiver(figure,
                         grid_spec,
                         algorithm,
                         observations_xy,
                         goals_xy,
                         evaluation_paths):
    min_x, max_x = algorithm._env.unwrapped.observation_x_bounds
    min_y, max_y = algorithm._env.unwrapped.observation_y_bounds

    nx = ny = 10
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)
    observations_xy = np.stack([X, Y], axis=-1).reshape(-1, 2)

    num_quiver_plots = goals_xy.shape[0]
    subplots_per_side = int(np.sqrt(num_quiver_plots))
    gridspec_0 = mpl.gridspec.GridSpecFromSubplotSpec(
        subplots_per_side, subplots_per_side, subplot_spec=grid_spec)

    quiver_axes = []
    quiver_axes_dict = {}
    for row in range(subplots_per_side-1, -1, -1):
        quiver_axes_dict[row] = {}
        for column in range(subplots_per_side):
            i = row * subplots_per_side + column
            ax = figure.add_subplot(
                gridspec_0[row, column],
                sharex=(
                    None if row == subplots_per_side - 1
                    else quiver_axes_dict[subplots_per_side-1][column]
                ),
                sharey=(
                    None if column == 0 else quiver_axes_dict[row][0]
                )
            )

            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

            quiver_axes_dict[row][column] = ax
            quiver_axes.append(ax)

            goal_position = goals_xy[i, :]
            observations_1 = get_observations(
                observations_xy,
                observations_xy,
                nx,
                ny,
                observation_type=None)
            observations_2 = get_observations(
                goal_position,
                observations_xy,
                nx,
                ny,
                velocities=None,
                observation_type=None)

            actions = np.zeros((observations_1.shape[0], 2))
            # inputs = algorithm._metric_learner._distance_estimator_inputs(
            #     observations_1, observations_2, actions)
            inputs = [observations_1, observations_2, actions]
            X = np.reshape(observations_xy[:, 0], (nx, ny))
            Y = np.reshape(observations_xy[:, 1], (nx, ny))

            UV = -algorithm._metric_learner.quiver_gradients([inputs])[0]

            GRADIENT_SCALE = RESOLUTION_MULTIPLIER
            U = np.reshape(UV[..., 0], (nx, ny)) * GRADIENT_SCALE
            V = np.reshape(UV[..., 1], (nx, ny)) * GRADIENT_SCALE

            M = np.hypot(U, V)

            quiver = ax.quiver(
                X, Y, U, V, M,
                # units='xy',
                # angles='xy',
                # scale_units='xy',
                # scale=1,
            )

            wall_collection, wall_rectangles = plot_walls(
                ax, algorithm._env.unwrapped.walls)

            goal_scatter = ax.scatter(
                *goal_position,
                s=(10 * RESOLUTION_MULTIPLIER) ** 2,
                color='blue',
                marker='*',
                label='goal')

    rightmost_rectangle = max(
        wall_rectangles, key=lambda x: x.xy[0] + x.get_width())
    handles, labels = quiver_axes[-1].get_legend_handles_labels()
    quiver_axes[-1].legend(
        handles=handles,
        labels=labels,
        loc='center',
        bbox_to_anchor=rightmost_rectangle.get_bbox(),
        bbox_transform=ax.transData)


def plot_lagrange_multipliers(figure,
                              grid_spec,
                              algorithm,
                              observations_xy,
                              goals_xy,
                              evaluation_paths,
                              num_heatmaps=16):
    from pdb import set_trace; from pprint import pprint; set_trace()
    pass


def point_2d_plotter(algorithm, iteration, training_paths, evaluation_paths):
    log_base_dir = os.getcwd()

    heatmap_dir = os.path.join(log_base_dir, 'vf_heatmap')
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)

    PLOT = (
        'distances',
        'Q',
        # 'distance-quiver',
        'V',
        # 'lagrange-multipliers'
    )

    num_heatmaps = 16
    subplots_per_side = int(np.sqrt(num_heatmaps))

    # goal_space = algorithm._env._env.unwrapped.goal_space
    # action_space = algorithm._env._env.unwrapped.action_space
    # # min_x, min_y = goal_space.low[:2] * 2
    # # max_x, max_y = goal_space.high[:2] * 2
    # min_x, min_y = 0.3, -0.3
    # max_x, max_y = 1.0, 0.3

    min_x, max_x = algorithm._env.unwrapped.observation_x_bounds
    min_y, max_y = algorithm._env.unwrapped.observation_y_bounds

    reset_position = algorithm._env._env.unwrapped.get_reset_position()
    # np.array([ 0.6890015 , -0.14968373])

    if (False and algorithm._metric_learner.distance_estimator.layers[0].name
        == 'float_lookup_layer'):
        X = algorithm.evaluation_observations[:, 0]
        Y = algorithm.evaluation_observations[:, 1]
        # x = np.arange(min_x, max_x).astype(np.float32)
        # y = np.arange(min_y, max_y).astype(np.float32)
        nx = int(np.sqrt(X.shape[0]))
        ny = int(np.sqrt(Y.shape[0]))
        x = np.unique(X)
        y = np.unique(Y)
        goals_x = x[
            np.linspace(0, nx, subplots_per_side, endpoint=False, dtype=int)]
        goals_y = y[
            np.linspace(0, ny, subplots_per_side, endpoint=False, dtype=int)]
    else:
        nx = ny = 100
        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        X, Y = np.meshgrid(x, y)

        goals_x = np.linspace(min_x, max_x, subplots_per_side + 2)[1:-1]
        goals_y = np.linspace(min_y, max_y, subplots_per_side + 2)[1:-1]

    observations_xy = np.stack([X, Y], axis=-1).reshape(-1, 2)

    # reverse to get the positions match the for-loop
    goals_y  = goals_y[::-1]
    goals_X, goals_Y = np.meshgrid(goals_x, goals_y)
    goals_xy = np.stack([goals_X, goals_Y], axis=-1).reshape(-1, 2)

    goal_estimate = algorithm._temporary_goal
    goals_xy[-2, :] = goal_estimate
    goals_xy[-3, :] = reset_position

    RESOLUTION = RESOLUTION_MULTIPLIER * 8.4
    fig = plt.figure(figsize=(RESOLUTION * len(PLOT), RESOLUTION), frameon=False)
    grid_spec = mpl.gridspec.GridSpec(1, len(PLOT))

    i = 0
    if 'distances' in PLOT:
        plot_distances(fig,
                       grid_spec[i],
                       algorithm,
                       observations_xy,
                       goals_xy,
                       num_heatmaps=num_heatmaps)
        distances_title = f"d(s1,s2) for {goals_xy.shape[0]} different s1"
        x_min, x_max = grid_spec[i].get_position(fig).extents[[0, 2]]
        distances_title_x_position = (x_max + x_min) / 2
        fig.text(*(distances_title_x_position, 0.9),
                 distances_title,
                 horizontalalignment='center',
                 size=15 * RESOLUTION_MULTIPLIER)
        i += 1

    if ('distance-quiver' in PLOT
        and algorithm._metric_learner.quiver_gradients is not None):
        x_delta = (max_x - min_x) / 10.0
        y_delta = (max_y - min_y) / 10.0
        quiver_goals = np.array((
            (min_x + x_delta, min_y + y_delta),
            (min_x + x_delta, max_y - y_delta),
            (max_x - x_delta, min_y + y_delta),
            (max_x - x_delta, max_y - y_delta),
        ))

        plot_distance_quiver(fig,
                             grid_spec[i],
                             algorithm,
                             observations_xy,
                             quiver_goals,
                             evaluation_paths)

        quiver_title = (
            # r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
            r"$\dfrac{d}{ds_{1}}d(s_{1}, g)$ for"
            f"  {quiver_goals.shape[0]} different g"
        )
        # quiver_title = f"d(s1,g)/ds1 for {quiver_goals.shape[0]} different g"
        x_min, x_max = grid_spec[i].get_position(fig).extents[[0, 2]]
        quiver_title_x_position = (x_max + x_min) / 2

        fig.text(*(quiver_title_x_position, 0.9),
                 quiver_title,
                 horizontalalignment='center',
                 size=15 * RESOLUTION_MULTIPLIER)
        i += 1

    if 'Q' in PLOT:
        plot_Q(fig,
               grid_spec[i],
               algorithm,
               observations_xy,
               goals_xy,
               num_heatmaps=num_heatmaps)
        i += 1

    if 'V' in PLOT:
        V_title = f"Value function with training/evaluation rollouts."
        x_min, x_max = grid_spec[i].get_position(fig).extents[[0, 2]]
        V_title_x_position = (x_max + x_min) / 2
        plot_V(fig,
               grid_spec[i],
               algorithm,
               observations_xy,
               goals_xy,
               training_paths,
               evaluation_paths,
               num_heatmaps=num_heatmaps)

        fig.text(*(V_title_x_position, 0.9),
                 V_title,
                 horizontalalignment='center',
                 size=15 * RESOLUTION_MULTIPLIER)

        i += 1

    if 'lagrange-multipliers' in PLOT:
        plot_lagrange_multipliers(fig,
                                  grid_spec[i],
                                  algorithm,
                                  observations_xy,
                                  goals_xy,
                                  evaluation_paths,
                                  num_heatmaps=num_heatmaps)
        i += 1

    fig.suptitle(f'iteration={iteration}',
                 fontsize=20 * RESOLUTION_MULTIPLIER)

    # for optimal_path in algorithm._optimal_paths:
    #     start, end = optimal_path[0, :], optimal_path[-1, :]
    #     ax.plot(optimal_path[1:-1, 0], optimal_path[1:-1, 1])
    #     ax.plot(*start, 'ro')
    #     ax.plot(*end, 'kx')

    # for observation_pair, preference in zip(
    #         algorithm._preference_pool.observations[:algorithm._preference_pool.size],
    #         algorithm._preference_pool.preferences[:algorithm._preference_pool.size]):
    #     ax.plot(observation_pair[:, 0], observation_pair[:, 1], 'k:', linewidth=0.5)
    #     ax.scatter(*observation_pair[preference], color='g', marker='o')
    #     ax.scatter(*observation_pair[1-preference], color='r', marker='o')

    # ax.text(
    #     1, 1,
    #     'queries used: {}'.format(algorithm._segment_pool.queries_used),
    #     transform=ax.transAxes)

    max_iteration_len = int(math.ceil(math.log10(
        algorithm._epoch_length * algorithm._n_epochs)))
    vf_heatmap_path = os.path.join(
        heatmap_dir,
        f'iteration-{iteration:0{max_iteration_len}}-V_pi.png')
    plt.savefig(vf_heatmap_path)
    fig.clf()
    plt.close(fig)
