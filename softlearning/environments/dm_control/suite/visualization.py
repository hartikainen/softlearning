import glob
import itertools
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import qhull

from .pond import compute_angular_deltas


def get_path_infos_orbit_pond(physics,
                              paths,
                              evaluation_type='training',
                              figure_save_path=None):
    infos = {}

    x, y = np.split(np.concatenate(tuple(itertools.chain(*[
        [
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2]
        ]
        for path in paths
    ]))), 2, axis=-1)

    def compute_cumulative_angle_travelled(path):
        xy = np.concatenate((
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2]
        ), axis=0)
        angular_deltas = compute_angular_deltas(
            xy[:-1, ...], xy[1:, ...], center=physics.pond_center_xyz)
        cumulative_angle_travelled = np.sum(angular_deltas)
        return cumulative_angle_travelled

    cumulative_angles_travelled = np.array([
        compute_cumulative_angle_travelled(path)
        for path in paths
    ])

    distances_from_water = np.linalg.norm(
        np.concatenate((x, y), axis=-1) - physics.pond_center_xyz[:2],
        ord=2,
        axis=1,
    ) - physics.pond_radius

    infos.update([
        (f"{metric_name}-{metric_fn_name}",
         getattr(np, metric_fn_name)(metric_values))
        for metric_name, metric_values in (
                ('distances_from_water', distances_from_water),
                ('cumulative_angles_travelled', cumulative_angles_travelled),
                ('angular_velocity_mean',
                 cumulative_angles_travelled / x.size))
        for metric_fn_name in ('mean', 'max', 'min', 'mean')
    ])

    if 'acceleration' in paths[0]['observations']:
        accelerations_xyz = np.concatenate(tuple(itertools.chain(*[
            [
                path['observations']['acceleration'],
                path['next_observations']['acceleration'][[-1]]
            ]
            for path in paths
        ])))
        infos.update([
            (f"{metric_name}-{metric_fn_name}",
             getattr(np, metric_fn_name)(metric_values))
            for metric_name, metric_values in (
                    ('accelerations-x', accelerations_xyz[:, 0]),
                    ('accelerations-y', accelerations_xyz[:, 1]),
                    ('accelerations-z', accelerations_xyz[:, 2]))
            for metric_fn_name in ('mean', 'max', 'min', 'mean')
        ])

    if 'velocity' in paths[0]['observations']:
        velocities_xyz = np.concatenate(tuple(itertools.chain(*[
            [
                path['observations']['velocity'],
                path['next_observations']['velocity'][[-1]]
            ]
            for path in paths
        ])))
        infos.update([
            (f"{metric_name}-{metric_fn_name}",
             getattr(np, metric_fn_name)(metric_values))
            for metric_name, metric_values in (
                    ('velocities-x', velocities_xyz[:, 0]),
                    ('velocities-y', velocities_xyz[:, 1]),
                    ('velocities-z', velocities_xyz[:, 2]))
            for metric_fn_name in ('mean', 'max', 'min', 'mean')
        ])

    histogram_margin = 3.0
    bins_per_unit = 2

    pond_center = physics.pond_center_xyz[:2]
    x_bounds = tuple(
        pond_center[0]
        + physics.pond_radius * np.array((-1, 1))
        + histogram_margin * np.array((-1, 1)))
    y_bounds = tuple(
        pond_center[1]
        + physics.pond_radius * np.array((-1, 1))
        + histogram_margin * np.array((-1, 1)))

    H, xedges, yedges = np.histogram2d(
        np.squeeze(x),
        np.squeeze(y),
        bins=(
            int(np.ptp(x_bounds) * bins_per_unit),
            int(np.ptp(y_bounds) * bins_per_unit),
        ),
        range=np.array((x_bounds, y_bounds)),
    )

    full_area = (
        (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0]))
    water_area = np.pi * physics.pond_radius ** 2 / 4
    support_of_total_area = (np.sum(H > 0) / H.size)
    support_of_walkable_area = (
        support_of_total_area * full_area
        / (full_area - water_area))

    infos.update({'support-1': support_of_walkable_area})

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    XY = np.stack((X, Y), axis=-1)

    valid_margin = np.sqrt(2 * (1.0 / bins_per_unit) ** 2)
    valid_bins = np.logical_and.reduce((
        (physics.pond_radius - valid_margin)
        <= np.linalg.norm(XY, ord=2, axis=-1),
        np.linalg.norm(XY, ord=2, axis=-1) < (physics.pond_radius + 3.0),
    ))

    support_of_valid_bins = (np.sum(H[valid_bins] > 0) / (
        H[valid_bins].size))

    infos.update({'support-2': support_of_valid_bins})

    log_base_dir = (
        os.getcwd()
        if figure_save_path is None
        else figure_save_path)
    heatmap_dir = os.path.join(log_base_dir, 'heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    previous_heatmaps = glob.glob(
        os.path.join(heatmap_dir, f"{evaluation_type}-iteration-*-heatmap.png"))
    heatmap_iterations = [
        int(re.search(f"{evaluation_type}-iteration-(\d+)-heatmap.png", x).group(1))
        for x in previous_heatmaps
    ]
    if not heatmap_iterations:
        iteration = 0
    else:
        iteration = int(max(heatmap_iterations) + 1)

    base_size = 12.8
    x_max = y_max = 3 * physics.pond_radius
    x_min = y_min = - x_max
    width = x_max - x_min
    height = y_max - y_min

    if width > height:
        figsize = (base_size, base_size * (height / width))
    else:
        figsize = (base_size * (width / height), base_size)

    figure, axis = plt.subplots(1, 1, figsize=figsize)

    axis.set_xlim((x_min - 1, x_max + 1))
    axis.set_ylim((y_min - 1, y_max + 1))

    color_map = plt.cm.get_cmap('tab10', len(paths))
    for i, path in enumerate(paths):
        positions = np.concatenate((
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2],
        ), axis=0)

        color = color_map(i)
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linestyle=':',
            linewidth=1.0,
            label='evaluation_paths' if i == 0 else None,
        )
        axis.scatter(
            *positions[0, :2],
            edgecolors='black',
            c=[color],
            marker='o',
            s=75.0,
        )
        axis.scatter(
            *positions[-1, :2],
            edgecolors='black',
            c=[color],
            marker='X',
            s=90.0,
        )

        if 'perturbed' in path.get('infos', {}):
            perturbed = np.array(path['infos']['perturbed'])
            perturbed_xy = positions[:-1, ...][perturbed]

            if 'perturbation' in path['infos']:
                perturbations = np.stack(
                    path['infos']['perturbation'], axis=0)[perturbed]
            elif 'original_action' in path['infos']:
                perturbations = np.stack(
                    path['infos']['original_action'], axis=0)[perturbed]
            else:
                raise NotImplementedError("TODO(hartikainen)")

            axis.quiver(
                perturbed_xy[:, 0],
                perturbed_xy[:, 1],
                perturbations[:, 0],
                perturbations[:, 1],
                units='xy',
                # units='width',
                angles='xy',
                # scale=1.0,
                scale_units='xy',
                # width=0.1,
                # headwidth=15,
                # headlength=10,
                linestyle='dashed',
                color=(*color[:3], 0.1),
                zorder=0,
            )

    pond_circle = mpl.patches.Circle(
        pond_center,
        physics.pond_radius,
        facecolor=(0.0, 0.0, 1.0, 0.3),
        edgecolor='blue',
        fill=True,
    )

    axis.add_patch(pond_circle)

    axis.grid(True, linestyle='-', linewidth=0.2)

    heatmap_path = os.path.join(
        heatmap_dir,
        f'{evaluation_type}-iteration-{iteration:05}-heatmap.png')
    plt.savefig(heatmap_path)
    figure.clf()
    plt.close(figure)

    return infos


def bridge_move_after_bridge_infos(physics, paths):
    xy = np.concatenate(tuple(itertools.chain(*[
        [
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2]
        ]
        for path in paths
    ])))

    (reward_bounds_x_low,
     reward_bounds_x_high,
     reward_bounds_y_low,
     reward_bounds_y_high) = physics.reward_bounds()

    where_in_reward_bounds = np.flatnonzero(np.logical_and.reduce((
        reward_bounds_x_low <= xy[:, 0],
        xy[:, 0] <= reward_bounds_x_high,
        reward_bounds_y_low <= xy[:, 1],
        xy[:, 1] <= reward_bounds_y_high)))

    if 0 < where_in_reward_bounds.size:
        min_x = np.min(xy[:, 0][where_in_reward_bounds])
        max_x = np.max(xy[:, 0][where_in_reward_bounds])
        min_y = np.min(xy[:, 1][where_in_reward_bounds])
        max_y = np.max(xy[:, 1][where_in_reward_bounds])
        ptp_x = max_x - min_x
        ptp_y = max_y - min_y
        try:
            convex_hull = ConvexHull(xy[where_in_reward_bounds])
            convex_hull_area = convex_hull.area
            convex_hull_volume = convex_hull.volume
        except qhull.QhullError:
            # ConvexHull fails when all the observations are along single axis.
            convex_hull_area = convex_hull_volume = 0.0
            convex_hull = None
    else:
        min_x = max_x = min_y = max_y = ptp_x = ptp_y = 0.0
        convex_hull_area = convex_hull_volume = 0.0
        convex_hull = None

    infos = {
        'after-bridge-min_x': min_x,
        'after-bridge-max_x': max_x,
        'after-bridge-min_y': min_y,
        'after-bridge-max_y': max_y,
        'after-bridge-ptp_x': ptp_x,
        'after-bridge-ptp_y': ptp_y,
        'after-bridge-convex_hull_area': convex_hull_area,
        'after-bridge-convex_hull_volume': convex_hull_volume,
    }

    return infos


def get_path_infos_bridge_move(physics,
                               paths,
                               evaluation_type='training',
                               figure_save_path=None):
    infos = bridge_move_after_bridge_infos(physics, paths)

    x, y = np.split(np.concatenate(tuple(itertools.chain(*[
        [
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2]
        ]
        for path in paths
    ]))), 2, axis=-1)

    water_left_pos = physics.named.model.geom_pos['water-left']
    water_right_pos = physics.named.model.geom_pos['water-right']
    water_left_size = physics.named.model.geom_size['water-left']
    water_right_size = physics.named.model.geom_size['water-right']

    (reward_bounds_x_low,
     reward_bounds_x_high,
     reward_bounds_y_low,
     reward_bounds_y_high) = physics.reward_bounds()

    bridge_x_low = (
        physics.named.model.geom_pos['bridge'][0]
        - physics.named.model.geom_size['bridge'][0])
    x_margin = y_margin = physics.named.model.geom_pos['bridge'][0] / 10
    xlim = np.array((
        bridge_x_low - x_margin, reward_bounds_x_high + x_margin))
    ylim = np.array((
        reward_bounds_y_low - y_margin, reward_bounds_y_high + y_margin))

    base_size = 12.8
    figure_width = reward_bounds_x_high - reward_bounds_x_low
    figure_height = reward_bounds_y_high - reward_bounds_y_low
    if figure_width > figure_height:
        figsize = (base_size, base_size * (figure_height / figure_width))
    else:
        figsize = (base_size * (figure_width / figure_height), base_size)

    figure, axis = plt.subplots(1, 1, figsize=figsize)

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    color_map = plt.cm.get_cmap('tab10', len(paths))
    for i, path in enumerate(paths):
        positions = np.concatenate((
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2],
        ), axis=0)

        color = color_map(i)
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linestyle=':',
            linewidth=1.0,
            label='evaluation_paths' if i == 0 else None,
        )
        axis.scatter(
            *positions[0, :2],
            edgecolors='black',
            c=[color],
            marker='o',
            s=75.0,
        )
        axis.scatter(
            *positions[-1, :2],
            edgecolors='black',
            c=[color],
            marker='X',
            s=90.0)

    water_left_rectangle = mpl.patches.Rectangle(
        xy=water_left_pos - water_left_size,
        width=water_left_size[0] * 2,
        height=water_left_size[1] * 2,
        fill=True)
    water_right_rectangle = mpl.patches.Rectangle(
        xy=water_right_pos - water_right_size,
        width=water_right_size[0] * 2,
        height=water_right_size[1] * 2,
        fill=True)

    water_patch_collection = mpl.collections.PatchCollection(
        (water_left_rectangle, water_right_rectangle),
        facecolor=(0.0, 0.0, 1.0, 0.3),
        edgecolor='blue')

    axis.add_collection(water_patch_collection)

    reward_bounds_rectangle = mpl.patches.Rectangle(
        xy=(reward_bounds_x_low, reward_bounds_y_low),
        width=(reward_bounds_x_high - reward_bounds_x_low),
        height=(reward_bounds_y_high - reward_bounds_y_low),
        facecolor='none',
        fill=False,
        edgecolor='black',
        linestyle=':')

    axis.add_patch(reward_bounds_rectangle)

    axis.grid(True, linestyle='-', linewidth=0.2)

    log_base_dir = (
        os.getcwd()
        if figure_save_path is None
        else figure_save_path)
    heatmap_dir = os.path.join(log_base_dir, 'heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    previous_heatmaps = glob.glob(
        os.path.join(heatmap_dir, f"{evaluation_type}-iteration-*-heatmap.png"))
    heatmap_iterations = [
        int(re.search(f"{evaluation_type}-iteration-(\d+)-heatmap.png", x).group(1))
        for x in previous_heatmaps
    ]
    if not heatmap_iterations:
        iteration = 0
    else:
        iteration = int(max(heatmap_iterations) + 1)

    heatmap_path = os.path.join(
        heatmap_dir,
        f'{evaluation_type}-iteration-{iteration:05}-heatmap.png')
    plt.savefig(heatmap_path)
    figure.clf()
    plt.close(figure)

    return infos
