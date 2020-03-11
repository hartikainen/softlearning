import glob
import itertools
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .pond import compute_angular_deltas


def get_path_infos_orbit_pond(physics,
                              paths,
                              evaluation_type='training',
                              figure_save_path=None):
    infos = {}

    x, y = np.split(np.concatenate(tuple(itertools.chain(*[
        [
            path['observations']['position'],
            path['next_observations']['position'][[-1]]
        ]
        for path in paths
    ]))), 2, axis=-1)

    def compute_cumulative_angle_travelled(path):
        xy = np.concatenate((
            path['observations']['position'],
            path['next_observations']['position'][[-1]]
        ), axis=0)
        angular_deltas = compute_angular_deltas(
            xy[:-1, ...], xy[1:, ...], center=physics.pond_center_xyz)
        cumulative_angle_travelled = np.sum(angular_deltas)
        return cumulative_angle_travelled

    cumulative_angles_travelled = np.array([
        compute_cumulative_angle_travelled(path)
        for path in paths
    ])

    velocities_xy = np.concatenate(tuple(itertools.chain(*[
        [
            path['observations']['velocity'],
            path['next_observations']['velocity'][[-1]]
        ]
        for path in paths
    ])))

    distances_from_water = np.linalg.norm(
        np.concatenate((x, y), axis=-1) - physics.pond_center_xyz[:2],
        ord=2,
        axis=1,
    ) - physics.pond_radius
    velocities = np.linalg.norm(velocities_xy, ord=2, axis=1)

    infos.update([
        (f"{metric_name}-{metric_fn_name}",
         getattr(np, metric_fn_name)(metric_values))
        for metric_name, metric_values in (
                ('distances_from_water', distances_from_water),
                ('velocities', velocities),
                ('cumulative_angles_travelled', cumulative_angles_travelled))
        for metric_fn_name in ('mean', 'min', 'mean')
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

    log_base_dir = os.getcwd()
    heatmap_dir = os.path.join(log_base_dir, 'heatmap')
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)

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
            path['observations']['position'],
            path['next_observations']['position'][[-1]],
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
            *positions[0],
            edgecolors='black',
            c=[color],
            marker='o',
            s=75.0,
        )
        axis.scatter(
            *positions[-1],
            edgecolors='black',
            c=[color],
            marker='X',
            s=90.0,
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


def get_path_infos_bridge_move(physics,
                               paths,
                               evaluation_type='training',
                               figure_save_path=None):
    infos = {}
    return infos
