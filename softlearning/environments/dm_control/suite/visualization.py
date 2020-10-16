import glob
import itertools
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import qhull
from scipy.spatial.transform import Rotation
import tree

from .pond import compute_angular_deltas

from softlearning.models.utils import flatten_input_structure
from softlearning.policies.deterministic_policy import DeterministicPolicy
from collections import defaultdict


def get_path_infos_platform_jump(physics,
                                 paths,
                                 evaluation_type,
                                 figure_save_path=None):
    left_platform_pos = physics.named.model.geom_pos['left-foot-platform']
    right_platform_pos = physics.named.model.geom_pos['right-foot-platform']

    infos = {}

    feet_platform_difference = np.concatenate([
        path['observations']['feet_platform_difference'] for path in paths
    ], axis=0)

    feet_heights = feet_platform_difference[:, [2, 5]]
    feet_min_heights = np.min(feet_heights, axis=-1)

    infos.update((
        ('left_foot_height-mean', np.mean(feet_heights[:, 0])),
        ('right_foot_height-mean', np.mean(feet_heights[:, 1])),
        ('feet_min_height-mean', np.mean(feet_min_heights)),
    ))

    log_base_dir = (
        os.getcwd()
        if figure_save_path is None
        else figure_save_path)
    heatmap_dir = os.path.join(log_base_dir, 'heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    previous_heatmaps = glob.glob(os.path.join(
        glob.escape(heatmap_dir),
        f"{evaluation_type}-iteration-*-heatmap.png"))
    heatmap_iterations = [
        int(re.search(f"{evaluation_type}-iteration-(\d+)-heatmap.png", x).group(1))
        for x in previous_heatmaps
    ]
    if not heatmap_iterations:
        iteration = 0
    else:
        iteration = int(max(heatmap_iterations) + 1)

    def round_to_most_significant_digit(x):
        return round(x, -int(np.floor(np.log10(abs(x)))))

    x_low, x_high = 0, 250
    xlim = np.array((x_low, x_high))
    y_low = -1.0 * round_to_most_significant_digit(
        physics.named.model.geom_size['left_left_foot'][0])
    y_high = 3.0
    ylim = np.array((y_low, y_high))

    base_size = 6.4
    # figure_width = reward_bounds_x_high - reward_bounds_x_low
    # figure_height = reward_bounds_y_high - reward_bounds_y_low
    figure_width = xlim[1] - xlim[0]
    figure_height = ylim[1] - ylim[0]
    if figure_width > figure_height:
        figsize = (base_size, base_size * (figure_height / figure_width))
    else:
        figsize = (base_size * (figure_width / figure_height), base_size)

    main_axis_height = figsize[1]
    figsize = (figsize[0], main_axis_height + base_size / 2)

    figure, axes = plt.subplots(
        (2 if not isinstance(physics.policy, DeterministicPolicy) else 1),
        1,
        figsize=figsize,
        gridspec_kw={
            # 'height_ratios': (
            #     [main_axis_height] + [base_size / 2]
            #     if not isinstance(physics.policy, DeterministicPolicy)
            #     else [main_axis_height]
            # ),
        })

    axes = np.atleast_1d(axes)
    axis = axes[0]

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    if len(axes) == 2:
        axes[1].set_xlim(xlim)

    if len(paths) <= 10:
        color_map = plt.cm.get_cmap('tab10')
    elif len(paths) <= 20:
        color_map = plt.cm.get_cmap('tab20')
    else:
        color_map = plt.cm.get_cmap('tab20', len(paths))

    for i, path in enumerate(paths):
        feet_heights = np.concatenate((
            path['observations']['feet_platform_difference'][..., [2, 5]],
            path['next_observations']['feet_platform_difference'][
                -1, ..., [2, 5]][None],
        ), axis=0)

        # feet_heights = feet_platform_differences[:, [2, 5]]
        feet_min_heights = np.min(feet_heights, axis=-1)

        head_heights = np.concatenate((
            path['observations']['head_height'][..., 0],
            path['next_observations']['head_height'][[-1], 0],
        ), axis=0)

        color = color_map(i)

        axis.plot(
            np.arange(head_heights.size),
            head_heights,
            color=(*color[:3], 0.3),
            linestyle='-',
            linewidth=0.5,
            marker='2',
        )
        axis.plot(
            np.arange(feet_heights[:, 0].size),
            feet_heights[:, 0],
            color=color,
            linestyle='-',
            linewidth=0.5,
            marker='3',
        )
        axis.plot(
            np.arange(feet_heights[:, 1].size),
            feet_heights[:, 1],
            color=color,
            linestyle='-',
            linewidth=0.5,
            marker='4',
        )

        if not isinstance(physics.policy, DeterministicPolicy):
            entropies = compute_entropies(
                physics.policy, path['observations'])
            axes[1].plot(
                np.arange(entropies.shape[0]),
                entropies,
                color=color,
                label=f'path-{i}')

    axis.grid(True, linestyle='-', linewidth=0.2)

    heatmap_path = os.path.join(
        heatmap_dir,
        f'{evaluation_type}-iteration-{iteration:05}-heatmap.pdf')
    plt.savefig(heatmap_path)
    figure.clf()
    plt.close(figure)

    return infos

def get_path_infos_stand(physics,
                         paths,
                         evaluation_type='training',
                         figure_save_path=None):
    infos = {}

    paths[0]['observations']['feet_platform_difference'][:, -1]
    x, y = np.split(np.concatenate(tuple(itertools.chain(*[
        [
            path['observations']['position'][..., :2],
            path['next_observations']['position'][[-1], ..., :2]
        ]
        for path in paths
    ]))), 2, axis=-1)

    if 'feet_velocity' in paths[0]['observations']:
        feet_velocities = np.concatenate([
            path['observations']['feet_velocity'] for path in paths
        ], axis=0)
        left_foot_velocities_xyz = feet_velocities[:, :3]
        right_foot_velocities_xyz = feet_velocities[:, 3:]

        left_foot_velocities = np.linalg.norm(
            left_foot_velocities_xyz, axis=1)
        right_foot_velocities = np.linalg.norm(
            right_foot_velocities_xyz, axis=1)

        left_foot_velocity_mean = np.mean(left_foot_velocities)
        right_foot_velocity_mean = np.mean(right_foot_velocities)

        feet_velocity_mean = np.mean(
            (left_foot_velocities + right_foot_velocities) / 2)

        infos.update((
            ('left_foot_velocity-mean', left_foot_velocity_mean),
            ('right_foot_velocity-mean', right_foot_velocity_mean),
            ('feet_velocity-mean', feet_velocity_mean),
        ))

    if 'feet_target_offset' in paths[0]['observations']:
        feet_target_offsets = np.concatenate([
            path['observations']['feet_target_offset'] for path in paths
        ], axis=0)

        non_zero_feet_target_offsets_mask = ~np.all(
            feet_target_offsets == 0, axis=1)
        left_foot_target_offsets, right_foot_target_offsets = np.split(
            feet_target_offsets[non_zero_feet_target_offsets_mask],
            [3],
            axis=-1)

        left_foot_target_distances = np.linalg.norm(
            left_foot_target_offsets, ord=2, axis=-1)
        right_foot_target_distances = np.linalg.norm(
            right_foot_target_offsets, ord=2, axis=-1)

        left_foot_target_distance_mean = np.mean(left_foot_target_distances)
        right_foot_target_distance_mean = np.mean(right_foot_target_distances)

        feet_target_distances_mean = np.mean((
            left_foot_target_distance_mean,
            right_foot_target_distance_mean))
        infos.update((
            ('left_foot_target_distance-mean',
             left_foot_target_distance_mean),
            ('right_foot_target_distance-mean',
             right_foot_target_distance_mean),
            ('feet_target_distance-mean', feet_target_distances_mean),
        ))

    results = defaultdict(list)
    info_keys = ('head_height', )
    for path in paths:
        for info_key in info_keys:
            info_values = np.array(path['observations'][info_key])
            results[info_key + '-first'].append(info_values[0])
            results[info_key + '-last'].append(info_values[-1])
            results[info_key + '-mean'].append(np.mean(info_values))
            results[info_key + '-median'].append(np.median(info_values))
            results[info_key + '-sum'].append(np.sum(info_values))
            if np.array(info_values).dtype != np.dtype('bool'):
                results[info_key + '-range'].append(np.ptp(info_values))

    aggregated_results = {}
    for key, value in results.items():
        aggregated_results[key + '-mean'] = np.mean(value)
        aggregated_results[key + '-median'] = np.median(value)
        aggregated_results[key + '-min'] = np.min(value)
        aggregated_results[key + '-max'] = np.max(value)

    infos.update(aggregated_results)

    origin = np.array((0.0, 0.0))
    distances_from_origin = np.linalg.norm(
        np.concatenate((x, y), axis=-1) - origin, ord=2, axis=1)

    infos.update([
        (f"{metric_name}-{metric_fn_name}",
         getattr(np, metric_fn_name)(metric_values))
        for metric_name, metric_values in (
                ('distances_from_origin', distances_from_origin),
        )
        for metric_fn_name in ('mean', 'max', 'min', 'median')
    ])

    log_base_dir = (
        os.getcwd()
        if figure_save_path is None
        else figure_save_path)
    heatmap_dir = os.path.join(log_base_dir, 'heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    previous_heatmaps = glob.glob(os.path.join(
        glob.escape(heatmap_dir),
        f"{evaluation_type}-iteration-*-heatmap.png"))
    heatmap_iterations = [
        int(re.search(f"{evaluation_type}-iteration-(\d+)-heatmap.png", x).group(1))
        for x in previous_heatmaps
    ]
    if not heatmap_iterations:
        iteration = 0
    else:
        iteration = int(max(heatmap_iterations) + 1)

    base_size = 6.4
    x_max = y_max = 4
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

    if len(paths) <= 10:
        color_map = plt.cm.get_cmap('tab10')
    elif len(paths) <= 20:
        color_map = plt.cm.get_cmap('tab20')
    else:
        color_map = plt.cm.get_cmap('tab20', len(paths))

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

    if 'feet_target_offset' in paths[0]['observations']:
        feet_target_offsets = np.concatenate([
            path['observations']['feet_target_offset'] for path in paths
        ], axis=0)
        feet_positions = np.concatenate([
            path['observations']['feet_position'] for path in paths
        ], axis=0)
        all_feet_target_positions = np.concatenate([
            path['observations']['feet_target_position'] for path in paths
        ], axis=0)
        non_zero_feet_target_offsets_mask = ~np.all(
            feet_target_offsets == 0, axis=1)
        freeze_start_mask = 1 + np.flatnonzero(
            np.diff(non_zero_feet_target_offsets_mask)
            & non_zero_feet_target_offsets_mask[1:])

        feet_target_positions = all_feet_target_positions[
            freeze_start_mask]

        axis.scatter(
            feet_target_positions[:, [0, 3]].reshape(-1),
            feet_target_positions[:, [1, 4]].reshape(-1),
            c=color_map(
                np.stack((
                    np.arange(feet_target_positions.shape[0]),
                    np.arange(feet_target_positions.shape[0]),
                )).T.reshape(-1),
                alpha=0.5),
            marker='*',
            s=90.0)

    axis.grid(True, linestyle='-', linewidth=0.2)

    heatmap_path = os.path.join(
        heatmap_dir,
        f'{evaluation_type}-iteration-{iteration:05}-heatmap.png')
    plt.savefig(heatmap_path)
    figure.clf()
    plt.close(figure)

    return infos


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
        for metric_fn_name in ('mean', 'max', 'min', 'median')
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
            for metric_fn_name in ('mean', 'max', 'min', 'median')
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
                    # ('velocities-z', velocities_xyz[:, 2]),
            )
            for metric_fn_name in ('mean', 'max', 'min', 'median')
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

    previous_heatmaps = glob.glob(os.path.join(
        glob.escape(heatmap_dir),
        f"{evaluation_type}-iteration-*-heatmap.png"))
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

    last_x_positions = [
        path['observations']['position'][-1, 0] for path in paths
    ]
    infos = {
        'x_position_last-min': np.min(last_x_positions),
        'x_position_last-max': np.max(last_x_positions),
        'x_position_last-mean': np.mean(last_x_positions),
        'x_position_last-median': np.median(last_x_positions),
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


def compute_entropies(policy, observations):
    num_actions = 100
    policy_inputs = flatten_input_structure({
        name: observations[name]
        for name in policy.observation_keys
    })
    tiled_policy_inputs = tree.map_structure(
        lambda x: np.reshape(
            np.tile(x[:, None, :], (1, num_actions, 1)),
            (-1, x.shape[-1])),
        policy_inputs)
    actions_flat = policy.actions_np(tiled_policy_inputs)
    log_pis_flat = policy.log_pis_np(
        tiled_policy_inputs, actions_flat)
    log_pis = np.reshape(
        log_pis_flat, (*tree.flatten(observations)[0].shape[:-1], num_actions))
    log_pis = np.mean(log_pis, axis=1)
    entropies = - 1.0 * log_pis
    return entropies


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
    water_left_size = physics.named.model.geom_size['water-left']
    water_left_quat = physics.named.model.geom_quat['water-left']
    water_left_euler = Rotation.from_quat(
        np.roll(water_left_quat, -1)).as_euler('xyz', degrees=True)
    np.testing.assert_equal(water_left_euler[:2], 0.0)
    water_left_angle = water_left_euler[-1]

    water_right_pos = physics.named.model.geom_pos['water-right']
    water_right_size = physics.named.model.geom_size['water-right']
    water_right_quat = physics.named.model.geom_quat['water-right']
    water_right_euler = Rotation.from_quat(
        np.roll(water_right_quat, -1)).as_euler('xyz', degrees=True)
    np.testing.assert_equal(water_right_euler[:2], 0.0)
    water_right_angle = water_right_euler[-1]

    (reward_bounds_x_low,
     reward_bounds_x_high,
     reward_bounds_y_low,
     reward_bounds_y_high) = physics.reward_bounds()

    bridge_x_low = (
        physics.named.model.geom_pos['bridge'][0]
        - physics.named.model.geom_size['bridge'][0])
    x_margin = y_margin = physics.named.model.geom_pos['bridge'][0] / 10

    x_low = min(bridge_x_low, 0.0) - x_margin
    xlim = np.array((x_low, reward_bounds_x_high + x_margin))
    ylim = np.array((
        reward_bounds_y_low - y_margin, reward_bounds_y_high + y_margin))

    base_size = 12.8
    # figure_width = reward_bounds_x_high - reward_bounds_x_low
    # figure_height = reward_bounds_y_high - reward_bounds_y_low
    figure_width = xlim[1] - xlim[0]
    figure_height = ylim[1] - ylim[0]
    if figure_width > figure_height:
        figsize = (base_size, base_size * (figure_height / figure_width))
    else:
        figsize = (base_size * (figure_width / figure_height), base_size)

    main_axis_height = figsize[1]
    figsize = (figsize[0], main_axis_height + base_size / 2)

    figure, axes = plt.subplots(
        (2 if not isinstance(physics.policy, DeterministicPolicy) else 1),
        1,
        figsize=figsize,
        gridspec_kw={
            'height_ratios': (
                [main_axis_height] + [base_size / 2]
                if not isinstance(physics.policy, DeterministicPolicy)
                else [main_axis_height]
            ),
        })

    axes = np.atleast_1d(axes)
    axis = axes[0]

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    if len(axes) == 2:
        axes[1].set_xlim(xlim)

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

        if not isinstance(physics.policy, DeterministicPolicy):
            entropies = compute_entropies(
                physics.policy, path['observations'])
            axes[1].plot(
                positions[:-1, 0],
                entropies,
                color=color_map(i),
                label=f'path-{i}')

    water_left_bottom_left_xy = water_left_pos + Rotation.from_quat(
        np.roll(water_left_quat, -1)).apply(
            (-1, -1, 1) * water_left_size)
    water_left_rectangle = mpl.patches.Rectangle(
        xy=water_left_bottom_left_xy[:2],
        width=water_left_size[0] * 2,
        height=water_left_size[1] * 2,
        angle=water_left_angle,
        fill=True)
    water_right_bottom_left_xy = water_right_pos + Rotation.from_quat(
        np.roll(water_right_quat, -1)).apply(
            (-1, -1, 1) * water_right_size)
    water_right_rectangle = mpl.patches.Rectangle(
        xy=water_right_bottom_left_xy,
        width=water_right_size[0] * 2,
        height=water_right_size[1] * 2,
        angle=water_right_angle,
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

    previous_heatmaps = glob.glob(os.path.join(
        glob.escape(heatmap_dir),
        f"{evaluation_type}-iteration-*-heatmap.png"))
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
