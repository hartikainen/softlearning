import glob
import itertools
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import numpy as np
from gym import utils
from gym.envs.mujoco.ant_v3 import AntEnv


class AntBridgeRunEnv(AntEnv):
    def __init__(self,
                 *args,
                 exclude_current_positions_from_observation=False,
                 forward_reward_weight=3.0,
                 after_bridge_reward=20.0,
                 bridge_length=10.0,
                 bridge_width=2.0,
                 **kwargs):
        utils.EzPickle.__init__(**locals())
        self.bridge_length = bridge_length
        self.bridge_width = bridge_width
        self.forward_reward_weight = forward_reward_weight
        self.after_bridge_reward = after_bridge_reward
        return super(AntBridgeRunEnv, self).__init__(
            *args,
            exclude_current_positions_from_observation=(
                exclude_current_positions_from_observation),
            **kwargs)

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        x_position, y_position = xy_position_after

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        observation = self._get_obs()

        fell_off_the_bridge = (
            x_position < self.bridge_length
            and self.bridge_width / 2 <= np.abs(y_position))

        after_bridge = self.bridge_length <= x_position

        if after_bridge:
            reward = self.after_bridge_reward

        done = self.done or fell_off_the_bridge

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,

            'after_bridge': after_bridge,
            'fell_off_the_bridge': fell_off_the_bridge
        }

        return observation, reward, done, info

    def get_path_infos(self, paths, evaluation_type='training'):
        infos = {}
        x, y = np.split(np.concatenate(tuple(itertools.chain(*[
            [
                path['observations']['observations'][:, :2],
                path['next_observations']['observations'][[-1], :2]
            ]
            for path in paths
        ]))), 2, axis=-1)
        bins_per_unit = 2
        x_bounds = np.array((self.bridge_length, self.bridge_length + 20.0))
        y_bounds = np.array((-20, 20.0))

        where_past_bridge = np.flatnonzero(np.logical_and.reduce((
            x_bounds[0] <= x,
            x <= x_bounds[1],
            y_bounds[0] <= y,
            y <= y_bounds[1])))

        if 0 < where_past_bridge.size:
            min_x = np.min(x[where_past_bridge])
            max_x = np.max(x[where_past_bridge])
            min_y = np.min(y[where_past_bridge])
            max_y = np.max(y[where_past_bridge])
            ptp_x = max_x - min_x
            ptp_y = max_y - min_y
            rectangle_area = ptp_x * ptp_y
            rectangle_support = rectangle_area / (
                np.ptp(x_bounds) * np.ptp(y_bounds))
            rectangle_x_support = ptp_x / np.ptp(x_bounds)
            rectangle_y_support = ptp_y / np.ptp(y_bounds)
        else:
            min_x = max_x = min_y = max_y = ptp_x = ptp_y = 0.0
            rectangle_area = rectangle_support = 0.0
            rectangle_x_support = rectangle_y_support = 0.0

        H, xedges, yedges = np.histogram2d(
            np.squeeze(x),
            np.squeeze(y),
            bins=(
                int(np.ptp(x_bounds) * bins_per_unit),
                int(np.ptp(y_bounds) * bins_per_unit),
            ),
            range=np.array((x_bounds, y_bounds)),
        )

        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        XY = np.stack((X, Y), axis=-1)

        valid_margin = np.sqrt(2 * (1.0 / bins_per_unit) ** 2)
        valid_bins = np.logical_and(
            0 <= XY[..., 0] - x_bounds[0],
            np.linalg.norm(XY - np.array((x_bounds[0], 0.0)), ord=2, axis=2)
            < np.ptp(x_bounds) / 2 + valid_margin
        )

        support_of_circle_bins = (0 < np.sum(H[valid_bins]) / (
            H[valid_bins].size))

        histogram_support = np.sum(H > 0) / H.size
        H_x = np.sum(H, axis=1)
        H_y = np.sum(H, axis=0)
        histogram_x_support = np.sum(H_x > 0) / H_x.size
        histogram_y_support = np.sum(H_y > 0) / H_y.size

        infos.update({
            'after-bridge-min_x': min_x,
            'after-bridge-max_x': max_x,
            'after-bridge-min_y': min_y,
            'after-bridge-max_y': max_y,
            'after-bridge-ptp_x': ptp_x,
            'after-bridge-ptp_y': ptp_y,
            'after-bridge-circle-support': support_of_circle_bins,
            'after-bridge-histogram_support': histogram_support,
            'after-bridge-histogram_x_support': histogram_x_support,
            'after-bridge-histogram_y_support': histogram_y_support,
            'after-bridge-rectangle_area': rectangle_area,
            'after-bridge-rectangle_support': rectangle_support,
            'after-bridge-rectangle_x_support': rectangle_x_support,
            'after-bridge-rectangle_y_support': rectangle_y_support,
        })

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

        base_size = 6.4
        x_min, x_max = np.array((0, self.bridge_length + 20))
        y_min, y_max = np.array((-1, 1.0)) * (20 + self.bridge_width)
        width = x_max - x_min
        height = y_max - y_min

        if width > height:
            figsize = (base_size, base_size * (height / width))
        else:
            figsize = (base_size * (width / height), base_size)

        figure, axis = plt.subplots(1, 1, figsize=figsize)

        axis.set_xlim((0, self.bridge_length + 20.0))
        axis.set_ylim(
            (-(20 + self.bridge_width), 20.0 + self.bridge_width))

        color_map = plt.cm.get_cmap('PuBuGn', len(paths))
        for i, path in enumerate(paths):
            positions = np.concatenate((
                path['observations']['observations'][:, :2],
                path['next_observations']['observations'][[-1], :2],
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
                color=color,
                marker='o',
                s=20.0,
            )
            axis.scatter(
                *positions[-1],
                color=color,
                marker='x',
                s=20.0,
            )

        wall_patch_collection = mpl.collections.PatchCollection(
            [
                mpl.patches.Rectangle(
                    (0, self.bridge_width / 2),
                    self.bridge_length,
                    20.0),
                mpl.patches.Rectangle(
                    (0, - self.bridge_width / 2 - 20),
                    self.bridge_length,
                    20.0),
            ],
            facecolor='blue',
            edgecolor=None)

        axis.add_collection(wall_patch_collection)

        axis.grid(True, linestyle='-', linewidth=0.2)

        heatmap_path = os.path.join(
            heatmap_dir,
            f'{evaluation_type}-iteration-{iteration:05}-heatmap.png')
        plt.savefig(heatmap_path)
        figure.clf()
        plt.close(figure)

        return infos
