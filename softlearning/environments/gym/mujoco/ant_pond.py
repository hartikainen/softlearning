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


class AntPondEnv(AntEnv):
    def __init__(self,
                 *args,
                 exclude_current_positions_from_observation=False,
                 pond_radius=1.0,
                 angular_velocity_max=float('inf'),
                 velocity_reward_weight=1.0,
                 **kwargs):
        utils.EzPickle.__init__(**locals())
        self.pond_radius = pond_radius
        self.angular_velocity_max = angular_velocity_max
        self.velocity_reward_weight = velocity_reward_weight
        self.pond_center = (-pond_radius - 3.0, 0)
        return super(AntPondEnv, self).__init__(
            *args,
            exclude_current_positions_from_observation=(
                exclude_current_positions_from_observation),
            **kwargs)

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        healthy_reward = self.healthy_reward

        angular_velocity = self.compute_angular_velocity(
            xy_position_after, xy_velocity)
        angular_velocity_reward = self.velocity_reward_weight * np.minimum(
            angular_velocity, self.angular_velocity_max)
        rewards = angular_velocity_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        distance_from_water = self.distance_from_pond_center(
            xy_position_after
        ) - self.pond_radius

        in_water = self.in_water(xy_position_after).item()

        reward = rewards - costs
        done = self.done or in_water
        observation = self._get_obs()
        info = {
            'angular_velocity': angular_velocity,
            'angular_velocity_reward': angular_velocity_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'distance_from_water': distance_from_water,
            'in_water': in_water,
        }

        return observation, reward, done, info

    def in_waters(self, states):
        in_waters = self.distances_from_pond_center(states) <= self.pond_radius
        return in_waters

    def in_water(self, state):
        in_water = self.in_waters(np.atleast_2d(state))[0]
        return in_water

    def distances_from_pond_center(self, states):
        states = np.atleast_2d(states)
        distances_from_pond_center = np.linalg.norm(
            states - self.pond_center, ord=2, keepdims=True, axis=-1)
        return distances_from_pond_center

    def distance_from_pond_center(self, state):
        distance_from_pond_center = self.distances_from_pond_center(
            np.atleast_2d(state)).item()
        return distance_from_pond_center

    def compute_angular_velocities(self, positions, velocities):
        positions = positions - velocities / 2
        positions_ = positions - self.pond_center

        r = np.linalg.norm(
            positions_,
            axis=-1,
            ord=2,
            keepdims=True)

        theta = (
            np.arctan2(velocities[..., 1], velocities[..., 0])
            - np.arctan2(positions_[..., 1], positions_[..., 0])
        )[..., np.newaxis]

        angular_velocities = (
            np.linalg.norm(velocities, ord=2, keepdims=True, axis=-1)
            * np.sin(theta)
            / (r / self.pond_radius))

        return angular_velocities

    def compute_angular_velocity(self, position, velocity):
        positions = np.atleast_2d(position)
        velocities = np.atleast_2d(velocity)
        angular_velocity = self.compute_angular_velocities(
            positions, velocities).item()
        return angular_velocity

    def get_path_infos(self, paths, evaluation_type='training'):
        infos = {}
        x, y = np.split(np.concatenate(tuple(itertools.chain(*[
            [
                path['observations']['observations'][:, :2],
                path['next_observations']['observations'][[-1], :2]
            ]
            for path in paths
        ]))), 2, axis=-1)

        histogram_margin = 5.0
        bins_per_unit = 5
        x_bounds = tuple(
            self.pond_center[0]
            + self.pond_radius * np.array((-1, 1))
            + histogram_margin * np.array((-1, 1)))
        y_bounds = tuple(
            self.pond_center[1]
            + self.pond_radius * np.array((-1, 1))
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

        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        XY = np.stack((X, Y), axis=-1)

        valid_margin = np.sqrt(2 * (1.0 / bins_per_unit) ** 2)
        valid_bins = np.logical_and.reduce((
            (self.pond_radius - valid_margin)
            <= np.linalg.norm(XY - self.pond_center, ord=2, axis=-1),
            np.linalg.norm(XY - self.pond_center, ord=2, axis=-1)
            < (self.pond_radius + 3.0),
        ))

        support_of_valid_bins = (np.sum(H[valid_bins] > 0) / (
            H[valid_bins].size))

        infos.update({'support': support_of_valid_bins})

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

        margin = 20
        x_min, x_max = (
            self.pond_center[0] - self.pond_radius - margin,
            self.pond_center[0] + self.pond_radius + margin + 3)
        y_min, y_max = (
            self.pond_center[1] - self.pond_radius - margin,
            self.pond_center[1] + self.pond_radius + margin)

        base_size = 6.4
        width = x_max - x_min
        height = y_max - y_min

        if width > height:
            figsize = (base_size, base_size * (height / width))
        else:
            figsize = (base_size * (width / height), base_size)

        figure, axis = plt.subplots(1, 1, figsize=figsize)

        axis.set_xlim((x_min, x_max))
        axis.set_ylim((y_min, y_max))

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

        pond_circle = mpl.patches.Circle(
            self.pond_center,
            self.pond_radius,
            facecolor=(0.0, 0.0, 1.0, 0.5),
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
