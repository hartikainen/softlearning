import glob
import itertools
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import numpy as np
from gym import utils
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv, mass_center
from scipy.spatial.transform import Rotation


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


class HumanoidPondEnv(HumanoidEnv):
    def __init__(self,
                 *args,
                 exclude_current_positions_from_observation=False,
                 pond_radius=1.0,
                 angular_velocity_max=float('inf'),
                 velocity_reward_weight=1.0,
                 experimental_angular_velocity_type=1,
                 reset_distance_range=(0.5, 2.0),
                 **kwargs):
        utils.EzPickle.__init__(**locals())
        self.pond_radius = pond_radius
        self.angular_velocity_max = angular_velocity_max
        self.velocity_reward_weight = velocity_reward_weight
        self.pond_center = (-pond_radius - 3.0, 0)
        self.cumulative_angle_travelled = 0.0
        self.cumulative_angular_velocity = 0.0
        self.experimental_angular_velocity_type = (
            experimental_angular_velocity_type)
        self._reset_distance_range = reset_distance_range
        result = super(HumanoidPondEnv, self).__init__(
            *args,
            exclude_current_positions_from_observation=(
                exclude_current_positions_from_observation),
            **kwargs)

        orientation = self.init_qpos[3:7]
        rotate_90_degree_quaternion = np.roll(
            Rotation.from_euler('z', 90, degrees=True).as_quat(), 1)
        new_orientation = quaternion_multiply(
            rotate_90_degree_quaternion, orientation)
        self.init_qpos[3:7] = new_orientation

        return result

    def reset_model(self, *args, **kwargs):
        self.cumulative_angle_travelled = 0.0
        self.cumulative_angular_velocity = 0.0

        rotate_90_degree_quaternion = np.roll(
            Rotation.from_euler('z', 90, degrees=True).as_quat(), 1)
        default_quaternion = quaternion_multiply(
            rotate_90_degree_quaternion, np.array([1., 0., 0., 0.]))

        random_angle = np.random.uniform(0, 2 * np.pi)
        rotate_by_angle_quaternion = np.roll(
            Rotation.from_euler('z', random_angle).as_quat(), 1)

        self.init_qpos[3:7] = quaternion_multiply(
            rotate_by_angle_quaternion, default_quaternion)

        random_distance = np.random.uniform(*self._reset_distance_range)

        xy = ((self.pond_radius + random_distance) * np.array((
            np.cos(random_angle),
            np.sin(random_angle),
        ))) + self.pond_center
        self.init_qpos[:2] = xy

        return super(HumanoidPondEnv, self).reset_model(*args, **kwargs)

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat.copy()
        qvel = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        # free joint = qpos[0:7]
        xyz, rotation = qpos[:3], qpos[3:7]

        xy_from_pond_center = xyz[:2] - self.pond_center
        angle_to_pond_center = np.arctan2(*xy_from_pond_center[::-1])
        pond_quaternion = np.roll(
            Rotation.from_euler('z', angle_to_pond_center).inv().as_quat(), 1)

        new_quaternion = quaternion_multiply(pond_quaternion, rotation)

        new_quaternion[-1] = np.abs(new_quaternion[-1])

        distance_from_water = self.distance_from_pond_center(
            xyz[:2]
        ) - self.pond_radius

        observation = np.concatenate((
            np.array(distance_from_water)[np.newaxis],
            np.array(qpos[2])[np.newaxis],
            new_quaternion,
            qpos[7:],
            qvel,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces))

        return observation

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        angle_before = np.arctan2(
            *(xy_position_before - self.pond_center)[::-1])
        angle_after = np.arctan2(
            *(xy_position_after - self.pond_center)[::-1])
        angle_travelled = np.arctan2(
            np.sin(angle_after - angle_before),
            np.cos(angle_after - angle_before))

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

        self.cumulative_angle_travelled += angle_travelled
        self.cumulative_angular_velocity += angular_velocity

        reward = rewards - costs
        done = self.done or in_water
        observation = self._get_obs()
        info = {
            'angular_velocity': angular_velocity,
            'angular_velocity_reward': angular_velocity_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'is_healthy': self.is_healthy,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'distance_from_water': distance_from_water,
            'in_water': in_water,

            'cumulative_angle_travelled': self.cumulative_angle_travelled,
            'cumulative_angular_velocity': self.cumulative_angular_velocity,
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

    def compute_angular_velocities_1(self, positions, velocities):
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

    def compute_angular_velocities_2(self, positions, velocities):
        positions1 = positions - velocities - self.pond_center
        positions2 = positions - self.pond_center
        angles1 = np.arctan2(positions1[..., 1], positions1[..., 0])
        angles2 = np.arctan2(positions2[..., 1], positions2[..., 0])
        angles = np.arctan2(
            np.sin(angles2 - angles1),
            np.cos(angles2 - angles1)
        )[..., np.newaxis]

        angular_velocities = angles * self.pond_radius

        return angular_velocities

    def compute_angular_velocities_3(self, positions, velocities):
        positions1 = positions - velocities / 2 - self.pond_center
        positions2 = positions + velocities / 2 - self.pond_center
        angles1 = np.arctan2(positions1[..., 1], positions1[..., 0])
        angles2 = np.arctan2(positions2[..., 1], positions2[..., 0])
        angles = np.arctan2(
            np.sin(angles2 - angles1),
            np.cos(angles2 - angles1)
        )[..., np.newaxis]

        angular_velocities = angles * self.pond_radius

        return angular_velocities

    def compute_angular_velocities(self, positions, velocities):
        if self.experimental_angular_velocity_type == 1:
            return self.compute_angular_velocities_1(positions, velocities)
        elif self.experimental_angular_velocity_type == 2:
            return self.compute_angular_velocities_2(positions, velocities)
        elif self.experimental_angular_velocity_type == 3:
            return self.compute_angular_velocities_3(positions, velocities)

        raise NotImplementedError(self.experimental_angular_velocity_type)

    def compute_angular_velocity(self, position, velocity):
        positions = np.atleast_2d(position)
        velocities = np.atleast_2d(velocity)
        angular_velocity = self.compute_angular_velocities(
            positions, velocities).item()
        return angular_velocity

    def get_path_infos(self,
                       paths,
                       evaluation_type='training',
                       figure_save_path=None):
        infos = {}
        x, y = np.split(np.concatenate(tuple(itertools.chain(*[
            [
                np.concatenate((
                    np.array(path['infos']['x_position'])[..., np.newaxis],
                    np.array(path['infos']['y_position'])[..., np.newaxis],
                ), axis=-1),
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

        if figure_save_path is None:
            log_base_dir = os.getcwd()
            figure_save_dir = os.path.join(log_base_dir, 'figures')
            if not os.path.exists(figure_save_dir):
                os.makedirs(figure_save_dir)

            previous_figure_saves = glob.glob(
                os.path.join(figure_save_dir, f"{evaluation_type}-iteration-*.png"))
            figure_save_iterations = [
                int(re.search(f"{evaluation_type}-iteration-(\d+).png", x).group(1))
                for x in previous_figure_saves
            ]
            if not figure_save_iterations:
                iteration = 0
            else:
                iteration = int(max(figure_save_iterations) + 1)

            figure_save_path = os.path.join(
                figure_save_dir,
                f'{evaluation_type}-iteration-{iteration:05}.png')

        margin = 20
        x_min, x_max = (
            self.pond_center[0] - self.pond_radius - margin,
            self.pond_center[0] + self.pond_radius + margin + 3)
        y_min, y_max = (
            self.pond_center[1] - self.pond_radius - margin,
            self.pond_center[1] + self.pond_radius + margin)

        base_size = 12.8
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
                np.array(path['infos']['x_position'])[..., np.newaxis],
                np.array(path['infos']['y_position'])[..., np.newaxis],
            ), axis=-1)
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

            if 'perturbed' in path['infos']:
                perturbed = np.array(path['infos']['perturbed'])
                perturbations = np.stack(
                    path['infos']['perturbation'], axis=0)[perturbed]
                # perturbations /= np.linalg.norm(
                #     perturbations, ord=2, keepdims=True, axis=-1)

                perturbed_xy = np.stack((
                    path['infos']['x_position'],
                    path['infos']['y_position']
                ), axis=-1)[perturbed]

                axis.quiver(
                    perturbed_xy[:, 0],
                    perturbed_xy[:, 1],
                    perturbations[:, 0],
                    perturbations[:, 1],
                    units='xy',
                    angles='xy',
                    scale=1.0,
                    scale_units='xy',
                    width=0.1,
                    headwidth=15,
                    headlength=10,
                    linestyle='dashed',
                    color=color,
                    zorder=0,
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

        plt.savefig(figure_save_path)
        figure.clf()
        plt.close(figure)

        return infos
