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


class AntRiverRunEnv(AntEnv):
    def __init__(self,
                 *args,
                 exclude_current_positions_from_observation=True,
                 river_y=-1.0,
                 **kwargs):
        utils.EzPickle.__init__(**locals())
        self.river_y = river_y
        return super(AntRiverRunEnv, self).__init__(
            *args,
            exclude_current_positions_from_observation=(
                exclude_current_positions_from_observation),
            **kwargs)

    @property
    def is_healthy(self):
        is_healthy = super(AntRiverRunEnv, self).is_healthy
        in_water = self.in_water(self.state_vector())
        is_healthy |= in_water
        return is_healthy

    def _get_obs(self, *args, **kwargs):
        observation = super(AntRiverRunEnv, self)._get_obs(*args, **kwargs)
        y = self.get_body_com("torso")[1].copy()
        observation = np.concatenate(([y], observation), axis=0)
        return observation

    def step(self, action):
        observation, reward, done, info = super(
            AntRiverRunEnv, self).step(action)
        assert observation.shape == (112, ), observation.shape
        y = observation[0]
        distance_from_river = np.abs(y - self.river_y)
        # reward_multiplier = - np.log(distance_from_river + 1e-1) + 1.5
        raw_reward = reward
        reward_multiplier = - 1.0 * distance_from_river + 2.0
        reward *= reward_multiplier

        info.update({
            'distance_from_river': np.abs(y - self.river_y),
            'in_water': self.in_water(self.state_vector()).item(),
            'raw_reward': raw_reward
        })

        return observation, reward, done, info

    def in_waters(self, states):
        in_waters = states[..., 1:2] < self.river_y
        return in_waters

    def in_water(self, state):
        in_water = self.in_waters(np.atleast_2d(state))[0]
        return in_water

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

        x_min, x_max = (-1.0, 200)
        y_min, y_max = (-1.0, 3.0)

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

        river_patch = mpl.patches.Rectangle(
            (-1.0, -1.0),
            200.0,
            1.0,
            facecolor=(0.0, 0.0, 1.0, 0.5),
            edgecolor='blue',
            fill=True,
        )

        axis.add_patch(river_patch)

        axis.grid(True, linestyle='-', linewidth=0.2)

        plt.savefig(figure_save_path)
        figure.clf()
        plt.close(figure)

        return infos
