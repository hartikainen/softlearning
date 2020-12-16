import collections

import numpy as np
from scipy.spatial.transform import Rotation
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.point_mass import (
    _DEFAULT_TIME_LIMIT,
    Physics as PointMassPhysics)

from softlearning.environments.dm_control.suite import bridge, visualization

from .common import make_model, KEY_GEOM_NAMES, SUITE


# 1.0 maximum, 0.2 for slow
DEFAULT_DESIRED_SPEED_ON_BRIDGE = 1.0
DEFAULT_DESIRED_SPEED_AFTER_BRIDGE = 1.0


@SUITE.add()
def bridge_run(*args, bridge_width=bridge.DEFAULT_BRIDGE_WIDTH, **kwargs):
    """Returns the BridgeRun task."""
    return tapering_bridge_run(
        *args,
        bridge_start_width=bridge_width,
        bridge_end_width=bridge_width,
        **kwargs)


@SUITE.add()
def tapering_bridge_run(time_limit=_DEFAULT_TIME_LIMIT,
                        random=None,
                        bridge_length=bridge.DEFAULT_BRIDGE_LENGTH,
                        bridge_start_width=bridge.DEFAULT_BRIDGE_WIDTH,
                        bridge_end_width=0.0,
                        bridge_offset=-1.0,
                        before_bridge_reward_type='constant',
                        before_bridge_reward_weight=1.0,
                        on_bridge_reward_type='x_velocity',
                        on_bridge_reward_weight=5.0,
                        after_bridge_reward_type='constant',
                        after_bridge_reward_weight=5.0,
                        desired_speed_before_bridge=DEFAULT_DESIRED_SPEED_ON_BRIDGE,
                        desired_speed_on_bridge=DEFAULT_DESIRED_SPEED_ON_BRIDGE,
                        desired_speed_after_bridge=DEFAULT_DESIRED_SPEED_AFTER_BRIDGE,
                        terminate_outside_of_reward_bounds=False,
                        randomize_initial_x_position=False,
                        environment_kwargs=None):
    """Returns the BridgeRun task."""
    environment_kwargs = environment_kwargs or {}
    base_model_string = make_model(walls_and_target=False)

    size_multiplier = 0.05

    water_map_length = 3 * size_multiplier
    water_map_width = 2 * size_multiplier
    water_map_dx = 0.5 * size_multiplier / 2
    water_map_dy = 0.5 * size_multiplier / 2

    xml_string = bridge.make_model(
        base_model_string,
        size_multiplier=size_multiplier,
        bridge_length=bridge_length * size_multiplier,
        bridge_start_width=bridge_start_width * size_multiplier,
        bridge_end_width=bridge_end_width * size_multiplier,
        bridge_offset=bridge_offset,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy,
    )
    physics = BridgeMovePhysics.from_xml_string(
        xml_string, common.ASSETS)
    task = BridgeMove(
        random=random,
        water_map_length=water_map_length,
        water_map_width=water_map_width,
        water_map_dx=water_map_dx,
        water_map_dy=water_map_dy,
        water_map_offset=(-water_map_length / 4, 0.0),
        before_bridge_reward_type=before_bridge_reward_type,
        before_bridge_reward_weight=before_bridge_reward_weight,
        on_bridge_reward_type=on_bridge_reward_type,
        on_bridge_reward_weight=on_bridge_reward_weight,
        after_bridge_reward_type=after_bridge_reward_type,
        after_bridge_reward_weight=after_bridge_reward_weight,
        desired_speed_before_bridge=desired_speed_before_bridge,
        desired_speed_on_bridge=desired_speed_on_bridge,
        desired_speed_after_bridge=desired_speed_after_bridge,
        terminate_outside_of_reward_bounds=terminate_outside_of_reward_bounds,
        randomize_initial_x_position=randomize_initial_x_position)

    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class BridgeMovePhysics(bridge.MovePhysicsMixin, PointMassPhysics):
    def __init__(self, *args,  **kwargs):
        self._agent_geom_ids = None
        self._fell_over_geom_ids = None
        return super(BridgeMovePhysics, self).__init__(*args, **kwargs)

    @property
    def agent_geom_ids(self):
        if self._agent_geom_ids is None:
            self._agent_geom_ids = np.array([
                self.model.name2id(geom_name, 'geom')
                for geom_name in KEY_GEOM_NAMES
            ])
        return self._agent_geom_ids

    @property
    def fell_over_geom_ids(self):
        return []

    def any_key_geom_in_water(self):
        key_geoms_xy = self.key_geom_positions()[:, :2]

        water_names = ('water-left', 'water-right')
        waters_xy = np.stack([
            self.named.model.geom_pos[water_name][:2]
            for water_name in water_names
        ], axis=0)
        waters_size = np.stack([
            self.named.model.geom_size[water_name][:2]
            for water_name in water_names
        ], axis=0)
        waters_angle = np.stack([
            Rotation.from_quat(
                np.roll(self.named.model.geom_quat[water_name], -1)
            ).as_euler('xyz')[-1:]
            for water_name in water_names
        ], axis=0)

        key_geoms_in_waters = bridge.point_inside_2d_rectangle(
            key_geoms_xy, waters_xy, waters_size, waters_angle)

        assert key_geoms_in_waters.shape == (
            *key_geoms_xy.shape[:-1], 2), (
                key_geoms_in_waters, key_geoms_in_waters.shape)

        any_key_geom_in_water = np.any(key_geoms_in_waters)

        return any_key_geom_in_water

    def torso_velocity(self):
        return self.velocity()

    def key_geom_positions(self):
        return self.named.data.xpos['pointmass'][np.newaxis, ...]

    def center_of_mass(self):
        return self.named.data.geom_xpos['pointmass']

    def torso_xmat(self):
        return self.named.data.geom_xmat['pointmass']

    def get_path_infos(self, *args, **kwargs):
        return visualization.get_path_infos_bridge_move(self, *args, **kwargs)

    def _get_orientation(self):
        return Rotation.from_euler('z', 0).as_quat()


class BridgeMove(bridge.MoveTaskMixin):
    def common_observations(self, physics):
        observation = collections.OrderedDict((
            ('position', physics.position()),
            ('velocity', physics.velocity()),
            # ('velocity', physics.velocity_to_pond()),
        ))
        return observation

    def upright_reward(self, physics):
        return 1.0

    def initialize_episode(self, physics):
        if self._randomize_initial_x_position:
            bridge_pos = physics.named.model.geom_pos['bridge']
            bridge_size = physics.named.model.geom_size['bridge']
            x_pos_max = (
                bridge_pos[0]
                + bridge_size[0]
                - np.atleast_1d(self._water_map_offset)[0])
            x_pos = np.random.uniform(0, x_pos_max)
        else:
            x_pos = 0.0

        physics.named.data.qpos['root_x'] = x_pos
        physics.named.data.qpos['root_y'] = 0.0
        physics.named.data.geom_xpos['pointmass'][:2] = (x_pos, 0.0)
        return super(BridgeMove, self).initialize_episode(physics)
