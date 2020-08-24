from pprint import pprint

from softlearning.environments.utils import get_environment
from softlearning.environments.dm_control.suite import quadruped
from softlearning.environments.dm_control.suite import boxhead
from softlearning.environments.dm_control import pond
from dm_control import suite
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
import numpy as np
import tree
from scipy.spatial.transform import Rotation


# env = dm_soccer.load(team_size=2, time_limit=10.0)

# env = suite.load(
#     domain_name="boxhead",
#     task_name="orbit_pond",
#     task_kwargs={
#         # 'friction': (0.001, 0.02, 0.02),
#         'roll_damping': 10.0
#     }
# )

# env = suite.load(domain_name="quadruped", task_name="bridge_run")
# env = suite.load(
#     # domain_name="quadruped",
#     domain_name="point_mass",
#     task_name="bridge_run",
#     task_kwargs={'bridge_length': 50.0},
# )

# env = suite.load(domain_name="point_mass", task_name="bridge_run")
env = suite.load(
    # domain_name="point_mass",
    # domain_name="quadruped",
    domain_name="humanoid",
    # task_name="bridge_run",
    # task_name="orbit_pond",
    task_name="tapering_bridge_run",
    # task_name="tapering_bridge_run",
    # task_name="run",
    task_kwargs={
        # 'make_1d': True,
        # 'lateral_control_magnitude': 3.0,
        # 'actuator_type': 'motor'
        # 'control_range_multiplier': 2.0,
        # 'upright_reward_type': '2-0'

        # humanoid default 0.7
        # quadruped default 1.5, should use 1.0?
        # 'friction': 1e-9,
        # 'bridge_width': 0.3,
        # 'bridge_width': 1.3,
        # 'bridge_length': 5.0,
        # 'randomize_initial_x_position': True,
        #
        'bridge_length': 10.0,
        'bridge_start_width': 0.5,
        'bridge_end_width': 0.5,
        # 'bridge_end_width': 0.0,
    },
)

# env = suite.load(domain_name="point_mass", task_name="easy")
# env = pond.load()

# breakpoint()

action_spec = env.action_spec()

# breakpoint()

step = 0


# Define a uniform random policy.
def random_policy(time_step):
    global step

    # del time_step  # Unused.
    # pprint(time_step.reward)
    def roundd(x):
        # try:
        #     return np.round(x, 2)
        # except Exception as e:
        #     breakpoint()
        #     pass
        pass

    # pprint(
    #     time_step.reward
    #     # tree.map_structure(
    #     #     # roundd,
    #     #     lambda x: np.round(x, 2) if x is not None else np.nan,
    #     #     {'rewards': time_step.reward, **time_step.observation})
    # )
    # print(time_step.observation['position'])

    # x-position at end = 2233 with defaults

    # orientation = time_step.observation['orientation']
    # orientation_to_pond = time_step.observation[
    #     'orientation_to_pond']

    # orientation_euler = Rotation.from_quat(
    #     np.roll(orientation, -1)).as_euler('xyz')
    # np.testing.assert_equal(orientation_euler[:2], 0)

    # orientation_to_pond_euler = Rotation.from_quat(
    #     np.roll(orientation_to_pond, -1)).as_euler('xyz')
    # np.testing.assert_equal(orientation_to_pond_euler[:2], 0)

    # step += 1
    # if step < 2:
    #     pprint(time_step.observation)
    #     print(f"orientation: {orientation_euler[-1]}, orientation_to_pond: {orientation_to_pond_euler[-1]}")

    # print(orientation_to_pond_euler[-1])
    # print(time_step.reward)
    # print(env.physics.velocity())
    # print(env.physics.torso_velocity())

    # if step < 100:
    #     return np.array((1.0, 0))
    # elif step < 175:
    #     return np.array((-1.0, 0))
    # elif step < 250:
    #     return np.array((0, 1.0))
    # else:
    #     return np.array((1.0, 0.0))

    # return np.array((0, 1.0))

    def random_action(action_spec):
        global step
        action = np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape)

        action = np.zeros_like(action)
        # action = (0.2, 0)
        # action = (1.0, - 20 * time_step.observation['position'][1] + np.random.uniform(-0.5, 0.5))
        # action = (1.0, np.clip(np.random.normal(0.0, 0.3), -1.0, 1.0))
        # print("action: ", action)

        # action = np.array((-0.02, 0.5)) + np.random.uniform([-0.5, 0.0], [0.5, 0.0])
        # action = (0.0, 0.3)
        # action = (0.3, 0.0, )
        # # action[-1] = 0.0
        step += 1
        # if step < 20:
        #     action = (1.0, 0)
        # else:
        #     action = np.zeros_like(action)

        # if 300 < step:
        #     breakpoint()
        #     action = (0.0, 0.0)

        # breakpoint()

        # if step == 300:
        #     breakpoint()
        #     pass

        # print(step)

        return action

    # print(step)
    # if step == 200:
    #     breakpoint()
    #     pass
    # pprint(tree.map_structure(lambda x: x.tolist(), time_step.observation[0]))
    # print(time_step.observation['position'])
    # breakpoint()

    # print(time_step.observation['orientation_to_pond'])
    # print(time_step.observation['velocity'])

    # print('position:', time_step.observation['position'])

    # if step == 10:
    #     breakpoint()
    #     pass

    # print([c for c in env.physics.data.contact])

    return tree.map_structure(random_action, action_spec)
    # return np.random.uniform(low=action_spec.minimum,
    #                          high=action_spec.maximum,
    #                          size=action_spec.shape)


# Launch the viewer application.

viewer.launch(env, policy=random_policy)
