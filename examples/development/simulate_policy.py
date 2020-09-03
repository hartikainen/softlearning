import argparse
from distutils.util import strtobool
import json
import os
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import tree
import pandas as pd

from softlearning.environments.utils import (
    get_environment_from_params, get_environment)
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.utils.video import save_video
from softlearning.utils.misc import set_seed


DEFAULT_RENDER_KWARGS = {
    'mode': 'human',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-kwargs', '-r',
                        type=json.loads,
                        default='{}',
                        help="Kwargs for rollouts renderer.")
    parser.add_argument('--evaluation-environment-kwargs',
                        type=json.loads,
                        default='{}',
                        help="Arguments passed to the evaluation environment.")
    parser.add_argument('--video-save-path',
                        type=Path,
                        default=None)
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args


def load_checkpoint(checkpoint_path, session=None):
    session = session or tf.compat.v1.keras.backend.get_session()
    checkpoint_path = checkpoint_path.rstrip('/')
    trial_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(trial_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    metadata_path = os.path.join(checkpoint_path, ".tune_metadata")
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = None

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    progress_path = os.path.join(trial_path, 'progress.csv')
    progress = pd.read_csv(progress_path)

    return picklable, variant, progress, metadata


def load_policy_and_environment(picklable, variant):
    environment_params = (
        variant['environment_params']['training']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    environment = get_environment_from_params(environment_params)

    policy = get_policy_from_variant(variant, environment)
    policy.set_weights(picklable['policy_weights'])

    return policy, environment


def simulate_policy_dm_control(policy,
                               environment_params,
                               checkpoint_path,
                               deterministic,
                               num_rollouts,
                               max_path_length,
                               render_kwargs,
                               video_save_path=None,
                               evaluation_environment_kwargs=None):
    universe = environment_params['universe']
    assert universe == 'dm_control'
    domain, task = environment_params['domain'], environment_params['task']

    environment = get_environment(
        universe,
        domain,
        task,
        {
            **environment_params['kwargs'],
            # 'perturb_body_kwargs': {
            #     'perturbation_strength': 500.0,
            #     'perturbation_direction': {
            #         "type": "towards",
            #         "target": "pond_center"
            #     },
            #     'perturbation_probability': None,
            #     'perturbation_frequency': 200,
            #     'perturbation_length': 1,
            # },
            # 'wind_kwargs': {
            #     'wind_strength': 10,
            #     'wind_direction': {
            #         "type": "towards",
            #         "target": "pond_center"
            #     },
            # },
        },
    )

    def policy_fn(time_step):
        observation = time_step.observation
        observation = tree.map_structure(lambda x: np.ravel(x), observation)
        observations = tree.map_structure(lambda x: x[None, ...], observation)
        actions = policy.actions_np(observations)
        action = tree.map_structure(lambda x: x[0], actions)
        # log_pis_np = policy.log_pis_np(observations, actions)
        # print(log_pis_np)
        return action

    if render_kwargs['mode'] == 'human':
        from dm_control import viewer
        viewer.launch(environment._env, policy=policy_fn)

    with policy.set_deterministic(deterministic):
        paths = rollouts(num_rollouts,
                         environment,
                         policy,
                         path_length=max_path_length,
                         render_kwargs=render_kwargs)

    successes = [
        max_path_length == path['terminals'].shape[0]
        for path in paths
    ]
    num_success = sum(successes)
    print(f"success probability: {num_success / len(paths)}")


def simulate_policy(checkpoint_path,
                    deterministic,
                    num_rollouts,
                    max_path_length,
                    render_kwargs,
                    video_save_path=None,
                    evaluation_environment_kwargs=None):
    set_seed(np.random.randint(0, 100000))

    checkpoint_path = checkpoint_path.rstrip('/')
    trial_path = os.path.dirname(checkpoint_path)
    simulate_policy_path = os.path.join(trial_path, 'simulate_policy')
    os.makedirs(simulate_policy_path, exist_ok=True)
    os.chdir(simulate_policy_path)

    picklable, variant, progress, metadata = load_checkpoint(checkpoint_path)
    policy, environment = load_policy_and_environment(picklable, variant)
    render_kwargs = {**DEFAULT_RENDER_KWARGS, **render_kwargs}

    environment_params = variant['environment_params']['training']
    universe = environment_params['universe']

    if universe == 'dm_control':
        return simulate_policy_dm_control(
            policy,
            environment_params,
            checkpoint_path,
            deterministic,
            num_rollouts,
            max_path_length,
            render_kwargs,
            video_save_path=video_save_path,
            evaluation_environment_kwargs=evaluation_environment_kwargs)

    domain, task = environment_params['domain'], environment_params['task']

    # assert domain in ('Humanoid', 'Hopper', 'Walker2d'), domain
    # assert task in ('MaxVelocity-v3', 'Standup-v2', 'Stand-v3', 'SimpleStand-v3', 'v3'), task

    if task == 'MaxVelocity-v3':
        environment_params['kwargs'].pop('max_velocity')


    # assert not environment_params['kwargs'], environment_params['kwargs']
    environment = get_environment(
        'gym',
        domain,
        task,
        {
            **environment_params['kwargs'],
            **evaluation_environment_kwargs,
            # 'perturb_random_action_kwargs': {
            #     'perturbation_probability': perturbation_probability
            # },
            # 'wind_kwargs': {
            #     'wind_strength': perturbation_strength
            # }
            # 'perturb_body_kwargs': {
            #     'perturbation_strength': perturbation_strength,
            #     'perturbation_length': 5,
            # },
            # 'perturb_body_kwargs': {
            #     'perturbation_strength': perturbation_strength,
            #     'perturbation_direction': (1.0, 0.0, 0.0),
            #     'perturbation_probability': perturbation_probability,
            #     # 'perturbation_probability': 0.5, # 1/10 @ 100, 0/20 @ 125
            #     # 'perturbation_probability': 0.25, # 2/10 @ 100, 4/10 @ 125
            #     # 'perturbation_probability': 0.125, # 7/10 @ 100, 4/10 @ 125
            #     'perturbation_frequency': None,
            #     'perturbation_length': 1,
            # }
        }
    )

    with policy.set_deterministic(deterministic):
        paths = rollouts(num_rollouts,
                         environment,
                         policy,
                         path_length=max_path_length,
                         render_kwargs=render_kwargs)


    # if 'Point2D' in environment.unwrapped.__class__.__name__:
    if hasattr(environment, 'get_path_infos'):
        infos = environment.get_path_infos(paths)

    else:
        num_paths = len(paths)
        if environment.unwrapped.__class__.__name__ == 'AntPondEnv':
            in_water = sum(int(path['infos']['in_water'][-1])
                           for path in paths)
            num_survived = sum(
                int(max_path_length <= path['terminals'].shape[0])
                for path in paths)
            result = (
                # f"perturbation_strength: {perturbation_strength}, "
                f"in_water: {in_water}, "
                f"num_survived: {num_survived}/{num_paths}")
        elif 'is_healthy' in paths[0]['infos']:
            num_survived = sum(int(path['infos']['is_healthy'][-1]) for path in paths)
            result = (
                # f"perturbation_strength: {perturbation_strength}, "
                # f"perturbation_probability: {perturbation_probability}, "
                f"num_survived: {num_survived}/{num_paths}")
        else:
            raise NotImplementedError

        # with open("/tmp/robustness-sweep-results.txt", "a") as f:
        #     f.write(result + '\n')

        print(result)
    # if video_save_path and render_kwargs.get('mode') == 'rgb_array':
    #     fps = 1 // getattr(environment, 'dt', 1/30)
    #     for i, path in enumerate(paths):
    #         video_save_dir = os.path.expanduser('/tmp/simulate_policy/')
    #         video_save_path = os.path.join(video_save_dir, f'episode_{i}.mp4')
    #         save_video(path['images'], video_save_path, fps=fps)

    return paths


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(session)

    args = parse_args()
    simulate_policy(**vars(args))
