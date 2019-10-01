import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf
import pandas as pd

from softlearning.environments.utils import (
    get_environment_from_params, get_environment)
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.utils.video import save_video


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
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args


def load_checkpoint(checkpoint_path, session=None):
    session = session or tf.keras.backend.get_session()
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


def simulate_policy(checkpoint_path,
                    deterministic,
                    num_rollouts,
                    max_path_length,
                    render_kwargs):
    checkpoint_path = checkpoint_path.rstrip('/')
    picklable, variant, progress, metadata = load_checkpoint(checkpoint_path)
    policy, environment = load_policy_and_environment(picklable, variant)
    render_kwargs = {**DEFAULT_RENDER_KWARGS, **render_kwargs}

    environment_params = variant['environment_params']['training']
    domain, task = environment_params['domain'], environment_params['task']

    assert domain in ('Humanoid', 'Hopper', 'Walker2d'), domain
    assert task in ('MaxVelocity-v3', 'Standup-v2', 'Stand-v3', 'v3'), task

    if task == 'MaxVelocity-v3':
        environment_params['kwargs'].pop('max_velocity')

    # (TODO):
    # More granular perturations
    # Make z to be 0-centered

    # environment = get_environment(
    #     'gym', domain, 'v3', {
    #         **environment_params['kwargs'],
    #     })

    # environment = get_environment(
    #     'gym', domain, 'HeightField-v0', {
    #         **environment_params['kwargs'],
    #         'field_z_range': (0, 0.25),
    #         # 'healthy_z_range': (1.0 - pothole_depth, 2.0 + pothole_depth)
    #     })

    # environment = get_environment(
    #     'gym', domain, 'Pothole-v0', {
    #         **environment_params['kwargs'],
    #         'pothole_depth': 1.0,
    #         # 'pothole_length': 0.25,
    #         # 'pothole_distance': 5.0,
    #         # 'healthy_z_range': (1.0 - pothole_depth, 2.0 + pothole_depth)
    #     })

    # environment = get_environment(
    #     'gym', domain, 'v3', {
    #         **environment_params['kwargs'],
    #         'perturb_random_action_kwargs': {
    #             'perturbation_probability': 0.0
    #         },
    #     })

    # environment = get_environment(
    #     'gym', domain, 'v3', {
    #         **environment_params['kwargs'],
    #         'perturb_noisy_action_kwargs': {
    #             'noise_scale': 2.0
    #         },
    #     })

    with policy.set_deterministic(deterministic):
        paths = rollouts(num_rollouts,
                         environment,
                         policy,
                         path_length=max_path_length,
                         render_kwargs=render_kwargs)

    if render_kwargs.get('mode') == 'rgb_array':
        fps = 1 // getattr(environment, 'dt', 1/30)
        for i, path in enumerate(paths):
            video_save_dir = os.path.expanduser('/tmp/simulate_policy/')
            video_save_path = os.path.join(video_save_dir, f'episode_{i}.mp4')
            save_video(path['images'], video_save_path, fps=fps)

    return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(**vars(args))
