import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf
import pandas as pd

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.misc.utils import save_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default='human',
                        choices=('human', 'rgb_array', None),
                        help="Mode to render the rollouts in.")
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
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    environment = get_environment_from_params(environment_params)

    policy = get_policy_from_variant(variant, environment, Qs=[None])
    policy.set_weights(picklable['policy_weights'])

    return policy, environment


def simulate_policy(checkpoint_path,
                    deterministic,
                    num_rollouts,
                    max_path_length,
                    render_mode):
    checkpoint_path = checkpoint_path.rstrip('/')
    picklable, variant, progress, metadata = load_checkpoint(checkpoint_path)
    policy, environment = load_policy_and_environment(picklable, variant)

    with policy.set_deterministic(deterministic):
        paths = rollouts(num_rollouts,
                         environment,
                         policy,
                         path_length=max_path_length,
                         render_mode=render_mode)

    return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(**vars(args))
