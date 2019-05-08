import argparse
from collections import defaultdict, OrderedDict
import glob
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from .simulate_policy import load_checkpoint, load_policy_and_environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path',
                        type=str,
                        help='Path to the experiment root.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=1)
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=False,
                        help="Evaluate policy deterministically.")
    parser.add_argument('--perturbation_probabilities',
                        type=float,
                        nargs='+',
                        # default=(0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0))
                        default=(
                            (0.0, 0.003, 0.01)
                            + tuple(np.round(np.linspace(0.03, 0.3, 10), 2))
                            + (1.0, )
                        ))

    args = parser.parse_args()

    return args


def evaluate_rollouts(paths, environment):
        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        diagnostics = OrderedDict([
            (f'{metric_name}-{metric_fn}',
             getattr(np, metric_fn)(metric_values))
            for metric_fn in ('mean', 'min', 'max', 'std')
            for metric_name, metric_values
            in zip(('total_returns', 'episode_lengths'),
                   (total_returns, episode_lengths))
        ])

        env_infos = environment.get_path_infos(paths)
        for key, value in env_infos.items():
            diagnostics[f'env_infos/{key}'] = value

        return diagnostics

    # paths_data = []

    # for path in paths:
    #     rewards = path['rewards']
    #     infos = path['infos']
    #     new_infos = defaultdict(list)
    #     for step_info in infos:
    #         for info_key, info_value in step_info.items():
    #             new_infos[info_key].append(info_value)

    #     paths_data.append({
    #         'rewards': rewards,
    #         **new_infos
    #     })

    # paths_data =
    # from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
    # pass


def simulate_perturbations(args):
    experiment_path = args.experiment_path
    deterministic = args.deterministic
    max_path_length = args.max_path_length
    num_rollouts = args.num_rollouts
    perturbation_probabilities = args.perturbation_probabilities

    output_dir = os.path.join(
        '/tmp',
        'perturbation-test',
        experiment_path.split('ray_results/gs/gym/')[-1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_data = defaultdict(list)
    variant_data = defaultdict(dict)

    trial_dirnames = tuple(os.walk(experiment_path))[0][1]
    for trial_dirname in trial_dirnames:
        trial_dir = os.path.join(experiment_path, trial_dirname)

        checkpoint_dirs = glob.iglob(os.path.join(trial_dir, 'checkpoint_*'))
        for checkpoint_dir in checkpoint_dirs:
            tf.keras.backend.clear_session()
            session = tf.keras.backend.get_session()
            picklable, variant, progress, metadata = load_checkpoint(
                checkpoint_dir, session=session)
            if metadata is None:
                continue
            policy, environment = load_policy_and_environment(
                picklable, variant)

            for perturbation_probability in perturbation_probabilities:
                assert environment._env._perturbation_probability is not None
                environment._env._perturbation_probability = (
                    perturbation_probability)

                with policy.set_deterministic(deterministic):
                    paths = rollouts(
                        num_rollouts,
                        environment,
                        policy,
                        path_length=max_path_length,
                        render_mode=None)

                path_data = evaluate_rollouts(paths, environment)
                path_data['iteration'] = metadata['iteration']
                result_data[(trial_dirname, perturbation_probability)].append(
                    path_data)
                variant_data[(trial_dirname, perturbation_probability)] = {
                    **variant,
                    'perturbation_probability': perturbation_probability
                }

    result_dataframes = {
        key: (pd.DataFrame(value)
              .sort_values('iteration')
              .reset_index(drop=True))
        for key, value in result_data.items()
    }

    for key in result_dataframes:
        trial_dirname, perturbation_probability = key
        trial_dir = os.path.join(
            output_dir,
            f'{trial_dirname}-{str(perturbation_probability)}')
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)

        dataframe  = result_dataframes[key]
        progress_path = os.path.join(trial_dir, 'progress.csv')
        dataframe.to_csv(progress_path, index=False)

        variant = variant_data[key]
        variant_path = os.path.join(trial_dir, 'params.json')
        with open(variant_path, 'w') as f:
            json.dump(variant, f)


if __name__ == '__main__':
    args = parse_args()

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    simulate_perturbations(args)
