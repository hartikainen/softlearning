import argparse
from collections import defaultdict, OrderedDict
import glob
from distutils.util import strtobool
import json
import os
import pickle
import re

import tensorflow as tf
import numpy as np
import pandas as pd
import ray

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


@ray.remote
def simulate_perturbations_for_trial(experiment_path,
                                     trial_dirname,
                                     perturbation_probabilities,
                                     output_dir,
                                     deterministic,
                                     num_rollouts,
                                     max_path_length):
    print(f"trial_dirname: {trial_dirname}")
    trial_dir = os.path.join(experiment_path, trial_dirname)

    perturbation_dataframes = {}
    for perturbation_probability in perturbation_probabilities:
        new_trial_dir = os.path.join(
            output_dir,
            f'{trial_dirname}-{str(perturbation_probability)}')

        if not os.path.exists(new_trial_dir):
            os.makedirs(new_trial_dir)

        progress_path = os.path.join(new_trial_dir, 'progress.csv')
        variant_path = os.path.join(new_trial_dir, 'params.json')

        if os.path.exists(progress_path):
            print(f"Progress for {trial_dir} already exist. Loading.")
            perturbation_dataframes[
                perturbation_probability] = pd.read_csv(progress_path)
        else:
            perturbation_dataframes[
                perturbation_probability] = pd.DataFrame()

    checkpoint_dirs = glob.iglob(os.path.join(trial_dir, 'checkpoint_*'))
    for checkpoint_dir in checkpoint_dirs:
        print(f"checkpoint_dir: {checkpoint_dir}")

        # checkpoint_id = int(re.match(
        #     'checkpoint_(\d+)',
        #     os.path.split(checkpoint_dir)[-1]
        # ).groups()[0])

        tf.keras.backend.clear_session()
        session = tf.keras.backend.get_session()
        picklable, variant, progress, metadata = load_checkpoint(
            checkpoint_dir, session=session)
        if metadata is None:
            continue
        policy, environment = load_policy_and_environment(
            picklable, variant)

        for perturbation_probability in perturbation_probabilities:
            print(f"perturbation_probability: {perturbation_probability}")

            new_trial_dir = os.path.join(
                output_dir,
                f'{trial_dirname}-{str(perturbation_probability)}')
            variant_path = os.path.join(new_trial_dir, 'params.json')

            if not os.path.exists(variant_path):
                variant[
                    'perturbation_probability'] = perturbation_probability
                with open(variant_path, 'w') as f:
                    json.dump(variant, f)

            iteration = metadata['iteration']
            if ('iteration' in perturbation_dataframes[
                    perturbation_probability].columns
                and iteration in perturbation_dataframes[
                    perturbation_probability]['iteration'].values):
                print(f"Iteration {iteration} already exists in"
                      " dataframe. Skipping")
                continue

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

            checkpoint_data = evaluate_rollouts(paths, environment)
            checkpoint_data['iteration'] = metadata['iteration']

            perturbation_dataframes[perturbation_probability] = (
                perturbation_dataframes[perturbation_probability]
                .append(checkpoint_data, ignore_index=True))

    return perturbation_dataframes


def simulate_perturbations(args):
    experiment_path = args.experiment_path
    deterministic = args.deterministic
    max_path_length = args.max_path_length
    num_rollouts = args.num_rollouts
    perturbation_probabilities = args.perturbation_probabilities

    output_dir = os.path.join(
        '/tmp',
        'perturbations',
        experiment_path.split('ray_results/gs/')[-1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ray.init()

    trial_dirnames = tuple(os.walk(experiment_path))[0][1]
    all_perturbation_dataframes = tuple(ray.get([
        simulate_perturbations_for_trial.remote(
            experiment_path,
            trial_dirname,
            perturbation_probabilities,
            output_dir,
            deterministic,
            num_rollouts,
            max_path_length
        )
        for trial_dirname in trial_dirnames
    ]))

    for trial_dirname, perturbation_dataframes in zip(
            trial_dirnames, all_perturbation_dataframes):
        for perturbation_probability, dataframe in perturbation_dataframes.items():
            new_trial_dir = os.path.join(
                output_dir,
                f'{trial_dirname}-{str(perturbation_probability)}')

            if not os.path.exists(new_trial_dir):
                os.makedirs(new_trial_dir)

            progress_path = os.path.join(new_trial_dir, 'progress.csv')

            dataframe = (dataframe
                         .sort_values('iteration')
                         .reset_index(drop=True))
            dataframe.to_csv(progress_path, index=False)


if __name__ == '__main__':
    args = parse_args()

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    simulate_perturbations(args)
