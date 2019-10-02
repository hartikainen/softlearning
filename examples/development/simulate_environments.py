import argparse
from collections import defaultdict, OrderedDict
import glob
from distutils.util import strtobool
import json
import os
import pickle
import re
from pprint import pprint

import tensorflow as tf
import numpy as np
import pandas as pd
import ray
from flatten_dict import flatten

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from .simulate_policy import load_checkpoint, load_policy_and_environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path',
                        type=str,
                        help='Path to the experiment root.')
    parser.add_argument('--evaluation-task',
                        type=str,
                        default='PerturbAction-v0')
    parser.add_argument('--desired-checkpoint', type=int)
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=1)
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=False,
                        help="Evaluate policy deterministically.")

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
def simulate_trial_in_environments(experiment_path,
                                   trial_dirname,
                                   environments_params,
                                   output_dir,
                                   desired_checkpoint,
                                   deterministic,
                                   num_rollouts,
                                   max_path_length):
    print(f"trial_dirname: {trial_dirname}, environments_params:")
    pprint(environments_params)
    trial_dir = os.path.join(experiment_path, trial_dirname)

    checkpoint_dirs = glob.glob(os.path.join(trial_dir, 'checkpoint_*'))
    checkpoint_ids = [
        int(re.match(
            'checkpoint_(\\d+)',
            os.path.split(checkpoint_dir)[-1]
        ).groups()[0])
        for checkpoint_dir in checkpoint_dirs
    ]

    # if np.max(checkpoint_ids) < desired_checkpoint:
    if desired_checkpoint not in checkpoint_ids:
        return

    checkpoint_dirs = [checkpoint_dirs[np.flatnonzero(np.equal(
        checkpoint_ids, desired_checkpoint
    ))[0]]]

    dataframe_parts = []
    assert checkpoint_dirs and str(desired_checkpoint) in checkpoint_dirs[0]
    for checkpoint_dir in checkpoint_dirs:
        print(f"checkpoint_dir: {checkpoint_dir}")

        tf.keras.backend.clear_session()
        session = tf.keras.backend.get_session()

        picklable, variant, progress, metadata = load_checkpoint(
            checkpoint_dir, session=session)
        policy, _ = load_policy_and_environment(picklable, variant)

        for name, environment_params in environments_params.items():
            print("environment_params")
            pprint(environment_params)
            evaluation_environment_params = {
                **variant['environment_params']['training'],
                **environment_params,
            }
            if ('terminate_when_unhealthy' in (variant
                                               ['environment_params']
                                               ['training'])):
                (evaluation_environment_params
                 ['kwargs']
                 ['terminate_when_unhealthy']) = (
                     variant
                     ['environment_params']
                     ['training']
                     .get('terminate_when_unhealthy', True))

            environment = get_environment_from_params(
                evaluation_environment_params)

            # updated_variant = {
            #     **variant,
            #     'environment_params': {
            #         'evaluation': environment_params
            #     }
            # }

            with policy.set_deterministic(deterministic):
                paths = rollouts(
                    num_rollouts,
                    environment,
                    policy,
                    path_length=max_path_length,
                    render_kwargs={})

            checkpoint_data = evaluate_rollouts(paths, environment)
            checkpoint_data.update(flatten(
                environment_params['kwargs'], reducer='path'))

            dataframe_parts.append(checkpoint_data)

    variant = {
        key: value for key, value in variant.items()
        if key != 'environment_params'
    }

    old_variant_path = os.path.join(trial_dir, 'params.json')
    with open(old_variant_path, 'r') as f:
        old_variant = json.load(f)

    old_environment_params = old_variant['environment_params']
    variant['old_environment_params'] = old_environment_params.copy()
    dataframe = pd.DataFrame(dataframe_parts)

    new_trial_dir = os.path.join(output_dir, trial_dirname)

    if not os.path.exists(new_trial_dir):
        os.makedirs(new_trial_dir)

    progress_path = os.path.join(new_trial_dir, 'progress.csv')

    # dataframe = (dataframe
    #              .sort_values('iteration')
    #              .reset_index(drop=True))
    dataframe.to_csv(progress_path, index=False)
    variant_path = os.path.join(new_trial_dir, 'params.json')
    with open(variant_path, 'w') as f:
        json.dump(variant, f, indent=2, sort_keys=True)

    return True


def simulate_perturbations(args):
    experiment_path = args.experiment_path
    deterministic = args.deterministic
    max_path_length = args.max_path_length
    num_rollouts = args.num_rollouts
    evaluation_task = args.evaluation_task

    if evaluation_task == 'Pothole-v0':
        environments_params = {
            f'pothole-depth-{pothole_depth}': {
                'task': evaluation_task,
                'kwargs': {
                    'pothole_depth': pothole_depth,
                }
            }
            # for pothole_depth in (0.1, 0.2, 0.4, 0.8)
            for pothole_depth in np.linspace(0.001, 1.0, 51)
        }
    elif evaluation_task == 'HeightField-v0':
        environments_params = {
            f'height-field-height-{field_z_max}': {
                'task': evaluation_task,
                'kwargs': {
                    'field_z_max': field_z_max,
                    'field_z_range': (0, field_z_max),
                }
            }
            for field_z_max in np.linspace(0, 0.5, 51)
        }
    elif evaluation_task == 'PerturbRandomAction-v0':
        environments_params = {
            f'perturbation-probability-{perturbation_probability}': {
                'kwargs': {
                    'perturb_random_action_kwargs': {
                        'perturbation_probability': perturbation_probability,
                    },
                }
            }
            for perturbation_probability in np.linspace(0, 0.5, 51)
        }
    elif evaluation_task == 'PerturbNoisyAction-v0':
        environments_params = {
            f'noise-scale-{noise_scale}': {
                'kwargs': {
                    'perturb_noisy_action_kwargs': {
                        'noise_scale': noise_scale,
                    },
                }
            }
            for noise_scale in np.linspace(0, 1.0, 51)
        }
    elif evaluation_task == 'PerturbBody-v0':
        environments_params = {
            f'perturbation-strength-{perturbation_strength}': {
                'kwargs': {
                    'perturb_body_kwargs': {
                        'perturbation_strength': perturbation_strength,
                    },
                }
            }
            for perturbation_strength in np.linspace(0, 50.0, 51)
        }
    elif evaluation_task == 'Wind-v0':
        environments_params = {
            f'wind-strength-{wind_strength}': {
                'kwargs': {
                    'wind_kwargs': {
                        'wind_strength': wind_strength,
                    },
                }
            }
            for wind_strength in np.linspace(0, 15.0, 51)
        }
    else:
        raise NotImplementedError(evaluation_task)

    output_dir = os.path.join(
        '/tmp',
        'perturbations',
        experiment_path.split('ray_results/gs/')[-1],
        evaluation_task,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ray.init(local_mode=False)

    trial_dirnames = tuple(os.walk(experiment_path))[0][1]

    results = ray.get([
        simulate_trial_in_environments.remote(
            experiment_path,
            trial_dirname,
            environments_params,
            output_dir,
            args.desired_checkpoint,
            deterministic,
            num_rollouts,
            max_path_length,
        )
        for trial_dirname in trial_dirnames
    ])

    succeeded_trial_dirnames = [
        x for result, x in zip(results, trial_dirnames) if result
    ]


if __name__ == '__main__':
    args = parse_args()

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    simulate_perturbations(args)
