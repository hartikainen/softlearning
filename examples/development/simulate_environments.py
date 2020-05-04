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
from softlearning.utils.dict import deep_update


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
                        default=True,
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
    import tensorflow as tf
    print(f"trial_dirname: {trial_dirname}")
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

    new_trial_dir = os.path.join(output_dir, trial_dirname)
    os.makedirs(new_trial_dir, exist_ok=True)

    assert checkpoint_dirs and str(desired_checkpoint) in checkpoint_dirs[0]
    dataframe_parts = []
    for checkpoint_dir in checkpoint_dirs:
        print(f"checkpoint_dir: {checkpoint_dir}")

        tf.compat.v1.keras.backend.clear_session()
        session = tf.compat.v1.keras.backend.get_session()

        picklable, variant, progress, metadata = load_checkpoint(
            checkpoint_dir, session=session)
        policy, _ = load_policy_and_environment(picklable, variant)

        for name, environment_params in environments_params.items():
            print("environment_params")
            pprint(environment_params)
            evaluation_environment_params = deep_update(
                variant['environment_params']['training'],
                environment_params,
            )
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

            if evaluation_environment_params['task'] == 'Stand-v3':
                evaluation_environment_params['task'] = 'SimpleStand-v3'

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
                    # render_kwargs={'mode': 'human'})

            checkpoint_data = evaluate_rollouts(paths, environment)
            checkpoint_data.update(flatten(
                environment_params['kwargs'], reducer='path'))

            dataframe_parts.append(checkpoint_data)

            if hasattr(environment, 'get_path_infos'):
                figure_save_dir = os.path.join(new_trial_dir, 'figures')
                os.makedirs(figure_save_dir, exist_ok=True)
                figure_save_path = os.path.join(
                    figure_save_dir, f'simulate-{desired_checkpoint}-{name}.png')
                infos = environment.get_path_infos(
                    paths, figure_save_path=figure_save_path)

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

    progress_path = os.path.join(new_trial_dir, 'progress.csv')

    # dataframe = (dataframe
    #              .sort_values('iteration')
    #              .reset_index(drop=True))
    dataframe.to_csv(progress_path, index=False)
    variant_path = os.path.join(new_trial_dir, 'params.json')
    with open(variant_path, 'w') as f:
        json.dump(variant, f, indent=2, sort_keys=True)

    return True


def filter_trials(experiment_path, trial_dirnames):
    if os.path.split(experiment_path.rstrip('/'))[-1] == '2019-10-07T20-04-53-robustness-td3-4':
        filtered_trial_dirnames = []
        for trial_dirname in trial_dirnames:
            variant_path = os.path.join(experiment_path, trial_dirname, 'params.json')
            with open(variant_path, 'r') as f:
                variant = json.load(f)
            if (variant['algorithm_params']['kwargs']['policy_train_every_n_steps'] == 1
                or variant['run_params']['seed'] in (
                    402,
                    9063,
                    4154,
                    9939,
                    1466,
                    8724,
                    2598,
                    7509,
                )):
                print(f"ignoring {trial_dirname}")
                continue

            filtered_trial_dirnames += [trial_dirname]
        trial_dirnames = filtered_trial_dirnames
    elif os.path.split(experiment_path.rstrip('/'))[-1] == '2019-10-24T13-52-30-no-termination-1':
        filtered_trial_dirnames = []
        for trial_dirname in trial_dirnames:
            variant_path = os.path.join(experiment_path, trial_dirname, 'params.json')
            with open(variant_path, 'r') as f:
                variant = json.load(f)
            if (variant['algorithm_params']['kwargs']['target_entropy']
                not in (9.0, 10.0, 11.0)):
                print(f"ignoring {trial_dirname}")
                continue

            filtered_trial_dirnames += [trial_dirname]
        trial_dirnames = filtered_trial_dirnames
    elif os.path.split(experiment_path.rstrip('/'))[-1] == '2019-10-24T14-35-20-no-termination-ddpg-1':
        filtered_trial_dirnames = []
        for trial_dirname in trial_dirnames:
            variant_path = os.path.join(experiment_path, trial_dirname, 'params.json')
            with open(variant_path, 'r') as f:
                variant = json.load(f)
            if (variant['sampler_params']['kwargs']['exploration_noise']
                not in (0.2, 0.3)):
                print(f"ignoring {trial_dirname}")
                continue

            filtered_trial_dirnames += [trial_dirname]
        trial_dirnames = filtered_trial_dirnames
    elif os.path.split(experiment_path.rstrip('/'))[-1] == '2019-06-08T05-35-29-perturbations-final-1':
        filtered_trial_dirnames = []
        for trial_dirname in trial_dirnames:
            variant_path = os.path.join(experiment_path, trial_dirname, 'params.json')
            with open(variant_path, 'r') as f:
                variant = json.load(f)
            if variant['replay_pool_params']['kwargs']['max_size'] != int(1e6):
                print(f"ignoring {trial_dirname}")
                continue

            filtered_trial_dirnames += [trial_dirname]
        trial_dirnames = filtered_trial_dirnames
    elif os.path.split(experiment_path.rstrip('/'))[-1] == '2019-11-14T17-15-12-robustness-DDPG-sweep-2':
        filtered_trial_dirnames = []
        for trial_dirname in trial_dirnames:
            variant_path = os.path.join(experiment_path, trial_dirname, 'params.json')
            with open(variant_path, 'r') as f:
                variant = json.load(f)
            if (variant['sampler_params']['kwargs']['exploration_noise'] != 0.03
                or variant['policy_params']['kwargs']['scale_identity_multiplier'] != 0.2):
                print(f"ignoring {trial_dirname}")
                continue

            filtered_trial_dirnames += [trial_dirname]
        trial_dirnames = filtered_trial_dirnames

    return trial_dirnames


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
            for perturbation_probability in np.linspace(0, 0.9, 21)
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
            for noise_scale in np.linspace(0, 1.0, 21)
        }
    elif evaluation_task == 'PerturbBody-v0':
        environments_params = {
            f'perturbation-strength-{perturbation_strength}': {
                'kwargs': {
                    'perturb_body_kwargs': {
                        'perturbation_strength': perturbation_strength,
                        'perturbation_length': 5,
                    },
                }
            }
            # for perturbation_strength in np.linspace(0, 500.0, 51)
            for perturbation_strength in np.linspace(0, 2000.0, 51)
        }
    elif evaluation_task == 'PerturbBody-v1':
        environments_params = {
            f'perturbation-strength-{perturbation_strength}': {
                'kwargs': {
                    'perturb_body_kwargs': {
                        'perturbation_strength': perturbation_strength,
                        'perturbation_length': 1,
                    },
                }
            }
            for perturbation_strength in np.linspace(0, 3000.0, 51)
        }
    elif evaluation_task == 'PerturbBody-v2':
        environments_params = {
            f'perturbation-probability-{perturbation_probability}': {
                'kwargs': {
                    'perturb_body_kwargs': {
                        'perturbation_strength': 125,
                        'perturbation_direction': (1.0, 0.0, 0.0),
                        'perturbation_probability': perturbation_probability,
                        'perturbation_frequency': None,
                        'perturbation_length': 1,
                    },
                }
            }
            for perturbation_probability in np.linspace(0.0, 0.6, 51)
        }
    elif evaluation_task == 'PerturbBody-AntPond-v0':
        environments_params = {
            f'perturbation-probability-{perturbation_probability}': {
                'kwargs': {
                    'perturb_body_kwargs': {
                        'perturbation_strength': 25.0,
                        'perturbation_direction': {
                            "type": "towards",
                            "target": "pond_center"
                        },
                        'perturbation_probability': perturbation_probability,
                        'perturbation_frequency': None,
                        'perturbation_length': 1,
                    },
                }
            }
            for perturbation_probability in np.linspace(0.0, 0.1, 51)
        }
    elif evaluation_task == 'PerturbBody-point_mass-orbit_pond-v0':
        environments_params = {
            f'perturbation-probability-{perturbation_probability}': {
                'kwargs': {
                    'perturb_body_kwargs': {
                        'perturbation_strength': 3.0,
                        'perturbation_direction': {
                            "type": "towards",
                            "target": "pond_center"
                        },
                        'perturbation_probability': perturbation_probability,
                        'perturbation_frequency': None,
                        'perturbation_length': 1,
                    },
                }
            }
            for perturbation_probability in np.linspace(0.0, 0.1, 51)
        }
    elif evaluation_task == 'Wind-v0':
        environments_params = {
            f'wind-strength-{wind_strength}': {
                'kwargs': {
                    'wind_kwargs': {
                        'wind_strength': wind_strength,
                        'wind_direction': 'random',
                    },
                }
            }
            for wind_strength in np.linspace(0, 60.0, 51)
        }
    elif evaluation_task == 'Wind-AntPond-v0':
        environments_params = {
            f'wind-strength-{wind_strength}': {
                'kwargs': {
                    'wind_kwargs': {
                        'wind_strength': wind_strength,
                        'wind_direction': {
                            "type": "towards",
                            "target": "pond_center"
                        },
                    },
                }
            }
            for wind_strength in np.linspace(0, 5.0, 51)
        }

    elif evaluation_task == 'Wind-AntPond-v1':
        # long wind
        environments_params = {
            f'wind-length-{wind_length}': {
                'kwargs': {
                    'wind_kwargs': {
                        'wind_strength': 6.0,
                        'wind_frequency': 200,
                        'wind_length': wind_length,
                        'wind_direction': {
                            "type": "towards",
                            "target": "pond_center"
                        },
                    },
                }
            }
            for wind_length in np.linspace(0, 50, 51)
        }

    elif evaluation_task == 'Wind-point_mass-orbit_pond-v0':
        environments_params = {
            f'wind-strength-{wind_strength}': {
                'kwargs': {
                    'wind_kwargs': {
                        'wind_strength': wind_strength,
                        'wind_direction': {
                            "type": "towards",
                            "target": "pond_center"
                        },
                    },
                }
            }
            for wind_strength in np.linspace(0, 50.0, 51)
        }

    else:
        raise NotImplementedError(evaluation_task)

    output_dir = os.path.join(
        '/tmp',
        'perturbations',
        (f"{experiment_path.split('ray_results/gs/')[-1].rstrip('/')}"
         f"-{args.desired_checkpoint}"
         f"{'' if deterministic else '-stochastic'}"),
        evaluation_task,
    )

    trial_dirnames = tuple(os.walk(experiment_path))[0][1]
    trial_dirnames = filter_trials(experiment_path, trial_dirnames)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ray.init(local_mode=False)

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

    if tf.test.is_gpu_available():
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(session)

    simulate_perturbations(args)
