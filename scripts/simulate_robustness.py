import argparse
import glob
import os
import pickle
from pprint import pprint, pformat
from distutils.util import strtobool
from shlex import quote
import subprocess

import numpy as np
import tensorflow as tf

from softlearning.utils.tune import is_trial_directory
from softlearning.utils.video import create_video_grid, save_video
from examples.development.simulate_policy import simulate_policy


EXPERIMENT_PATHS = {
    # "sac": os.path.expanduser(
    #     "~/ray_results/gs"
    #     "/gym/Humanoid/Stand-v3"
    #     "/gym/Humanoid/Stand-v3"
    #     "/2019-08-10T18-13-32-perturbations-2"),

    #
    # "sac": os.path.expanduser(
    #     "~/ray_results/gs"
    #     "/gym/Humanoid/Stand-v3"
    #     "/gym/Humanoid/Stand-v3"
    #     "2019-10-16T15-02-14-unbounded-scale-3"),
    # "ddpg": os.path.expanduser(
    #     "~/ray_results/gs"
    #     "/gym/Humanoid/Stand-v3"
    #     "/gym/Humanoid/Stand-v3"
    #     "/2019-08-12T21-15-01-perturbations-ddpg-1"),
    # "sac": os.path.expanduser(
    #     "~/ray_results/gs/gym/Humanoid/SimpleStand-v3/gym/Humanoid/SimpleStand-v3/2019-10-24T13-52-30-no-termination-1"),
    "sac": os.path.expanduser(
        "/tmp/perturbations/gym/Walker2d/NoTermination-v3/2019-11-23T22-46-00-robustness-1-24"),
    # "ddpg": os.path.expanduser(
    #     "~/ray_results/gs/gym/Humanoid/SimpleStand-v3/gym/Humanoid/SimpleStand-v3/2019-10-24T14-35-20-no-termination-ddpg-1"),
}

ROBUSTNESS_ENVIRONMENTS = {
    "PerturbRandomAction-v0": [
        {
            'perturb_random_action_kwargs': {
                'perturbation_probability': perturbation_probability,
            }
        }
        for perturbation_probability in np.linspace(0, 0.5, 6)
    ],
    "PerturbNoisyAction-v0": [
        {
            'perturb_noisy_action_kwargs': {
                'noise_scale': noise_scale,
            }
        }
        for noise_scale in np.linspace(0, 1.0, 6)
    ],
    "PerturbBody-v0": [
        {
            'perturb_body_kwargs': {
                'perturbation_strength': perturbation_strength,
                'perturbation_length': 5,
            }
        }
        for perturbation_strength in np.linspace(0, 500.0, 6)
    ],
    "PerturbBody-v1": [
        {
            'perturb_body_kwargs': {
                'perturbation_strength': perturbation_strength,
                'perturbation_length': 1,
            }
        }
        for perturbation_strength in np.linspace(0, 3000.0, 6)
    ],
    "Wind-v0": [
        {
            'wind_kwargs': {
                'wind_strength': wind_strength,
            },
        }
        for wind_strength in np.linspace(0, 40.0, 6)
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str, default='sac')
    parser.add_argument('checkpoint_id', type=int, default=100)
    parser.add_argument('environment', type=str, default='PerturbRandomAction-v0')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=5)
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args


def filter_trials(experiment_path, trial_filter):
    trials = []
    for trial_path in glob.glob(os.path.join(experiment_path, "*")):
        if not is_trial_directory(trial_path):
            continue

        variant_path = os.path.join(trial_path, 'params.pkl')
        with open(variant_path, 'rb') as f:
            variant = pickle.load(f)

        if not trial_filter(variant):
            continue

        trials += [trial_path]

    return trials


def get_trial_dirs(algorithm):
    if algorithm == 'sac':
        trial_dirs = filter_trials(
            EXPERIMENT_PATHS['sac'],
            trial_filter=lambda variant: (
                variant['algorithm_params']['kwargs']['target_entropy'] == 10)
        )
    elif algorithm == 'ddpg':
        trial_dirs = filter_trials(
            EXPERIMENT_PATHS['ddpg'],
            # trial_filter=lambda variant: (
            #     variant['policy_params']['kwargs']['scale_identity_multiplier'] == 0.2)
            trial_filter=lambda variant: (
                variant['sampler_params']['kwargs']['exploration_noise'] == 0.2)
        )
    else:
        raise NotImplementedError(algorithm)

    return trial_dirs


def main(algorithm,
         checkpoint_id,
         environment,
         max_path_length=1000,
         num_rollouts=3,
         deterministic=True):

    trial_dirs = get_trial_dirs(algorithm)
    checkpoint_dirs = [
        os.path.join(trial_dir, f"checkpoint_{checkpoint_id}")
        for trial_dir in trial_dirs
    ]
    environments_params = ROBUSTNESS_ENVIRONMENTS[environment]

    num_checkpoints = len(checkpoint_dirs)
    num_environments = len(environments_params)
    try:
        checkpoint_formatter = f"0{int(np.ceil(np.log10(num_checkpoints)))}d"
        environment_formatter = f"0{int(np.ceil(np.log10(num_environments)))}d"
    except Exception as e:
        from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
        pass

    video_directory = '/tmp/perturbations'
    video_filename_template = (
        f'{video_directory}'
        f'/{{environment}}'
        f'-{{algorithm}}'
        f'-{{i:{checkpoint_formatter}}}'
        f'-{{j:{environment_formatter}}}'
        f'{"-stochastic" if not deterministic else ""}.mp4')
    num_runs = num_checkpoints * num_environments

    all_frames = []
    # all_paths = []
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        print(f"checkpoint_dir: {checkpoint_dir}")
        checkpoint_frames = []
        # checkpoint_paths = []
        for j, environment_params in enumerate(environments_params):
            print(f"environment_params: {pformat(environment_params)}")
            environment_paths = simulate_policy(
                checkpoint_dir,
                deterministic,
                num_rollouts,
                max_path_length,
                render_kwargs={'mode': 'rgb_array', 'width': 250, 'height': 250},
                evaluation_environment_params=environment_params,
            )

            environment_frames = []
            for environment_path in environment_paths:
                path_frames = environment_path.pop('images')
                if path_frames.shape[0] < max_path_length:
                    path_frames = np.concatenate([
                        path_frames,
                        np.repeat(
                            np.zeros_like(path_frames[[-1]]),
                            max_path_length - path_frames.shape[0],
                            axis=0),
                    ])
                environment_frames += [path_frames]
            environment_frames = np.concatenate(environment_frames, axis=0)

            save_video(
                environment_frames,
                video_filename_template.format(
                    environment=environment,
                    algorithm=algorithm,
                    i=i,
                    j=j,
                    deterministic=deterministic),
                fps=67
            )

            del environment_frames
            # checkpoint_frames += [environment_frames]

            # checkpoint_paths += [environment_paths]

        # all_frames += [checkpoint_frames]
        # all_paths += [checkpoint_paths]

    output_id = "v"
    concatenate_command_parts = [
        "ffmpeg",
        *[
            f"-i {video_filename_template}".format(
                environment=environment,
                algorithm=algorithm,
                i=i,
                j=j,
                deterministic=deterministic,
            )
            for i in range(num_checkpoints)
            for j in range(num_environments)
        ],
        "-filter_complex",
        quote(";".join(
            [
                "".join(
                    [
                        f"[{i*num_environments+j}:v]"
                        for j in range(num_environments)
                    ]
                    + [f"hstack=inputs={num_environments}[row{i}]"]
                )
                for i in range(num_checkpoints)
            ] + [
                "".join(
                    [f"[row{i}]" for i in range(num_checkpoints)]
                    + [f"vstack=inputs={num_checkpoints}[{output_id}]"]
                )
            ]
        )),
        "-map",
        quote(f"[{output_id}]"),
        quote(os.path.join(video_directory, f"{environment}-{algorithm}{'-stochastic' if not deterministic else ''}.mp4"))
    ]
    concatenate_command = " ".join(concatenate_command_parts)

    print(f"runnning: {concatenate_command}")
    try:
        subprocess.call([concatenate_command], shell=True)
    except Exception as e:
        import ipdb; ipdb.set_trace(context=30)
        pass
    finally:
        pass

    return
    # Concatenate episodes
    # col_and_row_frames = [
    #     [
    #         np.concatenate(environment_frames, axis=0)
    #         for environment_frames in checkpoint_frames
    #     ]
    #     for checkpoint_frames in all_frames
    # ]

    # video_grid_frames = create_video_grid(col_and_row_frames)

    video_grid_frames = create_video_grid(all_frames)
    save_video(
        video_grid_frames,
        f'/tmp/perturbations/{environment}-{algorithm}.mp4',
        fps=67,
    )


if __name__ == '__main__':
    args = parse_args()

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    main(**vars(args))
