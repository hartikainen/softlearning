import argparse
import glob
import os
import pickle
from pprint import pprint, pformat

import numpy as np

from softlearning.utils.tune import is_trial_directory
from softlearning.utils.video import create_video_grid, save_video
from examples.development.simulate_policy import simulate_policy


EXPERIMENT_PATHS = {
    "sac": os.path.expanduser(
        "~/ray_results/gs"
        "/gym/Humanoid/Stand-v3"
        "/gym/Humanoid/Stand-v3"
        "/2019-08-10T18-13-32-perturbations-2"),
    "ddpg": os.path.expanduser(
        "~/ray_results/gs"
        "/gym/Humanoid/Stand-v3"
        "/gym/Humanoid/Stand-v3"
        "/2019-08-12T21-15-01-perturbations-ddpg-1"
    ),
}

ROBUSTNESS_ENVIRONMENTS = {
    "PerturbRandomAction-v0": [
        {
            'perturb_random_action_kwargs': {
                'perturbation_probability': perturbation_probability,
            }
        }
        for perturbation_probability in np.linspace(0, 0.6, 5)
    ],
    "PerturbNoisyAction-v0": [
        {
            'perturb_noisy_action_kwargs': {
                'noise_scale': noise_scale,
            }
        }
        for noise_scale in np.linspace(0, 1.0, 5)
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
                variant['algorithm_params']['kwargs']['target_entropy'] == 9)
        )
    elif algorithm == 'ddpg':
        trial_dirs = filter_trials(
            EXPERIMENT_PATHS['ddpg'],
            trial_filter=lambda variant: (
                variant['policy_params']['kwargs']['scale_identity_multiplier'] == 1)
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

    all_frames = []
    all_paths = []
    for checkpoint_dir in checkpoint_dirs:
        print(f"checkpoint_dir: {checkpoint_dir}")
        checkpoint_frames = []
        checkpoint_paths = []
        for environment_params in environments_params:
            print(f"environment_params: {pformat(environment_params)}")
            environment_paths = simulate_policy(
                checkpoint_dir,
                deterministic,
                num_rollouts,
                max_path_length,
                render_kwargs={'mode': 'rgb_array', 'width': 500, 'height': 500},
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

            checkpoint_frames += [environment_frames]
            checkpoint_paths += [environment_paths]

        all_frames += [checkpoint_frames]
        all_paths += [checkpoint_paths]

    # Concatenate episodes
    col_and_row_frames = [
        [
            np.concatenate(environment_frames, axis=0)
            for environment_frames in checkpoint_frames
        ]
        for checkpoint_frames in all_frames
    ]

    video_grid_frames = create_video_grid(col_and_row_frames)
    save_video(
        video_grid_frames,
        f'/tmp/perturbations/{environment}-{algorithm}.mp4',
        fps=67,
    )


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
