#!/bin/bash

declare -r SCRIPT_DIRECTORY="$(dirname $(realpath ${BASH_SOURCE[0]}))"
declare -r PROJECT_ROOT="$(dirname ${SCRIPT_DIRECTORY})"

RESULTS_BASE="${HOME}/ray_results/gs"
EXPERIMENT_PATHS=(
    # "${RESULTS_BASE}/gym/Hopper/MaxVelocity-v3/2019-07-27T20-44-40-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Hopper/MaxVelocity-v3/2019-07-27T20-49-38-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Walker2d/MaxVelocity-v3/2019-07-27T20-47-06-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Walker2d/MaxVelocity-v3/2019-07-27T20-51-59-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Hopper/MaxVelocity-v3/2019-07-30T04-59-10-perturbations-speed-limit-no-terminal-1"

    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/2019-05-10T05-08-36-perturbations-entropy-sweep-3"
    # "${RESULTS_BASE}/gym/Humanoid/Standup-v2/gym/Humanoid/Standup-v2/2019-08-03T02-09-37-perturbations-claude-1"
    # "${RESULTS_BASE}/gym/Hopper/v3/gym/Hopper/v3/2019-08-04T16-59-37-perturbations-claude-1/"

    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-08-06T21-10-09-perturbations-1/"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-08-10T18-13-32-perturbations-2/" # 200
    "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-08-12T21-15-01-perturbations-ddpg-1/" # 100

    # "${RESULTS_BASE}/gym/Hopper/v3/2019-09-03T12-24-29-perturbations-1/"
    # "${RESULTS_BASE}/gym/Walker2d/v3/2019-09-03T12-27-17-perturbations-1/"
)

EVALUATION_TASKS=(
    # "Pothole-v0"
    # "HeightField-v0"
    # "PerturbRandomAction-v0"
    # "PerturbNoisyAction-v0"
    "PerturbBody-v0"
    # "Wind-v0"
)


for EXPERIMENT_PATH in ${EXPERIMENT_PATHS[@]}; do
    for EVALUATION_TASK in ${EVALUATION_TASKS[@]}; do

        echo "EXPERIMENT_PATH: ${EXPERIMENT_PATH}; EVALUATION_TASK: ${EVALUATION_TASK}"

        python -m examples.development.simulate_environments \
               "${EXPERIMENT_PATH}" \
               --num-rollouts=25 \
               --evaluation-task="${EVALUATION_TASK}" \
               --desired-checkpoint=100

    done
done
