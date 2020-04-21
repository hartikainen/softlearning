#!/bin/bash

declare -r SCRIPT_DIRECTORY="$(dirname $(realpath ${BASH_SOURCE[0]}))"
declare -r PROJECT_ROOT="$(dirname ${SCRIPT_DIRECTORY})"

RESULTS_BASE="${HOME}/ray_results/gs"
EXPERIMENT_PATHS_AND_CHECKPOINT_IDS=(
    # "${RESULTS_BASE}/gym/Hopper/MaxVelocity-v3/2019-07-27T20-44-40-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Hopper/MaxVelocity-v3/2019-07-27T20-49-38-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Walker2d/MaxVelocity-v3/2019-07-27T20-47-06-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Walker2d/MaxVelocity-v3/2019-07-27T20-51-59-perturbations-speed-limit-2"
    # "${RESULTS_BASE}/gym/Hopper/MaxVelocity-v3/2019-07-30T04-59-10-perturbations-speed-limit-no-terminal-1"

    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/2019-05-10T05-08-36-perturbations-entropy-sweep-3"
    # "${RESULTS_BASE}/gym/Humanoid/Standup-v2/gym/Humanoid/Standup-v2/2019-08-03T02-09-37-perturbations-claude-1"
    # "${RESULTS_BASE}/gym/Hopper/v3/gym/Hopper/v3/2019-08-04T16-59-37-perturbations-claude-1/"

    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-08-06T21-10-09-perturbations-1/"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-08-10T18-13-32-perturbations-2/;200"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-08-12T21-15-01-perturbations-ddpg-1/;100"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-09-14T15-11-13-robustness-1/;600"

    # Dense
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-16T15-02-14-unbounded-scale-3/;150;True"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-16T15-02-14-unbounded-scale-3/;150;False"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-16T15-02-14-unbounded-scale-3/;200;True"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-16T15-02-14-unbounded-scale-3/;200;False"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-16T15-02-14-unbounded-scale-3/;250;True"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-16T15-02-14-unbounded-scale-3/;250;False"

    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-31T12-18-35-robustness-dense-td3-1/;250;True"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-31T12-18-35-robustness-dense-td3-1/;275;True"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-31T12-18-35-robustness-dense-td3-1/;300;True"
    # "${RESULTS_BASE}/gym/Humanoid/Stand-v3/gym/Humanoid/Stand-v3/2019-10-07T20-04-53-robustness-td3-4/;300"

    # Sparse
    # "${RESULTS_BASE}/gym/Humanoid/SimpleStand-v3/gym/Humanoid/SimpleStand-v3/2019-10-24T13-52-30-no-termination-1/;200;False"
    # "${RESULTS_BASE}/gym/Humanoid/SimpleStand-v3/gym/Humanoid/SimpleStand-v3/2019-10-24T13-52-30-no-termination-1/;200;True"
    # "${RESULTS_BASE}/gym/Humanoid/SimpleStand-v3/gym/Humanoid/SimpleStand-v3/2019-10-24T14-35-20-no-termination-ddpg-1/;150;True"

    # "${RESULTS_BASE}/gym/Hopper/v3/2019-09-03T12-24-29-perturbations-1/"
    # "${RESULTS_BASE}/gym/Walker2d/v3/2019-09-03T12-27-17-perturbations-1/"

    # "${RESULTS_BASE}/gym/Humanoid/v3/gym/Humanoid/v3/2019-09-20T14-48-41-robustness-2/;200;True"

    # "${RESULTS_BASE}/gym/Humanoid/v3/2019-06-08T05-35-29-perturbations-final-1/;200;True"
    # "${RESULTS_BASE}/gym/Humanoid/v3/gym/Humanoid/v3/2019-11-14T17-15-12-robustness-DDPG-sweep-2/;200;True"

    # Hopper/walker
    # "${RESULTS_BASE}/gym/Walker2d/NoTermination-v3/2019-11-23T22-46-00-robustness-1/;100;True"
    # "${RESULTS_BASE}/gym/Hopper/NoTermination-v3/2019-11-23T22-43-40-robustness-1/;100;True"

    # "${RESULTS_BASE}/gym/Walker2d/NoTermination-v3/2019-11-23T22-46-00-robustness-1/;99;True"
    # "${RESULTS_BASE}/gym/Hopper/NoTermination-v3/2019-11-23T22-43-40-robustness-1/;99;True"

    # "${RESULTS_BASE}/gym/Walker2d/NoTermination-v3/2019-11-23T22-46-00-robustness-1/;98;True"
    # "${RESULTS_BASE}/gym/Hopper/NoTermination-v3/2019-11-23T22-43-40-robustness-1/;98;True"

    # "${RESULTS_BASE}/gym/Walker2d/NoTermination-v3/2019-11-23T22-46-00-robustness-1/;97;True"
    # "${RESULTS_BASE}/gym/Hopper/NoTermination-v3/2019-11-23T22-43-40-robustness-1/;97;True"

    # "${RESULTS_BASE}/gym/Walker2d/NoTermination-v3/2019-11-23T22-46-00-robustness-1/;96;True"
    # "${RESULTS_BASE}/gym/Hopper/NoTermination-v3/2019-11-23T22-43-40-robustness-1/;96;True"

    # Pond-v0
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-21-39-support-Pond-SAC-2/;50;True"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-18-57-support-Pond-DDPG-2/;50;True"

    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-21-39-support-Pond-SAC-2/;50;False"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-18-57-support-Pond-DDPG-2/;50;False"

    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-21-39-support-Pond-SAC-2/;49;True"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-18-57-support-Pond-DDPG-2/;49;True"

    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-21-39-support-Pond-SAC-2/;49;False"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-11-26T19-18-57-support-Pond-DDPG-2/;49;False"

    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-12-04T18-14-47-support-Pond-SAC-5/;50;False"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-12-04T18-14-47-support-Pond-SAC-5/;50;True"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-12-04T18-11-47-support-Pond-DDPG-5/;50;False"
    # "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-12-04T18-11-47-support-Pond-DDPG-5/;50;True"
    "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-12-07T02-09-24-pond-dense-SAC-1/;50;True"
    "${RESULTS_BASE}/gym/Point2DEnv/Pond-v0/2019-12-07T02-09-24-pond-dense-SAC-1/;50;False"
)

EVALUATION_TASKS=(
    # "Pothole-v0"
    # "HeightField-v0"
    # "PerturbBody-v0"
    # "PerturbBody-v1"
    # "PerturbBody-v2"
    # "PerturbBody-AntPond-v0"
    # "Wind-v0"
    # "Wind-AntPond-v0"
    "PerturbRandomAction-v0"
    "PerturbNoisyAction-v0"
    # "Wind-point_mass-orbit_pond-v0"
    # "PerturbBody-point_mass-orbit_pond-v0"
)

for EXPERIMENT_PATH_AND_CHECKPOINT_ID in ${EXPERIMENT_PATHS_AND_CHECKPOINT_IDS[@]}; do
    IFS=";"; set -- ${EXPERIMENT_PATH_AND_CHECKPOINT_ID};
    experiment_path="${1}"
    checkpoint_id="${2}"
    deterministic="${3}"

    for evaluation_task in ${EVALUATION_TASKS[@]}; do

        python -m examples.development.simulate_environments \
               "${experiment_path}" \
               --num-rollouts=10 \
               --evaluation-task="${evaluation_task}" \
               --desired-checkpoint="${checkpoint_id}" \
               --deterministic="${deterministic}"

    done
done
