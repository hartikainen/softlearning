import os

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables

from examples.metric_learning.variants import get_variant_spec
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_ray)


def run_experiment(variant, reporter=None):
    if 'ray' in variant['mode']:
        set_seed(variant['run_params']['seed'])

    env = get_environment_from_variant(variant)
    replay_pool = get_replay_pool_from_variant(variant, env)
    sampler = get_sampler_from_variant(variant)
    Qs = get_Q_function_from_variant(variant, env)
    policy = get_policy_from_variant(variant, env, Qs)
    initial_exploration_policy = get_policy('UniformPolicy', env)

    algorithm = get_algorithm_from_variant(
        variant=variant,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        Qs=Qs,
        pool=replay_pool,
        sampler=sampler,
    )

    session = tf.keras.backend.get_session()
    initialize_tf_variables(session, only_uninitialized=True)

    # Do the training
    for diagnostics in algorithm.train():
        reporter(**diagnostics)


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = get_variant_spec(universe, domain, task, args.policy)
    variant_spec['mode'] = args.mode

    local_dir_base = (
        '~/ray_results/local'
        if args.mode in ('local', 'debug')
        else '~/ray_results')
    local_dir = os.path.join(local_dir_base, universe, domain, task)

    launch_experiments_ray(
        [variant_spec], args, local_dir, run_experiment)


if __name__ == '__main__':
    main()
