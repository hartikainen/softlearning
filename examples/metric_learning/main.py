import os
import copy

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.models.utils import get_metric_learner_from_variant
from softlearning.misc.utils import initialize_tf_variables

from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_ray)
from examples.development.main import ExperimentRunner
from .variants import get_variant_spec


class MetricExperimentRunner(ExperimentRunner):
    def _build(self):
        variant = copy.deepcopy(self._variant)

        env = self.env = get_environment_from_variant(variant)
        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, env))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(variant, env)
        policy = self.policy = get_policy_from_variant(variant, env, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', env))

        metric_learner = get_metric_learner_from_variant(variant, env)

        self.algorithm = get_algorithm_from_variant(
            variant=variant,
            env=env,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            metric_learner=metric_learner,
        )

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True


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
        [variant_spec], args, local_dir, MetricExperimentRunner)


if __name__ == '__main__':
    main()
