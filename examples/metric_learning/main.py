import os
import copy
import pickle
import sys

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.models.utils import get_metric_learner_from_variant
from softlearning.misc.utils import initialize_tf_variables

from examples.development.main import ExperimentRunner
from examples.instrument import run_example_local


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
            get_policy(variant['exploration_policy_params']['type'], env))

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
            session=self._session,
        )

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                pickleable = pickle.load(f)

        env = self.env = pickleable['env']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, env))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = pickleable['sampler']
        Qs = self.Qs = pickleable['Qs']
        # policy = self.policy = pickleable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, env, Qs))
        self.policy.set_weights(pickleable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', env))

        metric_learner = get_metric_learner_from_variant(self._variant, env)

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            env=self.env,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            metric_learner=metric_learner,
            session=self._session)
        self.algorithm.__setstate__(pickleable['algorithm'].__getstate__())

        initialize_tf_variables(self._session, only_uninitialized=True)

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)

        # TODO(hartikainen): target Qs should either be checkpointed
        # or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.metric_learning', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
