import os
import copy
import glob
import pickle
import sys

import tensorflow as tf
from ray import tune

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import (
    get_policy_from_variant, get_policy_from_params)
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from examples.development.main import ExperimentRunner

from softlearning.utils.tensorflow import initialize_tf_variables
from softlearning.models.feedforward import feedforward_model
from examples.instrument import run_example_local

tf.compat.v1.disable_eager_execution()


def create_discriminator(num_skills):
    discriminator = feedforward_model(
        hidden_layer_sizes=(512, 512),
        output_size=num_skills,
        activation='relu',
        output_activation='linear',
        name='discriminator',
    )
    return discriminator


class DiaynExperimentRunner(ExperimentRunner):
    def _build(self):
        variant = copy.deepcopy(self._variant)

        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        policy = self.policy = get_policy_from_variant(
            variant, training_environment)

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'], training_environment))

        num_discriminator_skills = (
            variant['discriminator_params']['kwargs']['num_skills'])
        discriminator = create_discriminator(num_discriminator_skills)

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            discriminator=discriminator,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.diayn', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
