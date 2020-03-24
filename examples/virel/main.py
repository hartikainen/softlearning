import copy
import sys

import tree

from softlearning.environments.utils import get_environment_from_params
from softlearning import algorithms
from softlearning import policies
from softlearning import value_functions
from softlearning import replay_pools
from softlearning import samplers

from examples.instrument import run_example_local
from examples.development import main as development_main


class ExperimentRunner(development_main.ExperimentRunner):
    def _build(self):
        variant = copy.deepcopy(self._variant)
        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        variant['f_params']['config'].update({
            'input_shapes': (
                self.training_environment.observation_shape,
                self.training_environment.action_shape),
        })
        fs = self.fs = tree.flatten(value_functions.get(variant['f_params']))

        variant['Q_params']['config'].update({
            'input_shapes': (
                training_environment.observation_shape,
                training_environment.action_shape),
        })
        Qs = self.Qs = tree.flatten(value_functions.get(variant['Q_params']))

        variant['policy_params']['config'].update({
            'action_range': (training_environment.action_space.low,
                             training_environment.action_space.high),
            'input_shapes': training_environment.observation_shape,
            'output_shape': training_environment.action_shape,
        })
        policy = self.policy = policies.get(variant['policy_params'])

        variant['replay_pool_params']['config'].update({
            'environment': training_environment,
        })
        replay_pool = self.replay_pool = replay_pools.get(
            variant['replay_pool_params'])

        variant['sampler_params']['config'].update({
            'environment': training_environment,
            'policy': policy,
            'pool': replay_pool,
        })
        sampler = self.sampler = samplers.get(variant['sampler_params'])

        variant['algorithm_params']['config'].update({
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'Qs': Qs,
            'fs': fs,
            'pool': replay_pool,
            'sampler': sampler
        })
        self.algorithm = algorithms.get(variant['algorithm_params'])

        self._built = True


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.virel', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
