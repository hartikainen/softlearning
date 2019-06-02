from ray import tune
import numpy as np

from flatten_dict import flatten, unflatten

from examples.metric_learning import (
    get_variant_spec as get_metric_learning_variant_spec)


def terminal(batch, where_resampled, environment):
    resampled_terminals = np.linalg.norm(
        batch['observations']['state_desired_goal'][where_resampled]
        - batch['goals']['state_desired_goal'][where_resampled],
        ord=2,
        axis=1,
        keepdims=True,
    ) < 0.1

    batch['terminals'][where_resampled] = resampled_terminals


def reward(batch, where_resampled, environment):
    resampled_actions = batch['actions'][where_resampled]
    resampled_observations = type(batch['observations'])(
        (key, values[where_resampled])
        for key, values in batch['observations'].items()
    )

    batch['rewards'][where_resampled] = environment.unwrapped.compute_rewards(
        resampled_actions, resampled_observations)


def update_batch(original_batch, resampled_batch, where_resampled):
    batch_flat = flatten(original_batch)
    resampled_batch_flat = flatten(resampled_batch)

    batch_flat[
        ('goals', 'state_desired_goal')
    ][where_resampled] = resampled_batch_flat[
        ('observations', 'state_observation')]

    updated_batch = unflatten(batch_flat)
    return updated_batch


def get_variant_spec(args):
    variant_spec = get_metric_learning_variant_spec(args)
    variant_spec['Q_params']['type'] = (
        'double_goal_conditioned_feedforward_Q_function')
    variant_spec['algorithm_params']['type'] = (
        'GoalConditionedMetricLearningAlgorithm')
    variant_spec['algorithm_params']['kwargs'].update({
        'eval_n_episodes': 1,
        'plot_distances': True,
    })
    variant_spec['exploration_policy_params']['type'] = (
        'GoalConditionedUniformPolicy')
    variant_spec['policy_params']['type'] = 'GoalConditionedGaussianPolicy'

    variant_spec['sampler_params']['type'] = 'GoalSampler'

    variant_spec['replay_pool_params'] = {
        'type': 'HindsightExperienceReplayPool',
        'kwargs': {
            'max_size': int(1e6),
            'her_strategy': {
                'type': 'future',
                'resampling_probability': 0.8,
            },
            'terminal_fn': tune.function(terminal),
            'reward_fn': tune.function(reward),
            'update_batch_fn': tune.function(update_batch),
        }
    }

    return variant_spec
