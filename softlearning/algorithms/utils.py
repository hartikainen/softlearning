from copy import deepcopy


def create_goal_conditioned_sac_algorithm(variant, *args, **kwargs):
    from .goal_conditioned_sac import GoalConditionedSAC

    algorithm = GoalConditionedSAC(*args, **kwargs)

    return algorithm


def create_her_sac_algorithm(variant, *args, **kwargs):
    from .goal_conditioned_sac import HERSAC

    algorithm = HERSAC(*args, **kwargs)

    return algorithm


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_goal_conditioned_metric_learning_algorithm(
        variant,
        *args,
        **kwargs):
    from .goal_conditioned_metric_learning_algorithm import (
        GoalConditionedMetricLearningAlgorithm)

    algorithm = GoalConditionedMetricLearningAlgorithm(*args, **kwargs)

    return algorithm


def create_metric_learning_algorithm(variant, *args, **kwargs):
    from .metric_learning_algorithm import MetricLearningAlgorithm

    algorithm = MetricLearningAlgorithm(*args, **kwargs)

    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL

    algorithm = SQL(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'GoalConditionedSAC': create_goal_conditioned_sac_algorithm,
    'HERSAC': create_her_sac_algorithm,
    'GoalConditionedMetricLearningAlgorithm': (
        create_goal_conditioned_metric_learning_algorithm),
    'MetricLearningAlgorithm': create_metric_learning_algorithm,
    'SQL': create_SQL_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
