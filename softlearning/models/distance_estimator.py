from softlearning.preprocessors.utils import get_preprocessor_from_params
from softlearning.models.feedforward import feedforward_model


def feedforward_distance_estimator(*args,
                                   preprocessors=None,
                                   name='feedforward_distance_estimator',
                                   **kwargs):
    model = feedforward_model(
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)

    return model


DISTANCE_ESTIMATORS = {
    'FeedforwardDistanceEstimator': feedforward_distance_estimator,
}


def get_distance_estimator_from_variant(variant, env, *args, **kwargs):
    distance_estimator_params = variant['distance_estimator_params']
    distance_estimator_type = distance_estimator_params['type']
    distance_estimator_kwargs = distance_estimator_params.get('kwargs', {})

    observation_shape = env.active_observation_shape
    action_shape = env.action_space.shape

    distance_input_type = (
        variant['metric_learner_params']['kwargs']['distance_input_type'])

    preprocessor_params = distance_estimator_kwargs.pop(
        'preprocessor_params', None)
    preprocessor = get_preprocessor_from_params(env, preprocessor_params)

    if distance_input_type == 'full':
        input_shapes = (observation_shape, observation_shape)
        preprocessors = (preprocessor, preprocessor)
    elif distance_input_type in ('xy_coordinates', 'xy_velocities'):
        input_shapes = (observation_shape, (2, ))
        preprocessors = (preprocessor, None)
    else:
        raise NotImplementedError(distance_input_type)

    if variant['metric_learner_params']['kwargs']['condition_with_action']:
        input_shapes = (input_shapes[0], action_shape, input_shapes[1])
    input_shape = tuple(
        sum(np.prod(shape, keepdims=True) for shape in input_shapes))
        preprocessors = (preprocessors[0], None, preprocessors[1])

    return DISTANCE_ESTIMATORS[distance_estimator_type](
        input_shapes=(input_shape, ),
        *args,
        preprocessors=preprocessors,
        **distance_estimator_kwargs,
        **kwargs)
