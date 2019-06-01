from collections import OrderedDict

from softlearning.preprocessors.utils import get_preprocessor_from_params
from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def feedforward_distance_estimator(input_shapes,
                                   *args,
                                   preprocessors=None,
                                   observation_keys=None,
                                   name='feedforward_distance_estimator',
                                   **kwargs):
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    model = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    model = PicklableModel(inputs_flat, model(preprocessed_inputs))
    model.observation_keys = observation_keys

    return model


DISTANCE_ESTIMATORS = {
    'FeedforwardDistanceEstimator': feedforward_distance_estimator,
}


def get_distance_estimator_from_variant(variant, env, *args, **kwargs):
    distance_estimator_params = variant['distance_estimator_params']
    distance_estimator_type = distance_estimator_params['type']
    distance_estimator_kwargs = distance_estimator_params.get('kwargs', {})

    observation_preprocessors_params = distance_estimator_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = distance_estimator_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    observation_shapes = OrderedDict((
        (key, value)
        for key, value in env.observation_shape.items()
        if key in observation_keys
    ))
    action_shape = env.action_shape
    input_shapes = {
        'observations': observation_shapes,
    }

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue
        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    preprocessors = {'observations': observation_preprocessors}

    condition_with_action = distance_estimator_kwargs.pop('condition_with_action')
    if condition_with_action:
        input_shapes['actions'] = action_shape
        preprocessors['actions'] = None

    distance_estimator = DISTANCE_ESTIMATORS[distance_estimator_type](
        input_shapes,
        *args,
        observation_keys=observation_keys,
        preprocessors=preprocessors,
        **distance_estimator_kwargs,
        **kwargs)
    distance_estimator.condition_with_action = condition_with_action

    return distance_estimator
