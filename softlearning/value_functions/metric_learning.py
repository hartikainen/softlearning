import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from . import utils


def create_state_action_goal_Q_function(observation_shape,
                                        action_shape,
                                        *args,
                                        observation_preprocessor=None,
                                        name="metric_Q",
                                        **kwargs):
    input_shapes = (observation_shape, observation_shape, action_shape)
    preprocessors = (observation_preprocessor, None, None)

    Q_function = feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)

    return Q_function
