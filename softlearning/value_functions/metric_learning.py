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


def main():
    from softlearning.environments.utils import get_environment
    env = get_environment('gym', 'Swimmer', 'v2', {})

    observations1_np = env.reset()[None]
    actions_np = env.action_space.sample()[None]
    observations2_np = env.step(actions_np[0])[0][None]

    # batch_size = 12
    # actions_np = np.random.uniform(
    #     0, 1, (batch_size, *action_shape)).astype(dtype=np.float32)
    # observations1_np = np.random.uniform(
    #     0, 1, (batch_size, *observation_shape)).astype(dtype=np.float32)
    # observations2_np = np.random.uniform(
    #     0, 1, (batch_size, *observation_shape)).astype(dtype=np.float32)

    actions_tf = tf.constant(actions_np, dtype=tf.float32)
    observations1_tf = tf.constant(observations1_np, dtype=tf.float32)
    observations2_tf = tf.constant(observations2_np, dtype=tf.float32)

    variant = {
        'Q_params': {
            'type': 'metric_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (64, 64),
            }
        },
    }

    Q = utils.get_Q_function_from_variant(variant, env)

    Q_np = Q.predict([observations1_np, observations2_np, actions_np])
    Q_tf = Q([observations1_tf, observations2_tf, actions_tf])


if __name__ == '__main__':
    main()
