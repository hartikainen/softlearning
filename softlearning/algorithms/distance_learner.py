import tensorflow as tf

tf.enable_eager_execution()


class DistanceLearner(object):
    def __init__(self):
        self._build()

    def _build(self):
        objective_observations = tf.keras.layers.Input(
            dtype='float32', name='objective_observations')
        step_constraint_observations = tf.keras.layers.Input(
            dtype='float32', name='step_constraint_observations')
        # step_constraint_distances = tf.keras.layers.Input(
        #     dtype='float32', name='step_constraint_distances')

        zero_constraint_observations = tf.keras.layers.Input(
            dtype='float32', name='zero_constraint_observations')

        triangle_inequality_constraint_observations = tf.keras.layers.Input(
            dtype='float32',
            name='triangle_inequality_constraint_observations')

        distance_estimator = tf.keras.Sequential((
            tf.keras.layers.Concatenate(axis=-1),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='softplus'),
        ))

        objective_distances = distance_estimator(objective_observations)
        step_constraint_distances = distance_estimator(
            step_constraint_distances)
        zero_constraint_distances = distance_estimator(
            zero_constraint_distances)
        triangle_inequality_constraint_distances = distance_estimator(
            triangle_inequality_constraint_distances)

        tf.keras.model.compile(
            optimizer='adam',
            loss={
                'objective_distances': lambda x, _: -x,
                'step_constraint_distances': None,
                'distance_output': 'binary_crossentropy',
            },
            loss_weights={'main_output': 1.0, 'aux_output': 1.0}
        )


def main():
    pass


if __name__ == '__main__':
    main()
