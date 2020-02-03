import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.utils.tensorflow import cast_and_concat


class EnsembleModel(tf.keras.Model):
    def __init__(self, models, *args, **kwargs):
        result = super(EnsembleModel, self).__init__(*args, **kwargs)
        self.models = models
        return result

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs = cast_and_concat(inputs)
        model_outputs = [model(inputs) for model in self.models]
        outputs = tf.stack(model_outputs, axis=-2)

        return outputs

    def get_config(self, *args, **kwargs):
        base_config = {}

        model_configs = [
            {
                'config': model.get_config(),
                'class_name': model.__class__.__name__,
            }
            for model in self.models
        ]
        config = {**base_config, 'model_configs': model_configs}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
        model_configs = config.pop('model_configs')
        models = [
            deserialize_layer(
                model_config, custom_objects=custom_objects)
            for model_config in model_configs
        ]
        return cls(models, **config)


class RandomPriorModel(tf.keras.Model):
    def __init__(self, model, prior, *args, **kwargs):
        result = super(RandomPriorModel, self).__init__(*args, **kwargs)
        self.model = model
        self.prior = prior
        return result

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs = cast_and_concat(inputs)
        model_output = self.model(inputs)
        prior_output = self.prior(inputs)
        # TODO(hartikainen): This should not be needed since prior always has
        # `trainable=False`. Remove once sanity-checked things work as
        # expected.
        output = model_output + tf.stop_gradient(prior_output)
        return output

    @tf.function(experimental_relax_shapes=True)
    def model_and_prior(self, inputs):
        inputs = cast_and_concat(inputs)
        model_output = self.model(inputs)
        prior_output = self.prior(inputs)
        # TODO(hartikainen): This should not be needed since prior always has
        # `trainable=False`. Remove once sanity-checked things work as
        # expected.
        return model_output, prior_output

    def get_config(self, *args, **kwargs):
        base_config = {}

        model_config = {
            'config': self.model.get_config(),
            'class_name': self.model.__class__.__name__,
        }
        prior_config = {
            'config': self.prior.get_config(),
            'class_name': self.prior.__class__.__name__,
        }
        config = {
            **base_config,
            'model_config': model_config,
            'prior_config': prior_config
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
        model_config = config.pop('model_config')
        model = deserialize_layer(
            model_config, custom_objects=custom_objects)
        prior_config = config.pop('prior_config')
        prior = deserialize_layer(
            prior_config, custom_objects=custom_objects)
        return cls(model, prior, **config)


def feedforward_random_prior_ensemble_model(ensemble_size,
                                            prior_kernel_initializer=None,
                                            **model_kwargs):
    models = [
        feedforward_model(
            **model_kwargs,
            name=f"ensemble-model-{i}")
        for i in range(ensemble_size)
    ]
    prior_kernel_initializer = (
        prior_kernel_initializer
        or tf.random_normal_initializer(stddev=0.1))
    priors = [
        feedforward_model(
            **model_kwargs,
            trainable=False,
            kernel_initializer=prior_kernel_initializer,
            name=f"ensemble-prior-{i}")
        for i in range(ensemble_size)
    ]
    random_prior_models = [
        RandomPriorModel(model, prior)
        for model, prior in zip(models, priors)
    ]
    random_prior_ensemble = EnsembleModel(random_prior_models)
    return random_prior_ensemble
