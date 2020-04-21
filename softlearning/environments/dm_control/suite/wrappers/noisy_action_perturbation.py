import dm_env
import numpy as np


__all__ = ['Wrapper']


class Wrapper(dm_env.Environment):
    """Rescale the action space of the environment."""
    def __init__(self, env, *args, noise_scale=0, **kwargs):
        assert np.issubdtype(env.action_spec().dtype, np.floating)
        self._env = env
        self._noise_scale = noise_scale
        return super(Wrapper, self).__init__(*args, **kwargs)

    def step(self, action):
        perturbed_action = self.action(action)
        perturbed = not np.array_equal(action, perturbed_action)
        time_step = self._env.step(perturbed_action)
        assert not any(x in time_step.observation for x in (
            'perturbed', 'original_action', 'perturbed_action'))
        time_step.observation.update({
            'perturbed': perturbed,
            'original_action': action,
            'perturbed_action': perturbed_action,
        })
        return time_step

    def action(self, action):
        noise = np.random.normal(
            scale=self._noise_scale, size=action.shape)

        action = action + noise

        return action

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        if name == '_env':
            return self.__getattribute__('_env')

        return getattr(self._env, name)
