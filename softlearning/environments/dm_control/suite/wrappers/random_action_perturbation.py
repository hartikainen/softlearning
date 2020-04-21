"""Wrapper that adds random perturbations to actions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import dm_env


_BOUNDS_MUST_BE_FINITE = (
    'All bounds in `env.action_spec()` must be finite, got: {action_spec}')


class Wrapper(dm_env.Environment):
    """Wraps a control environment and adds Gaussian noise to actions."""

    def __init__(self, env, *args, perturbation_probability=0.0, **kwargs):
        """Initializes a new action noise Wrapper.

        Args:
          env: The control suite environment to wrap.
          scale: The standard deviation of the noise, expressed as a fraction
            of the max-min range for each action dimension.

        Raises:
          ValueError: If any of the action dimensions of the wrapped environment are
            unbounded.
        """
        action_spec = env.action_spec()
        if not (np.all(np.isfinite(action_spec.minimum)) and
                np.all(np.isfinite(action_spec.maximum))):
            raise ValueError(_BOUNDS_MUST_BE_FINITE.format(
                action_spec=action_spec))
        self._minimum = action_spec.minimum
        self._maximum = action_spec.maximum
        self._perturbation_probability = perturbation_probability
        self._env = env
        return super(Wrapper, self).__init__(*args, **kwargs)

    def step(self, action):
        perturbed = np.random.rand() < self._perturbation_probability
        if perturbed:
            perturbed_action = np.random.uniform(self._minimum, self._maximum)
        else:
            perturbed_action = action
        time_step = self._env.step(perturbed_action)
        assert not any(x in time_step.observation for x in (
            'perturbed', 'original_action', 'perturbed_action'))
        time_step.observation.update({
            'perturbed': perturbed,
            'original_action': action,
            'perturbed_action': perturbed_action,
        })
        return time_step

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
