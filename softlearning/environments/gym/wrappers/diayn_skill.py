"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np

from gym import spaces
from gym import ObservationWrapper

STATE_KEY = 'state'


class DiaynSkillWrapper(ObservationWrapper):
    """Augment observations by pixel values."""
    def __init__(self, env, num_skills):
        assert isinstance(num_skills, int), (num_skills, type(num_skills))
        self._num_skills = num_skills
        super(DiaynSkillWrapper, self).__init__(env)

        wrapped_observation_space = env.observation_space
        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space,
                        (spaces.Dict, collections.MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if 'active_skill' in invalid_keys:
            raise ValueError("Duplicate or reserved observation key {!r}."
                             .format('active_skill'))

        if self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # skills_space = spaces.MultiBinary(n=num_skills)
        skills_space = spaces.Box(shape=(num_skills, ), low=0, high=1)
        self.observation_space.spaces['active_skill'] = skills_space

        self._active_skill = np.random.randint(0, self._num_skills)
        self._fixed_skill = None
        self._env = env

    def observation(self, observation):
        skill_observation = self._add_active_skill_to_observation(observation)
        return skill_observation

    def _add_active_skill_to_observation(self, observation):
        active_skill_one_hot = np.eye(
            self._num_skills, dtype=np.float32)[self._active_skill]

        if self._observation_is_dict:
            dict_observation = type(observation)(observation)
        else:
            dict_observation = collections.OrderedDict()
            dict_observation[STATE_KEY] = observation

        dict_observation['active_skill'] = active_skill_one_hot
        return dict_observation

    def reset(self, *args, **kwargs):
        if self._fixed_skill is not None:
            self._active_skill = self._fixed_skill
        else:
            self._active_skill = np.random.randint(0, self._num_skills)
        return super(DiaynSkillWrapper, self).reset(*args, **kwargs)

    def fix_skill(self, skill_id):
        self._fixed_skill = skill_id
