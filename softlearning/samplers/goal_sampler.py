from softlearning.models.utils import flatten_input_structure
from .simple_sampler import SimpleSampler


class GoalSampler(SimpleSampler):
    @property
    def _policy_input(self):
        observation = {
            key: self._current_observation[key][None, ...]
            for key in self.policy.observation_keys
        }
        goal = {
            key: self._current_observation[
                self.env.goal_key_map_inverse[key]
            ][None, ...]
            for key in self.policy.goal_keys
        }
        policy_input = flatten_input_structure({
            'observations': observation,
            'goals': goal,
        })

        return policy_input

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        full_observation = observation.copy()
        observation = {
            key: full_observation[key]
            for key in self.policy.observation_keys
        }
        goal = {
            key: full_observation[
                self.env.goal_key_map_inverse[key]
            ]
            for key in self.policy.goal_keys
        }
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'goals': goal,
            'infos': info,
        }

        return processed_observation
