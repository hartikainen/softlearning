from .simple_sampler import SimpleSampler


class GoalSampler(SimpleSampler):
    @property
    def _action_input(self):
        observation = self.env.convert_to_active_observation(
            self._current_observation)[None]
        goal = self._current_observation['desired_goal'][None]

        return [observation, goal]
