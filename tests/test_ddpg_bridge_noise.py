import pickle
from itertools import count

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import ray
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import img_as_float, io
from pprint import pprint

from softlearning.environments.utils import get_environment
from softlearning.samplers.utils import rollouts
from softlearning.utils.video import save_video


# register_environments()


class OptimalDDPGPolicy:
    def __init__(self, environment, noise_scale=1.0):
        self.environment = environment
        self.observation_keys = environment.observation_keys
        self.noise_distribution = tfp.distributions.Normal(
            loc=0.0, scale=noise_scale)

    def actions_np(self, *args):
        actions = np.array([(
            1.0, np.clip(self.noise_distribution.sample(), -1.0, 1.0)
        )])
        return actions

    def reset(self, *args):
        pass


@ray.remote
def evaluate(bridge_length, noise_scale, episode_length, num_episodes):
    environment = get_environment(
        'dm_control',
        'point_mass',
        'bridge_run',
        {'bridge_length': bridge_length})
    policy = OptimalDDPGPolicy(
        environment=environment,
        noise_scale=noise_scale)
    paths = rollouts(num_episodes,
                     environment,
                     policy,
                     path_length=episode_length,
                     render_kwargs=None)
    past_bridges = [not path['terminals'][-1] for path in paths]
    return np.mean(past_bridges)


def main():
    EPISODE_LENGTH = 200
    NUM_EPISODES = 25
    # task_kwargs={'time_limit': 1, 'bridge_length': 50.0},

    BRIDGE_LENGTHS = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    NOISE_SCALES = (3e-2, 1e-1, 2e-1, 3e-1, 1.0)

    ray.init()

    remotes = {
        (bridge_length, noise_scale):
        evaluate.remote(
            bridge_length, noise_scale, EPISODE_LENGTH, NUM_EPISODES)
        for bridge_length in BRIDGE_LENGTHS
        for noise_scale in NOISE_SCALES
    }

    keys, object_ids = list(zip(*remotes.items()))

    values = ray.get(list(object_ids))

    results = dict(zip(keys, values))

    dataframe = pd.DataFrame([
        {
            'bridge_length': bridge_length,
            'noise_scale': str(noise_scale),
            'result': result,
        }
        for (bridge_length, noise_scale), result in results.items()
    ])

    print(results)

    sns.lineplot(
        data=dataframe,
        x='bridge_length',
        y='result',
        hue='noise_scale',
        legend='brief',
    )

    # plt.show()
    plt.savefig('/tmp/ddpg-noise-test.pdf')


if __name__ == '__main__':
    main()
