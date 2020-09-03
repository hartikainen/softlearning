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
    fell_in_waters = [path['terminals'][-1] for path in paths]
    return np.mean(fell_in_waters)


def main():
    # results = {(1.0, 0.03): 1.0, (1.0, 0.1): 1.0, (1.0, 0.2): 1.0, (1.0, 0.3): 1.0, (1.0, 1.0): 1.0, (5.0, 0.03): 1.0, (5.0, 0.1): 0.96, (5.0, 0.2): 0.6, (5.0, 0.3): 0.52, (5.0, 1.0): 0.12, (10.0, 0.03): 1.0, (10.0, 0.1): 0.76, (10.0, 0.2): 0.52, (10.0, 0.3): 0.24, (10.0, 1.0): 0.0, (15.0, 0.03): 1.0, (15.0, 0.1): 0.64, (15.0, 0.2): 0.24, (15.0, 0.3): 0.08, (15.0, 1.0): 0.0, (20.0, 0.03): 1.0, (20.0, 0.1): 0.6, (20.0, 0.2): 0.12, (20.0, 0.3): 0.12, (20.0, 1.0): 0.0, (25.0, 0.03): 1.0, (25.0, 0.1): 0.56, (25.0, 0.2): 0.12, (25.0, 0.3): 0.0, (25.0, 1.0): 0.0}
    # results = {(1.0, 0.03): 0.0, (1.0, 0.1): 0.0, (1.0, 0.3): 0.0, (1.0, 1.0): 0.0, (5.0, 0.03): 0.0, (5.0, 0.1): 0.04, (5.0, 0.3): 0.36, (5.0, 1.0): 0.8, (10.0, 0.03): 0.0, (10.0, 0.1): 0.32, (10.0, 0.3): 0.72, (10.0, 1.0): 0.92, (15.0, 0.03): 0.0, (15.0, 0.1): 0.4, (15.0, 0.3): 0.96, (15.0, 1.0): 1.0, (20.0, 0.03): 0.0, (20.0, 0.1): 0.4, (20.0, 0.3): 1.0, (20.0, 1.0): 1.0, (25.0, 0.03): 0.12, (25.0, 0.1): 0.44, (25.0, 0.3): 0.92, (25.0, 1.0): 1.0, (30.0, 0.03): 0.08, (30.0, 0.1): 0.68, (30.0, 0.3): 1.0, (30.0, 1.0): 1.0, (35.0, 0.03): 0.04, (35.0, 0.1): 0.72, (35.0, 0.3): 0.96, (35.0, 1.0): 1.0, (40.0, 0.03): 0.12, (40.0, 0.1): 0.84, (40.0, 0.3): 1.0, (40.0, 1.0): 1.0, (45.0, 0.03): 0.08, (45.0, 0.1): 0.8, (45.0, 0.3): 1.0, (45.0, 1.0): 1.0, (50.0, 0.03): 0.08, (50.0, 0.1): 0.92, (50.0, 0.3): 1.0, (50.0, 1.0): 1.0}
    results = {(1.0, 0.03): 0.0, (1.0, 0.1): 0.0, (1.0, 0.3): 0.0, (1.0, 1.0): 0.0, (5.0, 0.03): 0.0, (5.0, 0.1): 0.04, (5.0, 0.3): 0.6, (5.0, 1.0): 0.84, (10.0, 0.03): 0.0, (10.0, 0.1): 0.16, (10.0, 0.3): 0.84, (10.0, 1.0): 1.0, (20.0, 0.03): 0.0, (20.0, 0.1): 0.44, (20.0, 0.3): 0.96, (20.0, 1.0): 1.0, (30.0, 0.03): 0.04, (30.0, 0.1): 0.56, (30.0, 0.3): 1.0, (30.0, 1.0): 1.0, (40.0, 0.03): 0.0, (40.0, 0.1): 0.8, (40.0, 0.3): 1.0, (40.0, 1.0): 1.0, (50.0, 0.03): 0.16, (50.0, 0.1): 0.64, (50.0, 0.3): 1.0, (50.0, 1.0): 1.0, (60.0, 0.03): 0.24, (60.0, 0.1): 0.96, (60.0, 0.3): 1.0, (60.0, 1.0): 1.0, (70.0, 0.03): 0.12, (70.0, 0.1): 0.96, (70.0, 0.3): 1.0, (70.0, 1.0): 1.0, (80.0, 0.03): 0.28, (80.0, 0.1): 0.92, (80.0, 0.3): 1.0, (80.0, 1.0): 1.0, (90.0, 0.03): 0.44, (90.0, 0.1): 0.96, (90.0, 0.3): 1.0, (90.0, 1.0): 1.0, (100.0, 0.03): 0.36, (100.0, 0.1): 0.92, (100.0, 0.3): 1.0, (100.0, 1.0): 1.0}
    # results = None
    if results is None:
        EPISODE_LENGTH = 200
        NUM_EPISODES = 25
        # task_kwargs={'time_limit': 1, 'bridge_length': 50.0},

        # BRIDGE_LENGTHS = (1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
        BRIDGE_LENGTHS = (1.0, 5.0, *np.arange(10.0, 105, 10.0))

        NOISE_SCALES = (3e-2, 1e-1, 3e-1, 1.0)

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
        print(results)

    dataframe = pd.DataFrame([
        {
            'bridge_length': bridge_length,
            'noise_scale': f"={noise_scale}",
            'result': result,
        }
        for (bridge_length, noise_scale), result in results.items()
    ])

    print(dataframe)

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
