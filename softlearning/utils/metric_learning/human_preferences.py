import os
import pickle
import json
import time
import glob
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import skimage.io


def record_query(query_id, observation_id, trial_directory, observation_value=None):
    preference_directory = os.path.join(trial_directory, 'preferences')

    metadata_path = os.path.join(preference_directory, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_queries = len(metadata['queries'])

    assert 0 <= query_id < num_queries, (query_id, num_queries)

    query = metadata['queries'][query_id]
    assert query_id == query['query_id'], (query_id, query['query_id'])
    assert query['best_observation_index'] == 'PENDING', (
        "The query has already been responded.")
    assert query['best_observation_value'] == 'PENDING', (
        "The query has already been responded.")
    assert 0 <= observation_id <= query['num_observations'], (
        f"Incorrect observation_id={observation_id}")

    metadata['queries'][query_id].update({
        'best_observation_index': observation_id,
        'best_observation_value': observation_value,
        'response_time': time.time()
    })
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print("Preference recorded succesfully:")
    pprint(metadata['queries'][query_id])


def view_all_queries(trial_directory):
    preference_directory = os.path.join(trial_directory, 'preferences')

    metadata_path = os.path.join(preference_directory, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_queries = len(metadata['queries'])

    image_grid = []
    current_target = None
    for query_id in range(num_queries):
        query = metadata['queries'][query_id]
        assert query_id == query['query_id'], (query_id, query['query_id'])
        best_observation_index = query['best_observation_index']
        best_observation_value = query['best_observation_value']

        query_directory = os.path.join(preference_directory, f'query-{query_id}')
        pickle_path = os.path.join(query_directory, 'observations.pkl')
        with open(pickle_path, 'rb') as f:
            new_observations = pickle.load(f)

        def get_images_from_observations(observations):
            image_key = next(
                key for key, values in new_observations.items()
                if values.ndim == 4 and values.dtype == np.uint8)

            new_images = observations[image_key]
            previous_target_image = (
                current_target[image_key][None, ...]
                if current_target is not None
                else 255 * np.ones_like(observations[image_key][[0]]))

            images = np.concatenate((new_images, previous_target_image), axis=0)
            return images

        def create_image_row(images, best_observation_index, height=480):
            if isinstance(best_observation_index, int):
                best_image = images[best_observation_index].copy()
                best_image = np.pad(
                    best_image[2:-2, 2:-2, :],
                    pad_width=((2, 2), (2, 2), (0, 0)),
                    mode='constant', constant_values=0.0)
                images[best_observation_index] = best_image

#             image_row = np.transpose(np.concatenate(np.transpose(
#                 images, axes=(0, 2, 1, 3))), axes=(1, 0, 2))

            if not isinstance(best_observation_index, int):
                pass

            repeat_times = int(np.ceil(height / images[0].shape[0]))

            image_row = [
                image.repeat(repeat_times, axis=0).repeat(repeat_times, axis=1)
                for image in images
            ]
            # image_row = image_row.repeat(10, axis=0).repeat(10, axis=1)
            return image_row

        images = get_images_from_observations(new_observations)
        image_row = create_image_row(images, best_observation_index, height=480)
        image_grid.append(image_row)

        if best_observation_index < new_observations['pixels'].shape[0]:
            current_target = type(new_observations)(
                (key, values[best_observation_index])
                for key, values in new_observations.items()
            )

    def show_image_grid(image_grid):
        nrows = len(image_grid)
        ncols = len(image_grid[0])
        figure, axes = plt.subplots(nrows, ncols, gridspec_kw={'wspace':0, 'hspace':0})

        for i, (images_row, axes_row) in enumerate(zip(image_grid, axes)):
            for j, (image, axis) in enumerate(zip(images_row, axes_row)):
                axis.imshow(image)

                axis.set_xticks([])
                axis.set_xticklabels([])
                axis.set_yticks([])
                axis.set_yticklabels([])

                if i == len(image_grid) - 1:
                    axis.set_xlabel(f'{j}', rotation=0, size='large')
                if j == 0:
                    axis.set_ylabel(f'{i}', rotation=90, size='large')

        figure.text(0.5, 0.1, 'observation id', ha='center', va='center', size='20')
        figure.text(0.1, 0.5, 'query id', ha='center', va='center', rotation='vertical', size='20')

    show_image_grid(image_grid)
