import numpy as np


MAZES_BY_TYPE = {
    'u-shape': (
        "#####",
        "#g  #",
        "### #",
        "#s  #",
        "#####",
    ),
}


def create_maze(maze_type, maze_kwargs):
    return np.array([
        [col for col in row]
        for row in MAZES_BY_TYPE[maze_type]
    ])
