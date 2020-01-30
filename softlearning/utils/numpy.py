import numpy as np


def custom_combinations(n_features, degree):
    from itertools import (
        chain,
        combinations_with_replacement as c_w_r
    )
    start = 1
    return list(chain.from_iterable(
        c_w_r(range(n_features), i) for i in range(start, degree + 1)))


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)
