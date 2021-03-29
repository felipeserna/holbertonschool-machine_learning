#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means.
Performs K-means on a dataset.
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Returns: C, clss, or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    d = X.shape[1]
    mini = np.amin(X, axis=0)
    maxi = np.amax(X, axis=0)

    # Initialize cluster centroids
    init = np.random.uniform(mini, maxi, size=(k, d))

    return init
