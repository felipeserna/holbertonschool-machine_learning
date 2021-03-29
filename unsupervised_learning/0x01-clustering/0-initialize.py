#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means
"""


import numpy as np


def initialize(X, k):
    """
    Returns: a numpy.ndarray of shape (k, d)
    containing the initialized centroids for each cluster, or None on failure
    """
    if k <= 0:
        return None

    d = X.shape[1]
    mini = np.amin(X, axis=0)
    maxi = np.amax(X, axis=0)

    # Initialize cluster centroids
    try:
        init = np.random.uniform(mini, maxi, size=(k, d))
    except Exception:
        return None

    return init
