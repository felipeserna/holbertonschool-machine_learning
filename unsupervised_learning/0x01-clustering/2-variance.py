#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a data set
"""


import numpy as np


def variance(X, C):
    """
    Returns: var, or None on failure
    - var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None

    # https://www.youtube.com/watch?v=xNfOheh-res&ab_channel=VictorLavrenko

    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    min_dist = np.min(dist, axis=-1)
    intra_var = np.sum(min_dist**2)
    total_var = np.sum(intra_var)

    return total_var
