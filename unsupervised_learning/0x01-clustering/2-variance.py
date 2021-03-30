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

    if X.shape[0] < C.shape[0] or X.shape[1] != C.shape[1]:
        return None

    dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    min_dist = np.min(dist, axis=0)
    intra_var = (min_dist**2).sum()
    total_var = (intra_var).sum()

    return total_var
