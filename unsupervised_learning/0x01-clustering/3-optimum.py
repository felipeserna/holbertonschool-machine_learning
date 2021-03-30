#!/usr/bin/env python3
"""
Tests for the optimum number of clusters by variance
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Returns: results, d_vars, or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None

    if type(kmax) is not int or kmax <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    if kmax <= kmin:
        return None, None

    return None
