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

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        if k == kmin:
            max_var = variance(X, C)

        total_var = variance(X, C)
        dvar = max_var - total_var
        d_vars.append(dvar)

    return results, d_vars
