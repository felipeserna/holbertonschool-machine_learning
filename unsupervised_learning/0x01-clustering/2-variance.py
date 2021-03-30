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

    return None
