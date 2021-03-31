#!/usr/bin/env python3
"""
Calculates the probability density function of a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    Returns: P, or None on failure
    - P is a numpy.ndarray of shape (n,)
      containing the PDF values for each data point
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    return None
