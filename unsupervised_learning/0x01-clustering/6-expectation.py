#!/usr/bin/env python3
"""
Calculates the expectation step in the EM algorithm for a GMM
"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Returns: g, l, or None, None on failure
    - g is a numpy.ndarray of shape (k, n) containing
      the posterior probabilities for each data point in each cluster
    - l is the total log likelihood
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None

    k = pi.shape[0]

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None, None

    if k != m.shape[0]:
        return None, None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None, None

    d = X.shape[1]

    if d != m.shape[0] or d != S.shape[0]:
        return None, None

    if S.shape[0] != S.shape[1] or d != S.shape[1]:
        return None, None

    return None, None
