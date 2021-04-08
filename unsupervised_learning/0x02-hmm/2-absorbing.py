#!/usr/bin/env python3
"""
Determines if a markov chain is absorbing
"""


import numpy as np


def absorbing(P):
    """
    Returns: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) is not 2:
        return False
    n, columns = P.shape

    if n != columns:
        return False

    if np.sum(P, axis=1).all() != 1:
        return False

    D = np.diagonal(P)

    if not np.any(D == 1):
        return False

    if np.all(D == 1):
        return True

    count = np.count_nonzero(D == 1)
    B = P[count:, count:]
    Id = np.eye(B.shape[0])

    try:
        if (np.any(np.linalg.inv(Id - B))):
            return True
    except np.linalg.LinAlgError:
        return False
