#!/usr/bin/env python3
"""
Calculates the intersection of obtaining this data
with the various hypothetical probabilities
"""


import numpy as np


def intersection(x, n, P, Pr):
    """
    Returns: a 1D numpy.ndarray containing the intersection
    of obtaining x and n with each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    err = "All values in P must be in the range [0, 1]"
    for prob in P:
        if prob < 0 or prob > 1:
            raise ValueError(err)

    err = "All values in Pr must be in the range [0, 1]"
    for prior in Pr:
        if prior < 0 or prior > 1:
            raise ValueError(err)

    if np.isclose(np.sum(Pr), 1) is False:
        raise ValueError("Pr must sum to 1")

    return None
