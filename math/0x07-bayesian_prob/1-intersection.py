#!/usr/bin/env python3
"""
Calculates the likelihood of obtaining this data
given various hypothetical probabilities of developing severe side effects.

Calculates the intersection of obtaining this data
with the various hypothetical probabilities.
"""


import numpy as np


def likelihood(x, n, P):
    """
    Returns: a 1D numpy.ndarray containing the likelihood
    of obtaining the data, x and n, for each probability in P, respectively
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

    for prob in P:
        if prob < 0 or prob > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    n_x_fact = np.math.factorial(n - x)

    likelihoods = \
        (n_fact / (x_fact * n_x_fact)) * (P ** x) * (1 - P) ** (n - x)

    return likelihoods


def intersection(x, n, P, Pr):
    """The Intersection"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) and x < 0:
        raise ValueError("x must be an integer that is",
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")
    A = likelihood(x, n, P)
    intersect = A * Pr

    return intersect
