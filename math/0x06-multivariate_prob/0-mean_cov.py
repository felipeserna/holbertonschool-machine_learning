#!/usr/bin/env python3
"""
Calculates the mean and covariance of a data set
"""


import numpy as np


def mean_cov(X):
    """
    Returns: mean, cov
    """
    n = len(X)
    d = len(X[0])

    if type(X) is not np.ndarray or len(X.shape) != 2:
        TypeError("X must be a 2D numpy.ndarray")

    if n < 2:
        ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0).reshape(1, d)

    deviation = X - mean

    cov = np.matmul(deviation.T, deviation) / (n - 1)

    return mean, cov
