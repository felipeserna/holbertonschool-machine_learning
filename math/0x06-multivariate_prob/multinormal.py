#!/usr/bin/env python3
"""
Represents a Multivariate Normal distribution
"""


import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        Class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        deviation = data - self.mean

        self.cov = np.matmul(deviation, deviation.T) / (n - 1)
