#!/usr/bin/env python3
"""
Represents a normal distribution
"""


class Normal:
    """Represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor
        - data is a list of the data to be used to estimate the distribution
        - mean is the mean of the distribution
        - stddev is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                # save mean and stddev as a floats
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # Calculate the mean and standard deviation of data
                self.mean = float(sum(data)/len(data))
                self.stddev = float(sum((x - self.mean)**2 for x in data) /
                                    len(data))**0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        """
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        z is the z-score
        """
        return (z*self.stddev + self.mean)
