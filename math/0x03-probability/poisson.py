#!/usr/bin/env python3
"""
Represents a poisson distribution
"""


class Poisson:
    """Represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        - data is a list of the data to be used to estimate the distribution
        - lambtha is the expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                # save lambtha as a float
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # calculate lambtha of the data
                self.lambtha = float(sum(data)/len(data))
