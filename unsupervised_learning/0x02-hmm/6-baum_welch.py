#!/usr/bin/env python3
"""
Performs the Baum-Welch algorithm for a hidden markov model
"""


import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None

    if Observations.shape[0] <= 0:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if Transition.shape[0] != Transition.shape[1]:
        return None, None

    if Transition.shape[0] <= 0:
        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if Emission.shape[0] != Transition.shape[0]:
        return None, None

    if Emission.shape[1] <= 0:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    if Initial.shape[0] != Transition.shape[0] or Initial.shape[1] != 1:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    return None, None
