#!/usr/bin/env python3
"""
Performs the forward algorithm for a hidden markov model
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Returns: P, F, or None, None on failure
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None

    if Observation.shape[0] <= 0:
        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if Emission.shape[0] <= 0 or Emission.shape[1] <= 0:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if Transition.shape[0] != Transition.shape[1]:
        return None, None

    if Emission.shape[0] != Transition.shape[0]:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    if Initial.shape[0] != Transition.shape[0] or Initial.shape[1] != 1:
        return None, None

    return None, None
