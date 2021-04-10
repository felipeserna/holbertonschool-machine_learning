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

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]

    N, _ = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None

    if not np.sum(Transition, axis=1).all():
        return None, None

    if not np.sum(Initial) == 1:
        return None, None

    F = np.zeros((N, T))

    # Initialization
    init_Obs = Observation[0]
    F[:, 0] = np.multiply(Initial.T, Emission[:, init_Obs])

    # Recursion
    for i in range(1, T):
        Obs_i = Observation[i]
        state = np.matmul(F[:, i - 1], Transition)
        F[:, i] = np.multiply(state, Emission[:, Obs_i])

    # Sum of path probabilities over all possible states
    # end of path
    P = np.sum(F[:, T - 1])

    return P, F
