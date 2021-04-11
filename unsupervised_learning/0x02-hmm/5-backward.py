#!/usr/bin/env python3
"""
Performs the backward algorithm for a hidden markov model
"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Returns: P, B, or None, None on failure
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

    Beta = np.zeros((N, T))
    # initialization
    Beta[:, T - 1] = np.ones(N)

    # recursion
    for t in range(T - 2, -1, -1):
        a = Transition
        b = Emission[:, Observation[t + 1]]
        c = Beta[:, t + 1]

        abc = a * b * c
        prob = np.sum(abc, axis=1)
        Beta[:, t] = prob

    # sum of path probabilities over all possible states
    # end of path
    P_first = Initial[:, 0] * Emission[:, Observation[0]] * Beta[:, 0]
    P = np.sum(P_first)

    return P, Beta
