#!/usr/bin/env python3
"""
Determines if a markov chain is absorbing
"""


import numpy as np


def absorbing(P):
    """
    Returns: True if it is absorbing, or False on failure
    """
    if np.all(np.diag(P) == 1):
        return True

    if P[0, 0] != 1:
        return False

    P = P[1:, 1:]

    if np.all(np.count_nonzero(P, axis=0) > 2):
        return True
    else:
        return False
