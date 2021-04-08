#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular markov chain
"""


import numpy as np


def regular(P):
    """
    Returns: a numpy.ndarray of shape (1, n) containing
    the steady state probabilities, or None on failure
    """
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None

    return None
