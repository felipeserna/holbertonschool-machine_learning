#!/usr/bin/env python3
"""
Determines if a markov chain is absorbing
"""


import numpy as np


def absorbing(P):
    """
    Returns: True if it is absorbing, or False on failure
    """
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return False

    return False
