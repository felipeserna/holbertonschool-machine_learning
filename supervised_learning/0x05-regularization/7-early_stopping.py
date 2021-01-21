#!/usr/bin/env python3
"""
Determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Returns: a boolean of whether the network should be stopped early,
    followed by the updated count
    """
    for i in range(count, patience + 1):
        if cost > opt_cost * threshold:
            return True, i
        else:
            return False, i
