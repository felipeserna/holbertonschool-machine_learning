#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix
"""


import numpy as np


def one_hot(labels, classes=None):
    """
    - The last dimension of the one-hot matrix must be the number of classes
    - Returns: the one-hot matrix
    """
    oh_matrix = np.zeros((labels.size, labels.max()+1))
    oh_matrix[np.arange(labels.size), labels] = 1
    return oh_matrix
