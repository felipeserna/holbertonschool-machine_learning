#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """
    - Cost is a tensor containing the cost of the network without
    L2 regularization
    - Returns: a tensor containing the cost of the network accounting
    for L2 regularization
    """
    l2_cost = tf.losses.get_regularization_losses()
    new_cost = cost + l2_cost
    return new_cost
