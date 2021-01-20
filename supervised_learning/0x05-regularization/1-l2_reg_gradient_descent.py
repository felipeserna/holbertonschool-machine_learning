#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using gradient descent
with L2 regularization
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a NN using gradient descent
    with L2 regularization
    """
    m = Y.shape[1]

    for i in reversed(range(L)):
        # if it is the last layer
        if i == L - 1:
            dZ = cache['A{}'.format(i + 1)] - Y

        # hidden layers
        else:
            dZa = np.matmul(weights['W{}'.format(i + 2)].T, dZ)
            # derivative of tanh function
            dZb = 1 - cache['A{}'.format(i + 1)] ** 2
            dZ = dZa * dZb

        dW = ((np.matmul(dZ, cache['A{}'.format(i)].T)) / m) + \
            (lambtha / m) * weights['W{}'.format(i + 1)]
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W{}'.format(i + 1)] = \
            weights['W{}'.format(i + 1)] \
            - (alpha * dW)
        weights['b{}'.format(i + 1)] = \
            weights['b{}'.format(i + 1)] \
            - (alpha * db)
