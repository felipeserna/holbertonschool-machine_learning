#!/usr/bin/env python3
"""
Computes to policy with a weight of a matrix.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes to policy with a weight of a matrix.
    """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / np.sum(exp)


def softmax_grad(softmax):
    """
    Softmax gradient
    """
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state
    and a weight matrix.
    Return: the action and the gradient (in this order)
    """
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])

    dsoftmax = softmax_grad(probs)[action, :]

    dlog = dsoftmax / probs[0, action]

    grad = np.dot(state.T, dlog[None, :])

    return action, grad
