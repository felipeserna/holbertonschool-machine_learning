#!/usr/bin/env python3
"""
Represents a bidirectional cell of an RNN
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        # weight for the hidden states in the forward direction
        self.Whf = np.random.normal(size=(i + h, h))

        # weight for the hidden states in the backward direction
        self.Whb = np.random.normal(size=(i + h, h))

        # weight for the outputs
        self.Wy = np.random.normal(size=(h + h, o))

        # bias for the hidden states in the forward direction
        self.bhf = np.zeros((1, h))

        # bias for the hidden states in the backward direction
        self.bhb = np.zeros((1, h))

        # bias for the outputs
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.
        Returns: h_next
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(concat @ self.Whf + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        for one time step.
        Returns: h_prev
        """
        concat = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(concat @ self.Whb + self.bhb)

        return h_prev
