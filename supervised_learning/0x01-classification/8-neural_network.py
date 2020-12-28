#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer
performing binary classification
"""


import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # weights vector for the hidden layer
        # default mean is 0
        # default stddev is 1
        self.W1 = np.random.normal(size=(nodes, nx))
        # The bias for the hidden layer. Upon instantiation,
        # it should be initialized with 0’s.
        self.b1 = np.zeros((nodes, 1))
        # The activated output for the hidden layer. Upon instantiation,
        # it should be initialized to 0
        self.A1 = 0
        # weights vector for the output neuron
        # default mean is 0
        # default stddev is 1
        self.W2 = np.random.normal(size=(1, nodes))
        # bias for the output neuron
        self.b2 = 0
        # activated output for the output neuron (prediction)
        self.A2 = 0
