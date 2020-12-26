#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """
        Class constructor
        nx is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # weights vector for the neuron
        # default mean is 0
        # default stddev is 1
        self.__W = np.random.normal(size=(1, nx))
        # bias for the neuron
        self.__b = 0
        # activated output of the neuron (prediction)
        self.__A = 0

    # getter functions
    @property
    def W(self):
        """Retrieves the weights vector"""
        return self.__W

    @property
    def b(self):
        """Retrieves the bias"""
        return self.__b

    @property
    def A(self):
        """Retrieves the activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        """
        Z = np.matmul(self.__W, X) + self.__b
        # sigmoid activation function
        self.__A = np.exp(Z)/(np.exp(Z) + 1)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        - Y is a numpy.ndarray with shape (1, m) that contains
          the correct labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing
          the activated output of the neuron (prediction) for each example
        """
        m = Y.shape[1]
        cost = (-1/m)*np.sum(np.multiply(Y, np.log(A)) +
                             np.multiply((1-Y), np.log(1.0000001 - A)))
        return cost
