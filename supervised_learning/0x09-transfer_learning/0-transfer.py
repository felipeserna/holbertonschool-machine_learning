#!/usr/bin/env python3
"""
Script that trains a convolutional neural network
to classify the CIFAR 10 dataset.

In the same file, write a function def preprocess_data(X, Y):
that pre-processes the data for your model
"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Returns: X_p, Y_p
    """
    