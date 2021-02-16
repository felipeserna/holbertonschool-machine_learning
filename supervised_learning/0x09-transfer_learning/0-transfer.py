#!/usr/bin/env python3
"""
Script that trains a convolutional neural network
to classify the CIFAR 10 dataset.

In the same file, write a function 'def preprocess_data(X, Y):'
that pre-processes the data for your model
"""


import tensorflow.keras as K


if __name__ == "__main__":
    # InceptionV3
    inc_v3 = K.applications.InceptionV3(include_top=False,
                                        weights="imagenet",
                                        input_tensor=None,
                                        input_shape=None,
                                        pooling=None,
                                        classes=10,
                                        classifier_activation="softmax")

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    model.save('cifar10.h5')


def preprocess_data(X, Y):
    """
    Returns: X_p, Y_p
    """
