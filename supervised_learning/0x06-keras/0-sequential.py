#!/usr/bin/env python3
"""
Builds a neural network with the Keras library
"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    - nx is the number of input features to the network
    - layers is a list containing the number of nodes
      in each layer of the network
    - activations is a list containing the activation functions used
      for each layer of the network
    - lambtha is the L2 regularization parameter
    - keep_prob is the probability that a node will be kept for dropout
    - You are not allowed to use the Input class
    - Returns: the keras model
    """
    # Linear stack of layers
    model = K.Sequential()

    # regularization scheme
    l2_reg = K.regularizers.l2(lambtha)

    # first layer "input_shape" argument
    model.add(K.layers.Dense(units=layers[0],
                             activation=activations[0],
                             kernel_regularizer=l2_reg,
                             input_shape=(nx,)))

    # Afterwards, we do automatic shape inference
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=l2_reg))

    return model
