#!/usr/bin/env python3
"""
Builds an inception block as described in
Going Deeper with Convolutions (2014)
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    F3R_layer = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                                activation='relu',
                                kernel_initializer='he_normal')(A_prev)

    F5R_layer = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                                activation='relu',
                                kernel_initializer='he_normal')(A_prev)

    max_pooling2d = K.layers.MaxPool2D(pool_size=3, strides=1,
                                       padding='same')(A_prev)

    F1_layer = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(A_prev)

    F3_layer = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(F3R_layer)

    F5_layer = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(F5R_layer)

    FPP_layer = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                                activation='relu',
                                kernel_initializer='he_normal')(max_pooling2d)

    concatenate = K.layers.Concatenate([F1_layer,
                                        F3_layer,
                                        F5_layer,
                                        FPP_layer])

    return concatenate
