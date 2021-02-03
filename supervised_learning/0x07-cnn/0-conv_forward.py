#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer of a neural network
"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding

    if padding == "same":
        ph = ((h_prev * (sh - 1)) - sh + kh) // 2
        pw = ((w_prev * (sw - 1)) - sw + kw) // 2

    if padding == "valid":
        ph = 0
        pw = 0

    padded_imgs = np.pad(A_prev,
                         pad_width=((0, 0),
                                    (ph, ph), (pw, pw),
                                    (0, 0)),
                         mode='constant',
                         constant_values=0)

    h_output = (padded_imgs.shape[1] - kh)//sh + 1
    w_output = (padded_imgs.shape[2] - kw)//sw + 1

    convoluted = np.zeros((m, h_output, w_output, c_new))

    for j in range(h_output):
        for i in range(w_output):
            for k in range(c_new):
                convoluted[:, j, i, k] = np.sum(padded_imgs[:,
                                                            j*sh: j*sh + kh,
                                                            i*sw: i*sw + kw] *
                                                W[:, :, :, k], axis=(1, 2, 3))
    output = activation(convoluted + b)
    return output
