#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer of a neural network
"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Returns: the output of the convolutional layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]

    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0

    pad_size = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    images_padded = np.pad(A_prev,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    h_new = int(((h_prev - kh + 2 * ph) / sh) + 1)
    w_new = int(((w_prev - kw + 2 * pw) / sw) + 1)

    A_new = np.zeros((m, h_new, w_new, c_new))

    for x in range(w_new):
        for y in range(h_new):
            for z in range(c_new):
                A_new[:, y, x, z] = \
                    (W[:, :, :, z] * images_padded[:,
                     y * sh: y * sh + kh,
                     x * sw: x * sw + kw,
                     :]).sum(axis=(1, 2, 3))

    return A_new
