#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    input_w, input_h, m = images.shape[2], images.shape[1], images.shape[0]
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]

    output_h = input_h
    output_w = input_w

    pad_along_height = max((filter_h - 1), 0)
    pad_along_width = max((filter_w - 1), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    output = np.zeros((m, output_h, output_w))

    images_padded = np.zeros((m, input_h + pad_along_height,
                              input_w + pad_along_width))
    images_padded[:, pad_top:-pad_bottom, pad_left:-pad_right] = images

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * images_padded[:, y: y + filter_h,
                               x: x + filter_w]).sum(axis=(1, 2))

    return output
