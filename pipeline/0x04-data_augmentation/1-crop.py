#!/usr/bin/env python3
"""
Performs a random crop of an image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Returns the cropped image
    """
    return tf.image.random_crop(
        image, size
    )
