#!/usr/bin/env python3
"""
Flips an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Returns the flipped image
    """
    return tf.image.flip_left_right(image)
