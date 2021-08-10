#!/usr/bin/env python3
"""
Randomly changes the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Returns the altered image
    """
    return tf.image.adjust_brightness(
        image, max_delta
    )
