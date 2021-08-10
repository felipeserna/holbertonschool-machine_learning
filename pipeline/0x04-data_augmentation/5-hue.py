#!/usr/bin/env python3
"""
Changes the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Returns the altered image
    """
    return tf.image.adjust_hue(
        image, delta
    )
