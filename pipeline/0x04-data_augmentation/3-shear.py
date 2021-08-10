#!/usr/bin/env python3
"""
Randomly shears an image
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Returns the sheared image
    """
    # Converts a PIL Image instance to a Numpy array.
    numpy_array = tf.keras.preprocessing.image.img_to_array(
        image
    )

    # Performs a random spatial shear of a Numpy image tensor.
    sheared_numpy = tf.keras.preprocessing.image.random_shear(
        numpy_array, intensity
    )

    # Converts a 3D Numpy array to a PIL Image instance.
    pil_image = tf.keras.preprocessing.image.array_to_img(
        sheared_numpy
    )

    return pil_image
