#!/usr/bin/env python3
"""
Randomly changes the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Returns the altered image
    """
    