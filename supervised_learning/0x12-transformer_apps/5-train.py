#!/usr/bin/env python3
"""
Creates and trains a transformer model
for machine translation of Portuguese to English
using our previously created dataset.
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Returns the trained model
    """
    return None
