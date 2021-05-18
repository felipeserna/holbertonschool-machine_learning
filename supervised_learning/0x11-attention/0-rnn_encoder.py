#!/usr/bin/env python3
"""
Class that inherits from `tensorflow.keras.layers.Layer`
to encode for machine translation
"""
import tensorflow as tf


class RNNEncoder:
    """
    Inherits from `tensorflow.keras.layers.Layer`
    to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        # Batch size
        self.batch = batch
        # Number of hidden units in the RNN cell
        self.units = units
        # Embedding layer that converts
        # words from the vocabulary into an embedding vector.
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # GRU layer
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Returns: a tensor of shape (batch, units)
        containing the initialized hidden states
        """
        init_hidden = tf.zeros([self.batch, self.units])
        return init_hidden

    def call(self, x, initial):
        """
        Returns: outputs, hidden
        """
        return None
