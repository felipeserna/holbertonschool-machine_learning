#!/usr/bin/env python3
"""
Class that inherits from tensorflow.keras.layers.Layer
to perform multi head attention.
https://www.tensorflow.org/tutorials/text/transformer
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Performs multi head attention
    """
    def __init__(self, dm, h):
        """
        Class constructor
        """
        super().__init__()
        # Number of heads
        self.h = h
        # Dimensionality of the model
        self.dm = dm
        # Depth of each attention head
        self.depth = dm // h
        # Dense layer used to generate the query matrix
        self.Wq = tf.keras.layers.Dense(units=dm)
        # Dense layer used to generate the key matrix
        self.Wk = tf.keras.layers.Dense(units=dm)
        # Dense layer used to generate the value matrix
        self.Wv = tf.keras.layers.Dense(units=dm)
        # Dense layer used to generate the attention output
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose the result such that the shape is
        (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Returns: output, weights
        """
        batch_size = tf.shape(Q)[0]

        # (batch_size, seq_len, d_model)
        Q = self.Wq(Q)
        # (batch_size, seq_len, d_model)
        K = self.Wk(K)
        # (batch_size, seq_len, d_model)
        V = self.Wv(V)

        # (batch_size, num_heads, seq_len_q, depth)
        Q = self.split_heads(Q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        K = self.split_heads(K, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        V = self.split_heads(V, batch_size)

        # scaled_attention.shape == (batch_size, h, seq_len_q, depth)
        # attention_weights.shape == (batch_size, h, seq_len_q, seq_len_k)
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = \
            tf.reshape(scaled_attention, (batch_size, -1, self.dm))

        # (batch_size, seq_len_q, d_model)
        output = self.linear(concat_attention)

        return output, weights
