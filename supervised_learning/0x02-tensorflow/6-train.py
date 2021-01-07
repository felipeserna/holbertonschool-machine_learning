#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network classifier
"""


import numpy as np
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier
    """
    x, y = create_placeholders(X_train, Y_train)
    y_pred = forward_prop(X_train, layer_sizes, activations)
    loss = calculate_loss(Y_valid, Y_train)
    accuracy = calculate_accuracy(Y_valid, Y_train)
    train_op = create_train_op(loss, alpha)

    init_op = tf.global_variables_initializer

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        (x, y).op.run()
        y_pred.op.run()
        loss.op.run()
        accuracy.op.run()
        train_op.op.run()
        # save the variables to disk
        save_path = saver.save(sess, save_path)
