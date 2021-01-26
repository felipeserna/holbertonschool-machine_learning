#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent.
Also analyzes validation data.
Also trains the model using early stopping
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Returns: the History object generated after training the model
    """
    if early_stopping and validation_data:
        callback = [K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience)]
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=callback)
    else:
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose, shuffle=shuffle,
                              validation_data=None)
    return history
