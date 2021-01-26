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
    if validation_data and early_stopping:
        e_s = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        History = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose, callbacks=[e_s],
                              validation_data=validation_data,
                              shuffle=shuffle)
    else:
        History = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose, callbacks=e_s,
                              validation_data=None,
                              shuffle=shuffle)

    return History
