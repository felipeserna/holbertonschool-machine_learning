#!/usr/bin/env python3
"""
Script that trains a convolutional neural network
to classify the CIFAR 10 dataset.
In the same file, write a function 'def preprocess_data(X, Y):'
that pre-processes the data for your model
"""


import tensorflow.keras as K


if __name__ == "__main__":
    # Dataset of 50,000 32x32 color training images and 10,000 test images,
    # labeled over 10 categories
    (x_train, y_train), (X, Y) = K.datasets.cifar10.load_data()

    # preprocessing
    x_train_p, y_train_p = preprocess_data(x_train, y_train)
    x_test_p, y_test_p = preprocess_data(X, Y)

    # InceptionV3
    #  loads weights pre-trained on ImageNet
    inc_v3 = K.applications.InceptionV3(include_top=False,
                                        weights="imagenet",
                                        input_tensor=None,
                                        input_shape=None,
                                        pooling='max')

    model = K.Sequential()
    model.add(inc_v3)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy'
                  metrics=['accuracy'])

    # training
    history = model.fit(x=x_train_p, y=y_train_p,
                        batch_size=32, epochs=5,
                        validation_data=(x_test_p, y_test_p))

    model.save('cifar10.h5')


def preprocess_data(X, Y):
    """
    Returns: X_p, Y_p
    """
    # Preprocessing needed in each Keras Application
    X_p = K.applications.inception_v3.preprocess_input(X)
    # Converts a label vector into a one-hot matrix
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p
