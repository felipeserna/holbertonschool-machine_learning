#!/usr/bin/env python3
"""
Script that trains a convolutional neural network
to classify the CIFAR 10 dataset.
In the same file, write a function 'def preprocess_data(X, Y):'
that pre-processes the data for your model
"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Returns: X_p, Y_p
    """
    # Preprocessing needed in each Keras Application
    X_p = K.applications.inception_v3.preprocess_input(X)
    # Converts a label vector into a one-hot matrix
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


if __name__ == "__main__":
    # Dataset of 50,000 32x32 color training images and 10,000 test images,
    # labeled over 10 categories
    (x_train, y_train), (X, Y) = K.datasets.cifar10.load_data()

    # preprocessing
    x_train_p, y_train_p = preprocess_data(x_train, y_train)
    x_test_p, y_test_p = preprocess_data(X, Y)

    input = K.Input(shape=(32, 32, 3))
    input = K.layers.UpSampling2D()(input)

    # DenseNet121
    #  loads weights pre-trained on ImageNet
    dense_121 = K.applications.DenseNet121(include_top=False,
                                           weights="imagenet",
                                           input_tensor=input,
                                           input_shape=None,
                                           pooling='max')

    output = dense_121.layers[-1].output
    output = K.layers.Flatten()(output)
    output = K.layers.Dense(512, activation='relu')(output)
    output = K.layers.Dropout(0.2)(output)
    output = K.layers.Dense(256, activation='relu')(output)
    output = K.layers.Dropout(0.2)(output)
    output = K.layers.Dense(10, activation='softmax')(output)

    model = K.models.Model(dense_121.input, output)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

    # training
    history = model.fit(x=x_train_p, y=y_train_p,
                        batch_size=256, epochs=20,
                        validation_data=(x_test_p, y_test_p),
                        verbose=1)

    model.save('cifar10.h5')
