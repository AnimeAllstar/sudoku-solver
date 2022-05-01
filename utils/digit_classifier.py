from tkinter import Label
import tensorflow as tf
import numpy as np


class DigitClassifier:
    def __init__(self):
        # create the layers
        self.model = tf.keras.Sequential ( [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        # build the neural network
        self.model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def fit(self):
        # load the dataset 
        mnist = tf.keras.datasets.mnist
        (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
        train_data = train_data / 255

        #train
        self.model.fit(train_data, train_labels, epochs=5)

    def predict(self, X):
        # X must be 3 dimensional (x, 28, 28)
        if X.ndim != 3:
            X.reshape(-1, 28, 28)

        # make prediction
        predict = self.model.predict(X)

        # return an array of predicted digit of the image 
        return np.argmax(predict, axis=1)

