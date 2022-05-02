import tensorflow as tf
import numpy as np
from os import path


class DigitClassifier:
    def __init__(self):
        if path.exists("saved_model/"):
            self.model = tf.keras.models.load_model("saved_model/")
        else:
            # create the layers
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )

            # build the neural network
            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

    def fit(self, X, y):
        # train the model
        self.model.fit(X, y, epochs=10)

        # save the model
        self.model.save("saved_model/")

    def predict(self, X):
        # X must be 3 dimensional (x, 28, 28)
        if X.ndim != 3:
            X.reshape(-1, 28, 28)

        # make prediction
        predict = self.model.predict(X)

        # return an array of predicted digit of the image
        return np.argmax(predict, axis=1)

    def evaluate(self, X, y):
        # evaluate the model
        loss, acc = self.model.evaluate(X, y)
        print("loss:", loss)
        print("acc:", acc)
