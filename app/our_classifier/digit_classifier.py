import tensorflow as tf
import numpy as np
from os import path


class DigitClassifier:
    def __init__(self):
        if path.exists("./our_classifier/saved_model/digit_classifier.h5"):
            self.model = tf.keras.models.load_model("./our_classifier/saved_model/digit_classifier.h5")
        else:
            # create the layers
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters=10,
                        kernel_size=3,
                        activation="relu",
                        input_shape=(28, 28, 1),
                    ),
                    tf.keras.layers.Conv2D(10, 3, activation="relu"),
                    tf.keras.layers.MaxPool2D(),
                    tf.keras.layers.Flatten(),
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
        self.model.save("./our_classifier/saved_model/digit_classifier.h5")

    def predict(self, X):
        # make prediction
        predict = self.model.predict(X)

        # return an array of predicted digit of the image
        return np.argmax(predict)

    def evaluate(self, X, y):
        # evaluate the model
        loss, acc = self.model.evaluate(X, y)
        print("loss:", loss)
        print("acc:", acc)
