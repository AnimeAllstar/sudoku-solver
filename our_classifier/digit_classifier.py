import tensorflow as tf
import numpy as np
from os import path


class DigitClassifier:
    def __init__(self):
        if path.exists("./our_classifier/saved_model/"):
            self.model = tf.keras.models.load_model("./our_classifier/saved_model/")
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
        self.model.save("./our_classifier/saved_model/")

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

        
