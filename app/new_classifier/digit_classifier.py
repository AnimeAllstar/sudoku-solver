import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
from os import path


class DigitClassifier:
    def __init__(self):
        if path.exists("./new_classifier/saved_model/digit_classifier.h5"):
            self.model = tf.keras.models.load_model("./new_classifier/saved_model/digit_classifier.h5")
        else:
            # create the layers
            self.model = tf.keras.Sequential(
                [
                    # first convolutional + pooling layer 
                    Conv2D(
                        filters=60, # number of output filters
                        kernel_size=5, # w and h of the filter
                        padding="same", # to avoid edges being ignored
                        activation="relu", # transformation
                        input_shape=(28, 28, 1), # 28x28 pixels and 1 grayscale channel
                    ),
                    MaxPool2D(), # combine the result of Conv2D 

                    # second convolutional + pooling layer
                    Conv2D(32, 3, padding="same", activation="relu"),
                    MaxPool2D(),  

                    # flatten input before passing to dense layer
                    Flatten(),

                    # first fully connected layer
                    Dense(65, activation="relu"),
                    Dropout(0.5), # drop neurons to prevent overfitting 

                    # output layer
                    Dense(10, activation="softmax"),
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
        self.model.fit(
            X, y, 
            batch_size=32, 
            epochs=10)

        # save the model
        self.model.save("./new_classifier/saved_model/digit_classifier.h5")

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
