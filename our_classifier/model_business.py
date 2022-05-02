import tensorflow as tf
from os import path
from our_classifier.digit_classifier import DigitClassifier

def train_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    model = DigitClassifier()
    model.fit(x_train, y_train)


def test_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    model = DigitClassifier()
    predict = model.predict(x_test)
    print(predict)


def eval_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = DigitClassifier()
    model.evaluate(x_test, y_test)

