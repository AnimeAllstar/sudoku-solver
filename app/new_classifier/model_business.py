import tensorflow as tf
import cv2 as cv
from our_classifier.digit_classifier import DigitClassifier
from utils.utils import read_img


def train_model():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
    X_train = X_train / 255.0

    model = DigitClassifier()
    model.fit(X_train, y_train)


def test_model():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test / 255.0
    model = DigitClassifier()
    predict = model.predict(X_test)
    print(predict)


def eval_model():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test / 255.0
    model = DigitClassifier()
    model.evaluate(X_test, y_test)


def test_with_single_image():
    X = read_img('./numbers/two.png')
    X = cv.bitwise_not(X)
    X = cv.resize(X, (28, 28))
    X = X.reshape((-1, 28, 28, 1)).astype('float32')
    X = X / 255.0
    model = DigitClassifier()
    predict = model.predict(X)
    print(predict)
