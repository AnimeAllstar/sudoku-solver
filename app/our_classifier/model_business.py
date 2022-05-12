import tensorflow as tf
import cv2 as cv
import numpy as np
import pandas as pd
from our_classifier.digit_classifier import DigitClassifier
from utils.utils import read_img


def train_model():
    file = pd.read_csv("./our_classifier/training_data/TMNIST_Data.csv")
    X_train = file.drop(columns={"names", "labels"})
    y_train = file[["labels"]]
    X_train = X_train.values.reshape(-1, 28, 28, 1).astype('float32')
    X_train = X_train.astype('float32')
    X_train = X_train / 255.0

    mnist = tf.keras.datasets.mnist
    (X_train_2, y_train_2), (X_test_2, y_test) = mnist.load_data()
    X_train_2 = X_train_2.reshape(-1, 28, 28, 1).astype('float32')
    X_train_2 = X_train_2.astype('float32')
    X_train_2 = X_train_2 / 255.0

    y_train_2 = y_train_2.reshape(-1, 1)
    X_train = np.concatenate((X_train, X_train_2))
    y_train = np.concatenate((y_train, y_train_2))
    model = DigitClassifier()
    model.fit(X_train, y_train)


def test_model():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    model = DigitClassifier()
    predict = model.predict(X_test)
    print(predict)


def eval_model():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    model = DigitClassifier()
    model.evaluate(X_test, y_test)


def test_with_single_image():
    X = read_img('./numbers/two.png')
    X = cv.bitwise_not(X)
    X = cv.resize(X, (28, 28))
    X = X.reshape((-1, 28, 28, 1))
    X = X.astype('float32')
    X = X / 255.0
    model = DigitClassifier()
    predict = model.predict(X)
    print(predict)
