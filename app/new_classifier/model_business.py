import tensorflow as tf
assert tf.__version__.startswith('2')
import cv2 as cv
import numpy as np
import pandas as pd
from new_classifier.digit_classifier import DigitClassifier
from utils.utils import read_img
from sklearn.model_selection import train_test_split
from os import path


def train_model():
    df = pd.read_csv("./our_classifier/training_data/TMNIST_Data.csv")
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] / 4))
    X_train = df_train.drop(columns={"names", "labels"})
    y_train = df_train[["labels"]]
    X_train = X_train.values.reshape(-1, 28, 28, 1).astype('float32')
    X_train = X_train / 255.0

    mnist = tf.keras.datasets.mnist
    (X_train_2, y_train_2), (X_test_2, y_test) = mnist.load_data()
    X_train_2 = X_train_2.reshape(-1, 28, 28, 1).astype('float32')
    X_train_2 = X_train_2 / 255.0

    y_train_2 = y_train_2.reshape(-1, 1)
    X_train = np.concatenate((X_train, X_train_2))
    y_train = np.concatenate((y_train, y_train_2))
    model = DigitClassifier()
    if model.model is None:
        model.build()
    model.fit(X_train, y_train)


def test_model():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test / 255.0
    model = DigitClassifier()
    if model.model is None:
        model.build()
    predict = model.predict(X_test)
    print(predict)


def eval_model():
    df = pd.read_csv("./our_classifier/training_data/TMNIST_Data.csv")
    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] / 4))
    X_test = df_test.drop(columns={"names", "labels"})
    y_test = df_test[["labels"]]
    X_test = X_test.values.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test / 255.0

    mnist = tf.keras.datasets.mnist
    (X_train_2, y_train_2), (X_test_2, y_test_2) = mnist.load_data()
    X_test_2 = X_test_2.reshape(-1, 28, 28, 1).astype('float32')
    X_test_2 = X_test_2 / 255.0

    y_test_2 = y_test_2.reshape(-1, 1)
    X_test = np.concatenate((X_test, X_test_2))
    y_test = np.concatenate((y_test, y_test_2))
    model = DigitClassifier()
    if model.model is None:
        model.build()
    model.evaluate(X_test, y_test)


def test_with_single_image():
    X = read_img('./numbers/two.png')
    X = cv.bitwise_not(X)
    X = cv.resize(X, (28, 28))
    X = X.reshape((-1, 28, 28, 1)).astype('float32')
    X = X / 255.0
    model = DigitClassifier()
    if model.model is None:
        model.build()
    predict = model.predict(X)
    print(predict)

def convert_to_lite():
    model = DigitClassifier().model
    if model.model is None:
        raise Exception('no saved model')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    our_tflite_model = converter.convert()
    open("./new_classifier/saved_model/digit_classifier.tflite", "wb").write(our_tflite_model)

