import tensorflow as tf
import cv2 as cv
from our_classifier.digit_classifier import DigitClassifier
from utils.utils import read_img


def train_model():
    file = pd.read_csv("./our_classifier/training_data/TMNIST_Data.csv")
    x_train = file.drop(columns={"names", "labels"})
    y_train = file[["labels"]]
    x_train = x_train.values.reshape(-1, 28, 28).astype('float32')
    x_train = x_train.astype('float32')
    x_train = x_train / 255.0

    mnist = tf.keras.datasets.mnist
    (x_train_2, y_train_2), (x_test, y_test) = mnist.load_data()
    y_train_2 = y_train_2.reshape(-1, 1)
    x_train = np.concatenate((x_train, x_train_2))
    y_train = np.concatenate((y_train, y_train_2))
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


def test_with_single_image():
    X = read_img('./numbers/two.png')
    X = cv.bitwise_not(X)
    X = cv.resize(X, (28, 28))
    X = X.reshape((-1, 28, 28))
    X = X / 255.0
    model = DigitClassifier()
    predict = model.predict(X)
    print(predict)
