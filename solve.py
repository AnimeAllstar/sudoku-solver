from utils.grid_to_array import grid_to_array
from utils.utils import read_img, display_imgs
from utils.extract_grid import extract_grid

import tensorflow as tf
from utils.digit_classifier import DigitClassifier


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


def main():
    # Restriction on image:
    # no more than 1 sudoku shown - see sudoku_angled.png
    # no objects greater than sudoku shown - see real_sudoku.jpg
    img = read_img("./images/sudoku.png")

    # isolate sudoku from rest of the image
    img_grid = extract_grid(img)
    array_grid = grid_to_array(img_grid)

    # display_imgs([img, img_grid], ["original", "grid"])


if __name__ == "__main__":
    main()
