from utils.utils import read_img, display_imgs
from utils.extract_grid import extract_grid

import tensorflow as tf
import numpy as np
from utils.digit_classifier import DigitClassifier

def test_model():
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    model = DigitClassifier()
    predict = model.predict(test_data)
    print(predict)


def main():
    # Restriction on image:
    # no more than 1 sudoku shown - see sudoku_angled.png
    # no objects greater than sudoku shown - see real_sudoku.jpg
    img = read_img("./images/sudoku_angled.png")

    # isolate sudoku from rest of the image
    img_grid = extract_grid(img)
    
    display_imgs([img, img_grid], ["original", "grid"])


if __name__ == "__main__":
    main()
