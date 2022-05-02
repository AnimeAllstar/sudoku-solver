from utils.utils import read_img, display_imgs
from utils.plot_grid import plot_grid

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
    img = read_img("./images/sudoku.png")
    img_grid = plot_grid(img)
    display_imgs([img, img_grid], ["original", "grid"]) 
    


if __name__ == "__main__":
    main()
