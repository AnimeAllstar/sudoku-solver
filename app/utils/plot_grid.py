import cv2 as cv
import numpy as np


def plot_grid(img):
    """
    Plot lines using Hough transform
    """

    # finds edges in an image using the Canny algorithm
    edges = cv.Canny(img, 50, 150, apertureSize=3)

    # finds lines in a binary image using the standard Hough transform
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    # color image to draw lines on
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # draw lines on image_color
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img_color
