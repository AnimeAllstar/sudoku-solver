import cv2 as cv
import numpy as np

from utils.utils import distance_to_origin


def process_image(img):
    # blur the image to reduce noise
    img_proc = cv.GaussianBlur(img, (9, 9), 0)

    # segmentation using thresholding
    img_proc = cv.adaptiveThreshold(
        img_proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2
    )

    # invert the colour of the image, so that the grid lines are in white
    img_proc = cv.bitwise_not(img_proc, img_proc)

    # define a kernel and dilate the image using the kernel
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
    img_proc = cv.dilate(img_proc, kernel)

    return img_proc


def find_grid(img):
    # find contours of the image
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    # sort the contour such that contours[0] is the largest polygon
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # find the largest 4-sided contrours
    for c in contours:
        epsilon = 0.015 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            return approx
    return None


def order_the_corners(grid_points):
    # convert the data points to vector of tuples
    corners = [(grid[0][0], grid[0][1]) for grid in grid_points]

    # calculate the distance to origin for each corner
    magnitudes = {}
    for i in range(len(corners)):
        magnitudes[i] = distance_to_origin(corners[i][0], corners[i][1])

    # sort the corners based on the distance to origin
    sorted_magnitudes = list(
        {k: v for k, v in sorted(magnitudes.items(), key=lambda item: item[1])}.items()
    )

    corners_dict = {}

    # define which corner is top left, top right, bottom left and bottom right using the magnitudes
    corners_dict["top_left"] = corners[sorted_magnitudes[0][0]]
    corners_dict["bottom_right"] = corners[sorted_magnitudes[3][0]]
    if corners[sorted_magnitudes[1][0]][0] > corners[sorted_magnitudes[2][0]][0]:
        corners_dict["top_right"] = corners[sorted_magnitudes[1][0]]
        corners_dict["bottom_left"] = corners[sorted_magnitudes[2][0]]
    else:
        corners_dict["top_right"] = corners[sorted_magnitudes[2][0]]
        corners_dict["bottom_left"] = corners[sorted_magnitudes[1][0]]

    return corners_dict


def extract_grid(img):
    # process the image
    img_proc = process_image(img)

    # find the contours
    grid_contour = find_grid(img_proc)

    if grid_contour is None:
        return None
    # convert that to points
    corners = order_the_corners(grid_contour)

    # calculate the width
    bottom_len = distance_to_origin(
        int(corners["bottom_left"][0] - corners["bottom_right"][0]),
        int(corners["bottom_left"][1] - corners["bottom_right"][1]),
    )
    top_len = distance_to_origin(
        int(corners["top_left"][0] - corners["top_right"][0]),
        int(corners["top_left"][1] - corners["top_right"][1]),
    )
    width = max(int(bottom_len), int(top_len))

    # calculate the height
    left_len = distance_to_origin(
        int(corners["top_left"][0] - corners["bottom_left"][0]),
        int(corners["top_left"][1] - corners["bottom_left"][1]),
    )
    right_len = distance_to_origin(
        int(corners["top_right"][0] - corners["bottom_right"][0]),
        int(corners["top_right"][1] - corners["bottom_right"][1]),
    )
    height = max(int(left_len), int(right_len))

    # define the dimension
    dimensions = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    # convert the data points to numpy array
    corners = np.array(
        [
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ],
        dtype="float32",
    )

    # calculate the perspective transform matrix
    real_grid = cv.getPerspectiveTransform(corners, dimensions)

    # warp the image
    img_warped = cv.warpPerspective(img_proc, real_grid, (width, height))

    # invert the image
    return img_warped
