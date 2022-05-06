from operator import truediv
import cv2 as cv
import numpy as np
from our_classifier.digit_classifier import DigitClassifier
from utils.extract_grid import process_image


def cell_has_digit(cell):
    """
    returns true if the cell has a digit
    """
    num_white_pixel = np.sum(cell == 255)
    num_black_pixel = np.sum(cell == 0)
    if num_white_pixel / num_black_pixel > 0.3:
        return 1
    return 0


def find_digit(cell):  # not yet done
    # change it back to image type
    cell = cell.astype(np.uint8)
    # find the contours
    digit_contour = cv.findContours(cell, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    digit_contour = digit_contour[0] if len(digit_contour) == 2 else digit_contour[1]

    # focus on smaller area
    min_area = 0.50 * cell.shape[0] * cell.shape[1]
    digit_contour = [c for c in digit_contour if cv.contourArea(c) < min_area]

    # sort the contour in desending order
    digit_contour = sorted(digit_contour, key=cv.contourArea, reverse=True)

    for c in digit_contour:
        # get the approximate height and width of the digit
        x, y, w, h = cv.boundingRect(c)

        # uncomment to check if the image is cropped
        # if (x, y, w, h) == (0,0,53,50), then the cell is not cropped
        # print(x, y, w, h)

        # if it is not cropped
        if x < 4 or y < 4:
            x += 5
            y += 5
        if w >= cell.shape[0] or w >= cell.shape[1]:
            w -= 5
        if h >= cell.shape[0] or h >= cell.shape[1]:
            h -= 5

        # crop the image
        ROI = cell[y : y + h, x : x + w]
        return ROI


def grid_to_array(grid):
    """
    returns a numpy array of the grid
    """
    # compute the size of each cell
    (h, w) = grid.shape
    cell_h, cell_w = h // 9, w // 9

    # empty 9x9 array to store the cells
    cropped_cells = np.zeros((9, 9, cell_h, cell_w))

    # empty 9x9 array to store the digits
    digits = np.zeros((9, 9))

    # our model
    model = DigitClassifier()

    for i in range(9):
        for j in range(9):
            cropped_cells[i][j] = grid[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
            ]

            # uncomment to display each cell, press esc to continue to next cell
            # cv.imshow(f"cropped ({i}, {j})", cell_proc)
            # cv.waitKey(0)

            # crop out the center of the grid to check if there's digit
            tmp_cell = grid[
                i * cell_h + cell_h // 4 : i * cell_h + 3 * (cell_h // 4),
                j * cell_w + cell_w // 4 : j * cell_w + 3 * (cell_w // 4),
            ]

            # not yet done
            if cell_has_digit(tmp_cell):
                # crop the image
                digit = find_digit(cropped_cells[i][j])

                # resize for prediction
                digit = cv.resize(digit, (28, 28))
                digit = digit.astype('float32')
                digit = digit.reshape((-1, 28, 28, 1))
                digit = digit / 255.0

                # predict
                digits[i][j] = model.predict(digit)
            else:
                digits[i][j] = -1

    print(digits)

    return digits
