from operator import truediv
import cv2 as cv
import numpy as np
from our_classifier.digit_classifier import DigitClassifier


def cell_has_digit(cell):
    """
    returns true if the cell has a digit
    """

    # obtain the ratio white : black
    num_black_pixel = np.sum(cell == 255)
    num_white_pixel = np.sum(cell == 0)

    if num_black_pixel / num_white_pixel > 0.1:
        return 1
    return 0



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
    digits = np.zeros((9,9))
    
    # our model
    model = DigitClassifier()

    for i in range(9):
        for j in range(9):
            cropped_cells[i][j] = grid[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
            ]
            
            # uncomment to display each cell, press esc to continue to next cell
                # cv.imshow(f"cropped ({i}, {j})", cells_centers[i][j])
                # cv.waitKey(0)

            # crop to the center of the image (focus on to the digit)
            tmp_cell = grid[ i * cell_h + cell_h // 6 : i * cell_h + 5 * (cell_h // 6), j * cell_w + cell_w // 6 : j * cell_w + 5 * (cell_w // 6)]
            
            if cell_has_digit(tmp_cell):
                cell = cv.resize(tmp_cell, (28, 28))
                # cv.imshow(f"cropped ({i}, {j})", cell)
                # cv.waitKey(0)
                cell = cell.astype('float32')
                cell = cell.reshape(-1, 28, 28)
                cell = cell / 255.0
                digits[i][j] = model.predict(cell)
            else:
                digits[i][j] = -1

            
            # TODO: Instead of storing images into cropped_cells,
            # use grid[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] to get the value, and store the values
            # if cropped_cells[i][j] is empty, then mark array[i][j] as empty (-1)
            # if it is not empty, add it to X
            # pass X to the model get predictions for all non-empty cells
            # update the array with the predictions
    
    print(digits)

    return digits
