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

def find_digit(cell): # not yet done
    # change it back to image type  
    cell = cell.astype(np.uint8)
    # find the contours 
    digit_contour, h = cv.findContours(cell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # digit_contour = sorted(digit_contour, key=cv.contourArea, reverse=True)
    digit_contour = digit_contour[0] if len(digit_contour) == 2 else digit_contour[1]


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
            cropped_cells[i][j] = grid[i * cell_h : (i + 1) * cell_h, 
                                       j * cell_w : (j + 1) * cell_w]

            # uncomment to display each cell, press esc to continue to next cell
            # cv.imshow(f"cropped ({i}, {j})", cell_proc)
            # cv.waitKey(0)

            # crop out the center of the grid to check if there's digit 
            tmp_cell = grid[i * cell_h + cell_h // 6 : i * cell_h + 5 * (cell_h // 6), 
                            j * cell_w + cell_w // 6 : j * cell_w + 5 * (cell_w // 6)]

            # not yet done
            if cell_has_digit(tmp_cell):
                cell = cv.resize(cropped_cells[i][j], (28, 28))
                cell = cell.astype('float32')
                cell = cell.reshape((-1, 28, 28))
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
