import cv2 as cv
import numpy as np


def grid_to_array(grid):
    """
    returns a numpy array of the grid
    """

    # computer size of each cell
    (h, w) = grid.shape
    cell_h, cell_w = h // 9, w // 9

    # Empty 9x9 array to store the cells
    cropped_cells = np.zeros((9, 9, cell_h, cell_w))

    for i in range(9):
        for j in range(9):
            cropped_cells[i][j] = grid[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
            ]
            # uncomment to display each cell, press esc to continue to next cell
            # cv.imshow(f"cropped ({i}, {j})", cropped_cells[i][j])
            # cv.waitKey(0)

            # TODO: Insread of storing images into cropped_cells,
            # use grid[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] to get the value, and store the values
            # if cropped_cells[i][j] is empty, then mark array[i][j] as empty (-1)
            # if it is not empty, add it to X
            # pass X to the model get predictions for all non-empty cells
            # update the array with the predictions

    return cropped_cells
