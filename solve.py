from solver.sudoku import Sudoku
from solver.test import test
from utils.grid_to_array import grid_to_array
from utils.utils import read_img, display_imgs
from utils.extract_grid import extract_grid
from our_classifier.model_business import (
    eval_model,
    train_model,
    test_with_single_image,
)


def main():
    # Restriction on image:
    # no more than 1 sudoku shown - see sudoku_angled.png
    # no objects greater than sudoku shown - see real_sudoku.jpg
    img = read_img("./images/sudoku.png")

    # isolate sudoku from rest of the image
    img_grid = extract_grid(img)
    array_grid = grid_to_array(img_grid)
    
    # this will not work currently since the number extraction is not 100% accurate
    # the app will have a method to edit the extracted grid before the sudoku is solved
    # sudoku = Sudoku(array_grid)
    # sudoku.solve()
    # sudoku.show(solution=True)
    
    # display_imgs([img, img_grid], ["original", "grid"])


if __name__ == "__main__":
    # main()
    test()
