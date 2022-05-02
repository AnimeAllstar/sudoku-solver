from utils.utils import read_img, display_imgs
from utils.extract_grid import extract_grid


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
