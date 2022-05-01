from utils.utils import read_img, display_imgs
from utils.plot_grid import plot_grid


def main():
    img = read_img("./images/sudoku.png")
    img_grid = plot_grid(img)
    display_imgs([img, img_grid], ["original", "grid"])


if __name__ == "__main__":
    main()
