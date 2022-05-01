import cv2 as cv
import sys


def read_img(img_path, grayscale=True):
    """
    Reads an image from a path and returns it as a numpy array.
    """
    if grayscale:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(img_path)

    if img is None:
        print(f"{img_path} could not be read.")
        sys.exit(1)

    return img


def display_img(img, window_name="image"):
    """
    Displays an image.
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


def display_imgs(imgs, window_names=None):
    """
    Displays a list of images.
    """
    if window_names is None:
        window_names = [f"image {i}" for i in range(len(imgs))]
    for img, window_name in zip(imgs, window_names):
        cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
