import cv2 as cv
import numpy as np

def process_image(img):
    # blur the image to reduce noise
    img_proc = cv.GaussianBlur(img, (9,9), 0)

    # segmentation using thresholding  
    img_proc = cv.adaptiveThreshold(img_proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)

    # invert the colour of the image, so that the grid lines are in white 
    img_proc = cv.bitwise_not(img_proc, img_proc)

    # define a kernel and dilate the image using the kernel
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    img_proc = cv.dilate(img_proc, kernel)

    return img_proc

def find_grid(img):
    # find contours of the image
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # sort the contour such that contours[0] is the largest polygon
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    # find the largest 4-sided contrours 
    for c in contours:
        epsilon = 0.015 *cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            return approx

def order_the_corners(grid_points):
    # convert the data points to vector of tuples
    corners = [(grid[0][0], grid[0][1]) for grid in grid_points]

    # reorder the corners in the following order:
    # top left, top right, bottom right, bottom left
    return corners[1], corners[0], corners[3], corners[2]

def extract_grid(img):
    # process the image
    img_proc = process_image(img)

    # find the contours
    grid_contour = find_grid(img_proc)

    # convert that to points
    corners = order_the_corners(grid_contour)

    # calculate the width 
    width_1 = np.sqrt(((corners[3][0] - corners[2][0]) ** 2) + ((corners[3][1] - corners[2][1]) ** 2))
    width_2 = np.sqrt(((corners[0][0] - corners[1][0]) ** 2) + ((corners[0][1] - corners[1][1]) ** 2))
    width = max(int(width_1), int(width_2))

    # calculate the height
    height_1 = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) + ((corners[0][1] - corners[3][1]) ** 2))
    height_2 = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
    height = max(int(height_1), int(height_2))

    # define the dimension
    dimensions = np.array([[0, 0], [width - 1, 0], 
                          [width - 1, height - 1],
                          [0, height - 1]], dtype="float32")

    # convert the data points to numpy array
    corners = np.array(corners, dtype="float32")

    # calculate the perspective transform matrix 
    real_grid = cv.getPerspectiveTransform(corners, dimensions)
    
    # warp the perspective to grab the screen
    return cv.warpPerspective(img_proc, real_grid, (width, height))

