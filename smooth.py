import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8') 

# Images are smoothed to reduce noise and reduce detail
# Noise is random variation of brightness or color in images, from the camera sensor or environment (problems in lighting)
# Blurring is done by averaging the pixel values of the neighboring pixels
# Gaussian Blur (most common) but there are other types of blurring, for different effects

# Averaging
average = cv.blur(img, (7, 7)) # Image, kernel size (width, height)
average2 = cv.blur(img, (3, 3)) # Image, kernel size (width, height)
cv.imshow('Average Blur', average)
cv.imshow('Average Blur 2', average2)
# The larger the kernel size, the more the image is blurred

# Gaussian Blur
gauss = cv.GaussianBlur(img, (7, 7), 0) # Image, kernel size, sigmaX (standard deviation in the x direction)
gauss2 = cv.GaussianBlur(img, (3, 3), 0) # Image, kernel size, sigmaX (standard deviation in the x direction)
# SigmaX is the standard deviation in the x direction, if 0, then it is calculated from the kernel size
# Greater sigmaX, the more blurred the image
cv.imshow('Gaussian Blur', gauss)
cv.imshow('Gaussian Blur 2', gauss2)

# Median Blur
median = cv.medianBlur(img, 3) # Image, kernel size (must be an odd number), is effective in removing salt and pepper noise
median2 = cv.medianBlur(img, 7) # Image, kernel size (must be an odd number), is assumed to be 7x7
# Used in advanced image processing, not as common as Gaussian blur
# Gives a more natural look to the image, better at preserving edges than Gaussian blur and averaging
cv.imshow('Median Blur', median)
cv.imshow('Median Blur 2', median2)
# Not meant to have a large kernel size, 3x3 is common

# Bilateral Blur
bilateral = cv.bilateralFilter(img, 10, 35, 25) # Image, diameter of each pixel neighborhood, sigma color, sigma space
bilateral2 = cv.bilateralFilter(img, 15, 50, 75) # Image, diameter of each pixel neighborhood, sigma color, sigma space
# Diameter of each pixel neighborhood: larger value means more pixels will be included in the blurring
# Sigma color: filter sigma in the color space, larger value means colors further apart will be mixed together
# Sigma space: filter sigma in the coordinate space, larger value means pixels further apart will influence each other
# Retains edges while smoothing, removes noise while keeping edges sharp
cv.imshow('Bilateral Blur', bilateral)
cv.imshow('Bilateral Blur 2', bilateral2)

cv.waitKey(0)