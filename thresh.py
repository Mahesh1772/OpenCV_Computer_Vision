import cv2 as cv

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)
# Thresholding is the binarization of an image, converting it to black and white
# anthing above a certain threshold is turned white, below is turned black

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) # Image, threshold, max value, type
# Pixels below 150 are turned black, above 150 are turned white (255)
# threshold is the value we used, thresh is the image
# higher threshold value, less white pixels
cv.imshow('Simple Thresholded Image', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) # Image, threshold, max value, type
cv.imshow('Simple Thresholded Inverse Image', thresh_inv)
# Pixels below 150 are turned white, above 150 are turned black (0)
# higher threshold value, more white pixels

# Adaptive Thresholding
# Instead of using a global value as the threshold, we use the mean of the neighborhood area
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 13) 
# Image, max value, adaptive method, type, block size, constant
# adaptive method: ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C is the method of thresholding
# block size: size of the area to calculate the mean == kernel size
# higher block size, more white pixels
# constant: a constant that is subtracted from the mean or weighted mean calculated
# constant increases the threshold value, making the image lighter
cv.imshow('Adaptive Thresholded Image', adaptive_thresh)

adaptive_thresh_gauss = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 13)
cv.imshow('Adaptive Thresholded Gaussian Image', adaptive_thresh_gauss)

cv.waitKey(0)