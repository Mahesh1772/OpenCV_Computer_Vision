import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)
# Almost the same as edge detection, but we are looking for the gradient of the image

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F) # Image, data type
lap = np.uint8(np.absolute(lap)) # Convert to absolute values and then to image specific data type
cv.imshow('Laplacian', lap)
# Laplacian is a second derivative, it calculates the gradient of the image
# It looks like chalk on a blackboard

# Sobel
# Used in advanced edge detection
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0) # Image, data type, x order, y order
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1) # Y axis specific gradient
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)
# Sobel is a first derivative, it calculates the gradient of the image in the x and y direction

canny = cv.Canny(gray, 150, 175) # Image, threshold1, threshold2
cv.imshow('Canny', canny)
# Canny is a multi-step algorithm to detect a wide range of edges in images
# Cleaner edges than the other two, used often in edge detection

cv.waitKey(0)