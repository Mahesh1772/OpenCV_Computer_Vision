import cv2 as cv

img = cv.imread('Resources/Photos/cat_large2.jpg')

cv.imshow('Cat', img)

cv.waitKey(0)