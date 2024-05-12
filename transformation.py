import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img) # window name (can be anything), image object

def translate(img, x, y):
    # -x --> Left
    # -y --> Up
    # x --> Right
    # y --> Down
    transMat = np.float32([[1, 0, x], [0, 1, y]]) # Transformation matrix
    dimensions = (img.shape[1], img.shape[0]) # Width, Height
    return cv.warpAffine(img, transMat, dimensions) # Image, transformation matrix, dimensions

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

translated2 = translate(img, -100, -100)
cv.imshow('Translated 2', translated2)

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2] # Height, Width

    if rotPoint is None:
        rotPoint = (width//2, height//2) # Width, Height

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0) # Rotation matrix
    dimensions = (width, height) # Width, Height
    return cv.warpAffine(img, rotMat, dimensions) # Image, rotation matrix, dimensions

rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)
# The image will be rotated around the center of the image, anti-clockwise

rotated2 = rotate(img, -45)
cv.imshow('Rotated 2', rotated2)
# The image will be rotated around the center of the image, clockwise

rotated3 = rotate(rotated2, -45)
cv.imshow('Rotated 3', rotated3)
# The image will be rotated to total -90 degrees, but edges will be cut off
# This is because in first rotation, the image was rotated to -45 degrees, so the edges were cut off
# In the second rotation, the image was rotated to -45 degrees again, so the edges which were already cut off remained cut off

rotated_rotPoint = rotate(img, 45, (img.shape[1]//3, img.shape[0]//3))
cv.imshow('Rotated Rot Point', rotated_rotPoint)
# The image will be rotated around the specified point, anti-clockwise

flip = cv.flip(img, 0) # 0 is the flip code, 0 means vertical flip, 1 means horizontal flip, -1 means both vertical and horizontal flip
cv.imshow('Flip', flip)

cv.waitKey(0) # 0 means infinite time, 1000 means 1 second