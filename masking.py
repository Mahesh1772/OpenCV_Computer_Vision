import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Masking allows us to focus on certain parts of the image, while ignoring the rest
blank = np.zeros(img.shape[:2], dtype='uint8') # Create a blank image with the same shape as the image but no color channels
# Mask must be the same size as the image, but can have a single color channel
cv.imshow('Blank', blank)

mask = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1) # Image, center, radius, color, thickness
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask=mask) # Image 1, Image 2, mask
cv.imshow('Masked Image', masked)

mask2 = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1) # Image, top-left corner, bottom-right corner, color, thickness
cv.imshow('Mask 2', mask2)

masked2 = cv.bitwise_and(img, img, mask=mask2) # Image 1, Image 2, mask
cv.imshow('Masked Image 2', masked2)

weird_shape = cv.bitwise_and(mask, mask2)
cv.imshow('Weird Shape', weird_shape)

masked_weird_shape = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Masked Weird Shape', masked_weird_shape)

cv.waitKey(0)