import cv2 as cv
import numpy as np

# AND, OR, XOR, NOT : bitwise operations
# Pixel is truned on (1) if it is on in both images and off (0) if it is off in both images

blank = np.zeros((400, 400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1) # Image, top-left corner, bottom-right corner, color, thickness
# One parameter in color means the color is white as this is a blank image

circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1) # Image, center, radius, color, thickness
# One parameter in color means the color is white as this is a blank image

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# Bitwise AND
bitwise_and = cv.bitwise_and(rectangle, circle) # Image 1, Image 2
cv.imshow('Bitwise AND', bitwise_and)
# Pixel is truned on (1) if it is on in both images and off (0) if it is off in both images

# Bitwise OR --> Union of the images
bitwise_or = cv.bitwise_or(rectangle, circle) # Image 1, Image 2
cv.imshow('Bitwise OR', bitwise_or)
# Pixel is truned on (1) if it is on in either image and off (0) if it is off in both images

# Bitwise XOR --> Non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle) # Image 1, Image 2
cv.imshow('Bitwise XOR', bitwise_xor)
# Pixel is truned on (1) if it is on in one image but not both and off (0) if it is off in both images or on in both images
# XOR = OR - AND

# Bitwise NOT
bitwise_not = cv.bitwise_not(rectangle) # Image
cv.imshow('Bitwise NOT', bitwise_not)
bitwise_not2 = cv.bitwise_not(circle) # Image
cv.imshow('Bitwise NOT 2', bitwise_not2)
# Pixel is inverted, turned on (1) if it is off (0) and off (0) if it is on (1)

cv.waitKey(0)