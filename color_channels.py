import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)  

blank = np.zeros(img.shape[:2], dtype='uint8') # Create a blank image with the same shape as the image but no color channels

# There are 3 color channels: Blue, Green, Red
# The image is made up of these 3 color channels

# Split the image into color channels
b, g, r = cv.split(img) # Split the image into its color channels
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)
# The image is split into its color channels, but the image is not displayed in color
# lighter the color, the higher the intensity of that color in the image (Ex: white = high intensity, black = low intensity)

blue = cv.merge([b, blank, blank]) # Merge the blue color channel with a blank image = blue color channel, black, black
green = cv.merge([blank, g, blank]) # Merge the green color channel with a blank image = black, green color channel, black
red = cv.merge([blank, blank, r]) # Merge the red color channel with a blank image = black, black, red color channel
# The blank image is used to fill in the other color channels with blank, so only the desired color channel is shown
# Since the blank image has no color channels, when merged with the color channel, it will not show any color
cv.imshow('Blue1', blue)
cv.imshow('Green1', green)
cv.imshow('Red1', red)

print(img.shape) # (height, width, color channels)
print(b.shape) # (height, width) --> only 2D since it is a single color channel
print(g.shape)
print(r.shape)

merged = cv.merge([b, g, r]) # Merge the color channels back together
cv.imshow('Merged', merged)

cv.waitKey(0)
