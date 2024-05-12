import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8') 
# 500*500 pixel image,3 channels (BGR)
cv.imshow('Blank', blank)

blank[200:300, 300:400] = 0, 255, 0 # Green square within said pixel range
# blank[:] = 0, 255, 0 # Entire image will be green
# the last 3 values are the BGR values, so (0, 255, 0) is green
cv.imshow('Green', blank)

cv.rectangle(blank, (0, 0), (500, 250), (0, 255, 0), thickness=cv.FILLED) # -1 can also be used instead of cv.FILLED
# cv.rectangle(blank, (0, 0), (500, 250), (0, 255, 0), thickness=2) # Thickness of the rectangle
# left top corner, right bottom corner, color, thickness
cv.imshow('Rectangle', blank)

cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 0, 0), thickness=-1)
# blank.shape[1]//2 is the width, blank.shape[0]//2 is the height, image is divided into 4 quadrants
cv.imshow('Rectangle 2', blank)

cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=-1)
# center, radius, color, thickness
cv.imshow('Circle', blank)

cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)
# start, end, color, thickness
cv.imshow('Line', blank)

cv.putText(blank, 'Hello, my name is Aman', (225, 225), cv.FONT_HERSHEY_TRIPLEX, 0.2, (0, 255, 255), 2)
# text, origin, font, scale, color, thickness
cv.imshow('Text', blank)

cv.waitKey(0)