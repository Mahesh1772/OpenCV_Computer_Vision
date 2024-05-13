import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)

# Color spaces is a way to represent colors in an image: Grayscale, BGR, HSV, L*a*b*, etc.

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray) # Grayscale image shows the intensity of the color in the image

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # HSV is a color space that attempts to represent colors the way humans perceive it
cv.imshow('HSV', hsv) # Hue, Saturation, Value

# BGR to L*a*b*
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB) # L*a*b* is a color space that attempts to be more perceptually uniform
cv.imshow('L*a*b*', lab) # Lightness, a, b

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

plt.imshow(img)
plt.show()
# Matplotlib reads the image as RGB, OpenCV reads the image as BGR --> colors are inverted
plt.imshow(rgb)
plt.show() # This will show the image in the correct colors

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV --> BGR', hsv_bgr)

# L*a*b* to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('L*a*b* --> BGR', lab_bgr)

# RGB to BGR
rgb_bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
cv.imshow('RGB --> BGR', rgb_bgr)

# HSV to L*a*b*
hsv_lab = cv.cvtColor(hsv_bgr, cv.COLOR_BGR2LAB)
cv.imshow('HSV --> L*a*b*', hsv_lab)
# For conversions without BGR, the conversion is done in two steps
# Ex: HSV --> BGR --> L*a*b*

cv.waitKey(0)