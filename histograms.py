import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Histograms are a great way to visualize the color/pixel distribution of an image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Grayscale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256]) # List of images, list of channels, mask, histSize, ranges
# One image, grayscale == 0, no mask, 256 bins, range 0-256

# Masking allows us to focus on certain parts of the image, while ignoring the rest
blank = np.zeros(img.shape[:2], dtype='uint8') # Create a blank image with the same shape as the image but no color channels
# Mask must be the same size as the image, but can have a single color channel
cv.imshow('Blank', blank)

circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1) # Image, center, radius, color, thickness
cv.imshow('Mask', circle)

masked = cv.bitwise_and(gray, gray, mask=circle) # Image 1, Image 2, mask
cv.imshow('Masked Image', masked)

color_masked = cv.bitwise_and(img, img, mask=circle) # Image 1, Image 2, mask
cv.imshow('Masked Color Image', color_masked)

gray_hist2 = cv.calcHist([gray], [0], masked, [256], [0, 256]) # List of images, list of channels, mask, histSize, ranges
cv.imshow('Masked Gray Histogram', gray_hist)

plt.figure()
plt.title('Grayscale Histogram Masked')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist2)
plt.show()

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.show()

# The darkness of image which most of the pixels are has the highest peak in the histogram

# Color Histogram
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title('Color Histogram')
plt.show()

for i, col in enumerate(colors):
    hist2 = cv.calcHist([color_masked], [i], masked, [256], [0, 256])
    plt.plot(hist2, color=col)
    plt.xlim([0, 256])
plt.title('Color Histogram Masked')
plt.show()

cv.waitKey(0)