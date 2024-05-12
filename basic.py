import cv2 as cv

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img) # window name (can be anything), image object

# Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert the BGR image --> grayscale
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT) # Image, kernel size, border type
# Kernel size must be odd numbers, higher the kernel size, more the blur
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175) # Image, threshold values
# Threshold values are the minimum and maximum values of the gradient
cv.imshow('Canny Edges', canny)
canny2 = cv.Canny(blur, 125, 175) # Image, threshold values
cv.imshow('Canny Edges 2', canny2)
# The blur image will have less edges than the original image

# Dilating the image
dilated = cv.dilate(canny, (7, 7), iterations=3) # Image, kernel size, iterations
# Kernel size must be odd numbers, higher the kernel size, more the dilation
# Iterations increase the thickness of the edges
cv.imshow('Dilated', dilated)
dilated2 = cv.dilate(canny2, (7, 7), iterations=1) # Image, kernel size, iterations
cv.imshow('Dilated 2', dilated2)

# Eroding the image
eroded = cv.erode(dilated, (7, 7), iterations=3) # Image, kernel size, iterations
# Removes the dilated edges, opposite of dilation, gets back the original image
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC) # Image, dimensions, interpolation
# Interpolation is the method used to resize the image, INTER_LINEAR is the default method
# INTER_AREA is used for shrinking the image, INTER_CUBIC is used for zooming the image
# INTER_LANCZOS4 is used for zooming the image, INTER_NEAREST is used for shrinking the image
# INTER_CUBIC is slower but better than INTER_LINEAR
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400] # Height, Width
# Height is the y-axis, Width is the x-axis
cv.imshow('Cropped', cropped)

cv.waitKey(0) # 0 means infinite time, 1000 means 1 second