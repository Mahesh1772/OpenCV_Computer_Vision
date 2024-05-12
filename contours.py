import cv2 as cv

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img) # window name (can be anything), image object

# Countours are the boundaries of the objects in an image, used for object detection

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert the BGR image --> grayscale
cv.imshow('Gray Cats', gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT) # Image, kernel size, border type
cv.imshow('Blur Cats', blur)

canney = cv.Canny(img, 125, 175) # Image, threshold values
cv.imshow('Canny Cats', canney)

canny2 = cv.Canny(blur, 125, 175) # Image, threshold values
cv.imshow('Canny Cats 2', canny2)

countours, hierarchies = cv.findContours(canney, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # Image, retrieval mode, approximation method
countours2, hierarchies2 = cv.findContours(canny2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # Image, retrieval mode, approximation method
# countours is a list of all the contours in the image
# hierarchies is the hierarchy of the contours, Ex: if one contour is inside another, then the inner contour is a child of the outer contour
# RETR_LIST is retrieves all the contours
# RETR_EXTERNAL retrieves the external contours only
# RETR_TREE retrieves all the contours and creates a full family "hierarchy list"
# CHAIN_APPROX_SIMPLE  retrieves only the endpoints of the contours, it compresses horizontal, vertical, and diagonal segments and leaves only their end points
# CHAIN_APPROX_NONE retrieves all the contours
print(f'{len(countours)} contours found')
print(f'{len(countours2)} contours found in blurred image')

cv.waitKey(0) # 0 means infinite time, 1000 means 1 second