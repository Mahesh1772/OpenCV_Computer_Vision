"""
Haar Cascade Classifier is used to detect faces in an image.
Face detection : Detects the presence of a face in an image
Face recognition : Identifies the face in an image
Face verification : Verifies the face in an image
face detection is done using classifiers, which are pre-trained models.
Classfiers : Haar Cascade Classifier, Local Binary Patterns Histograms (LBPH), Convolutional Neural Networks (CNN)

Can copy cascade files from OpenCV GitHub repository
"""

import cv2 as cv

img = cv.imread('Resources/Photos/group 2.jpg')
cv.imshow('Lady', img)

# Face detection does not need color as it only detects the conturs/ edges in the face 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Load the Haar Cascade Classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) # Image, scale factor, min neighbors
# scaleFactor: how much the image size is reduced at each image scale
# minNeighbors: how many neighbors each candidate rectangle should have to retain it (how many rectangles can be combined to form a face)
# faces_rect is a list of rectangles that contain the faces, rectangles are the coordinates of the faces
# We can use this list to draw rectangles around the faces

print(f'Number of faces found = {len(faces_rect)}')

faces_rect2 = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
print(f'Number of faces found = {len(faces_rect2)}') # More neighbors, less faces detected
# Faces with accessories are not detected as faces, or face not perpendicular to the camera

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) # Image, start point, end point, color, thickness
    # Draw a rectangle around the face
    # x, y is the top left corner of the rectangle
    # x+w, y+h is the bottom right corner of the rectangle
cv.imshow('Detected Faces', img)

# Face detection is not perfect, it can detect other objects as faces
# Anything that looks like a face can be detected as a face, we need to tune the parameters to get better results

cv.waitKey(0)