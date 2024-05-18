# Buliding a face recognition system using OpenCV

import cv2 as cv
import numpy as np
import os

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# List of people we want to recognize

DIR = r'C:\Users\bsiva\Desktop\mahesh\H3 Dynamics\OpenCV_Computer_Vision\Resources\Faces\train'
# Directory where the images are stored

people1 = []
for i in os.listdir(DIR):
    people1.append(i) # List of all folder names in the directory
# same as people

# Method which goes through the images in the directory and creates a list of the images and makes it the training data

labels = []  # Labels of the faces in the training data (Who's face it is)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features_mat = []  # List to store the cv::Mat objects

def create_train():
    for person in people:
        path = os.path.join(DIR, person)  # Path to the person's folder
        label = people.index(person)  # Index of the person in the people list

        for img in os.listdir(path):
            img_path = os.path.join(path, img)  # Path to the image
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) # Detect the faces in the image

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]  # Crop the face from the image
                # Resize the face ROI to a fixed size, e.g., (100, 100)
                faces_roi_resized = cv.resize(faces_roi, (100, 100)) # crop the face and resize it to 100x100
                features_mat.append(faces_roi_resized)
                labels.append(label)

create_train()

print('Training done ---------------')

face_recognizer = cv.face.LBPHFaceRecognizer_create()  # instantiate the face recognizer

face_recognizer.save('face_trained.yml')  # Save the trained face recognizer

# Train the face recognizer on the features list and the labels list

# Convert features_mat to a NumPy array
features = np.array(features_mat, dtype='object')
labels = np.array(labels, dtype='int32')  # Convert labels to a NumPy array of integers

face_recognizer.train(features_mat, labels)  # Train with the correct format

np.save('features.npy', features)
np.save('labels.npy', labels)