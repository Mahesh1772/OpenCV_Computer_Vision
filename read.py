import cv2 as cv

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat', img) # window name (can be anything), image object
cv.waitKey(1000) # 0 means infinite time, 1000 means 1 second

img_large = cv.imread('Resources/Photos/cat_large.jpg')
cv.imshow('Cat Large', img_large)
cv.waitKey(0) # When any key is pressed, the window will close

# The size of the read image is too large to fit in the screen, but it will still be displayed in the window!!
# The cv.waitKey() function will be same regardless of the number of images, so both the images will be displayed 
# at the same time and will close when any key is pressed once, will stay open otherwise

capture = cv.VideoCapture('Resources/Videos/dog.mp4') # 0 for webcam, specify path for video file
# capture is the object of the video capture class


while True:
    isTrue, frame = capture.read() # isTrue is a boolean value representing if the frame is read successfully
    
    if isTrue:
        cv.imshow('Video', frame) # Display the frame with the window name 'Video'

    if cv.waitKey(20) & 0xFF==ord('d'): # If 'd' key is pressed, break the loop
        break
    
capture.release()
cv.destroyAllWindows() # Closes all windows
