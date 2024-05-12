import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img_large = cv.imread('Resources/Photos/cat_large.jpg')
resized_large = rescaleFrame(img_large, scale=0.2)
cv.imshow('Cat Large Resized', resized_large)
cv.waitKey(0)

def changeRes(width, height):
    # Only for Live Videos
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read() # isTrue is a boolean value representing if the frame is read successfully
    
    if isTrue:
        resized_frame = rescaleFrame(frame, scale=0.2)
        cv.imshow('Video', frame) # Display the frame with the window name 'Video'
        cv.imshow('Video Resized', resized_frame)

    if cv.waitKey(20) & 0xFF==ord('d'): # If 'd' key is pressed, break the loop
        break
    
capture.release()
cv.destroyAllWindows() # Closes all windows
