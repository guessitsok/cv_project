import os
import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame_flip = cv.flip(frame, 1)

    # Display the resulting frame
    # frame = cv.flip()
    cv.imshow('Comdined videos', frame_flip)

    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
