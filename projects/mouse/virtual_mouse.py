import numpy as np
import cv2
import time

#Wdth and height of the camera
CAMERA_WIDTH , CAMERA_HEIGHT = 640, 480
########################################

cap = cv2.VideoCapture(0)
cap.set(3,CAMERA_WIDTH)
cap.set(4,CAMERA_HEIGHT)

ctime = 0
ptime = 0

while True:
    success, img = cap.read()

    

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime 

    cv2.putText(img, f'FPS: {int(fps)}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
    cv2.imshow("Fingers", img)

    cv2.waitKey(1)
