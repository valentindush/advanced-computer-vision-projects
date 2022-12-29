import cv2
import time
import numpy as np
import hand_detector as hd
import math

#Wdth and height of the camera
CAMERA_WIDTH , CAMERA_HEIGHT = 640, 480
########################################

cap = cv2.VideoCapture(0)
cap.set(3,CAMERA_WIDTH)
cap.set(4,CAMERA_HEIGHT)

ctime = 0
ptime = 0

detector = hd.hand_detector()

while True:
    success, img = cap.read()

    img = detector.find_hands(img,draw=False)
    lm_list = detector.find_position(img, draw=False)

    #Landmarks to be usef are 4 nd 8

    if len(lm_list) != 0:
        x1,y1 = lm_list[4][1], lm_list[4][2]
        x2,y2 = lm_list[8][1],lm_list[8][2]
        cx,cy = (x1 + x2)//2, (y1 + y2)//2

        #Draw circles
        cv2.circle(img, (x1, y1),7, (255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2),7, (255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2), (255,0,255),3)
        cv2.circle(img, (cx, cy),7, (255,0,255),cv2.FILLED)

        #Length
        length = math.hypot(x2 - x1, y2 - y1)

        if length<30:
            cv2.circle(img, (cx, cy),7, (0,255,0),cv2.FILLED)
        
        if length>130:
            cv2.circle(img, (cx, cy),7, (0,0,255),cv2.FILLED)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime  

    cv2.putText(img, f'FPS: {int(fps)}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)
