import cv2
import time
import numpy as np
import hand_detector as hd
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#Wdth and height of the camera
CAMERA_WIDTH , CAMERA_HEIGHT = 640, 480
########################################

cap = cv2.VideoCapture(0)
cap.set(3,CAMERA_WIDTH)
cap.set(4,CAMERA_HEIGHT)

ctime = 0
ptime = 0

detector = hd.hand_detector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
min_vol = vol_range[0]
max_vol = vol_range[1]

vol = 0
vol_bar = 400

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



        vol = np.interp(length, [30,130], [min_vol, max_vol])
        vol_bar = np.interp(length, [30,130],[400, 150])
        vol_percentage = np.interp(length, [30,150], [0,100])
        volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(img, (50, 150), (85,400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85,400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'Vol: {int(vol_percentage)}%',(50,450),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime  

    cv2.putText(img, f'FPS: {int(fps)}%',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)
