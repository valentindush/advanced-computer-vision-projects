import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

prev_time = 0
curr_time = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx,cy), 6, (0,0,255), cv.FILLED)
            

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv.putText(img, str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv.imshow("Video", img)
    cv.waitKey(1)