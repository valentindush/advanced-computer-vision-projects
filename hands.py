import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#for FPS

prev_time = 0
curr_time = 0

while True:
    success, frame = cap.read()
    imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 6:
                    cv.circle(frame, (cx,cy), 15, (255,0,255), cv.FILLED)

            mpDraw.draw_landmarks(frame,handLandmarks,mpHands.HAND_CONNECTIONS)
    
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv.imshow("video", frame)

    cv.waitKey(1)