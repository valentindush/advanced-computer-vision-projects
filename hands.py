import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()
    imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handLandmarks,mpHands.HAND_CONNECTIONS)

    cv.imshow("video", frame)

    cv.waitKey(1)