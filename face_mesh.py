import cv2 as cv
import mediapipe as mp
import time

ctime = 0
ptime = 0

mp_facemesh = mp.solutions.face_mesh
face_mesh = mp_facemesh.FaceMesh(max_num_faces=2)

mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1,circle_radius=1,color=(0, 255, 0))

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lms,landmark_drawing_spec=draw_spec,connection_drawing_spec=draw_spec)

            for id,lm in enumerate(face_lms.landmark):
                h,w,c = img.shape
                x,y = int(lm.x*w), int(lm.y*h)
                print(id, x ,y)

    curr_time = time.time()
    fps = 1/(curr_time - ptime)
    ptime = curr_time



    cv.putText(img, f'FPS: {int(fps)}',(10,70),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    cv.imshow('Vide0',img)
    cv.waitKey(1)
