import cv2 as cv
import mediapipe as mp
import time

ctime = 0
ptime = 0

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(0.75)

mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = face_detection.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            bbox_c = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            
            #Bounding Box
            bbox = int(bbox_c.xmin * w), int(bbox_c.ymin * h), \
                int(bbox_c.width * w), int(bbox_c.height * h)
            
            cv.rectangle(img, bbox, (255,0,255),2)

            #percentage
            cv.putText(img, f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_DUPLEX,0.9,(255,0,255),2)


    curr_time = time.time()
    fps = 1/(curr_time - ptime)
    ptime = curr_time

    cv.putText(img, str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv.imshow('Vide0',img)
    cv.waitKey(1)
