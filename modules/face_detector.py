import mediapipe as mp
import cv2 as cv
import time

class Face_detector:
    def __init__(self,min_detection_conf = 0.5):
       
        self.min_detection_conf =  min_detection_conf
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(self.min_detection_conf)
        self.mp_draw = mp.solutions.drawing_utils


    def find_faces(self,img,draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.face_detection.process(img_rgb)

        bboxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mp_draw.draw_detection(img, detection)
                bbox_c = detection.location_data.relative_bounding_box
                h,w,c = img.shape
                
                #Bounding Box
                bbox = int(bbox_c.xmin * w), int(bbox_c.ymin * h), \
                    int(bbox_c.width * w), int(bbox_c.height * h)

                bboxes.append([id,bbox,detection.score])
                
                img = self.draw(img,bbox)

                #percentage
                cv.putText(img, f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-15),cv.FONT_HERSHEY_DUPLEX,0.7,(255,0,255),2)

        return img, bboxes



    
    def draw(self, img, bbox, l=20, t=3, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv.rectangle(img, bbox, (255, 0, 255),rt)

        #Top left x,y
        cv.line(img, (x,y), (x+l,y), (255, 0, 255), t)
        cv.line(img, (x,y), (x, y+l), (255,0,255),t)

        #Top right x1, y
        cv.line(img, (x1,y), (x1 - l,y), (255, 0, 255), t)
        cv.line(img, (x1,y), (x1, y + l), (255,0,255),t)

        #Bottom left x,y1
        cv.line(img, (x,y1), (x+l,y1), (255, 0, 255), t)
        cv.line(img, (x,y1), (x, y1-l), (255,0,255),t)

        #Bottom right x1, y1
        cv.line(img, (x1,y1), (x1 - l,y1), (255, 0, 255), t)
        cv.line(img, (x1,y1), (x1, y1 - l), (255,0,255),t)

        return img

def main():

    prev_time = 0
    curr_time = 0
    cap = cv.VideoCapture(0)

    face_detector = Face_detector()

    while True:
        success, img = cap.read()

        img, bboxes = face_detector.find_faces(img)

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv.putText(img, f'FPS: {int(fps)}',(10,70),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        cv.imshow("Video", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()