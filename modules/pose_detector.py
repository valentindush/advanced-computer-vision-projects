import mediapipe as mp
import cv2 as cv
import time

class Pose_detector:
    def __init__(self, mode=False, up_body=False, smooth=True, detection_conf=0.5, track_conf=0.5):
        self.mode= mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils


    def detect_pose(self,img,draw=True):
        image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
    
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img,self.results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)

        return img  

    def get_position(self,img,draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy), 8, (0,255,0), cv.FILLED)
        
        return lm_list

def main():

    prev_time = 0
    curr_time = 0
    cap = cv.VideoCapture(0)

    pose_detector = Pose_detector()

    while True:
        success, img = cap.read()

        img = pose_detector.detect_pose(img)
        lm_list = pose_detector.get_position(img)

        if len(lm_list) != 0:
            print(lm_list[0])

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv.putText(img, str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

        cv.imshow("Video", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()