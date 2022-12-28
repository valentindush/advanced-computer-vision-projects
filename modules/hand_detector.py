import cv2 as cv
import mediapipe as mp
import time

class hand_detector:

    def __init__(self, mode=False,max_hands=2,detection_conf=0.5,track_conf=0.5):
        self.mode = mode
        self.max_hands=max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    
    def find_hands(self,img,draw=True):
        imageRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLandmarks,self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def find_position(self,img,hand_no=0,draw=True):

        lm_list = []

        if self.results.multi_hand_landmarks:

            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #append
                lm_list.append([id,cx,cy])

                if draw:
                    cv.circle(img, (cx,cy), 10, (140,0,55), cv.FILLED)

        return lm_list

def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    ctime = 0
    detector = hand_detector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmks = detector.find_position(img)
        if len(lmks) != 0:
            print(lmks[4])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img, f'FPS: {int(fps)}',(10,70),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        cv.imshow("hands",img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()