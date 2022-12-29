import mediapipe as mp
import cv2 as cv
import time

class FaceMeshDetector:
    def __init__(self,max_num_faces = 2):
       
        self.max_num_faces =  max_num_faces
        self.mp_face_mesh= mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=self.max_num_faces)
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1,circle_radius=1,color=(0, 255, 0))

    def find_face_mesh(self,img,draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)

        faces_lms = []

        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms,landmark_drawing_spec=self.draw_spec,connection_drawing_spec=self.draw_spec)

                face = []

                for id,lm in enumerate(face_lms.landmark):
                    h,w,c = img.shape
                    x,y = int(lm.x*w), int(lm.y*h)
                    face.append([x,y])

                faces_lms.append(face)

        return img,faces_lms
    

def main():

    prev_time = 0
    curr_time = 0
    cap = cv.VideoCapture(0)

    face_mesh = FaceMeshDetector(2)

    while True:
        success, img = cap.read()
        img, faces = face_mesh.find_face_mesh(img)
        
        if len(faces) != 0:
            print(len(faces))

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv.putText(img, f'FPS: {int(fps)}',(10,70),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        cv.imshow("Video", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()