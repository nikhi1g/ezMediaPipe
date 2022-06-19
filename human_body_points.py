import mediapipe as mp
import cv2  # OpenCV for camera




class human_body_points:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2) # integer because in pixels not measurements


    def start(self):
        # Get Realtime Webcam Feed
        cap = cv2.VideoCapture(0)
        # Initiate holistic model
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # cap = cv2.VideoCapture(2)
            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Flip Image to get mirror effect
                image = cv2.flip(image, 1)

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                '''
                globals
                '''
                image_height = image.shape[0]
                image_width = image.shape[1]
                '''
                end globals
                '''

                for index, landmark in enumerate(results.face_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    if index == 10:
                        cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                        print(landmark_x, landmark_y)

                # 1_Draw face landmark
                self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                          self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                                          self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1))

                # 2_Draw Left hand landmark
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(150, 255, 0), thickness=2, circle_radius=2),
                                          self.mp_drawing.DrawingSpec(color=(150, 255, 0), thickness=2, circle_radius=2))

                # 3_Draw Right hand landmark
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(150, 255, 0), thickness=2, circle_radius=2),
                                          self.mp_drawing.DrawingSpec(color=(150, 255, 0), thickness=2, circle_radius=2))

                # 4_Draw Pose detection landmark
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                                          self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

                cv2.imshow('Raw webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'): # structured way of breaking
                    break




                # cv2.waitKey(10) # in code way of breaking


        cap.release()
        cv2.destroyAllWindows()

#get point, return location

if __name__ == '__main__':
    person_detection = human_body_points()
    person_detection.start()
