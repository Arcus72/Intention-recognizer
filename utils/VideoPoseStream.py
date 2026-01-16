import cv2
import time
from utils.skeleton_vectors import skeleton_to_feature_vector

class VideoPoseStream:
    def __init__(self, generate_skeleton, categorize_skeleton):
        self.generate_skeleton = generate_skeleton
        self.categorize_skeleton = categorize_skeleton

        self.cap = cv2.VideoCapture(0)

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        if not self.cap.isOpened():
            print("Błąd: Nie można otworzyć kamery.")
            self.cap = None

    def add_fps_text(self, frame):
        self.frame_count += 1

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= 1.0:

            self.fps = self.frame_count / elapsed_time

            self.frame_count = 0
            self.start_time = current_time

        fps_text = f"FPS: {self.fps:.2f}"

        frame_with_text = cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        return frame_with_text


    def activate(self):
        if not self.cap:
            return

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Błąd: Nie można odczytać klatki.")
                break

            frame = self.add_fps_text(frame)
            df_skeleton = self.generate_skeleton(frame)
            category = self.categorize_skeleton(df_skeleton)

            cv2.putText(
                frame,
                category,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Kolor zielony
                2,
                cv2.LINE_AA
            )

            cv2.imshow('Kamerka OpenCV z FPS (q -> exit)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()