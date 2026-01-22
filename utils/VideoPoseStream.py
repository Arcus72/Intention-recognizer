from datetime import datetime
import cv2
import time

class VideoPoseStream:
    def __init__(self, generate_skeleton, categorize_skeleton,camera_id=0):
        self.generate_skeleton = generate_skeleton
        self.categorize_skeleton = categorize_skeleton

        self.cap = cv2.VideoCapture(camera_id)

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
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        return frame_with_text

    def get_distance_1d_x(self, df_skeleton, id1=7, id2=8):
        point1 = df_skeleton[df_skeleton['id'] == id1].iloc[0]
        point2 = df_skeleton[df_skeleton['id'] == id2].iloc[0]

        distance_x = abs(point1['x_px'] - point2['x_px'])

        return distance_x


    def draw_posture(self,frame, df_skeleton, category):
        min_x = df_skeleton["x_px"].min()
        max_x = df_skeleton["x_px"].max()

        min_y = df_skeleton["y_px"].min()
        max_y = df_skeleton["y_px"].max()
        min_y = min_y -  abs(int(self.get_distance_1d_x(df_skeleton, 7, 8) ))

        cv2.putText(
            frame,
            category,
            (min_x, min_y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),  # Kolor zielony
            1,
            cv2.LINE_AA
        )

        if category == "Aggression":
            color = (0, 0, 255)
        elif category == "Fear":
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)


        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color , 2)

    def get_dangerous_level(self, intentions):
        if "Aggression" in intentions:
            if "Fear" in intentions:
                if "Neutrality" in intentions:
                    return 40
                else:
                    return 100
            else:
                return 60

        if "Fear" in intentions and  "Aggression" not in intentions:
            if "Neutrality" not in intentions:
                return 30
            else:
                return 10
        return 0

    def save_image(self, img):
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        display_time = now.strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(img, display_time, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        file_path = f"save/alert_{timestamp_str}.jpg"
        cv2.imwrite(file_path, img)

    def activate(self):
        if not self.cap:
            return
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Błąd: Nie można odczytać klatki.")
                continue
            original_frame = frame.copy()
            frame = self.add_fps_text(frame)
            df_skeletons = self.generate_skeleton(frame)
            frame_intentions = []
            print("-----------------------------------")
            for df_skeleton in df_skeletons:

                if df_skeleton is not None:
                    category = self.categorize_skeleton(df_skeleton)
                    print(category)
                    print(df_skeleton)

                    frame_intentions.append(category)
                    self.draw_posture(frame, df_skeleton, category)

            dangerous_level = self.get_dangerous_level(frame_intentions)

            if dangerous_level == 100 and self.frame_count == 0:
                self.save_image(original_frame)
                pass

            cv2.putText(
                frame,
                f"Danger: {dangerous_level}%",
                (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if dangerous_level == 100 else (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow('Press q to exit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()