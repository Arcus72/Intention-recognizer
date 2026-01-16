import cv2
import mediapipe as mp
import pandas as pd
import os

class Skeleton_generator():
    def __init__(self,static_image_mode = True ):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode) # True - for photo (best quality), False - for video

    @staticmethod
    def normalize(df, col_name, new_col_name):
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        df[new_col_name] = (df[col_name] - min_val) / (max_val - min_val)
        return df


    def process_photos(self, pictures_location, data_outcome_folder):
        pictures_names = os.listdir(pictures_location)

        for picture_name in pictures_names:
            full_picture_path = os.path.join(pictures_location, picture_name)
            print(full_picture_path)

            image = cv2.imread(full_picture_path)

            df_skeleton= self.generate(image)

            if df_skeleton is None:
                continue
            df_skeleton.to_csv(os.path.join(data_outcome_folder, picture_name.split(".")[0] + ".csv"))


    def generate(self, image):
        # konwersja BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"Nie wykryto postaci na obrazie")
            return None

        h, w, _ = image.shape
        data = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_px, y_px, z_px = int(lm.x * w), int(lm.y * h),  int(lm.z * h)
            data.append([id, x_px, y_px, z_px, lm.visibility])

        df = pd.DataFrame(data, columns=["id", "x_px", "y_px", "z_px", "visibility"])

        # cutting end normalization part
        min_x = df["x_px"].min()
        min_y = df["y_px"].min()
        min_z = df["z_px"].min()

        df["x_px"] = df["x_px"] - min_x
        df["y_px"] = df["y_px"] - min_y
        df["z_px"] = df["z_px"] - min_z

        df = self.normalize(df, "x_px", "x")
        df = self.normalize(df, "y_px", "y")
        df = self.normalize(df, "z_px", "z")


        important_points = [8, 7, 20, 16, 14, 12, 11, 13, 15, 19, 24, 23, 26, 25, 26, 28, 32 ,27, 31]
        df = df.iloc[important_points]
        # End of normalization

        # Draws skeleton
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return df