import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time


class Skeleton_generator():
    def __init__(self, max_num_poses = 5):
        model_path = 'assets/pose_landmarker_heavy.task'

        base_options = python.BaseOptions(model_asset_path=model_path)

        mode = vision.RunningMode.VIDEO

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode= mode,
            num_poses=max_num_poses,
            min_pose_detection_confidence=0.8
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)



    def process_photos(self, pictures_location, data_outcome_folder):

        pictures_names = os.listdir(pictures_location)

        for picture_name in pictures_names:
            full_picture_path = os.path.join(pictures_location, picture_name)
            print(full_picture_path)

            image = cv2.imread(full_picture_path)
            df_skeleton = self.generate(image)
            if df_skeleton is None:
                continue

            df_skeleton.to_csv(os.path.join(data_outcome_folder, picture_name.split(".")[0] + ".csv"))

    @staticmethod
    def normalize(df, col_name, new_col_name):
        min_val = df[col_name].min()
        max_val = df[col_name].max()

        if max_val - min_val == 0:
            df[new_col_name] = 0
        else:
            df[new_col_name] = (df[col_name] - min_val) / (max_val - min_val)
        return df

    def process_photos(self, pictures_location, data_outcome_folder):
        pictures_names = os.listdir(pictures_location)
        for picture_name in pictures_names:
            full_picture_path = os.path.join(pictures_location, picture_name)
            print(full_picture_path)
            image = cv2.imread(full_picture_path)
            df_skeleton = self.generate(image)[0]
            if df_skeleton is None:
                continue

            df_skeleton.to_csv(os.path.join(data_outcome_folder, picture_name.split(".")[0] + ".csv"))

    def generate(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.pose_landmarks:
            print("Nie wykryto postaci")
            return [None]

        h, w, _ = image.shape
        list_of_skeletons_df = []

        for pose_landmarks in detection_result.pose_landmarks:
            data = []
            for id, lm in enumerate(pose_landmarks):
                x_px, y_px, z_px = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                data.append([id, x_px, y_px, z_px, lm.visibility])

            df = pd.DataFrame(data, columns=["id", "x_px", "y_px", "z_px", "visibility"])

            min_x, min_y, min_z = df["x_px"].min(), df["y_px"].min(), df["z_px"].min()
            # TODO: Different way of reducint x, y, z
            df_copy = df.copy()
            # df_copy["x_px"] -= min_x
            # df_copy["y_px"] -= min_y
            # df_copy["z_px"] -= min_z

            df_copy = self.normalize( df_copy, "x_px", "x")
            df_copy = self.normalize( df_copy, "y_px", "y")
            df_copy = self.normalize( df_copy, "z_px", "z")

            df = df_copy.copy()

            important_points = [0, 8, 7, 20, 16, 14, 12, 11, 13, 15, 19, 24, 23, 26, 25, 28, 32, 27, 31]
            df = df[df['id'].isin(important_points)].copy()

            list_of_skeletons_df.append(df)

        return list_of_skeletons_df