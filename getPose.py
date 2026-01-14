import cv2
import mediapipe as mp
import pandas as pd
import os

# inicjalizacja MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # tryb do analizy zdjęć
mp_drawing = mp.solutions.drawing_utils

def extract_pose_from_image(image, save_csv=True):
  # konwersja BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # przetwarzanie przez MediaPipe
    results = pose.process(image_rgb)

    # sprawdź czy wykryto sylwetkę
    if not results.pose_landmarks:
        print(f"Nie wykryto postaci na obrazie")
        return None

    # pobierz współrzędne punktów
    h, w, _ = image.shape
    data = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        x_px, y_px = int(lm.x * w), int(lm.y * h)
        data.append([id, lm.x, lm.y, lm.z, lm.visibility, x_px, y_px])

    # utwórz DataFrame
    df = pd.DataFrame(data, columns=["id", "x", "y", "z", "visibility", "x_px", "y_px"])



    return df, results