import time

from utils.VideoPoseStream import VideoPoseStream
from utils.Skeleton_generator import Skeleton_generator
from utils.recognize_intention import recognize_intention
from utils.image_process_fn import resize_image_to_width
from utils.train_models import train_models
import cv2
import os

def runPrediction():
    skeleton_generator = Skeleton_generator(False)
    videoPoseStream = VideoPoseStream(skeleton_generator.generate, recognize_intention)
    videoPoseStream.activate()

def runPhotoProcessing(pictures_location, data_outcome_folder):
    skeleton_generator = Skeleton_generator(True)

    pictures_names = os.listdir(pictures_location)
    for picture_name in pictures_names:
        full_picture_path = os.path.join(pictures_location, picture_name)
        print(full_picture_path)

        cv2.namedWindow("Processing", cv2.WINDOW_AUTOSIZE)
        image = cv2.imread(full_picture_path)

        image = resize_image_to_width(image, 400)
        df_skeleton = skeleton_generator.generate(image)
        cv2.imshow("Processing (any key to skip)", image)

        df_skeleton.to_csv(os.path.join(data_outcome_folder, picture_name.split(".")[0] + ".csv"))

        # cv2.waitKey(3000)

if __name__ == '__main__':
    print("1. Uruchom rozpoznawanie intencji")
    print("2. Zaaktualizuj zbiór danych")
    print("3. Statystyki")
    choice = input("Wybierz jedną z poniższych opcji: ")

    if choice == "1":
        runPrediction()
    elif choice == "2":
        pictures_location = "dataset\pictures"
        data_outcome_folder = "dataset\processed_data"
        runPhotoProcessing(pictures_location, data_outcome_folder)
        train_models()

