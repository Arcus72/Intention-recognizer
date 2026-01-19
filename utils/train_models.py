from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.models.random_forest import create_model as rf_model
from utils.models.neural_network import create_model as nn_model
from utils.models.knn_algorithm import create_model as knn_model
from utils.models.model_io import save_model
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

def csv_to_features(path):
    df = pd.read_csv(path)
    df = df.set_index("id")

    #return skeleton_to_feature_vector(df)

    # Jeśli korzystamy z wektorów to zakomentować linijki poniżej, a odkomentować return powyżej
    feature_cols = ["x", "y", "z", "visibility"]
    df = df[feature_cols]

    return df
    #.to_numpy().flatten() - test test

def label_from_filename(filename):
    name = filename.lower()
    if "agresja" in name:
        return "Aggression"
    if "neutralnosc" in name:
        return "Neutrality"
    if "lek" in name:
        return "Fear"
    return "Uncertain"

LEFT_RIGHT_PAIRS = [
    (7, 8),
    (11, 12), (13, 14), (15, 16),
    (19, 20),
    (23, 24), (25, 26), (27, 28),
    (31, 32)
]

def flip_pose(df):
    df = df.copy()

    df["x"] = -df["x"]

    for left, right in LEFT_RIGHT_PAIRS:
        df.loc[[left, right]] = df.loc[[right, left]].values

    return df

def rotate_y(df, angle_range=(-15, 15)):
    df = df.copy()
    angle = np.deg2rad(np.random.uniform(*angle_range))

    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x = df["x"].values
    z = df["z"].values

    df["x"] = cos_a * x - sin_a * z
    df["z"] = sin_a * x + cos_a * z

    return df

def scale_pose(df, scale_range=(0.9, 1.1)):
    df = df.copy()
    scale = np.random.uniform(*scale_range)
    df[["x", "y", "z"]] *= scale
    return df

def add_noise(df, sigma=0.01):
    df = df.copy()
    noise = np.random.normal(0, sigma, size=(len(df), 3))
    df[["x", "y", "z"]] += noise
    return df

def augment_pose(df):
    if np.random.rand() < 0.5:
        df = rotate_y(df)
    if np.random.rand() < 0.5:
        df = scale_pose(df)
    if np.random.rand() < 0.3:
        df = add_noise(df)
    return df

def train_models():
    X = []
    y = []

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "dataset" / "processed_data"

    for file in DATA_DIR.glob("*.csv"):
        df = csv_to_features(file)

        X.append(df.to_numpy().flatten())
        y.append(label_from_filename(file.name))

        flipped_df = flip_pose(df)
        X.append(flipped_df.to_numpy().flatten())
        y.append(label_from_filename(file.name))

        aug_df = augment_pose(df)
        X.append(aug_df.to_numpy().flatten())
        y.append(label_from_filename(file.name))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = rf_model()
    nn = nn_model()
    knn = knn_model()

    rf.fit(X_scaled, y_enc)
    nn.fit(X_scaled, y_enc)
    knn.fit(X_scaled, y_enc)

    save_model(rf, "random_forest")
    save_model(nn, "neural_network")
    save_model(knn, "knn")

    joblib.dump(scaler, "trained_models/scaler.joblib")
    joblib.dump(le, "trained_models/label_encoder.joblib")

    print("Zaktualizowano modele")