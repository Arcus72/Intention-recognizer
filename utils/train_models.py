from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.models.random_forest import create_model as rf_model
from utils.models.neural_network import create_model as nn_model
from utils.models.knn_algorithm import create_model as knn_model
from utils.models.model_io import save_model
from pathlib import Path
import pandas as pd
import joblib

def csv_to_features(path):
    df = pd.read_csv(path)

    #return skeleton_to_feature_vector(df)

    # Jeśli korzystamy z wektorów to zakomentować linijki poniżej, a odkomentować return powyżej
    feature_cols = ["x", "y", "z", "visibility"]
    df = df[feature_cols]

    return df.to_numpy().flatten()

def label_from_filename(filename):
    name = filename.lower()
    if "agresja" in name:
        return "Aggression"
    if "neutralna" in name or "spokoj" in name:
        return "Neutrality"
    if "pobudzenie" in name:
        return "High-Energy"
    if "lek" in name:
        return "Fear"
    return "Niejednoznaczna"

def train_models(): 
    X = []
    y = []

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "dataset" / "processed_data"

    for file in DATA_DIR.glob("*.csv"):
        X.append(csv_to_features(file))
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