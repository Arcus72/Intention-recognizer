from utils.models.model_io import load_model
from collections import deque
import numpy as np
import joblib

_PROB_BUFFER = deque(maxlen=10)
_model = load_model("random_forest")

scaler = joblib.load("trained_models/scaler.joblib")
label_encoder = joblib.load("trained_models/label_encoder.joblib")

def recognize_intention(df_skeleton):
    if df_skeleton is None:
        return "no_person"

    # Jeśli korzystamy z wektorów to te trzy linijki zakomentować, a skeleton_to_feature odkomentować
    feature_cols = ["x", "y", "z", "visibility"]
    df = df_skeleton[feature_cols]
    df = df.to_numpy().flatten()

    features = scaler.transform(df.reshape(1, -1))

    probs = _model.predict_proba(features)[0]
    _PROB_BUFFER.append(probs)

    if len(_PROB_BUFFER) < 10:
        return "Analizing..."

    avg_probs = np.mean(_PROB_BUFFER, axis=0)

    best_idx = np.argmax(avg_probs)
    best_prob = avg_probs[best_idx]

    if best_prob < 0.25:
        return "Uncertain"

    return label_encoder.inverse_transform([best_idx])[0]