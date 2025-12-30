import numpy as np
import pandas as pd

LANDMARKS = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}

# Połączenia stawów – "kości"
BONES = [
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]

def get_point(df, name):
    idx = LANDMARKS[name]
    row = df[df["id"] == idx].iloc[0]
    return np.array([row["x_px"], row["y_px"]], dtype=np.float32)

def normalize_points(df):
    left_hip = get_point(df, "LEFT_HIP")
    right_hip = get_point(df, "RIGHT_HIP")
    center = (left_hip + right_hip) / 2

    df = df.copy()
    df["x_norm"] = df["x_px"] - center[0]
    df["y_norm"] = df["y_px"] - center[1]
    return df

def normalize_scale(df):
    left_shoulder = get_point(df, "LEFT_SHOULDER")
    right_shoulder = get_point(df, "RIGHT_SHOULDER")

    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    if shoulder_width < 1e-5:
        shoulder_width = 1.0

    df = df.copy()
    df["x_norm"] /= shoulder_width
    df["y_norm"] /= shoulder_width
    return df

def compute_bone_vectors(df):
    df = normalize_points(df)
    df = normalize_scale(df)

    vectors = []
    for start, end in BONES:
        p1 = get_point(df, start)
        p2 = get_point(df, end)

        p1_norm = np.array([df.loc[df["id"] == LANDMARKS[start], "x_norm"].values[0],
                            df.loc[df["id"] == LANDMARKS[start], "y_norm"].values[0]])
        p2_norm = np.array([df.loc[df["id"] == LANDMARKS[end], "x_norm"].values[0],
                            df.loc[df["id"] == LANDMARKS[end], "y_norm"].values[0]])

        v = p2_norm - p1_norm
        vectors.extend(v.tolist())

    return np.array(vectors, dtype=np.float32)

def skeleton_to_feature_vector(df):
    if df is None:
        return None
    return compute_bone_vectors(df)
