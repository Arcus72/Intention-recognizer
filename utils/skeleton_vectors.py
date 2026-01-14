import numpy as np

BONES = [
    (11, 13),  # LEFT_SHOULDER -> LEFT_ELBOW
    (13, 15),  # LEFT_ELBOW -> LEFT_WRIST
    (12, 14),  # RIGHT_SHOULDER -> RIGHT_ELBOW
    (14, 16),  # RIGHT_ELBOW -> RIGHT_WRIST
    (23, 25),  # LEFT_HIP -> LEFT_KNEE
    (25, 27),  # LEFT_KNEE -> LEFT_ANKLE
    (24, 26),  # RIGHT_HIP -> RIGHT_KNEE
    (26, 28),  # RIGHT_KNEE -> RIGHT_ANKLE
]

def skeleton_to_feature_vector(df_skeleton):
    if df_skeleton is None:
        return None

    hip_center_x = (df_skeleton.loc[23, "x"] + df_skeleton.loc[24, "x"]) / 2
    hip_center_y = (df_skeleton.loc[23, "y"] + df_skeleton.loc[24, "y"]) / 2

    shoulder_width = df_skeleton.loc[12, "x"] - df_skeleton.loc[11, "x"]
    if shoulder_width == 0:
        shoulder_width = 1e-6

    norm_points = {}
    for idx, row in df_skeleton.iterrows():
        norm_x = (row["x"] - hip_center_x) / shoulder_width
        norm_y = (row["y"] - hip_center_y) / shoulder_width
        norm_points[idx] = (norm_x, norm_y)

    features = []
    for start, end in BONES:
        start_pt = np.array(norm_points[start])
        end_pt = np.array(norm_points[end])
        vec = end_pt - start_pt
        features.extend(vec.tolist())

    return np.array(features)
