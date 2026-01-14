from utils.skeleton_vectors import skeleton_to_feature_vector

def recognize_intention(df_skeleton):
    if df_skeleton is None:
        return "no_person"

    features = skeleton_to_feature_vector(df_skeleton)

    return "none"
