from utils.skeleton_vectors import skeleton_to_feature_vector

def recognize_intention(df_skeleton):
    features = skeleton_to_feature_vector(df_skeleton)

    if features is None:
        return "no_person"

    # Przykładowa, bardzo prosta logika:
    # np. ręka w górze – sprawdzamy y wektora bark → łokieć
    # features[1] = y komponent pierwszej kości (LEFT_SHOULDER -> LEFT_ELBOW)
    if features[1] < -0.5:
        return "reka_w_gore"

    # Domyślnie brak gestu
    return "none"
