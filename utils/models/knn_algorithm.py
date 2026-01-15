from sklearn.neighbors import KNeighborsClassifier

def create_model():
    return KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    )