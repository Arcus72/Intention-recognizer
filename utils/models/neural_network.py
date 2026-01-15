from sklearn.neural_network import MLPClassifier

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(256,128),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )