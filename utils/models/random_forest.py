from sklearn.ensemble import RandomForestClassifier

def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )