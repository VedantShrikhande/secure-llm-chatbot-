import joblib

class HateClassifier:
    def __init__(self, vect_path="filters/vectorizer.pkl", model_path="filters/hate_model.pkl"):
        self.vect = joblib.load(vect_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> int:
        X = self.vect.transform([text])
        return self.model.predict(X)[0]
