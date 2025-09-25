# backend/train_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# === Load Dataset ===
# Adjust paths if needed (your Kaggle CSVs should be in data/)
entries_path = "data/entries.csv"
targets_path = "data/targets.csv"

entries = pd.read_csv(entries_path)
targets = pd.read_csv(targets_path)

# Ensure proper column names (adjust if different in your CSVs)
texts = entries["text"].astype(str)
labels = targets["label"].astype(int)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# === Vectorization ===
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Train Model ===
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# === Evaluate ===
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Save Vectorizer + Model ===
os.makedirs("filters", exist_ok=True)
joblib.dump(vectorizer, "filters/vectorizer.pkl")
joblib.dump(model, "filters/hate_model.pkl")

print("Saved vectorizer.pkl and hate_model.pkl inside filters/")
