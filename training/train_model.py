import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.makedirs("models", exist_ok=True)

X = np.load("data/X.npy")
y = np.load("data/y.npy")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=54
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=54
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(model, "models/alphabet_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Model and encoder saved")