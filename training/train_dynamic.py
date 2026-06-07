import os
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 🔹 Paths
# -----------------------------
DATA_PATH = "data/sequences"
MODEL_PATH = "models/dynamic_model.h5"
ENCODER_PATH = "models/label_encoder_dynamic.pkl"

# -----------------------------
# 🔹 Load dataset
# -----------------------------
X = []
y = []

print("📂 Loading sequences...")

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(label_path, file))
            X.append(seq)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Total samples: {len(X)}")
print(f"📐 Shape: {X.shape}")

# -----------------------------
# 🔹 Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder
os.makedirs("models", exist_ok=True)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

print(f"✅ Labels encoded: {len(set(y_encoded))} classes")

# -----------------------------
# 🔹 Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# 🔹 Model
# -----------------------------
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dense(len(set(y_encoded)), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 🔹 Training
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("🚀 Training started...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# -----------------------------
# 🔹 Save model
# -----------------------------
model.save(MODEL_PATH)

print("✅ Model saved!")