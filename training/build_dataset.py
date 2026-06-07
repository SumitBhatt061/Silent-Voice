import os
import cv2
import numpy as np
import mediapipe as mp

from utils.geometry import get_angle, FINGER_JOINTS  # ✅ FIX: no longer duplicated here

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "asl_dataset")
)

print("Resolved DATA_DIR:", DATA_DIR)

X, y = [], []

detected = 0
skipped = 0

# ----------------------------
# PROCESS DATASET
# ----------------------------
for label in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing label: {label}")

    for img_name in os.listdir(folder_path):

        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        image = cv2.resize(image, (512, 512))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            skipped += 1
            continue

        detected += 1

        # ----------------------------
        # LANDMARKS
        # ----------------------------
        lm = result.multi_hand_landmarks[0].landmark
        landmarks = np.array([[p.x, p.y, p.z] for p in lm])

        # ----------------------------
        # NORMALIZATION
        # ----------------------------
        base = landmarks[0]
        landmarks = landmarks - base

        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val

        # ----------------------------
        # XY FEATURES (42)
        # ----------------------------
        xy_features = landmarks[:, :2].flatten()

        # ----------------------------
        # ANGLE FEATURES (12)
        # ----------------------------
        angles = [
            get_angle(landmarks[a], landmarks[b], landmarks[c])
            for a, b, c in FINGER_JOINTS  # ✅ FIX: was a local list defined inside the loop
        ]

        # ----------------------------
        # FINAL FEATURES (54)
        # ----------------------------
        features = np.concatenate([xy_features, angles])

        X.append(features)
        y.append(label.upper())

# ----------------------------
# SAVE DATA
# ----------------------------
X = np.array(X)
y = np.array(y)

print("\nFINAL DATASET SIZE:", X.shape)
print("Detected:", detected)
print("Skipped:", skipped)

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Dataset saved successfully ✅")