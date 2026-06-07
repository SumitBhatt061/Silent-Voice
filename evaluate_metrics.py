import os
import cv2
import mediapipe as mp
import numpy as np
from models.static_model import StaticModel
from processing.feature_extractor import extract_features
import config

# 1. Setup MediaPipe specifically for TOUGH static images
mp_hands = mp.solutions.hands
hands_processor = mp_hands.Hands(
    static_image_mode=True,       # Required for unrelated photos
    max_num_hands=1,
    min_detection_confidence=0.3, # Lowered to catch "G" and "H" which are flat
    model_complexity=1            # Use 1 for better accuracy on static shots
)

model = StaticModel(config.MODEL_PATH)
DATASET_PATH = "data/asl_alphabet_test"
SKIP_CLASSES = ['j', 'z']

correct, total, detection_failures = 0, 0, 0
class_stats = {}

print("\n===== STATIC IMAGE EVALUATION MODE =====\n")

# Use os.listdir and filter manually to ensure we see the files
files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in files:
    label = img_name.split("_")[0].lower()
    if label in SKIP_CLASSES:
        continue

    img_path = os.path.join(DATASET_PATH, img_name)
    img = cv2.imread(img_path)
    if img is None: continue

    if label not in class_stats:
        class_stats[label] = {'correct': 0, 'total': 0}

    class_stats[label]['total'] += 1
    total += 1

    # --- IMAGE PREPROCESSING FOR MEDIAPIPE ---
    # 1. Resize to a standard size (MediaPipe likes ~640px)
    img = cv2.resize(img, (640, 640))
    # 2. Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. Optional: Improve contrast (Helps if images are dark)
    img_rgb = cv2.normalize(img_rgb, None, 0, 255, cv2.NORM_MINMAX)

    result = hands_processor.process(img_rgb)

    if result.multi_hand_landmarks:
        # Convert landmarks to the list format [[x, y, z], ...]
        landmarks = []
        for res in result.multi_hand_landmarks[0].landmark:
            landmarks.append([res.x, res.y, res.z])
            
        features = extract_features(landmarks)
        pred, conf = model.predict(features)
        
        # Clean up strings for comparison
        pred_clean = str(pred).lower().strip()
        
        if pred_clean == label:
            correct += 1
            class_stats[label]['correct'] += 1
        else:
            print(f"❌ MISMATCH: {label.upper()} -> Pred: {pred_clean.upper()} ({int(conf*100)}%)")
    else:
        detection_failures += 1
        print(f"⚠️ NO HAND: {img_name} (MediaPipe couldn't find it)")

# --- FINAL REPORT ---
print("\n" + "="*30)
if total > 0:
    det_rate = ((total - detection_failures) / total) * 100
    print(f"TOTAL IMAGES:    {total}")
    print(f"DETECTION RATE:  {det_rate:.2f}%")
    
    if (total - detection_failures) > 0:
        acc = (correct / total) * 100
        print(f"MODEL ACCURACY:  {acc:.2f}%")
    
    print("-" * 30)
    for l in sorted(class_stats.keys()):
        s = class_stats[l]
        c_acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0
        print(f" {l.upper():<5}: {c_acc:>6.1f}%")
else:
    print("Zero images found in directory.")
print("="*30)