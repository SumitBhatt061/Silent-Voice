import os
import numpy as np
import imageio
import cv2
import mediapipe as mp

from utils.sequence_utils import normalize_sequence
from processing.feature_extractor_2hand import extract_both_hands, add_velocity

VIDEO_FOLDER = "data/filtered_dataset"
SAVE_PATH = "data/sequences"

SEQUENCE_LENGTH = 30
MIN_FRAMES_REQUIRED = 15

# ✅ MediaPipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

processed = 0
skipped = 0

labels = os.listdir(VIDEO_FOLDER)

print(f"📂 Total labels found: {len(labels)}")

for label in labels:
    label_path = os.path.join(VIDEO_FOLDER, label)

    if not os.path.isdir(label_path):
        continue

    clean_label = label.replace(" ", "_").lower()
    video_files = os.listdir(label_path)

    print(f"\n📌 Processing label: {label} → {clean_label} ({len(video_files)} videos)")

    for video_file in video_files:

        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_path = os.path.join(label_path, video_file)

        try:
            reader = imageio.get_reader(video_path)
        except Exception as e:
            print(f"❌ Failed to read {video_file}: {e}")
            skipped += 1
            continue

        sequence = []

        for frame in reader:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb)

                # =========================
                # ✅ FIX STARTS HERE
                # =========================
                if results.multi_hand_landmarks:

                    hand_landmarks = results.multi_hand_landmarks

                    # extract "Left"/"Right"
                    handedness = [
                        h.classification[0].label
                        for h in results.multi_handedness
                    ] if results.multi_handedness else ["Right"] * len(hand_landmarks)

                    features = extract_both_hands(hand_landmarks, handedness)

                else:
                    continue  # skip frame
                # =========================
                # ✅ FIX ENDS HERE
                # =========================

                if np.all(features == 0):
                    continue

                sequence.append(features)

            except Exception as e:
                print(f"⚠️ Frame error in {video_file}: {e}")
                continue

        if len(sequence) < MIN_FRAMES_REQUIRED:
            print(f"⚠️ Skipped {video_file} (too short: {len(sequence)} frames)")
            skipped += 1
            continue

        sequence = np.array(sequence)

        # motion
        sequence = add_velocity(sequence)

        # normalize
        sequence = normalize_sequence(sequence, SEQUENCE_LENGTH)

        save_label_path = os.path.join(SAVE_PATH, clean_label)
        os.makedirs(save_label_path, exist_ok=True)

        save_name = os.path.splitext(video_file)[0] + ".npy"
        save_file = os.path.join(save_label_path, save_name)

        np.save(save_file, sequence)

        print(f"✅ {video_file} → {clean_label}")
        processed += 1

print("\n📊 SUMMARY")
print(f"✅ Processed: {processed}")
print(f"⚠️ Skipped: {skipped}")