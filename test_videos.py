import imageio
import cv2
import mediapipe as mp
import numpy as np

# 👉 CHANGE THIS PATH
VIDEO_PATH = r"C:/Users/prhtt/OneDrive/Desktop/silentVoice/data/words/SORRY/62778761952392-SORRY.mp4"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Read video using imageio
reader = imageio.get_reader(VIDEO_PATH)

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

sequence = []
frame_count = 0

print("✅ Video opened with imageio")

for frame in reader:
    frame_count += 1

    # Convert formats
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            keypoints = []

            # Extract 21 landmarks
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            sequence.append(keypoints)

            print(f"✅ Frame {frame_count} - Hand detected")

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        print(f"❌ Frame {frame_count} - No hand")

    cv2.imshow("Test", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cv2.destroyAllWindows()

print("\n🎯 FINAL OUTPUT")
print("Total frames:", frame_count)
print("Sequence length (hand frames):", len(sequence))

# OPTIONAL: convert to numpy
sequence = np.array(sequence)
print("Sequence shape:", sequence.shape)