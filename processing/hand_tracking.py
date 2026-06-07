import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7
        )

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            all_hands = []
            handedness_labels = []

            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):

                # 🔹 Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                all_hands.append(landmarks)

                # 🔹 FIX: Extract "Left"/"Right"
                if result.multi_handedness:
                    label = result.multi_handedness[i].classification[0].label
                else:
                    label = "Right"  # fallback

                handedness_labels.append(label)

            return all_hands, result.multi_hand_landmarks, handedness_labels

        return None, None, None

    def draw_landmarks(self, frame, hand_landmarks):
        if hand_landmarks:
            for hand in hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame