import numpy as np
from utils.geometry import get_angle, FINGER_JOINTS  # ✅ FIX: no longer duplicated here


def extract_features(all_hands, handedness):
    if all_hands is None or handedness is None:
        return None

    # ----------------------------
    # Pick ONLY one hand (Right preferred)
    # ----------------------------
    selected_hand = None

    for i, hand_info in enumerate(handedness):
        # handedness entries can be raw strings or MediaPipe objects
        if isinstance(hand_info, str):
            label = hand_info
        else:
            label = hand_info.classification[0].label

        if label == "Right":
            selected_hand = all_hands[i]
            break

    # Fallback: if Right hand not detected, use whatever is available
    if selected_hand is None:
        selected_hand = all_hands[0]

    landmarks = np.array(selected_hand)

    # ----------------------------
    # Normalize (wrist-relative + scale)
    # ----------------------------
    base = landmarks[0]
    landmarks = landmarks - base

    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks = landmarks / max_val

    # ----------------------------
    # XY features — 21 landmarks × 2 = 42 values
    # ----------------------------
    xy_features = landmarks[:, :2].flatten()

    # ----------------------------
    # Angle features — 12 joints = 12 values
    # ----------------------------
    angles = [
        get_angle(landmarks[a], landmarks[b], landmarks[c])
        for a, b, c in FINGER_JOINTS
    ]

    # ----------------------------
    # Final: 42 + 12 = 54 features
    # ----------------------------
    return np.concatenate([xy_features, angles])