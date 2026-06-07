import numpy as np


# -----------------------------
# 🔹 Normalize a single hand
# -----------------------------
def normalize_hand(coords):
    """
    Normalize landmarks relative to wrist (landmark 0)
    and scale by max absolute value — same as static extractor.
    """
    base = coords[0]  # wrist
    coords = coords - base

    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val

    return coords


# -----------------------------
# 🔹 Extract both hands (FIXED)
# -----------------------------
def extract_both_hands(hand_landmarks, handedness):
    """
    Inputs:
    - hand_landmarks: list of MediaPipe hand landmark objects
    - handedness: list of "Left" / "Right" strings

    Output:
    - 126 features (63 left + 63 right), normalized per hand
    """

    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if hand_landmarks is None or len(hand_landmarks) == 0:
        return np.concatenate([left_hand.flatten(), right_hand.flatten()])

    for hand, hand_type in zip(hand_landmarks, handedness):

        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])

        # ✅ FIX: normalize before storing (was missing before)
        coords = normalize_hand(coords)

        if hand_type == "Left":
            left_hand = coords
        else:
            right_hand = coords

    left_flat = left_hand.flatten()
    right_flat = right_hand.flatten()

    return np.concatenate([left_flat, right_flat])


# -----------------------------
# 🔹 Add velocity (motion)
# -----------------------------
def add_velocity(sequence):
    """
    Appends per-frame velocity (diff) to the feature vector.

    Input:  (T, F)
    Output: (T, F*2)
    """

    if len(sequence) < 2:
        velocity = np.zeros_like(sequence)
        return np.concatenate([sequence, velocity], axis=1)

    velocity = np.diff(sequence, axis=0)

    # pad first frame with zeros so shape stays (T, F)
    first = np.zeros_like(velocity[0])
    velocity = np.vstack([first, velocity])

    return np.concatenate([sequence, velocity], axis=1)