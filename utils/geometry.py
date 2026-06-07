import numpy as np


# Shared across feature_extractor.py and dataset builder
def get_angle(a, b, c):
    """
    Computes the angle at point b formed by points a-b-c.
    Works with 2D or 3D numpy arrays.
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


# Joint triplets used for finger angle features
# Covers all 4 fingers (index→pinky), 3 joints each
FINGER_JOINTS = [
    (0, 5, 6),  (5, 6, 7),  (6, 7, 8),   # index
    (0, 9, 10), (9, 10, 11),(10, 11, 12), # middle
    (0, 13, 14),(13, 14, 15),(14, 15, 16),# ring
    (0, 17, 18),(17, 18, 19),(18, 19, 20),# pinky
]