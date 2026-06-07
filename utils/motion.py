import numpy as np

def compute_motion(sequence):
    """
    sequence shape: (frames, 63)
    returns: average movement
    """
    sequence = np.array(sequence)

    if len(sequence) < 2:
        return 0

    diffs = []

    for i in range(1, len(sequence)):
        diff = np.linalg.norm(sequence[i] - sequence[i - 1])
        diffs.append(diff)

    return np.mean(diffs)