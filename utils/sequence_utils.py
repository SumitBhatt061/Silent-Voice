import numpy as np

def normalize_sequence(sequence, length=30):
    sequence = np.array(sequence)

    if len(sequence) > length:
        indices = np.linspace(0, len(sequence) - 1, length).astype(int)
        return sequence[indices]

    elif len(sequence) < length:
        padding = np.repeat(sequence[-1][np.newaxis, :], length - len(sequence), axis=0)
        return np.vstack((sequence, padding))

    return sequence