import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

from utils.sequence_buffer import SequenceBuffer
from utils.sequence_utils import normalize_sequence
from processing.feature_extractor_2hand import add_velocity

# ⚠️ IMPORTANT: add_velocity is also called in prepare_sequences.py during training.
# The model input shape expects (SEQUENCE_LENGTH, features * 2) because of this.
# If you ever remove or change add_velocity here, you MUST retrain the model,
# and update prepare_sequences.py to match — and vice versa.


class DynamicModel:
    def __init__(self, model_path, labels, seq_length=30, smooth_window=5):
        self.model = load_model(model_path)
        self.labels = labels
        self.seq_length = seq_length

        self.buffer = SequenceBuffer(seq_length)
        self.pred_history = deque(maxlen=smooth_window)

    def predict(self, features):
        if features is None:
            return None, 0

        if np.all(features == 0):
            return None, 0

        self.buffer.add(features)

        if not self.buffer.is_full():
            return None, 0

        sequence = np.array(self.buffer.get())

        # ⚠️ See warning at top of file before modifying this
        sequence = add_velocity(sequence)

        sequence = normalize_sequence(sequence, self.seq_length)

        sequence = np.expand_dims(sequence, axis=0)

        prediction = self.model.predict(sequence, verbose=0)[0]

        pred_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        self.pred_history.append(pred_index)
        final_index = max(set(self.pred_history), key=self.pred_history.count)

        final_label = self.labels[final_index]

        return final_label, confidence

    def reset(self):
        self.buffer.clear()
        self.pred_history.clear()