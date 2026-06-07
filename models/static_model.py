import joblib
import numpy as np
import config


class StaticModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # ✅ FIX: was hardcoded "models/label_encoder.pkl"
        self.encoder = joblib.load(config.LABEL_ENCODER_STATIC_PATH)

    def predict(self, features):
        if features is None:
            return "", 0

        proba = self.model.predict_proba([features])[0]
        max_prob = float(np.max(proba))

        pred_index = np.argmax(proba)
        label = self.encoder.inverse_transform([pred_index])[0]

        return label, max_prob