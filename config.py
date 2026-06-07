# =========================
# MODEL PATHS
# =========================
MODEL_PATH = "models/alphabet_model.pkl"
LABEL_ENCODER_STATIC_PATH = "models/label_encoder.pkl"   # ✅ FIX: was hardcoded in StaticModel

DYNAMIC_MODEL_PATH = "models/dynamic_model.h5"
LABEL_ENCODER_DYNAMIC_PATH = "models/label_encoder_dynamic.pkl"

# =========================
# SEQUENCE SETTINGS
# =========================
SEQUENCE_LENGTH = 30          # ✅ FIX: was defined twice (second overwrote first)
BUFFER_SIZE = 30

# =========================
# FEATURE DIMENSIONS
# =========================
STATIC_FEATURE_DIM = 54       # ✅ FIX: was magic number scattered across main.py
RAW_FEATURE_DIM = 63          # full feature vector before slicing

# =========================
# CONFIDENCE THRESHOLDS
# =========================
STATIC_CONF = 0.7
DYNAMIC_CONF = 0.5

# =========================
# TIMING
# =========================
NO_HAND_CONFIRM_FRAMES = 15   # ✅ FIX: was magic number in main.py (confirm_threshold)

# =========================
# FLAGS
# =========================
USE_STATIC_ONLY = False