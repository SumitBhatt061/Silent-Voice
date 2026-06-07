import cv2
import numpy as np
import pickle

from capture.camera import Camera
from processing.hand_tracking import HandTracker
from processing.feature_extractor import extract_features
from processing.feature_extractor_2hand import extract_both_hands

from models.static_model import StaticModel
from models.dynamic_model import DynamicModel

from utils.buffer import SequenceBuffer
from output.display import show_prediction
import config

from sentence.sentence_builder import build_sentence
from ttl.speaker import init_tts, speak_text

def main():
    camera = Camera()
    tracker = HandTracker()

    static_model = StaticModel(config.MODEL_PATH)

    with open(config.LABEL_ENCODER_DYNAMIC_PATH, "rb") as f:
        labels = pickle.load(f).classes_

    dynamic_model = DynamicModel(config.DYNAMIC_MODEL_PATH, labels)

    print("🔊 Initializing TTS")
    init_tts()

    static_buffer = SequenceBuffer(config.SEQUENCE_LENGTH)
    dynamic_buffer = SequenceBuffer(10)

    word = ""
    last_prediction = ""
    no_hand_counter = 0

    current_pred = ""
    stable_pred = ""
    dynamic_stable_pred = ""
    confidence = 0.0

    prev_stable_pred = ""
    prev_dynamic_pred = ""

    mode = "static"
    last_corrected_sentence = ""

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        all_hands, hand_landmarks, handedness = tracker.get_landmarks(frame)
        frame = tracker.draw_landmarks(frame, hand_landmarks)

        if hand_landmarks is not None:
            if mode == "dynamic":
                if handedness is not None:
                    features = extract_both_hands(hand_landmarks, handedness)
                else:
                    features = None

                if features is not None:
                    pred, conf = dynamic_model.predict(features)
                else:
                    pred, conf = None, 0.0

                if pred is not None:
                    current_pred = pred
                    confidence = conf

                    if conf >= config.DYNAMIC_CONF:
                        dynamic_buffer.add(pred)

                    dynamic_stable_pred = dynamic_buffer.get_most_common()

                    if dynamic_stable_pred:
                        dynamic_stable_pred = dynamic_stable_pred.lower()

                        if dynamic_stable_pred != prev_dynamic_pred:
                            word += " " + dynamic_stable_pred.upper()
                            print("Dynamic:", dynamic_stable_pred)

                            prev_dynamic_pred = dynamic_stable_pred
                            dynamic_buffer.clear()
                            dynamic_model.reset()
                else:
                    current_pred = "Recording..."
                    confidence = 0.0

                stable_pred = ""
                last_prediction = ""
                static_buffer.clear()

            else:  # Static Mode Execution Block
                if handedness is not None:
                    features = extract_features([all_hands[0]], [handedness[0]])
                else:
                    features = None

                if features is not None:
                    if len(features) > config.RAW_FEATURE_DIM:
                        features = features[:config.RAW_FEATURE_DIM]
                    elif len(features) < config.RAW_FEATURE_DIM:
                        features = np.pad(features, (0, config.RAW_FEATURE_DIM - len(features)))

                    static_input = features[:config.STATIC_FEATURE_DIM]
                    pred, conf = static_model.predict(static_input)

                    current_pred = pred
                    confidence = conf

                    if conf >= config.STATIC_CONF:
                        static_buffer.add(pred)

                    stable_pred = static_buffer.get_most_common()

                    if stable_pred:
                        stable_pred = stable_pred.lower()

                        if stable_pred == "del":
                            if stable_pred != prev_stable_pred and len(word) > 0:
                                word = word[:-1]
                                static_buffer.clear()
                                prev_stable_pred = stable_pred
                        elif stable_pred == "space":
                            if stable_pred != prev_stable_pred:
                                word += " "
                                static_buffer.clear()
                                prev_stable_pred = stable_pred
                        elif stable_pred != prev_stable_pred:
                            if stable_pred.isalpha():
                                last_prediction = stable_pred.upper()
                                prev_stable_pred = stable_pred

                no_hand_counter = 0
        else:
            current_pred = ""
            stable_pred = ""
            dynamic_stable_pred = ""
            confidence = 0.0
            no_hand_counter += 1

            if no_hand_counter >= config.NO_HAND_CONFIRM_FRAMES:
                if last_prediction != "":
                    word += last_prediction
                    last_prediction = ""
                    static_buffer.clear()

                dynamic_model.reset()
                dynamic_buffer.clear()
                prev_dynamic_pred = ""

        print(f"[MODE={mode}] WORD={word}")

        # --- REFACTORED INTERFACE GENERATION ---
        # Delegate context compilation directly to our custom UI utility module.
        active_stable = stable_pred if mode == "static" else dynamic_stable_pred
        
        dashboard_canvas = show_prediction(
            frame=frame,
            current_pred=current_pred,
            stable_pred=active_stable,
            word=word,
            confidence=confidence,
            mode=mode,
            corrected_sentence=last_corrected_sentence
        )

        # Full-Screen Deployment Window Hooks
        cv2.namedWindow("SilentVoice", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("SilentVoice", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Display our clean composed dashboard canvas instead of the dirty raw video matrix frame
        cv2.imshow("SilentVoice", dashboard_canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            mode = "static"
            dynamic_model.reset()
            dynamic_buffer.clear()
            print("STATIC MODE")

        elif key == ord('d'):
            mode = "dynamic"
            static_buffer.clear()
            print("DYNAMIC MODE")

        elif key == ord(' '):
            if word.strip():
                print(f"\n🧠 Processing (offline): {word}")
                corrected = build_sentence(word.strip())
                last_corrected_sentence = corrected
                print(f"✅ Sentence: {corrected}")
                speak_text(corrected)
                
                # State Reset Sequences
                word = ""
                last_prediction = ""
                static_buffer.clear()
                dynamic_buffer.clear()
                dynamic_model.reset()
                prev_stable_pred = ""
                prev_dynamic_pred = ""
            else:
                print("⚠️ No input to process")

        elif key == ord('c'):
            word = ""
            last_prediction = ""
            last_corrected_sentence = ""
            static_buffer.clear()
            dynamic_buffer.clear()
            dynamic_model.reset()
            prev_stable_pred = ""
            prev_dynamic_pred = ""
            print("CLEARED")

        elif key == 27: # Escape Key mapping code
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()