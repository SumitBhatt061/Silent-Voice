import cv2
import config
from processing.hand_tracking import HandTracker
from processing.feature_extractor import extract_features
from models.static_model import StaticModel

def test_single_image():
    # 1. Setup
    tracker = HandTracker()
    model = StaticModel(config.MODEL_PATH)
    
    image_path = "data/asl_alphabet_test/I_test.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Image not found at {image_path}")
        return

    # 2. Detection
    # landmarks: the raw numeric data for the model
    # hand_landmarks: the MediaPipe object used for drawing
    landmarks, hand_landmarks = tracker.get_landmarks(image)

    if hand_landmarks:
        print("HAND DETECTED ✅")
        
        # 3. Draw the landmarks onto the image
        # NOTE: tracker.draw_landmarks must return the modified image
        annotated_image = tracker.draw_landmarks(image.copy(), hand_landmarks)
        
        # 4. Predict
        features = extract_features(landmarks)
        prediction, confidence = model.predict(features)
        
        # 5. Overlay Text (This shows WHAT was detected)
        display_text = f"Detected: {prediction} ({round(confidence * 100, 2)}%)"
        
        # Draw a background rectangle for text readability
        cv2.rectangle(annotated_image, (5, 10), (450, 60), (0, 0, 0), -1)
        cv2.putText(annotated_image, display_text, (10, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(f"Prediction: {prediction}")
        
        # 6. Show the result
        cv2.imshow("Detection Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("NO HAND DETECTED ❌")
        cv2.imshow("No Hand Found", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    test_single_image()