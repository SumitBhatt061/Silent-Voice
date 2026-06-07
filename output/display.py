import cv2
import numpy as np

def draw_wrapped_text(img, text, position, max_width, font, font_scale, color, thickness, line_spacing=30):
    """
    Helper function to wrap long sentences so they don't overlap across sections.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if line_width < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
        
    x, y = position
    for line in lines:
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_spacing

def show_prediction(frame, current_pred, stable_pred, word, confidence, mode, corrected_sentence):
    # --- CONFIGURATION ---
    h, w = frame.shape[:2]
    sidebar_w = 340   
    bottom_h = 140    # Slightly increased height for wrapped text breathing room
    bg_color = (24, 24, 27)      
    panel_color = (32, 32, 35)   
    accent_color = (255, 170, 50) 
    
    # Create the complete canvas dashboard layout
    canvas = np.zeros((h + bottom_h, w + sidebar_w, 3), dtype=np.uint8)
    canvas[:] = bg_color

    # Insert camera feed on the left side
    canvas[0:h, 0:w] = frame
    
    # ----------------------------------------------------
    # SIDEBAR PANEL (Right Rail)
    # ----------------------------------------------------
    cv2.rectangle(canvas, (w, 0), (w + sidebar_w, h), panel_color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header & Mode Badge
    cv2.putText(canvas, "SYSTEM STATUS", (w + 25, 40), font, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    
    mode_color = (100, 255, 150) if mode == "static" else (100, 200, 255)
    cv2.rectangle(canvas, (w + 180, 23), (w + sidebar_w - 20, 47), (45, 45, 50), -1)
    cv2.putText(canvas, f"MODE: {mode.upper()}", (w + 195, 40), font, 0.4, mode_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (w + 25, 60), (w + sidebar_w - 25, 60), (60, 60, 60), 1)

    # Current Raw Prediction Component
    cv2.putText(canvas, "LIVE RECOGNITION", (w + 25, 95), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    display_pred = current_pred if current_pred else "Scanning..."
    cv2.putText(canvas, f"{display_pred}", (w + 25, 130), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Confidence Metrics Indicator Bar
    bar_x, bar_y = w + 25, 150
    bar_w, bar_h = 220, 6
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    if confidence > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), accent_color, -1)
    cv2.putText(canvas, f"{int(confidence*100)}%", (bar_x + bar_w + 12, bar_y + 7), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # Stabilized Result Component Card
    cv2.putText(canvas, "STABILIZED OUTPUT", (w + 25, 215), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (w + 20, 230), (w + sidebar_w - 20, 290), (45, 45, 50), -1)
    
    display_stable = stable_pred if stable_pred else "---"
    cv2.putText(canvas, f"{display_stable}", (w + 35, 272), font, 1.1, (100, 255, 150), 2, cv2.LINE_AA)

    # Hotkeys Instruction Guide footer
    cv2.putText(canvas, "KEYBOARD SHORTCUTS", (w + 25, h - 110), font, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
    controls = [
        "S: Static Mode  |  D: Dynamic Mode",
        "SPACE: Trigger NLP Speak & Reset",
        "C: Clear System Canvas Buffer",
        "ESC: Secure Shutdown Workspace"
    ]
    for i, line in enumerate(controls):
        cv2.putText(canvas, line, (w + 25, (h - 85) + (i * 22)), font, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # ----------------------------------------------------
    # FOOTER PANEL (Sentence Construction Track)
    # ----------------------------------------------------
    total_width = w + sidebar_w
    mid_x = total_width // 2
    max_text_width = mid_x - 50 # Prevents text from hitting the boundaries/dividers
    
    # Card Border Dividers
    cv2.line(canvas, (0, h), (total_width, h), (50, 50, 55), 2)
    cv2.line(canvas, (mid_x, h), (mid_x, h + bottom_h), (50, 50, 55), 1)

    # Left Column: Raw Word Sequence Buffer
    cv2.putText(canvas, "ACCUMULATED CHARACTER BUFFER", (25, h + 30), font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    raw_sentence = word if word.strip() else "Awaiting system gesture signatures..."
    raw_color = (255, 255, 255) if word.strip() else (90, 90, 95)
    
    # Wrapped text handling instead of standard putText
    draw_wrapped_text(canvas, raw_sentence, (25, h + 65), max_text_width, font, 0.7, raw_color, 2, line_spacing=25)

    # Right Column: AI / LLM Context-Corrected Output
    cv2.putText(canvas, "CORRECTED SENTENCE (TTS)", (mid_x + 25, h + 30), font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    ai_sentence = corrected_sentence if corrected_sentence else "Awaiting natural language conversion step..."
    ai_color = (100, 200, 255) if corrected_sentence else (90, 90, 95)
    
    # Wrapped text handling instead of standard putText
    draw_wrapped_text(canvas, ai_sentence, (mid_x + 25, h + 65), max_text_width, font, 0.7, ai_color, 2, line_spacing=25)

    return canvas