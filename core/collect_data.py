import cv2
import json
import os
import time
import numpy as np
import mediapipe as mp
import sys

# Ensure parent directory is in import path to access core and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.classifier import get_invariant_features

def open_camera():
    """Tries to open the camera using backend configuration from config.py."""
    print("[Data Collector] Initializing camera...")
    for backend in config.CAMERA_BACKENDS:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        if not cap.isOpened():
            print(f"[Data Collector] Backend {backend} failed to open")
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        time.sleep(config.CAMERA_WARMUP_SLEEP_S)
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            cap.read()

        ret, frame = cap.read()
        if ret and frame is not None and frame.mean() >= config.MIN_FRAME_BRIGHTNESS:
            print(f"[Data Collector] Camera successfully opened using backend {backend}")
            return cap
        cap.release()
    return None

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.json")
    
    # Load existing dataset if it exists
    dataset = []
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
            print(f"[Data Collector] Loaded existing dataset with {len(dataset)} samples.")
        except Exception as e:
            print(f"[Data Collector WARNING] Failed to load existing dataset: {e}")
            
    # Count existing samples by label
    counts = {"rock": 0, "paper": 0, "scissors": 0, "unknown": 0}
    for item in dataset:
        lbl = item.get("label")
        if lbl in counts:
            counts[lbl] += 1

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    cap = open_camera()
    if cap is None:
        print("[Data Collector ERROR] Could not open any camera backend. Exiting.")
        return

    print("\n" + "="*50)
    print(" GESTURE DATA COLLECTOR")
    print("="*50)
    print("Hold the following keys while waving/tilting your hand:")
    print("  'r' -> Record ROCK")
    print("  'p' -> Record PAPER")
    print("  's' -> Record SCISSORS")
    print("  'u' -> Record UNKNOWN / NOISE (thumbs up, open hand moving, etc.)")
    print("  'ESC' -> Save and Exit")
    print("  'q' -> Quit WITHOUT saving")
    print("="*50 + "\n")

    last_record_time = 0.0
    record_cooldown = 0.05  # Max 20 Hz recording to prevent identical duplicate frames

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Data Collector] Frame capture error.")
            break

        # Flip frame horizontally to mimic mirrored display
        if config.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        # Convert color to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        label_to_record = None
        
        # Check keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            # Save and exit
            print(f"[Data Collector] Saving dataset with {len(dataset)} samples to {dataset_path}...")
            try:
                with open(dataset_path, "w") as f:
                    json.dump(dataset, f, indent=2)
                print("[Data Collector] Save successful! Exiting.")
            except Exception as e:
                print(f"[Data Collector ERROR] Failed to save dataset: {e}")
            break
        elif key == ord('q'):
            print("[Data Collector] Exiting without saving changes.")
            break
        elif key == ord('r'):
            label_to_record = "rock"
        elif key == ord('p'):
            label_to_record = "paper"
        elif key == ord('s'):
            label_to_record = "scissors"
        elif key == ord('u'):
            label_to_record = "unknown"

        # If hand detected, draw skeleton and record if key pressed
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check record cooldown
                now = time.time()
                if label_to_record and (now - last_record_time >= record_cooldown):
                    lm = hand_landmarks.landmark
                    try:
                        # Extract invariant features
                        features = get_invariant_features(lm)
                        
                        # Store raw coordinates for visual debugging/re-projection if needed
                        raw_coords = [[pt.x, pt.y, pt.z] for pt in lm]
                        
                        dataset.append({
                            "label": label_to_record,
                            "features": features.tolist(),
                            "raw_landmarks": raw_coords
                        })
                        
                        counts[label_to_record] += 1
                        last_record_time = now
                        print(f"Recorded [{label_to_record.upper()}] - Total: {counts[label_to_record]}")
                    except Exception as e:
                        print(f"[Data Collector] Recording error: {e}")

        # Display counts and instructions on frame
        h, w = frame.shape[:2]
        y0, dy = 40, 30
        
        # Overlay translucent box for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Dataset Summary:", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"  Rock:     {counts['rock']}", (20, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"  Paper:    {counts['paper']}", (20, y0 + dy*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"  Scissors: {counts['scissors']}", (20, y0 + dy*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"  Unknown:  {counts['unknown']}", (20, y0 + dy*4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"  Total:    {len(dataset)}", (20, y0 + dy*5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw recording overlay if active
        if label_to_record:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"RECORDING {label_to_record.upper()}", (w - 240, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Gesture Data Collector", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
