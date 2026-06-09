import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp

# Ensure parent directory is in path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.classifier import get_invariant_features

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rock_frames_dir = os.path.join(base_dir, "rock_frames")
    output_path = os.path.join(base_dir, "core", "real_rock_features.json")
    
    print("[Extractor] Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.2  # Match low threshold for hand detection robustness
    )
    
    if not os.path.exists(rock_frames_dir):
        print(f"[Extractor ERROR] rock_frames directory not found at: {rock_frames_dir}")
        sys.exit(1)
        
    # Get all jpg files
    files = sorted([f for f in os.listdir(rock_frames_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    print(f"[Extractor] Found {len(files)} image frames to process.")
    
    real_features = []
    processed_count = 0
    skipped_count = 0
    
    for filename in files:
        img_path = os.path.join(rock_frames_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [Warning] Failed to load {filename}. Skipping.")
            skipped_count += 1
            continue
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            # Extract the 60-D pose-invariant features
            features = get_invariant_features(lm)
            real_features.append(features.tolist())
            processed_count += 1
        else:
            # Skip frame if MediaPipe cannot detect a hand
            print(f"  [Warning] No hand detected in {filename}. Skipping.")
            skipped_count += 1
            
    mp_hands.close()
    
    print(f"[Extractor] Completed. Successfully processed: {processed_count}, Skipped: {skipped_count}.")
    
    # Save the feature vectors to a new file
    with open(output_path, "w") as f:
        json.dump(real_features, f, indent=2)
    print(f"[Extractor] Saved features to {output_path}")

if __name__ == "__main__":
    main()
