import numpy as np
import json
import os
import random
import sys

# Ensure parent directory is in import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.classifier import classify_gesture_heuristic, get_invariant_features

class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def random_rotation_matrix(max_angle_deg=45.0):
    """Generates a random 3D rotation matrix within the specified angle bounds."""
    theta_x = np.radians(random.uniform(-max_angle_deg, max_angle_deg))
    theta_y = np.radians(random.uniform(-max_angle_deg, max_angle_deg))
    theta_z = np.radians(random.uniform(-max_angle_deg, max_angle_deg))
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
                   
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
                   
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
                   
    return np.dot(Rz, np.dot(Ry, Rx))

def make_hand_landmarks(gesture_type, rotation_matrix, scale, translation, is_left=False, noise_level=0.003):
    """
    Constructs a 21-landmark hand skeleton template for the specified gesture,
    optionally mirrors it for a left hand, applies joint noise, and transforms it in 3D.
    """
    joints = [np.zeros(3) for _ in range(21)]
    joints[0] = np.array([0.0, 0.0, 0.0]) # Wrist
    
    # 1. Base MCP coordinates relative to wrist
    joints[1] = np.array([0.06, 0.06, -0.02])   # Thumb CMC
    joints[2] = np.array([0.11, 0.12, -0.03])   # Thumb MCP
    joints[5] = np.array([0.05, 0.22, 0.0])     # Index MCP
    joints[9] = np.array([0.0, 0.23, 0.0])      # Middle MCP
    joints[13] = np.array([-0.04, 0.22, 0.0])   # Ring MCP
    joints[17] = np.array([-0.08, 0.19, -0.01])  # Pinky MCP
    
    # helper to construct finger chain (mcp -> pip -> dip -> tip)
    def add_finger(mcp_idx, ext_dir, curl_factor):
        # Bone lengths
        mcp_to_pip = 0.075
        pip_to_dip = 0.050
        dip_to_tip = 0.038
        
        # Curl rotation angles (radians)
        pip_angle = curl_factor * 1.3
        dip_angle = curl_factor * 1.4
        tip_angle = curl_factor * 0.9
        
        # PIP joint
        dir_pip = ext_dir * np.cos(pip_angle) - np.array([0.0, 0.0, 1.0]) * np.sin(pip_angle)
        joints[mcp_idx + 1] = joints[mcp_idx] + dir_pip * mcp_to_pip
        
        # DIP joint
        tot_dip = pip_angle + dip_angle
        dir_dip = ext_dir * np.cos(tot_dip) - np.array([0.0, 0.0, 1.0]) * np.sin(tot_dip)
        joints[mcp_idx + 2] = joints[mcp_idx + 1] + dir_dip * pip_to_dip
        
        # TIP joint
        tot_tip = tot_dip + tip_angle
        dir_tip = ext_dir * np.cos(tot_tip) - np.array([0.0, 0.0, 1.0]) * np.sin(tot_tip)
        joints[mcp_idx + 3] = joints[mcp_idx + 2] + dir_tip * dip_to_tip

    if gesture_type == "paper":
        # Spread out wider to satisfy config.MIN_PAPER_SPREAD
        idx_dir = np.array([0.40, 0.92, 0.0]);  idx_dir /= np.linalg.norm(idx_dir)
        mid_dir = np.array([0.0, 1.0, 0.0]);   mid_dir /= np.linalg.norm(mid_dir)
        rng_dir = np.array([-0.40, 0.92, 0.0]); rng_dir /= np.linalg.norm(rng_dir)
        pky_dir = np.array([-0.72, 0.69, 0.0]); pky_dir /= np.linalg.norm(pky_dir)

        add_finger(5, idx_dir, 0.0)
        add_finger(9, mid_dir, 0.0)
        add_finger(13, rng_dir, 0.0)
        add_finger(17, pky_dir, 0.0)
        # Thumb out (lateral)
        joints[3] = joints[2] + np.array([0.07, 0.04, -0.01])
        joints[4] = joints[3] + np.array([0.06, 0.02, -0.01])
        
    else:
        # Default natural finger spread for rock, scissors, thumbs_up, and random
        idx_dir = np.array([0.22, 0.97, 0.0]);  idx_dir /= np.linalg.norm(idx_dir)
        mid_dir = np.array([0.0, 1.0, 0.0]);   mid_dir /= np.linalg.norm(mid_dir)
        rng_dir = np.array([-0.22, 0.97, 0.0]); rng_dir /= np.linalg.norm(rng_dir)
        pky_dir = np.array([-0.42, 0.91, 0.0]); pky_dir /= np.linalg.norm(pky_dir)

    if gesture_type == "paper":
        pass
    elif gesture_type == "rock":
        add_finger(5, idx_dir, 0.96)
        add_finger(9, mid_dir, 0.96)
        add_finger(13, rng_dir, 0.96)
        add_finger(17, pky_dir, 0.96)
        # Thumb folded
        joints[3] = joints[2] + np.array([0.02, 0.04, 0.03])
        joints[4] = joints[3] + np.array([-0.01, 0.03, 0.03])
        
    elif gesture_type == "scissors":
        add_finger(5, idx_dir, 0.0)
        add_finger(9, mid_dir, 0.0)
        add_finger(13, rng_dir, 0.96)
        add_finger(17, pky_dir, 0.96)
        # Thumb folded
        joints[3] = joints[2] + np.array([0.02, 0.04, 0.03])
        joints[4] = joints[3] + np.array([-0.01, 0.03, 0.03])
        
    elif gesture_type == "thumbs_up":
        # Fingers curled, thumb pointing UP
        add_finger(5, idx_dir, 0.96)
        add_finger(9, mid_dir, 0.96)
        add_finger(13, rng_dir, 0.96)
        add_finger(17, pky_dir, 0.96)
        # Thumb pointing straight UP (local Y-direction)
        joints[3] = joints[2] + np.array([0.0, 0.07, 0.03])
        joints[4] = joints[3] + np.array([0.0, 0.05, 0.03])
        
    else: # random noise hand
        add_finger(5, idx_dir, random.random())
        add_finger(9, mid_dir, random.random())
        add_finger(13, rng_dir, random.random())
        add_finger(17, pky_dir, random.random())
        joints[3] = joints[2] + np.array([random.uniform(-0.04, 0.06), random.uniform(-0.04, 0.06), random.uniform(-0.03, 0.03)])
        joints[4] = joints[3] + np.array([random.uniform(-0.04, 0.06), random.uniform(-0.04, 0.06), random.uniform(-0.03, 0.03)])

    # Mirror X coordinates if Left hand
    if is_left:
        for i in range(21):
            joints[i][0] = -joints[i][0]

    # Add Gaussian joint noise
    for i in range(21):
        joints[i] += np.random.normal(0, noise_level, 3)

    # Apply rotation, scale, and translation
    transformed = []
    for j in joints:
        rotated = np.dot(rotation_matrix, j)
        final_pt = (rotated * scale) + translation
        transformed.append(final_pt)
        
    return [Landmark(pt[0], pt[1], pt[2]) for pt in transformed]

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.json")
    
    print("[Bootstrapper] Starting hybrid dataset generation...")
    dataset = []
    counts = {"rock": 0, "paper": 0, "scissors": 0, "unknown": 0}
    target_samples = 600

    # 1. Load and Augment Real Rock Features
    real_rock_path = os.path.join(os.path.dirname(__file__), "real_rock_features.json")
    if not os.path.exists(real_rock_path):
        print(f"[Bootstrapper ERROR] Real rock features file not found at {real_rock_path}")
        print("Please run core/extract_real_features.py first.")
        sys.exit(1)

    with open(real_rock_path, "r") as f:
        real_rock_features = json.load(f)

    if len(real_rock_features) == 0:
        print("[Bootstrapper ERROR] Real rock features file is empty.")
        sys.exit(1)

    print(f"[Bootstrapper] Loaded {len(real_rock_features)} real rock samples. Augmenting to {target_samples}...")
    for _ in range(target_samples):
        base_features = random.choice(real_rock_features)
        base_features = np.array(base_features, dtype=np.float32)
        # Add very slight Gaussian noise
        noise = np.random.normal(0, 0.005, size=base_features.shape)
        noisy_features = base_features + noise
        
        dataset.append({
            "label": "rock",
            "features": noisy_features.tolist(),
            "raw_landmarks": []
        })
        counts["rock"] += 1

    # Generators mapping for the remaining synthetic gestures (excl. rock)
    generators = {
        "paper": "paper",
        "scissors": "scissors",
        "thumbs_up": "unknown",
        "random": "unknown"
    }

    for gen_type, target_label in generators.items():
        attempts = 0
        samples_recorded = 0
        
        while samples_recorded < target_samples and attempts < 25000:
            attempts += 1
            
            # Random scaling (representing user moving hand closer/farther)
            scale = random.uniform(0.12, 0.40)
            
            # Random translation (camera frame positioning)
            translation = np.array([
                random.uniform(0.1, 0.9), 
                random.uniform(0.1, 0.9), 
                random.uniform(-0.15, 0.15)
            ])
            
            # Tilt / Rotation: allows 45-deg tilt for main moves, full 360-deg for noise
            max_angle = 45.0 if target_label != "unknown" else 180.0
            rot = random_rotation_matrix(max_angle)
            
            # Alternate handedness
            is_left = (random.random() < 0.5)
            handedness = "Left" if is_left else "Right"
            
            # Create joints
            lm = make_hand_landmarks(gen_type, rot, scale, translation, is_left=is_left)
            
            # Test with heuristic (except for thumbs_up which is directly UNKNOWN)
            if gen_type == "thumbs_up":
                pred = "unknown"
            else:
                res = classify_gesture_heuristic(lm, handedness)
                pred = res["gesture"]
            
            # Accept if matches intended target class
            if pred == target_label:
                features = get_invariant_features(lm)
                dataset.append({
                    "label": target_label,
                    "features": features.tolist(),
                    "raw_landmarks": [[pt.x, pt.y, pt.z] for pt in lm]
                })
                counts[target_label] += 1
                samples_recorded += 1

        print(f"  Generated {samples_recorded} samples for {gen_type.upper()} label target [{target_label.upper()}] (attempts: {attempts})")

    # Save to JSON
    print(f"\n[Bootstrapper] Generation complete. Saving {len(dataset)} samples to {dataset_path}...")
    try:
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print("[Bootstrapper] Save successful.")
    except Exception as e:
        print(f"[Bootstrapper ERROR] Failed to save dataset: {e}")
        return

    # Call train_model.py automatically to compile model_weights.json
    print("\n" + "="*50)
    print(" COMPILING MLP MODEL WEIGHTS...")
    print("="*50)
    import core.train_model
    core.train_model.main()

if __name__ == "__main__":
    main()
