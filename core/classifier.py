import numpy as np
import json
import os
from utilities import hand_axis, palm_side_axis, lm_xyz, dist, dot
import config

# ── MLP Model Setup ──────────────────────────────────────────────────────────
_weights = None
_last_weights_mtime = 0.0
weights_path = os.path.join(os.path.dirname(__file__), "model_weights.json")

def load_weights():
    """Dynamically loads or reloads the MLP weights if the JSON file is updated."""
    global _weights, _last_weights_mtime
    if os.path.exists(weights_path):
        try:
            mtime = os.path.getmtime(weights_path)
            if _weights is None or mtime > _last_weights_mtime:
                with open(weights_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct weight matrices and biases as numpy arrays
                _weights = {
                    "W1": np.array(data["W1"], dtype=np.float32),
                    "b1": np.array(data["b1"], dtype=np.float32),
                    "W2": np.array(data["W2"], dtype=np.float32),
                    "b2": np.array(data["b2"], dtype=np.float32),
                    "classes": data["classes"]
                }
                # Support 3-layer MLP if W3 is present in the export
                if "W3" in data:
                    _weights["W3"] = np.array(data["W3"], dtype=np.float32)
                    _weights["b3"] = np.array(data["b3"], dtype=np.float32)
                
                _last_weights_mtime = mtime
                print(f"[Classifier] Loaded/updated MLP weights from model_weights.json (classes: {_weights['classes']})")
        except Exception as e:
            print(f"[Classifier WARNING] Failed to load MLP weights: {e}")

# Initial load attempt
load_weights()


def get_invariant_features(lm) -> np.ndarray:
    """
    Transforms raw landmarks into a 60-dimensional local hand coordinate frame
    invariant to scale, translation, and rotation.

    Uses robust Gram-Schmidt orthogonalization with an explicit sign-lock on the
    X-axis so the frame never flips when the hand faces the camera (flat Paper
    pose) or rotates past the camera plane (gimbal lock condition).
    """
    # Convert landmarks to numpy arrays
    pts = [np.array([l.x, l.y, l.z], dtype=np.float32) if hasattr(l, 'x')
           else np.array(l[:3], dtype=np.float32) for l in lm]

    wrist      = pts[0]
    middle_mcp = pts[9]
    index_mcp  = pts[5]
    pinky_mcp  = pts[17]
    
    # ── Step 1: Scale normalization ────────────────────────────────────────
    hand_size = np.linalg.norm(middle_mcp - wrist)
    if hand_size < 1e-6:
        hand_size = 1e-6

    # ── Step 2: Y-axis ─────────────────────────────────────────────────────
    # Wrist to middle MCP: hand length direction.
    y_axis = (middle_mcp - wrist) / hand_size

    # ── Step 3: X-axis with Gram-Schmidt + sign lock ────────────────────────
    # Candidate: index MCP to pinky MCP (hand width direction).
    v_x = index_mcp - pinky_mcp
    # Gram-Schmidt: project out the Y component so X is strictly perpendicular to Y.
    x_candidate = v_x - np.dot(v_x, y_axis) * y_axis
    x_norm = np.linalg.norm(x_candidate)
    if x_norm > 1e-6:
        x_axis = x_candidate / x_norm
    else:
        # Degenerate: index->pinky nearly parallel to wrist->middle (extreme side view).
        # Use world [0,1,0] projected out of Y as a secondary reference.
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_fb = fallback - np.dot(fallback, y_axis) * y_axis
        x_fbn = np.linalg.norm(x_fb)
        x_axis = x_fb / x_fbn if x_fbn > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Sign lock: ensure X consistently points toward world +X.
    # Without this a small hand rotation can flip dot(x_axis, [1,0,0]) negative,
    # inverting the 60-D feature vector and feeding garbage into the MLP.
    if np.dot(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32)) < 0.0:
        x_axis = -x_axis

    # ── Step 4: Z-axis = cross(y, x) ───────────────────────────────────────
    # cross(y, x) — not cross(x, y) — gives Z pointing out of the palm
    # consistently for both left and right hands.
    z_axis = np.cross(y_axis, x_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        # Fully degenerate (y_axis approx x_axis) — fallback to world Z.
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # ── Step 5: Final re-orthogonalization pass ─────────────────────────────
    # Remove any residual drift accumulated through sign-lock and cross products.
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    xn = np.linalg.norm(x_axis)
    x_axis = x_axis / xn if xn > 1e-6 else x_axis
    z_axis = np.cross(y_axis, x_axis)   # recompute from clean X and Y
    zn = np.linalg.norm(z_axis)
    z_axis = z_axis / zn if zn > 1e-6 else z_axis

    # ── Step 6: Project joints 1–20 into the local frame ─────────────────────
    # Wrist (joint 0) is always the local origin — not projected.
    features = []
    for i in range(1, 21):
        rel_pos = pts[i] - wrist
        px = np.dot(rel_pos, x_axis) / hand_size
        py = np.dot(rel_pos, y_axis) / hand_size
        pz = np.dot(rel_pos, z_axis) / hand_size
        features.extend([px, py, pz])

    return np.array(features, dtype=np.float32)


def mlp_predict(features) -> tuple[str, float]:
    """Runs a raw numpy-based feedforward pass on the loaded MLP model weights."""
    if _weights is None:
        return "unknown", 0.0
    
    # Reshape input to (1, 60)
    x = features.reshape(1, -1)
    
    # Layer 1
    h1 = np.dot(x, _weights["W1"]) + _weights["b1"]
    h1 = np.maximum(0, h1)  # ReLU
    
    # Layer 2 / Output
    if "W3" in _weights:
        h2 = np.dot(h1, _weights["W2"]) + _weights["b2"]
        h2 = np.maximum(0, h2)  # ReLU
        logits = np.dot(h2, _weights["W3"]) + _weights["b3"]
    else:
        logits = np.dot(h1, _weights["W2"]) + _weights["b2"]
        
    # Softmax
    logits = logits.flatten()
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    pred_class = _weights["classes"][idx]
    
    return pred_class, confidence


def classify_gesture_heuristic(lm, handedness="Right") -> dict:
    """Original rule-based hand gesture classifier used as a robust fallback."""
    TIPS  = [8,  12, 16, 20]
    PIPS  = [6,  10, 14, 18]
    MCPS  = [5,  9,  13, 17]

    axis      = hand_axis(lm)
    side_axis = palm_side_axis(lm)
    hand_size = max(dist(lm_xyz(lm, 0), lm_xyz(lm, 9)), 1e-6)

    extended = [False] * 4
    for i in range(4):
        tip = lm_xyz(lm, TIPS[i])
        pip = lm_xyz(lm, PIPS[i])
        mcp = lm_xyz(lm, MCPS[i])

        projection    = dot(tip - pip, axis) / hand_size
        tip_mcp_dist  = dist(tip, mcp)
        pip_mcp_dist  = dist(pip, mcp)

        cond_proj = projection > (config.EXTENSION_THRESHOLD / 0.2)
        cond_dist = tip_mcp_dist > (pip_mcp_dist * config.DISTANCE_RATIO)
        extended[i] = cond_proj and cond_dist

    thumb_tip = lm_xyz(lm, 4)
    thumb_ip  = lm_xyz(lm, 3)
    hand_side_axis = side_axis if handedness == "Right" else -side_axis
    thumb_lateral = abs(dot(thumb_tip - thumb_ip, hand_side_axis)) / hand_size
    thumb_extended = thumb_lateral > (config.THUMB_THRESHOLD / 0.2)

    n_ext      = sum(extended)
    index_up   = extended[0]
    middle_up  = extended[1]
    ring_up    = extended[2]
    pinky_up   = extended[3]

    if n_ext == 0:
        high_thumb = thumb_lateral > (config.HIGH_THUMB_THRESHOLD / 0.2)
        if high_thumb:
            return {"gesture": "unknown", "confidence": 0.30, "valid": False,
                    "extended_fingers": [False, False, False, False], "thumb_extended": True,
                    "handedness": handedness, "reason": "thumbs_up"}

        curls = []
        for i in range(4):
            val = dot(lm_xyz(lm, TIPS[i]) - lm_xyz(lm, PIPS[i]), axis) / hand_size
            curls.append(1.0 - max(0.0, min(val * 3.0, 1.0)))
        avg_curl = sum(curls) / len(curls)
        confidence = 0.70 + 0.30 * avg_curl
        return {"gesture": "rock", "confidence": confidence, "valid": True,
                "extended_fingers": [False, False, False, False], "thumb_extended": thumb_extended,
                "handedness": handedness, "reason": ""}

    elif n_ext == 1 and (ring_up or pinky_up):
        return {"gesture": "rock", "confidence": 0.55, "valid": True,
                "extended_fingers": extended, "thumb_extended": thumb_extended,
                "handedness": handedness, "reason": "edge_rock"}

    elif index_up and middle_up and not ring_up and not pinky_up:
        index_tip  = lm_xyz(lm, 8)
        middle_tip = lm_xyz(lm, 12)
        tip_sep    = dist(index_tip, middle_tip) / hand_size

        scaled_min_sep = config.MIN_SCISSORS_SEP / 0.2
        if tip_sep < scaled_min_sep:
            return {"gesture": "unknown", "confidence": 0.40, "valid": False,
                    "reason": "scissors_tips_too_close"}

        confidence = 0.65 + 0.35 * min(tip_sep / (scaled_min_sep * 2), 1.0)
        return {"gesture": "scissors", "confidence": confidence, "valid": True,
                "extended_fingers": extended, "thumb_extended": thumb_extended,
                "handedness": handedness, "reason": ""}

    elif n_ext >= 3:
        ext_tips = [lm_xyz(lm, TIPS[i]) for i in range(4) if extended[i]]
        spread = 0.0
        if len(ext_tips) >= 2:
            spread = sum(dist(ext_tips[i], ext_tips[i+1]) for i in range(len(ext_tips)-1)) / (len(ext_tips)-1)
            spread = spread / hand_size

        scaled_min_spread = config.MIN_PAPER_SPREAD / 0.2
        if spread < scaled_min_spread:
            return {"gesture": "unknown", "confidence": 0.45, "valid": False,
                    "reason": "paper_spread_too_low"}

        confidence = 0.60 + 0.40 * (n_ext / 4.0)
        return {"gesture": "paper", "confidence": confidence, "valid": True,
                "extended_fingers": extended, "thumb_extended": thumb_extended,
                "handedness": handedness, "reason": ""}

    else:
        return {"gesture": "unknown", "confidence": 0.0, "valid": False,
                "extended_fingers": extended, "thumb_extended": thumb_extended,
                "handedness": handedness, "reason": "ambiguous"}


def classify_gesture(lm, handedness="Right") -> dict:
    """Master gesture classification function attempting MLP prediction before falling back to rules."""
    load_weights()
    if _weights is not None:
        try:
            features = get_invariant_features(lm)
            pred_class, confidence = mlp_predict(features)
            if confidence >= config.MLP_CONFIDENCE_MIN:
                return {
                    "gesture": pred_class,
                    "confidence": confidence,
                    "valid": (pred_class != "unknown"),
                    "extended_fingers": [False] * 4,
                    "thumb_extended": False,
                    "handedness": handedness,
                    "reason": "" if pred_class != "unknown" else "mlp_unknown"
                }
            # Low confidence -> fall through to heuristic
        except Exception as e:
            print(f"[Classifier WARNING] MLP prediction failed: {e}. Falling back to heuristic.")

    return classify_gesture_heuristic(lm, handedness)
