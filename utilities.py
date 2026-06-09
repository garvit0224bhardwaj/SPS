import numpy as np

def lm_xyz(landmark, idx) -> np.ndarray:
    lm = landmark[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def hand_axis(lm) -> np.ndarray:
    """Wrist(0) → Middle MCP(9). Orientation-agnostic hand 'up' vector."""
    return normalize(lm_xyz(lm, 9) - lm_xyz(lm, 0))

def palm_side_axis(lm) -> np.ndarray:
    """Index MCP(5) → Pinky MCP(17). Lateral axis for thumb detection."""
    return normalize(lm_xyz(lm, 5) - lm_xyz(lm, 17))

def palm_centre(lm) -> np.ndarray:
    """Average of wrist + 4 MCP joints. Used for jitter detection."""
    return np.mean([lm_xyz(lm, i) for i in [0, 5, 9, 13, 17]], axis=0)
