import numpy as np
from core.bootstrap_dataset import make_hand_landmarks

def make_rock_landmarks():
    """Generates a list of 21 Landmark objects representing a rock gesture."""
    rot = np.eye(3, dtype=np.float32)
    scale = 0.25
    translation = np.zeros(3, dtype=np.float32)
    return make_hand_landmarks("rock", rot, scale, translation)

