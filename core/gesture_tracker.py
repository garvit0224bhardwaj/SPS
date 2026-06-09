from collections import deque
import config


class GestureTracker:
    """Push-only rolling buffer. Evaluation logic lives in commitment_detector.py."""

    def __init__(self):
        self.buffer = deque(maxlen=config.BUFFER_SIZE)

    def push(self, gesture: str, confidence: float, timestamp: float):
        """Called by InferenceThread under gesture_lock."""
        self.buffer.append({
            "gesture":    gesture,
            "confidence": confidence,
            "timestamp":  timestamp,
        })

    def get_snapshot(self) -> list:
        """Thread-safe copy for commitment_detector."""
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
