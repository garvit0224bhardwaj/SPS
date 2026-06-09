import config

class TemporalStabilizer:
    def __init__(self, shared_state):
        self.shared = shared_state
        self.stable_gesture    = "unknown"
        self.stable_confidence = 0.0
        self.pending_gesture   = None
        self.pending_count     = 0
        self.ema_confidence    = 0.0

    def reset(self):
        self.stable_gesture    = "unknown"
        self.stable_confidence = 0.0
        self.pending_gesture   = None
        self.pending_count     = 0
        self.ema_confidence    = 0.0

    def update(self, raw_result: dict) -> tuple[str, float]:
        g = raw_result["gesture"]
        c = raw_result["confidence"]

        # Always advance EMA regardless of validity
        self.ema_confidence = config.CONFIDENCE_SMOOTH_A * c + (1 - config.CONFIDENCE_SMOOTH_A) * self.ema_confidence

        if not raw_result["valid"]:
            # One bad frame does not erase confirmed gesture
            return self.stable_gesture, self.ema_confidence * 0.5

        if g == self.stable_gesture:
            self.pending_gesture = None
            self.pending_count   = 0
            self.stable_confidence = self.ema_confidence
            return self.stable_gesture, self.stable_confidence

        # New candidate — need HYSTERESIS_FRAMES consecutive confirmations
        if g == self.pending_gesture:
            self.pending_count += 1
            if self.pending_count >= config.HYSTERESIS_FRAMES:
                self.stable_gesture    = g
                self.stable_confidence = self.ema_confidence
                self.pending_gesture   = None
                self.pending_count     = 0
                
                # Push event to the pipeline
                self.shared.gesture_events.put({
                    "gesture": self.stable_gesture,
                    "confidence": self.stable_confidence
                })
        else:
            self.pending_gesture = g
            self.pending_count   = 1

        return self.stable_gesture, self.stable_confidence
