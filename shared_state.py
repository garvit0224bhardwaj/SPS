import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SharedState:
    # Frame (frame_lock) ───────────────────────────────────────────────────────
    frame_lock:         threading.Lock = field(default_factory=threading.Lock)
    latest_frame:       Optional[np.ndarray] = None
    frame_id:           int   = 0
    capture_ts:         float = 0.0

    # Gesture (gesture_lock) ───────────────────────────────────────────────────
    gesture_lock:       threading.Lock = field(default_factory=threading.Lock)
    gesture_events:     queue.Queue = field(default_factory=queue.Queue)
    hand_detected:      bool  = False
    stable_gesture:     str   = "unknown"
    stable_confidence:  float = 0.0
    handedness:         str   = "Right"
    landmark_ref:       Optional[list] = None
    landmark_frame_id:  int   = -1
    last_inference_ts:  float = 0.0
    reset_gesture_history: bool = False
    fallback_to_rock:   bool  = False   # Set by contour fail-safe when MP drops Rock
    game_phase:         str   = "IDLE"  # Written by state machine; read by inference thread

    # Telemetry (telemetry_lock) ───────────────────────────────────────────────
    telemetry_lock:     threading.Lock = field(default_factory=threading.Lock)
    capture_fps:        float = 0.0
    inference_fps:      float = 0.0
    render_fps:         float = 0.0
    last_latency_ms:    float = 0.0
    dropped_frames:     int   = 0

    # System ───────────────────────────────────────────────────────────────────
    degradation_level:  int   = 0      # 0=full, 1=half-res, 2=skip+ROI
    camera_error:       bool  = False

shared = SharedState()
