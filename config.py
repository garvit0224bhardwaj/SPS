import cv2

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON_MIN_VERSION = (3, 11)    # Enforce at startup in main.py

# ── Windows Startup Grace Period ──────────────────────────────────────────────
STARTUP_GRACE_PERIOD_S = 10.0  # Allow threads to initialize before watchdog
# (Windows DShow + MediaPipe init can take 3-5s)

# ── Camera ────────────────────────────────────────────────────────────────────
# Backend ladder: try in order until a working frame is received
CAMERA_BACKENDS = [
    cv2.CAP_DSHOW,   # Most stable on Windows (DirectShow)
    cv2.CAP_MSMF,    # Windows Media Foundation (fallback)
    cv2.CAP_ANY,     # Emergency fallback (OS chooses)
]
CAMERA_INDEX = 0
TARGET_WIDTH = 640      # Lowered from 1280 -> 3x fewer pixels for CPU optimization
TARGET_HEIGHT = 480      # Lowered from 720 -> 3x fewer pixels for CPU optimization
TARGET_FPS = 30       # Lowered from 60 to free CPU resources
CAMERA_FOURCC = 'MJPG'   # MJPEG reduces USB bandwidth, improves FPS

FALLBACK_RESOLUTIONS = [
    (1280, 720,  60),
    (1280, 720,  30),
    (640,  480,  30),
]

CAMERA_WARMUP_FRAMES = 30       # Discard this many frames after open
CAMERA_WARMUP_SLEEP_S = 0.5      # Sleep before warmup reads (exposure settle)
MIN_FRAME_BRIGHTNESS = 10       # frame.mean() — reject black frames
MIRROR_CAMERA = True     # Flip horizontal — feels natural to users

# ── MediaPipe ─────────────────────────────────────────────────────────────────
# 0=fastest, 1=more reliable. 0 chosen for minimum latency.
MODEL_COMPLEXITY = 0
MAX_NUM_HANDS = 1
# Balanced: locks onto the knuckle-profile camera angle reliably
MIN_DETECTION_CONFIDENCE = 0.2
# Balanced: maintains stable tracking without accepting marginal detections
MIN_TRACKING_CONFIDENCE = 0.2

# ── Gesture Geometry ──────────────────────────────────────────────────────────
EXTENSION_THRESHOLD = 0.03    # Dot-product projection for extended finger
DISTANCE_RATIO = 1.28    # Raised from 1.15 -> kills paper bias on blurry/partial hands
THUMB_THRESHOLD = 0.04    # Lateral projection for thumb
# Thumb is VERY extended → thumbs-up → reject as unknown
HIGH_THUMB_THRESHOLD = 0.12
MIN_SCISSORS_SEP = 0.04    # Min index/middle tip separation (normalised)
MIN_PAPER_SPREAD = 0.09    # Raised from 0.06 -> requires convincing spread for paper

# ── MLP Classifier ────────────────────────────────────────────────────────────
MLP_CONFIDENCE_MIN = 0.68   # Below this, fall back to heuristic
MIN_HAND_BBOX_AREA = 8000    # px² — reject tiny/distant hands
MIN_VISIBILITY = 0.60    # Per-landmark visibility threshold

# ── Temporal Stabiliser ───────────────────────────────────────────────────────
# Consecutive frames to confirm gesture change (tuned for 1Euro filter)
HYSTERESIS_FRAMES = 3
# EMA alpha for confidence smoothing (slower decay = less flicker)
CONFIDENCE_SMOOTH_A = 0.20

# ── Gesture Tracker ───────────────────────────────────────────────────────────
BUFFER_SIZE = 10
RECENCY_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 2, 3, 3]
CONFIDENCE_MIN = 0.62    # Minimum weighted agreement to accept

# ── Sliding Window Shoot Capture ─────────────────────────────────────────────
# Grace period before sampling begins (ms) — lets the throw settle
SHOOT_GRACE_PERIOD_MS = 100
# Duration of the sampling window after the grace period (ms)
SHOOT_WINDOW_DURATION_MS = 500

# ── State Machine Timing (ms) ─────────────────────────────────────────────────
HAND_LOST_GRACE_MS = 1500  # Hand loss tolerance during COUNTDOWN
# Extended to 1500ms — brief quality blips should never break a round
WAITING_STABLE_MS = 400   # Time in HAND_DETECTED before → COUNTDOWN

# Countdown: one word per tick
COUNTDOWN_WORDS = ["READY", "ROCK", "PAPER", "SCISSORS"]
COUNTDOWN_TICK_MS = 750   # Each word displayed for this duration
# (750ms x 3 = 2.25s total)

SHOOT_DISPLAY_MS = 200   # "SHOOT!" stays visible this long

RESULT_REVEAL_DELAY_MS = 40    # Tiny pause before showing counter-move
# (feels intelligent; human cannot react in 40ms)
RESULT_DISPLAY_MS = 2500  # How long result stays on screen
SCOREBOARD_MS = 2000

NO_HAND_TIMEOUT_MS = 1200  # Lose hand during COUNTDOWN → reset

# ── Anti-Cheat / Commitment Detector ─────────────────────────────────────────
COMMITMENT_MAX_TRANSITIONS = 2    # More than this many gesture switches → invalid
# Shannon entropy above this → gesture too random
COMMITMENT_ENTROPY_THRESHOLD = 1.20
# Minimum total snapshot frames (including unknowns) before the commitment
# detector will evaluate entropy. Below this → insufficient_data rejection.
# Prevents a single hallucinated frame from auto-passing with len(known) < 3.
MIN_COMMITMENT_FRAMES = 5

# ── Renderer / UI ─────────────────────────────────────────────────────────────
WINDOW_TITLE = "Can You Beat The Machine?"
COUNTDOWN_FONT_SIZE = 140
OVERLAY_ALPHA = 160

SHOW_SKELETON = True  # Always True for event — builds trust
SHOW_GESTURE_LABEL = True
SHOW_HAND_GUIDE = True  # Semi-transparent guide in IDLE + HAND_DETECTED
DEBUG_OVERLAY_DEFAULT = False  # Toggle with F3


# Confidence tier thresholds for skeleton rendering
CONF_TIER_WEAK = 0.35  # Below this: faint outline (20% opacity)
CONF_TIER_STABLE = 0.60  # Above this: bright skeleton (100% opacity)
CONF_TIER_LOCKED = 0.80  # Above this: green glow

# Countdown word icons (displayed alongside each word)
SHOW_COUNTDOWN_ICONS = True  # Show rock/paper/scissors asset next to each word

# ── Audio ─────────────────────────────────────────────────────────────────────
ENABLE_AUDIO = True
AUDIO_FREQUENCY = 44100
AUDIO_BUFFER = 512   # Small buffer = low audio latency
AUDIO_CHANNELS = 2
MASTER_VOLUME = 0.7   # 0.0–1.0

# ── Adaptive Degradation ──────────────────────────────────────────────────────
DEGRADE_THRESHOLD_FPS = 20    # Below this → Level 1
DEGRADE_CRITICAL_FPS = 12    # Below this → Level 2
ROI_MARGIN_FACTOR = 1.30  # ROI crop enlargement
REACQUIRE_EVERY_N = 45    # Frames between global reacquisition

# ── Watchdog Relaxed Timeouts (event-tolerant) ────────────────────────────────
# Seconds (was 0.5) — tolerates MediaPipe stalls
INFERENCE_HEARTBEAT_TIMEOUT = 3.0
# Seconds (was 1.0) — tolerates Windows driver
CAPTURE_HEARTBEAT_TIMEOUT = 5.0

# ── Latency Logging ───────────────────────────────────────────────────────────
LOG_LATENCY = True
LATENCY_WARN_MS = 70

# ── Jitter Logging ────────────────────────────────────────────────────────────
# Palm centre is computed from wrist + 4 MCP joints (normalized 0–1).
# A delta > this value between consecutive frames counts as a jitter spike.
JITTER_WARN_THRESHOLD = 0.04     # ~2–3% of frame width — tune if too noisy

# ── Frame-Drop Logging ────────────────────────────────────────────────────────
# Log a FRAME_DROP event after this many consecutive camera read failures.
FRAME_DROP_WARN_COUNT = 5

# ── Game Session Logging ──────────────────────────────────────────────────────
ENABLE_SESSION_LOGGING = True
# base path; session folders are created underneath logs/
SESSION_LOG_PATH = "logs/game_session.csv"

# ── Inference Thread Rate Cap ──────────────────────────────────────────────────
INFERENCE_TARGET_FPS = 30      # Capped inference rate to free CPU
