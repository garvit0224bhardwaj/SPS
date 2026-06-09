"""
session_logger.py
-----------------
Records two CSV files per session under logs/session_NNN/:

  rounds.csv  — one row per game round (valid or rejected)
  events.csv  — one row per notable system event:
                  JITTER_SPIKE, LATENCY_SPIKE, STATE_TRANSITION,
                  DEGRADATION_CHANGE, INFERENCE_RESTART,
                  CAMERA_ERROR, FRAME_DROP

Sessions are numbered automatically. Each new run of the app
creates the next session folder (session_001, session_002, …).
"""

import os
import csv
import datetime
import threading
import config

# ── Internal state ─────────────────────────────────────────────────────────────
_lock           = threading.RLock()   # RLock allows re-entrant acquisition (init_logger -> _write_event)
_session_dir    = None
_rounds_path    = None
_events_path    = None
_initialized    = False

# ── Column headers ─────────────────────────────────────────────────────────────
_ROUND_HEADERS = [
    "Timestamp",
    "Session",
    "RoundNumber",
    "Status",            # VALID | INVALID
    "Outcome",           # machine_win | player_win | tie | invalid
    "PlayerGesture",
    "ComputerGesture",
    "InvalidReason",
    "PreScoreRobo",
    "PreScorePlayer",
    "PreScoreStreak",
    "PostScoreRobo",
    "PostScorePlayer",
    "PostScoreStreak",
    "PalmVelocity",
    "SettleTimeout",
    "Confidence",
]

_EVENT_HEADERS = [
    "Timestamp",
    "Session",
    "EventType",         # see list above
    "Detail",            # human-readable description
    "Value",             # numeric value if applicable, else N/A
    "ExtraA",            # optional extra column A
    "ExtraB",            # optional extra column B
]

# ── Round counter ───────────────────────────────────────────────────────────────
_round_number = 0


def _next_session_number(log_root: str) -> int:
    """Scan log_root for session_NNN dirs and return the next number."""
    if not os.path.exists(log_root):
        return 1
    existing = [
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d)) and d.startswith("session_")
    ]
    if not existing:
        return 1
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return (max(nums) + 1) if nums else 1


def init_logger():
    """
    Call once at startup.
    Creates logs/session_NNN/ and writes CSV headers.
    Does nothing if ENABLE_SESSION_LOGGING is False.
    """
    global _session_dir, _rounds_path, _events_path, _initialized, _round_number

    if not config.ENABLE_SESSION_LOGGING:
        return

    log_root = os.path.dirname(config.SESSION_LOG_PATH) or "logs"

    with _lock:
        if _initialized:
            return

        os.makedirs(log_root, exist_ok=True)
        session_num = _next_session_number(log_root)
        _session_dir = os.path.join(log_root, f"session_{session_num:03d}")
        os.makedirs(_session_dir, exist_ok=True)

        _rounds_path = os.path.join(_session_dir, "rounds.csv")
        _events_path = os.path.join(_session_dir, "events.csv")
        _round_number = 0

        with open(_rounds_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_ROUND_HEADERS)

        with open(_events_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_EVENT_HEADERS)

        _initialized = True

        # Log the very first event: session started
        _write_event(
            session_num,
            "SESSION_START",
            f"Session {session_num:03d} started",
            value="N/A",
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def log_round(
    status: str,
    outcome: str,
    player_gesture,
    computer_gesture,
    invalid_reason,
    pre_score_robo: int,
    pre_score_player: int,
    pre_score_streak: int,
    post_score_robo: int,
    post_score_player: int,
    post_score_streak: int,
    palm_velocity,
    settle_timeout,
    confidence,
):
    """Append one row to rounds.csv."""
    if not config.ENABLE_SESSION_LOGGING or not _initialized:
        return

    global _round_number
    _round_number += 1
    ts = _now()
    session_num = _session_num_from_dir(_session_dir)

    with _lock:
        with open(_rounds_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ts,
                f"session_{session_num:03d}",
                _round_number,
                status,
                outcome,
                player_gesture  or "N/A",
                computer_gesture or "N/A",
                invalid_reason  or "N/A",
                pre_score_robo,
                pre_score_player,
                pre_score_streak,
                post_score_robo,
                post_score_player,
                post_score_streak,
                f"{palm_velocity:.4f}" if isinstance(palm_velocity, float) else (palm_velocity if palm_velocity is not None else "N/A"),
                "TIMEOUT" if settle_timeout else "SETTLED",
                f"{confidence:.4f}" if confidence is not None else "N/A",
            ])


def log_event(event_type: str, detail: str, value=None, extra_a=None, extra_b=None):
    """
    Append one row to events.csv.

    event_type examples (use these exact strings for easy filtering):
        JITTER_SPIKE        – palm moved too fast between frames
        LATENCY_SPIKE       – capture-to-inference latency exceeded budget
        STATE_TRANSITION    – game state changed (e.g. IDLE → COUNTDOWN)
        DEGRADATION_CHANGE  – watchdog changed resolution/frame-skip level
        INFERENCE_RESTART   – watchdog restarted the inference thread
        CAMERA_ERROR        – camera could not be read / opened
        FRAME_DROP          – capture thread had consecutive read failures
        SESSION_START       – (internal) session opened
    """
    if not config.ENABLE_SESSION_LOGGING or not _initialized:
        return
    session_num = _session_num_from_dir(_session_dir)
    _write_event(session_num, event_type, detail, value, extra_a, extra_b)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _write_event(session_num, event_type, detail, value=None, extra_a=None, extra_b=None):
    with _lock:
        if _events_path is None:
            return
        with open(_events_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                _now(),
                f"session_{session_num:03d}",
                event_type,
                detail,
                value    if value   is not None else "N/A",
                extra_a  if extra_a is not None else "N/A",
                extra_b  if extra_b is not None else "N/A",
            ])


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _session_num_from_dir(d: str) -> int:
    if d is None:
        return 0
    try:
        return int(os.path.basename(d).split("_")[1])
    except (IndexError, ValueError):
        return 0
