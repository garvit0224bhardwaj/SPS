"""
calibrate_velocity.py  —  SPS Physical Stop Detection Calibration Tool
=======================================================================

Runs an AUTOMATED sequence of game-like phases to profile palm velocity
across every stage of a real round.  No key presses needed — just follow
the on-screen prompts.

Phase flow (per round):
  1. LOCKING_IN    — hold still for 5 s  (records baseline / tremor noise)
  2. ROCK          — "ROCK" banner (750 ms, same as game)
  3. PAPER         — "PAPER" banner (750 ms)
  4. SCISSORS      — "SCISSORS" banner (750 ms)
  5. SHOOT         — banner + throw window  (records mid-throw velocity)
  6. POST_SHOOT    — hold pose for 3 s     (records settle-down velocity)
  → Repeat for N rounds, then print summary stats and save CSV.

CSV columns recorded every inference frame:
  round, phase, timestamp_s,
  raw_velocity, smoothed_velocity, is_settled,
  hand_detected, frames_since_detected, reentry_spike,
  bbox_area, hand_size_norm

The 'reentry_spike' column is 1 on the first frame the hand re-appears
after being absent for ≥ 3 frames — useful for diagnosing the ghost-lock
problem (hand exits frame then returns with a huge velocity spike).

Usage:
  venv\\Scripts\\python calibrate_velocity.py [--rounds N] [--out logs/vc.csv]
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ── Import from project ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
from utilities import palm_centre, dist, lm_xyz

# ── Constants (override locally so the tool is self-contained) ────────────────
LOCK_IN_DURATION_S   = 5.0      # Seconds of "hold still" before countdown
POST_SHOOT_S         = 3.0      # Seconds to hold pose after throw
COUNTDOWN_TICK_MS    = config.COUNTDOWN_TICK_MS   # 750 ms per word (same as game)
COUNTDOWN_WORDS      = config.COUNTDOWN_WORDS      # ["ROCK", "PAPER", "SCISSORS"]
VELOCITY_THRESHOLD   = config.VELOCITY_THRESHOLD
VELOCITY_WINDOW_SIZE = config.VELOCITY_WINDOW_SIZE
REENTRY_GAP_FRAMES   = 3        # ≥ this many lost frames = "re-entry" event

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_GREEN  = (50,  220, 80)
C_YELLOW = (30,  220, 220)
C_RED    = (50,  50,  230)
C_CYAN   = (220, 220, 50)
C_ORANGE = (40,  160, 255)
C_PURPLE = (200, 60,  200)
C_GREY   = (160, 160, 160)

# ── Phase definitions ─────────────────────────────────────────────────────────
PHASE_LOCKING_IN = "locking_in"
PHASE_ROCK       = "rock"
PHASE_PAPER      = "paper"
PHASE_SCISSORS   = "scissors"
PHASE_SHOOT      = "shoot"
PHASE_POST_SHOOT = "post_shoot"

PHASE_COLOUR = {
    PHASE_LOCKING_IN: C_GREEN,
    PHASE_ROCK:       C_CYAN,
    PHASE_PAPER:      C_CYAN,
    PHASE_SCISSORS:   C_CYAN,
    PHASE_SHOOT:      C_RED,
    PHASE_POST_SHOOT: C_YELLOW,
}


# ─────────────────────────────────────────────────────────────────────────────
# Open camera (replicates main.py backend ladder)
# ─────────────────────────────────────────────────────────────────────────────
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        fourcc = cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[Camera] Opened with backend {backend}")
            return cap
        cap.release()
    raise RuntimeError("Could not open camera with any backend.")


# ─────────────────────────────────────────────────────────────────────────────
# HUD helpers
# ─────────────────────────────────────────────────────────────────────────────
def put_text(img, text, x, y, color=C_WHITE, scale=0.7, thick=2, shadow=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (x+1, y+1), font, scale, C_BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, max_val, color, label):
    """Horizontal progress bar."""
    ratio = min(value / max(max_val, 1e-6), 1.0)
    cv2.rectangle(img, (x, y), (x+w, y+h), C_GREY, 1)
    fill_w = int(w * ratio)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1)
    put_text(img, f"{label}: {value:.4f}", x + w + 8, y + h - 2, C_WHITE, 0.55, 1, False)


def draw_velocity_graph(img, history, x, y, w, h, threshold):
    """Mini rolling graph of smoothed velocity."""
    cv2.rectangle(img, (x, y), (x+w, y+h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), C_GREY, 1)

    vals = list(history)
    if len(vals) < 2:
        return

    max_v = max(max(vals), threshold * 2, 0.05)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i / (len(vals)-1) * w)
        py = y + h - int(v / max_v * h)
        pts.append((px, py))

    for i in range(len(pts)-1):
        col = C_GREEN if vals[i] < threshold else C_RED
        cv2.line(img, pts[i], pts[i+1], col, 2)

    # Threshold line
    ty = y + h - int(threshold / max_v * h)
    cv2.line(img, (x, ty), (x+w, ty), C_YELLOW, 1)
    put_text(img, f"thresh", x+2, ty-3, C_YELLOW, 0.40, 1, False)


# ─────────────────────────────────────────────────────────────────────────────
# Stats summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(rows):
    import statistics
    print("\n" + "="*65)
    print("  CALIBRATION SUMMARY")
    print("="*65)

    phases_order = [PHASE_LOCKING_IN, PHASE_ROCK, PHASE_PAPER,
                    PHASE_SCISSORS, PHASE_SHOOT, PHASE_POST_SHOOT]

    all_phases = set(r["phase"] for r in rows)
    for ph in phases_order:
        if ph not in all_phases:
            continue
        vels = [float(r["smoothed_velocity"]) for r in rows if r["phase"] == ph]
        raw  = [float(r["raw_velocity"])       for r in rows if r["phase"] == ph]
        if not vels:
            continue
        print(f"\n  Phase: {ph.upper()}")
        print(f"    Frames recorded : {len(vels)}")
        print(f"    Smoothed vel  mean:{np.mean(vels):.4f}  "
              f"median:{np.median(vels):.4f}  "
              f"P95:{np.percentile(vels,95):.4f}  "
              f"max:{max(vels):.4f}")
        print(f"    Raw vel       mean:{np.mean(raw):.4f}  "
              f"median:{np.median(raw):.4f}  "
              f"P95:{np.percentile(raw,95):.4f}  "
              f"max:{max(raw):.4f}")

    # Re-entry spike analysis
    spikes = [float(r["raw_velocity"]) for r in rows if int(r["reentry_spike"]) == 1]
    gaps   = [int(r["frames_since_detected"]) for r in rows
              if int(r["reentry_spike"]) == 1]
    if spikes:
        print(f"\n  Re-Entry Spikes ({len(spikes)} events):")
        print(f"    Velocity at re-entry -- mean:{np.mean(spikes):.4f}  "
              f"max:{max(spikes):.4f}  P95:{np.percentile(spikes,95):.4f}")
        print(f"    Gap length (frames)  -- mean:{np.mean(gaps):.1f}  "
              f"max:{max(gaps)}")
        print(f"\n  [!] These spikes are the ghost-lock cause.")
        print(f"     Consider ignoring the first ~{int(np.mean(gaps))+2} frames")
        print(f"     after hand re-detection.")
    else:
        print("\n  No re-entry events recorded (hand stayed in frame).")

    print(f"\n  Suggested VELOCITY_THRESHOLD = "
          f"just above P95 of locking_in smoothed vel")
    lock_p95 = np.percentile(
        [float(r["smoothed_velocity"]) for r in rows if r["phase"] == PHASE_LOCKING_IN],
        95
    ) if any(r["phase"] == PHASE_LOCKING_IN for r in rows) else "N/A"
    shoot_mean = np.mean(
        [float(r["smoothed_velocity"]) for r in rows if r["phase"] == PHASE_SHOOT]
    ) if any(r["phase"] == PHASE_SHOOT for r in rows) else "N/A"

    if lock_p95 != "N/A":
        print(f"    locking_in P95 smoothed = {lock_p95:.4f}")
        print(f"    -> Recommended VELOCITY_THRESHOLD ~= {lock_p95 * 1.2:.4f}")
    if shoot_mean != "N/A":
        print(f"    shoot mean smoothed     = {shoot_mean:.4f}")
        print(f"    -> Recommended MIN_THROW_VELOCITY ~= {shoot_mean * 0.5:.4f}")

    print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration loop
# ─────────────────────────────────────────────────────────────────────────────
def run(num_rounds: int, out_path: str):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    cap = open_camera()
    hands_model = mp.solutions.hands.Hands(
        model_complexity=config.MODEL_COMPLEXITY,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    fieldnames = [
        "round", "phase", "timestamp_s",
        "raw_velocity", "smoothed_velocity", "is_settled",
        "hand_detected", "frames_since_detected", "reentry_spike",
        "bbox_area", "hand_size_norm",
    ]

    csv_file = open(out_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # State
    velocity_history    = deque(maxlen=VELOCITY_WINDOW_SIZE)
    graph_history       = deque(maxlen=120)  # ~2s at 60fps for the graph
    palm_prev           = None
    frames_since_det    = 0   # consecutive frames with NO hand
    reentry_countdown   = 0   # counts down from 3 after re-entry so spike frames are flagged
    last_gap_length     = 0   # gap length that caused the current re-entry window
    all_rows            = []

    current_round  = 1
    start_ts       = time.monotonic()

    # ── Phase sequencer ───────────────────────────────────────────────────────
    # Each phase is (label, duration_s).  Shoot has no fixed duration — it
    # ends when we transition to POST_SHOOT, which we do automatically 750ms
    # after SHOOT starts (mirrors SHOOT_DISPLAY_MS logic) plus we let user
    # throw and settle.  POST_SHOOT then runs for POST_SHOOT_S.
    TICK_S = COUNTDOWN_TICK_MS / 1000.0

    def make_schedule():
        """Build list of (phase_label, duration_s) for one round."""
        return [
            (PHASE_LOCKING_IN, LOCK_IN_DURATION_S),
            (PHASE_ROCK,       TICK_S),
            (PHASE_PAPER,      TICK_S),
            (PHASE_SCISSORS,   TICK_S),
            (PHASE_SHOOT,      TICK_S + 0.2),  # brief, then post-shoot begins
            (PHASE_POST_SHOOT, POST_SHOOT_S),
        ]

    schedule      = make_schedule()
    phase_idx     = 0
    phase_start   = time.monotonic()
    current_phase = schedule[0][0]

    cv2.namedWindow("Velocity Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Velocity Calibration", 1000, 600)

    rounds_desc = f"{num_rounds}" if num_rounds > 0 else "infinite"
    print(f"\n[Calibrate] Starting {rounds_desc} round(s). Writing to: {out_path}")
    print("[Calibrate] Press Q or ESC to stop, save data, and view statistics.\n")

    run_start = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        now_s = time.monotonic()

        # Mirror (natural feel)
        if config.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        # ── MediaPipe inference ───────────────────────────────────────────────
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results     = hands_model.process(rgb)
        hand_det    = results.multi_hand_landmarks is not None
        reentry     = 0
        raw_vel     = 0.0
        bbox_area   = 0.0
        hand_sz_n   = 0.0
        smoothed    = 0.0

        if hand_det:
            if frames_since_det >= REENTRY_GAP_FRAMES:
                # Hand just came back after a gap — start a 3-frame spike window.
                # Frame 0: palm_prev=None → vel=0 (no diff yet)
                # Frame 1: first real position diff → the actual spike lands here
                # Frame 2: still elevated from smoothing window
                reentry_countdown = 3
                last_gap_length   = frames_since_det
                palm_prev = None      # Reset prev so velocity is 0 on re-entry frame
                velocity_history.clear()

            if reentry_countdown > 0:
                reentry = 1
                reentry_countdown -= 1
            else:
                reentry = 0

            frames_since_det = 0

            lm_data = results.multi_hand_landmarks[0]
            lm      = lm_data.landmark

            # Palm position + hand size (same as inference_thread)
            palm_now  = palm_centre(lm)
            hand_sz_n = float(max(dist(lm_xyz(lm, 0), lm_xyz(lm, 9)), 1e-6))

            # 2D velocity, normalized by hand size
            if palm_prev is not None:
                raw_vel = float(np.linalg.norm(palm_now[:2] - palm_prev[:2])) / hand_sz_n
            palm_prev = palm_now

            velocity_history.append(raw_vel)
            smoothed = float(np.mean(velocity_history)) if velocity_history else 0.0
            graph_history.append(smoothed)

            # Bounding box area (in canonical resolution pixels)
            xs = [l.x for l in lm]; ys = [l.y for l in lm]
            h_px, w_px = config.TARGET_HEIGHT, config.TARGET_WIDTH
            bbox_area = (max(xs)-min(xs)) * w_px * (max(ys)-min(ys)) * h_px

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm_data, mp.solutions.hands.HAND_CONNECTIONS)

        else:
            frames_since_det += 1
            palm_prev = None
            velocity_history.clear()
            graph_history.append(0.0)

        is_settled = (smoothed < VELOCITY_THRESHOLD) and hand_det

        # ── Write CSV row ─────────────────────────────────────────────────────
        row = {
            "round":                current_round,
            "phase":                current_phase,
            "timestamp_s":          f"{now_s - run_start:.4f}",
            "raw_velocity":         f"{raw_vel:.6f}",
            "smoothed_velocity":    f"{smoothed:.6f}",
            "is_settled":           int(is_settled),
            "hand_detected":        int(hand_det),
            # If we're in a re-entry window, record the gap that caused it;
            # otherwise record current consecutive-miss count (0 when hand present).
            "frames_since_detected": last_gap_length if reentry else (frames_since_det if not hand_det else 0),
            "reentry_spike":        reentry,
            "bbox_area":            f"{bbox_area:.1f}",
            "hand_size_norm":       f"{hand_sz_n:.6f}",
        }
        writer.writerow(row)
        all_rows.append(row)

        # ── Phase timer ───────────────────────────────────────────────────────
        phase_elapsed = now_s - phase_start
        _, phase_dur  = schedule[phase_idx]
        phase_remain  = max(0.0, phase_dur - phase_elapsed)

        if phase_elapsed >= phase_dur:
            phase_idx += 1
            if phase_idx >= len(schedule):
                # Round complete
                if num_rounds > 0 and current_round >= num_rounds:
                    break   # All rounds done
                current_round += 1
                schedule   = make_schedule()
                phase_idx  = 0
            phase_start   = now_s
            current_phase = schedule[phase_idx][0]

        # ── Draw HUD ──────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent left panel
        cv2.rectangle(overlay, (0, 0), (340, h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Phase banner (top center)
        ph_col   = PHASE_COLOUR.get(current_phase, C_WHITE)
        banner   = current_phase.upper().replace("_", " ")
        if current_phase in (PHASE_ROCK, PHASE_PAPER, PHASE_SCISSORS):
            banner = current_phase.upper()
        elif current_phase == PHASE_SHOOT:
            banner = "SHOOT!"
        elif current_phase == PHASE_POST_SHOOT:
            banner = "HOLD POSE"
        elif current_phase == PHASE_LOCKING_IN:
            banner = "LOCKING IN"

        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        bx = (w - bw) // 2
        by = 100
        # Shadow + main text
        cv2.putText(frame, banner, (bx+3, by+3), cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_BLACK, 8, cv2.LINE_AA)
        cv2.putText(frame, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 2.2, ph_col, 5, cv2.LINE_AA)

        # Round counter
        if num_rounds > 0:
            put_text(frame, f"Round {current_round} / {num_rounds}", w//2 - 80, 130, C_GREY, 0.65, 1)
        else:
            put_text(frame, f"Round {current_round}", w//2 - 40, 130, C_GREY, 0.65, 1)

        # Phase timer bar (center top)
        timer_w = 400
        timer_x = (w - timer_w) // 2
        ratio_done = min(phase_elapsed / max(phase_dur, 1e-6), 1.0)
        cv2.rectangle(frame, (timer_x, 145), (timer_x+timer_w, 158), (50,50,50), -1)
        cv2.rectangle(frame, (timer_x, 145), (timer_x + int(timer_w*ratio_done), 158), ph_col, -1)
        put_text(frame, f"{phase_remain:.1f}s", timer_x + timer_w + 6, 157, C_GREY, 0.5, 1, False)

        # ── Left panel metrics ────────────────────────────────────────────────
        px, py = 8, 40

        put_text(frame, "VELOCITY CALIBRATION", px, py, C_WHITE, 0.65, 2)
        py += 28
        cv2.line(frame, (px, py), (330, py), C_GREY, 1)
        py += 18

        det_col = C_GREEN if hand_det else C_RED
        put_text(frame, f"Hand: {'DETECTED' if hand_det else 'MISSING'}", px, py, det_col, 0.62, 2)
        py += 24

        if not hand_det and frames_since_det > 0:
            put_text(frame, f"  Lost {frames_since_det} frames ago", px, py, C_RED, 0.55, 1)
            py += 20

        if reentry:
            put_text(frame, "  !! RE-ENTRY SPIKE !!", px, py, C_ORANGE, 0.62, 2)
            py += 22

        py += 5
        put_text(frame, f"Raw vel:     {raw_vel:.4f}", px, py, C_WHITE, 0.58, 1)
        py += 22
        put_text(frame, f"Smoothed:    {smoothed:.4f}", px, py, C_WHITE, 0.58, 1)
        py += 22

        s_col = C_GREEN if is_settled else C_RED
        put_text(frame, f"Settled: {'YES' if is_settled else 'NO '}", px, py, s_col, 0.65, 2)
        py += 26

        # Velocity bars
        draw_bar(frame, px, py, 200, 12, raw_vel, 0.2, C_CYAN, "raw")
        py += 22
        draw_bar(frame, px, py, 200, 12, smoothed, 0.2, C_ORANGE, "smt")
        py += 28

        cv2.line(frame, (px, py), (330, py), C_GREY, 1); py += 12

        put_text(frame, f"BBox area:  {bbox_area:.0f} px²", px, py, C_GREY, 0.52, 1)
        py += 20
        put_text(frame, f"Hand size:  {hand_sz_n:.4f}", px, py, C_GREY, 0.52, 1)
        py += 20
        put_text(frame, f"Threshold:  {VELOCITY_THRESHOLD:.4f}", px, py, C_YELLOW, 0.52, 1)
        py += 20
        put_text(frame, f"Win size:   {VELOCITY_WINDOW_SIZE}", px, py, C_GREY, 0.52, 1)
        py += 20
        put_text(frame, f"Rows logged: {len(all_rows)}", px, py, C_GREY, 0.52, 1)
        py += 20

        cv2.line(frame, (px, py), (330, py), C_GREY, 1); py += 12
        put_text(frame, "Press Q / ESC to quit", px, py, C_GREY, 0.48, 1)

        # ── Velocity graph (bottom right) ────────────────────────────────────
        g_w, g_h = 380, 100
        g_x = w - g_w - 12
        g_y = h - g_h - 40
        draw_velocity_graph(frame, graph_history, g_x, g_y, g_w, g_h, VELOCITY_THRESHOLD)
        put_text(frame, "Smoothed vel (rolling 2s)", g_x, g_y - 8, C_GREY, 0.48, 1, False)

        cv2.imshow("Velocity Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            print("[Calibrate] Quit early by user.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    hands_model.close()
    cv2.destroyAllWindows()
    csv_file.flush()
    csv_file.close()

    print(f"\n[Calibrate] CSV saved to: {out_path}  ({len(all_rows)} rows)")
    if all_rows:
        print_summary(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPS Velocity Calibration — automated phase profiler"
    )
    parser.add_argument(
        "--rounds", type=int, default=-1,
        help="Number of rounds to run (default: -1 for infinite)"
    )
    parser.add_argument(
        "--out", type=str, default="logs/velocity_calibration.csv",
        help="Output CSV path (default: logs/velocity_calibration.csv)"
    )
    args = parser.parse_args()
    run(args.rounds, args.out)
