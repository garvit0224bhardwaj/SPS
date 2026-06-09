import threading
import time
import cv2
import numpy as np
import mediapipe as mp
import config
from utilities import lm_xyz, dist
from core.classifier import classify_gesture
from core.session_logger import log_event
from core.one_euro_filter import LandmarkOneEuroFilter

class FilteredLandmark:
    __slots__ = ['x', 'y', 'z']
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def log_warning(msg):
    print(f"[Inference WARNING] {msg}")

class InferenceThread(threading.Thread):
    def __init__(self, shared_state, stabilizer, gesture_tracker, telemetry):
        super().__init__(daemon=True)
        self.shared = shared_state
        self.stabilizer = stabilizer
        self.gesture_tracker = gesture_tracker
        self.telemetry = telemetry
        self.running = True
        self.last_heartbeat = time.monotonic()

        self.last_processed_id = -1
        self.scale_factor      = 1.0          # Degradation: 0.5 at Level 1
        self.skip_frames       = False        # Degradation: True at Level 2
        self.frame_counter     = 0
        self._frames_lost      = 0            # Consecutive frames with no hand
        self._reentry_cooldown = 0            # Suppresses spike frames on re-detection

        # Initialize 21 One Euro Filters for landmark smoothing
        self._lm_filters = [LandmarkOneEuroFilter(min_cutoff=0.1, beta=0.5) for _ in range(21)]

        # Pre-allocated BGR/RGB buffers to prevent memory allocation GC overhead
        self._bgr_buf = None
        self._rgb_buf = None

        # Timing for inference rate limiting
        self._next_inf_tick = 0.0
        self._inf_interval  = 1.0 / config.INFERENCE_TARGET_FPS


    def run(self):
        hands = mp.solutions.hands.Hands(
            model_complexity=config.MODEL_COMPLEXITY,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        try:
            self._run_loop(hands)
        finally:
            # Always release MediaPipe handle to avoid Windows DLL leak
            try:
                hands.close()
            except Exception:
                pass

    def _run_loop(self, hands):
        while self.running:
            # ── Inference Rate Limiting (Frame Cap) ──────────────────────────
            now = time.monotonic()
            sleep_s = self._next_inf_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._next_inf_tick = time.monotonic() + self._inf_interval

            # ── Read latest frame ─────────────────────────────────────────────
            with self.shared.frame_lock:
                if self.shared.frame_id == self.last_processed_id or self.shared.latest_frame is None:
                    # Update heartbeat so watchdog doesn't kill us while we wait for camera
                    with self.shared.gesture_lock:
                        self.shared.last_inference_ts = time.monotonic()
                    time.sleep(0.001)
                    continue

                # Pre-allocate and reuse BGR buffer to avoid memory allocation GC pressure
                if self._bgr_buf is None or self._bgr_buf.shape != self.shared.latest_frame.shape:
                    self._bgr_buf = np.empty_like(self.shared.latest_frame)
                np.copyto(self._bgr_buf, self.shared.latest_frame)

                fid      = self.shared.frame_id
                cap_ts   = self.shared.capture_ts
            self.last_processed_id = fid
            frame = self._bgr_buf

            # ── Check for history reset request ──────────────────────────────
            reset_requested = False
            with self.shared.gesture_lock:
                if getattr(self.shared, "reset_gesture_history", False):
                    reset_requested = True
                    self.shared.reset_gesture_history = False

            if reset_requested:
                self.stabilizer.reset()
                self._lm_filters = [LandmarkOneEuroFilter(min_cutoff=0.1, beta=0.5) for _ in range(21)]

            # ── Frame skip (degradation Level 2) ─────────────────────────────
            self.frame_counter += 1
            if self.skip_frames and self.frame_counter % 2 == 0:
                continue

            # ── Adaptive scaling (degradation Level 1) ────────────────────────
            if self.scale_factor < 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * self.scale_factor), int(h * self.scale_factor)))

            # ── Inference ────────────────────────────────────────────────────
            infer_start = time.monotonic()
            if self._rgb_buf is None or self._rgb_buf.shape != frame.shape:
                self._rgb_buf = np.empty_like(frame)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
            if config.MIRROR_CAMERA:
                cv2.flip(self._rgb_buf, 1, dst=self._rgb_buf)
            results     = hands.process(self._rgb_buf)
            infer_ms    = (time.monotonic() - infer_start) * 1000

            self.last_heartbeat = time.monotonic()
            self.telemetry.record_inference(infer_ms)

            with self.shared.gesture_lock:
                self.shared.last_inference_ts = time.monotonic()

            # Measure latency against budget
            total_ms = (time.monotonic() - cap_ts) * 1000
            if config.LOG_LATENCY and total_ms > config.LATENCY_WARN_MS:
                log_warning(f"Latency {total_ms:.1f}ms exceeds {config.LATENCY_WARN_MS}ms budget")
                log_event(
                    "LATENCY_SPIKE",
                    f"Capture-to-inference latency exceeded budget",
                    value=f"{total_ms:.1f}ms",
                    extra_a=f"budget={config.LATENCY_WARN_MS}ms",
                    extra_b=f"deg_level={self.shared.degradation_level}",
                )
            self.telemetry.record_latency(total_ms)


            # ── No hand ───────────────────────────────────────────────────────
            if results.multi_hand_landmarks is None:
                self._frames_lost += 1

                # ── Contour fail-safe (SHOOT phase only) ──────────────────────────
                # MediaPipe sometimes drops a tight Rock fist because the landmark
                # model cannot find individual finger joints.  Before wiping hand
                # state, run a cheap OpenCV blob check on the current frame to see
                # whether a hand-sized high-contrast region is still present.  If
                # it is, we set the fallback flag so the state machine can award
                # Rock rather than treating the round as a miss.
                _contour_fallback_set = False
                with self.shared.gesture_lock:
                    _in_shoot = self.shared.game_phase == "SHOOT"
                if _in_shoot:
                    _contour_fallback_set = self._contour_blob_check(frame)
                    if _contour_fallback_set:
                        with self.shared.gesture_lock:
                            self.shared.fallback_to_rock = True
                        # Do NOT wipe hand state — let state machine resolve this frame
                        continue

                # Wipe stale 1€ filters on hand loss so the first frame back
                # doesn't produce a huge smoothing artifact.
                self._lm_filters = [LandmarkOneEuroFilter(min_cutoff=0.1, beta=0.5) for _ in range(21)]
                with self.shared.gesture_lock:
                    self.shared.hand_detected  = False
                    self.shared.landmark_ref   = None  # Prevent ghost skeleton on renderer
                continue

            lm_data    = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            lm         = lm_data.landmark

            # ── Quality filter ─────────────────────────────────────────────────────
            if not self._landmarks_pass_quality(lm, frame.shape):
                self._frames_lost += 1
                self._lm_filters = [LandmarkOneEuroFilter(min_cutoff=0.1, beta=0.5) for _ in range(21)]
                with self.shared.gesture_lock:
                    self.shared.hand_detected = False
                    self.shared.landmark_ref  = None  # Prevent ghost skeleton on renderer
                continue

            # Apply 1 Euro Filter to landmarks
            now = time.monotonic()
            filtered_lm = []
            for i, point in enumerate(lm):
                fx, fy, fz = self._lm_filters[i].process(now, point.x, point.y, point.z)
                filtered_lm.append(FilteredLandmark(fx, fy, fz))
            lm = filtered_lm  # Replace raw landmarks with filtered ones

            # ── Re-entry cooldown ──────────────────────────────────────────────
            # Hand has passed quality filter — it's genuinely back in frame.
            if self._frames_lost >= 3:
                # Gap of 3+ frames means re-entry. Suppress classification for
                # 2 frames so motion blur from re-entry doesn't corrupt gesture.
                self._reentry_cooldown = 2
            self._frames_lost = 0

            # ── Read current game phase ────────────────────────────────────────
            with self.shared.gesture_lock:
                _game_phase = self.shared.game_phase

            # ── COUNTDOWN fast-path: skeleton display only ─────────────────────
            # Do not classify during countdown — only push skeleton for rendering.
            if _game_phase == "COUNTDOWN":
                with self.shared.gesture_lock:
                    self.shared.hand_detected    = True
                    self.shared.handedness        = handedness
                    self.shared.landmark_ref      = lm
                    self.shared.landmark_frame_id = fid
                continue

            # ── Re-entry cooldown guard ────────────────────────────────────────
            # Push skeleton but skip classification while cooldown is active.
            if self._reentry_cooldown > 0:
                self._reentry_cooldown -= 1
                with self.shared.gesture_lock:
                    self.shared.hand_detected    = True
                    self.shared.handedness        = handedness
                    self.shared.landmark_ref      = lm
                    self.shared.landmark_frame_id = fid
                continue

            # ── Classify and push ──────────────────────────────────────────────
            # All phases except COUNTDOWN: run the full classify pipeline and push
            # the result to shared state.  The state machine reads stable_gesture
            # during the SHOOT window to build its majority-vote buffer.
            raw_result = classify_gesture(lm, handedness)
            stable_g, stable_c = self.stabilizer.update(raw_result)

            with self.shared.gesture_lock:
                self.shared.hand_detected      = True
                self.shared.stable_gesture     = stable_g
                self.shared.stable_confidence  = stable_c
                self.shared.handedness         = handedness
                self.shared.landmark_ref       = lm
                self.shared.landmark_frame_id  = fid

    def _landmarks_pass_quality(self, lm, frame_shape) -> bool:
        # Check 1: Bbox area (using config dimensions so it's immune to degradation scaling)
        xs = [l.x for l in lm]
        ys = [l.y for l in lm]
        h, w = config.TARGET_HEIGHT, config.TARGET_WIDTH
        bbox_area = (max(xs) - min(xs)) * w * (max(ys) - min(ys)) * h
        if bbox_area < config.MIN_HAND_BBOX_AREA:
            return False

        # Check 2: Per-landmark visibility — allow up to 4 partially-visible joints.
        # GUARD: MediaPipe Hands NormalizedLandmark.visibility is always 0.0 in most
        # versions (it is only populated for Pose landmarks, not Hand landmarks).
        # We only apply this filter when MediaPipe is actually providing real data
        # (i.e. at least one landmark has visibility > 0.01).
        vis_values = [l.visibility for l in lm]
        if max(vis_values) > 0.01:
            low_vis = sum(1 for v in vis_values if v < config.MIN_VISIBILITY)
            if low_vis > 4:
                return False

        return True

    def _contour_blob_check(self, frame: np.ndarray) -> bool:
        """Lightweight contour check: returns True if a hand-sized high-contrast
        blob is present in *frame* even though MediaPipe returned no landmarks.

        Pipeline (all ops in-place on small arrays — negligible CPU cost):
          1. Grayscale conversion
          2. 5×5 Gaussian blur  (suppresses sensor noise & fine texture)
          3. Otsu auto-threshold (adapts to ambient lighting without a fixed value)
          4. 3×3 morphological close × 2 iterations (reunites fragmented fist silhouette)
          5. findContours (EXTERNAL only — ignores holes inside the blob)
          6. Area gate: between BLOB_MIN_AREA and BLOB_MAX_AREA

        Returns True as soon as one qualifying contour is found.
        """
        try:
            h, w = frame.shape[:2]
            frame_px = h * w

            # ── 1. Grayscale ────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 2. Gaussian blur ────────────────────────────────────────────
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # ── 3. Otsu threshold ───────────────────────────────────────────
            _, thresh = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # ── 4. Morphological close ──────────────────────────────────────
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # ── 5. Find external contours ───────────────────────────────────
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # ── 6. Area gate ────────────────────────────────────────────────
            # Lower bound: half of the normal quality-filter floor so a partially
            #              occluded fist still qualifies.
            # Upper bound: 60 % of full frame area to reject background blobs.
            blob_min = config.MIN_HAND_BBOX_AREA * 0.5
            blob_max = frame_px * 0.60

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if blob_min <= area <= blob_max:
                    return True  # Found a hand-sized blob — trigger fail-safe

            return False

        except Exception:
            # Never crash the inference loop on a contour error
            return False
