import cv2
import time
import threading
import config
from core.session_logger import log_event


def log(msg):
    print(f"[Capture] {msg}")

def _apply_resolution(cap, w, h, fps):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # If camera rejected resolution (>10% off), try first fallback
    if actual_w == 0 or abs(actual_w - w) / max(w, 1) > 0.1:
        for (fw, fh, ffps) in config.FALLBACK_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  fw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
            cap.set(cv2.CAP_PROP_FPS,          ffps)
            if abs(cap.get(cv2.CAP_PROP_FRAME_WIDTH) - fw) / max(fw, 1) < 0.1:
                break

def _open_camera():
    for backend in config.CAMERA_BACKENDS:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        if not cap.isOpened():
            log(f"Backend {backend} failed to open")
            continue

        # Force MJPEG (reduces USB bandwidth, improves FPS on consumer cams)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))

        # Attempt resolution + FPS
        _apply_resolution(cap, config.TARGET_WIDTH, config.TARGET_HEIGHT, config.TARGET_FPS)

        # Camera warm-up: sleep first, then discard N frames
        # Without this: first frames may be black/wrong-exposure/wrong-size
        time.sleep(config.CAMERA_WARMUP_SLEEP_S)
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            cap.read()    # discard

        # Verify we get a real, non-black, correctly-sized frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue
        if frame.mean() < config.MIN_FRAME_BRIGHTNESS:
            log("Frame too dark — black screen, trying next backend")
            cap.release()
            continue
        if frame.shape[0] < 100 or frame.shape[1] < 100:
            log("Frame dimensions invalid")
            cap.release()
            continue

        # Log what we actually got (may differ from what we requested)
        actual_w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        log(f"Camera OK: {actual_w}x{actual_h}@{actual_fps}fps via backend {backend}")
        return cap

    # All backends failed
    log("FATAL: All camera backends failed")
    log_event("CAMERA_ERROR", "All camera backends failed — cannot open camera")
    return None


class CaptureThread(threading.Thread):
    def __init__(self, shared_state, telemetry):
        super().__init__(daemon=True)
        self.shared = shared_state
        self.telemetry = telemetry
        self.running = True
        self.last_heartbeat = time.monotonic()

    def run(self):
        cap = _open_camera()
        if cap is None:
            self.shared.camera_error = True
            return

        consecutive_failures = 0

        while self.running:
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures == config.FRAME_DROP_WARN_COUNT:
                    log_event(
                        "FRAME_DROP",
                        f"Camera read failed {consecutive_failures} times consecutively",
                        value=str(consecutive_failures),
                    )
                if consecutive_failures > 30:
                    log("Attempting camera reopen")
                    log_event("CAMERA_ERROR", "Too many consecutive read failures — reopening camera")
                    self.last_heartbeat = time.monotonic() + 10.0 # Grace period during reconnect
                    cap.release()
                    cap = _open_camera()
                    if cap is None:
                        self.shared.camera_error = True
                        return
                    # Fix #7: Clear error flag — camera is back, allow recovery
                    self.shared.camera_error = False
                    consecutive_failures = 0
                time.sleep(0.001)
                continue


            consecutive_failures = 0

            # Mirror before storing (natural feel for users)
            if config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            with self.shared.frame_lock:
                self.shared.latest_frame = frame
                self.shared.frame_id    += 1
                self.shared.capture_ts   = now

            self.last_heartbeat = now
            if self.telemetry:
                self.telemetry.record_capture_frame(now)

        cap.release()
