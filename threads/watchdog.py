import time
from threads.inference_thread import InferenceThread
import config
from core.session_logger import log_event


def log_warning(msg):
    print(f"[Watchdog WARNING] {msg}")

def log_info(msg):
    print(f"[Watchdog INFO] {msg}")

class Watchdog:
    def __init__(self, shared_state, state_machine, telemetry, stabilizer, gesture_tracker):
        self.shared = shared_state
        self.state_machine = state_machine
        self.telemetry = telemetry
        self.stabilizer = stabilizer
        self.gesture_tracker = gesture_tracker
        
        self.capture_thread = None
        self.inference_thread_ref = None
        self._inference_restart_count = 0
        self._degradation_level = 0

    def register(self, capture_thread, inference_thread_ref):
        self.capture_thread = capture_thread
        self.inference_thread_ref = inference_thread_ref

    def tick(self, now_mono: float):
        # Check inference heartbeat
        with self.shared.gesture_lock:
            last_inf = self.shared.last_inference_ts

        if (now_mono - last_inf) > config.INFERENCE_HEARTBEAT_TIMEOUT:
            log_warning(f"Inference stalled {(now_mono-last_inf)*1000:.0f}ms")
            self._restart_inference_thread()
            self.state_machine.handle_inference_stall()

        # Check capture heartbeat
        if self.capture_thread and (now_mono - self.capture_thread.last_heartbeat) > config.CAPTURE_HEARTBEAT_TIMEOUT:
            log_warning("Capture thread stalled")
            # self._restart_capture_thread() # Optional based on design
            self.state_machine.handle_camera_lost()

        # Check FPS for degradation
        if self.telemetry.inference_fps() > 0:
            if self.telemetry.inference_fps() < config.DEGRADE_CRITICAL_FPS:
                self._apply_degradation(2)
            elif self.telemetry.inference_fps() < config.DEGRADE_THRESHOLD_FPS:
                self._apply_degradation(1)
            elif self.telemetry.inference_fps() > config.DEGRADE_THRESHOLD_FPS + 5 and self._degradation_level > 0:
                self._recover_degradation()

    def _apply_degradation(self, level: int):
        if level <= self._degradation_level: return    # already at this level or worse
        prev = self._degradation_level
        self._degradation_level = level
        inf_thread = self.inference_thread_ref[0]
        if level == 1:
            inf_thread.scale_factor = 0.5
            log_info("Degradation L1: half resolution")
        elif level == 2:
            inf_thread.scale_factor  = 0.5
            inf_thread.skip_frames   = True
            log_info("Degradation L2: half resolution + frame skip")
        self.telemetry.record_degradation(level)
        log_event(
            "DEGRADATION_CHANGE",
            f"Degradation level raised",
            value=f"L{prev}->L{level}",
            extra_a=f"inf_fps={self.telemetry.inference_fps():.1f}",
            extra_b=f"cap_fps={self.telemetry.capture_fps():.1f}",
        )


    def _recover_degradation(self):
        prev = self._degradation_level
        self._degradation_level -= 1
        inf_thread = self.inference_thread_ref[0]
        if self._degradation_level == 0:
            inf_thread.scale_factor = 1.0
            inf_thread.skip_frames  = False
            log_info("Recovered to L0: full resolution")
        elif self._degradation_level == 1:
            inf_thread.skip_frames  = False
            log_info("Recovered to L1: half resolution")
        log_event(
            "DEGRADATION_CHANGE",
            f"Degradation level recovered",
            value=f"L{prev}->L{self._degradation_level}",
            extra_a=f"inf_fps={self.telemetry.inference_fps():.1f}",
            extra_b=f"cap_fps={self.telemetry.capture_fps():.1f}",
        )


    def _restart_inference_thread(self):
        old = self.inference_thread_ref[0]
        old.running = False
        old.join(timeout=1.0)
        
        # We must reset last_inference_ts otherwise we get an infinite restart loop
        with self.shared.gesture_lock:
            self.shared.last_inference_ts = time.monotonic()
            
        new = InferenceThread(self.shared, self.stabilizer, self.gesture_tracker, self.telemetry)
        new.start()
        self.inference_thread_ref[0] = new
        self._inference_restart_count += 1
        log_info(f"Inference thread restarted (#{self._inference_restart_count})")
        log_event(
            "INFERENCE_RESTART",
            f"Inference thread was restarted by watchdog",
            value=f"#{self._inference_restart_count}",
            extra_a=f"inf_fps={self.telemetry.inference_fps():.1f}",
        )

