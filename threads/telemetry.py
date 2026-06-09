from collections import deque
import time

class Telemetry:
    def __init__(self, shared_state):
        self.shared = shared_state
        self.capture_ts_ring  = deque(maxlen=60)
        self.infer_ts_ring    = deque(maxlen=60)
        self.render_ts_ring   = deque(maxlen=60)
        self.latency_log      = deque(maxlen=200)

    def record_capture_frame(self, ts):
        self.capture_ts_ring.append(ts)

    def record_render_frame(self, ts):
        self.render_ts_ring.append(ts)

    def record_inference(self, ms):
        self.infer_ts_ring.append(time.monotonic())
        with self.shared.telemetry_lock:
            self.shared.last_latency_ms = ms

    def record_latency(self, ms):
        self.latency_log.append(ms)

    def record_degradation(self, level):
        self.shared.degradation_level = level

    def capture_fps(self):
        return self._fps(self.capture_ts_ring)

    def inference_fps(self):
        return self._fps(self.infer_ts_ring)

    def render_fps(self):
        return self._fps(self.render_ts_ring)

    def _fps(self, ring):
        if len(ring) < 2: return 0.0
        elapsed = ring[-1] - ring[0]
        return (len(ring) - 1) / elapsed if elapsed > 0 else 0.0

    def debug_string(self):
        return (f"CAP:{self.capture_fps():.0f} "
                f"INF:{self.inference_fps():.0f} "
                f"REN:{self.render_fps():.0f} "
                f"LAT:{self.shared.last_latency_ms:.1f}ms "
                f"DEG:{self.shared.degradation_level} "
                f"DROPS:{self.shared.dropped_frames}")
