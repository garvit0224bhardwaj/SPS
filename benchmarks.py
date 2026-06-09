"""
benchmarks.py  -  SPS In-App Session Profiler
==============================================
This module attaches to the LIVE main.py process and records REAL performance
data as the game runs.  It does NOT run standalone.

How it works
------------
  - Import SessionProfiler in main.py
  - Call profiler.start() after threads are up
  - Call profiler.stop() before pygame.quit()
  - A background daemon thread samples every SAMPLE_INTERVAL seconds

What is saved
-------------
  benchmarks/
      session_001.json    <- full per-sample time-series + summary
      session_002.json    <- each run gets a new session number
      ...
      benchmark_log.csv   <- one row per session, for easy before/after reports

Metrics recorded (from the LIVE application)
--------------------------------------------
  - Capture FPS         (real camera thread delivery rate)
  - Inference FPS       (real MediaPipe throughput)
  - Render FPS          (real Pygame frame rate)
  - Inference Latency   (ms, live from telemetry ring buffer)
  - CPU Usage %         (system-wide via psutil)
  - RAM (RSS) MB        (this process's resident set)
  - Dropped Frames      (shared_state counter)
  - Degradation Level   (0=full, 1=half-res, 2=skip+ROI)
  - Gesture Counts      (rock / paper / scissors detected)
  - Hand Detection Rate (% of samples where hand was visible)
  - CPU Temperature     (if available)
"""

import csv
import json
import os
import platform
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    print("[Profiler WARNING] psutil not found - CPU/RAM metrics disabled. "
          "Run: pip install psutil")

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_INTERVAL   = 2.0          # seconds between samples
BENCHMARKS_DIR    = "benchmarks" # directory for all output files
CSV_LOG_FILE      = os.path.join(BENCHMARKS_DIR, "benchmark_log.csv")

CSV_FIELDS = [
    "session", "timestamp", "duration_s", "device",
    "cpu_model", "cpu_cores_logical", "cpu_cores_physical",
    "total_ram_mb", "cpu_freq_mhz",
    "capture_fps_mean", "capture_fps_min", "capture_fps_max",
    "inference_fps_mean", "inference_fps_min", "inference_fps_max",
    "render_fps_mean", "render_fps_min", "render_fps_max",
    "latency_mean_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms",
    "cpu_mean_pct", "cpu_max_pct",
    "ram_mean_mb", "ram_peak_mb",
    "cpu_temp_max_c",
    "total_dropped_frames", "degradation_level_mean",
    "hand_detection_rate_pct",
    "gestures_rock", "gestures_paper", "gestures_scissors",
    "total_gestures",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_session_number() -> int:
    """Return the next available session number based on existing files."""
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    existing = [
        f for f in os.listdir(BENCHMARKS_DIR)
        if f.startswith("session_") and f.endswith(".json")
    ]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.replace("session_", "").replace(".json", "")))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def _read_cpu_temp() -> float | None:
    """Read CPU temperature from available platform sources."""
    if sys.platform == "win32":
        try:
            import subprocess
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace "
                 "root/wmi | Select -ExpandProperty CurrentTemperature"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = float(result.stdout.strip().split("\n")[0])
                return (raw / 10.0) - 273.15
        except Exception:
            pass

    if _PSUTIL:
        try:
            temps = psutil.sensors_temperatures()
            for key in ("cpu_thermal", "coretemp", "acpitz", "cpu-thermal"):
                if key in temps and temps[key]:
                    return temps[key][0].current
        except Exception:
            pass

    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path) as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    return None


def _system_info() -> dict:
    info = {
        "os":                  platform.system(),
        "os_version":          platform.version(),
        "machine":             platform.machine(),
        "python_version":      platform.python_version(),
        "cpu_model":           platform.processor() or "unknown",
        "cpu_cores_logical":   os.cpu_count() or 1,
        "cpu_cores_physical":  1,
        "total_ram_mb":        0,
        "cpu_freq_mhz":        None,
    }
    if _PSUTIL:
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or 1
        info["total_ram_mb"]       = round(psutil.virtual_memory().total / 1024**2, 1)
        try:
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"] = round(freq.current, 1)
        except Exception:
            pass

    # Raspberry Pi model (if running on Pi)
    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model") as f:
                info["board_model"] = f.read().strip("\x00")
        except Exception:
            pass

    return info


def _stat(values: list, pct: float) -> float:
    if not values:
        return 0.0
    idx = max(0, int(len(values) * pct) - 1)
    return round(sorted(values)[idx], 2)


# ── SessionProfiler ───────────────────────────────────────────────────────────

class SessionProfiler:
    """
    Attach to the live SPS application and record real performance data.

    Usage in main.py::

        from benchmarks import SessionProfiler
        profiler = SessionProfiler(shared, telemetry)
        profiler.start()
        # ... game loop ...
        profiler.stop()
    """

    def __init__(self, shared_state, telemetry):
        """
        Parameters
        ----------
        shared_state : SharedState
            The live SharedState instance from main.py.
        telemetry : Telemetry
            The live Telemetry instance from main.py.
        """
        self.shared    = shared_state
        self.telemetry = telemetry

        self._running      = False
        self._thread       = None
        self._start_time   = None
        self._session_num  = _next_session_number()

        # Per-sample ring buffers
        self._samples: list[dict] = []

        # Gesture accumulator (tracked across the whole session)
        self._gesture_counts: dict[str, int] = defaultdict(int)
        self._last_gesture   = "unknown"

        # psutil process handle
        self._proc = psutil.Process() if _PSUTIL else None
        if self._proc:
            try:
                self._proc.cpu_percent(interval=None)
            except Exception:
                pass

        print(f"[Profiler] Session {self._session_num:03d} initialised  "
              f"(saves to {BENCHMARKS_DIR}/)")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background sampling thread."""
        if self._running:
            return
        self._running    = True
        self._start_time = time.monotonic()
        self._thread     = threading.Thread(
            target=self._sample_loop,
            name="SessionProfiler",
            daemon=True,
        )
        self._thread.start()
        print(f"[Profiler] Recording started  (sample every {SAMPLE_INTERVAL}s)")

    def stop(self):
        """Stop sampling, compute summary, and save results."""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=SAMPLE_INTERVAL + 1)

        duration_s = time.monotonic() - (self._start_time or time.monotonic())
        self._save(duration_s)

    # ── Sampling loop ─────────────────────────────────────────────────────────

    def _sample_loop(self):
        while self._running:
            try:
                self._take_sample()
            except Exception as ex:
                print(f"[Profiler WARNING] Sample error: {ex}")
            time.sleep(SAMPLE_INTERVAL)

    def _take_sample(self):
        ts = time.monotonic()

        # --- FPS and latency from live telemetry ---
        cap_fps  = self.telemetry.capture_fps()
        inf_fps  = self.telemetry.inference_fps()
        ren_fps  = self.telemetry.render_fps()

        with self.shared.telemetry_lock:
            latency_ms     = self.shared.last_latency_ms
            dropped        = self.shared.dropped_frames
            degradation    = self.shared.degradation_level

        with self.shared.gesture_lock:
            hand_detected  = self.shared.hand_detected
            gesture        = self.shared.stable_gesture
            confidence     = self.shared.stable_confidence

        # --- Gesture counting (only count new gestures, not repeats) ---
        if gesture not in ("unknown", "") and gesture != self._last_gesture:
            self._gesture_counts[gesture] += 1
            self._last_gesture = gesture

        # --- CPU / RAM via psutil ---
        cpu_pct = None
        ram_mb  = None
        if _PSUTIL:
            try:
                cpu_pct = self._proc.cpu_percent(interval=None)
                ram_mb  = self._proc.memory_info().rss / 1024**2
            except Exception:
                pass

        # --- Temperature ---
        temp_c = _read_cpu_temp()

        sample = {
            "ts":            round(ts - self._start_time, 2),
            "capture_fps":   round(cap_fps,   1),
            "inference_fps": round(inf_fps,   1),
            "render_fps":    round(ren_fps,   1),
            "latency_ms":    round(latency_ms, 1),
            "dropped":       dropped,
            "degradation":   degradation,
            "hand_detected": int(hand_detected),
            "gesture":       gesture,
            "confidence":    round(confidence, 3),
            "cpu_pct":       round(cpu_pct, 1) if cpu_pct is not None else None,
            "ram_mb":        round(ram_mb,  1) if ram_mb  is not None else None,
            "temp_c":        round(temp_c,  1) if temp_c  is not None else None,
        }
        self._samples.append(sample)

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self, duration_s: float):
        if not self._samples:
            print("[Profiler] No samples collected - nothing saved.")
            return

        sysinfo  = _system_info()
        summary  = self._compute_summary(duration_s, sysinfo)

        session_data = {
            "session":       self._session_num,
            "timestamp":     datetime.now().isoformat(),
            "duration_s":    round(duration_s, 1),
            "system":        sysinfo,
            "summary":       summary,
            "samples":       self._samples,
            "gesture_counts": dict(self._gesture_counts),
        }

        # ---- JSON ----
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)
        json_path = os.path.join(
            BENCHMARKS_DIR, f"session_{self._session_num:03d}.json"
        )
        with open(json_path, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        # ---- CSV ----
        write_header = not os.path.exists(CSV_LOG_FILE)
        row = {
            "session":                self._session_num,
            "timestamp":              session_data["timestamp"],
            "duration_s":             round(duration_s, 1),
            "device":                 sysinfo.get("board_model", sysinfo.get("cpu_model", "unknown")),
            "cpu_model":              sysinfo.get("cpu_model"),
            "cpu_cores_logical":      sysinfo.get("cpu_cores_logical"),
            "cpu_cores_physical":     sysinfo.get("cpu_cores_physical"),
            "total_ram_mb":           sysinfo.get("total_ram_mb"),
            "cpu_freq_mhz":           sysinfo.get("cpu_freq_mhz"),
            **{k: summary.get(k) for k in CSV_FIELDS if k in summary},
        }
        with open(CSV_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._print_summary(summary, json_path)

    def _compute_summary(self, duration_s: float, sysinfo: dict) -> dict:
        s = self._samples

        def vals(key):
            return [x[key] for x in s if x.get(key) is not None]

        def safe_mean(lst):
            return round(statistics.mean(lst), 2) if lst else None

        def safe_max(lst):
            return round(max(lst), 2) if lst else None

        def safe_min(lst):
            return round(min(lst), 2) if lst else None

        cap   = vals("capture_fps")
        inf   = vals("inference_fps")
        ren   = vals("render_fps")
        lat   = vals("latency_ms")
        cpu   = vals("cpu_pct")
        ram   = vals("ram_mb")
        temp  = vals("temp_c")
        hand  = vals("hand_detected")
        deg   = vals("degradation")
        drops = [x["dropped"] for x in s]

        return {
            # FPS
            "capture_fps_mean":       safe_mean(cap),
            "capture_fps_min":        safe_min(cap),
            "capture_fps_max":        safe_max(cap),
            "inference_fps_mean":     safe_mean(inf),
            "inference_fps_min":      safe_min(inf),
            "inference_fps_max":      safe_max(inf),
            "render_fps_mean":        safe_mean(ren),
            "render_fps_min":         safe_min(ren),
            "render_fps_max":         safe_max(ren),
            # Latency
            "latency_mean_ms":        safe_mean(lat),
            "latency_p95_ms":         _stat(lat, 0.95),
            "latency_p99_ms":         _stat(lat, 0.99),
            "latency_max_ms":         safe_max(lat),
            # CPU / RAM
            "cpu_mean_pct":           safe_mean(cpu),
            "cpu_max_pct":            safe_max(cpu),
            "ram_mean_mb":            safe_mean(ram),
            "ram_peak_mb":            safe_max(ram),
            "cpu_temp_max_c":         safe_max(temp) if temp else "N/A",
            # Quality
            "total_dropped_frames":   max(drops) - min(drops) if len(drops) > 1 else drops[0] if drops else 0,
            "degradation_level_mean": safe_mean(deg),
            "hand_detection_rate_pct": round(statistics.mean(hand) * 100, 1) if hand else 0.0,
            # Gestures
            "gestures_rock":          self._gesture_counts.get("rock", 0),
            "gestures_paper":         self._gesture_counts.get("paper", 0),
            "gestures_scissors":      self._gesture_counts.get("scissors", 0),
            "total_gestures":         sum(self._gesture_counts.values()),
        }

    def _print_summary(self, summary: dict, json_path: str):
        w = 63
        sep = "=" * w
        print(f"\n{sep}")
        print(f"  SESSION {self._session_num:03d} PROFILER REPORT")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(sep)
        print(f"  Samples collected        : {len(self._samples)}")
        print(f"  {'─'*55}")
        print(f"  Capture FPS   mean/min/max: "
              f"{summary['capture_fps_mean']} / {summary['capture_fps_min']} / {summary['capture_fps_max']}")
        print(f"  Inference FPS mean/min/max: "
              f"{summary['inference_fps_mean']} / {summary['inference_fps_min']} / {summary['inference_fps_max']}")
        print(f"  Render FPS    mean/min/max: "
              f"{summary['render_fps_mean']} / {summary['render_fps_min']} / {summary['render_fps_max']}")
        print(f"  {'─'*55}")
        print(f"  Latency mean / p95 / p99  : "
              f"{summary['latency_mean_ms']} / {summary['latency_p95_ms']} / {summary['latency_p99_ms']} ms")
        print(f"  Latency max               : {summary['latency_max_ms']} ms")
        print(f"  {'─'*55}")
        if summary['cpu_mean_pct'] is not None:
            print(f"  CPU mean / peak           : {summary['cpu_mean_pct']} / {summary['cpu_max_pct']} %")
            print(f"  RAM mean / peak           : {summary['ram_mean_mb']} / {summary['ram_peak_mb']} MB")
        print(f"  CPU Temp max              : {summary['cpu_temp_max_c']} C")
        print(f"  {'─'*55}")
        print(f"  Hand detection rate       : {summary['hand_detection_rate_pct']} %")
        print(f"  Dropped frames (delta)    : {summary['total_dropped_frames']}")
        print(f"  Degradation level (mean)  : {summary['degradation_level_mean']}")
        print(f"  {'─'*55}")
        print(f"  Gestures  rock/paper/scissors : "
              f"{summary['gestures_rock']} / {summary['gestures_paper']} / {summary['gestures_scissors']}")
        print(f"  Total gesture detections  : {summary['total_gestures']}")
        print(sep)
        print(f"  JSON  -> {json_path}")
        print(f"  CSV   -> {CSV_LOG_FILE}")
        print(sep)
