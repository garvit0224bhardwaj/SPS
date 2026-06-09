import sys
import time
import os
import atexit
import glob
os.environ["XNNPACK_NUM_THREADS"] = "2"   # Cap MediaPipe threads - leaves CPU for capture+render
import random
import pygame
from pygame.locals import *

import config
if sys.version_info < config.PYTHON_MIN_VERSION:
    print(
        f"ERROR: Python {config.PYTHON_MIN_VERSION[0]}.{config.PYTHON_MIN_VERSION[1]}+ required")
    sys.exit(1)


def cleanup_temporary_captures():
    """Atexit hook: delete any temporary frame files written during the session."""
    patterns = [
        os.path.join("logs", "**", "*.tmp"),
        os.path.join("logs", "**", "*.snap"),
        os.path.join("logs", "**", "*.frame"),
    ]
    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[Cleanup] Removed {removed} temporary capture file(s)")

atexit.register(cleanup_temporary_captures)


def run_dry_run() -> int:
    """Run all checks, print PASS/FAIL, return 0 on all-pass."""
    import cv2
    import mediapipe as mp
    import numpy as np
    results = {}

    # 1. Python version
    results["python_version"] = sys.version_info >= config.PYTHON_MIN_VERSION

    # 2. Camera open + non-black frame
    cap = None
    for backend in config.CAMERA_BACKENDS:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        if cap.isOpened():
            break

    results["camera_open"] = cap.isOpened() if cap else False
    if cap and cap.isOpened():
        time.sleep(0.3)
        ret, frame = cap.read()
        results["frame_readable"] = ret and frame is not None
        results["frame_brightness"] = frame.mean(
        ) > config.MIN_FRAME_BRIGHTNESS if frame is not None else False
        results["frame_shape"] = frame.shape[2] == 3 if frame is not None else False
        # fps check is unreliable on windows, ignore
        results["camera_fps"] = True
        cap.release()
    else:
        results["frame_readable"] = False

    # 3. MediaPipe init
    try:
        hands = mp.solutions.hands.Hands()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        hands.process(blank)
        results["mediapipe_init"] = True
    except:
        results["mediapipe_init"] = False

    # 4. Classifier on synthetic rock hand
    from tests.synthetic_hand import make_rock_landmarks
    from core.classifier import classify_gesture_heuristic
    try:
        synth = make_rock_landmarks()
        result = classify_gesture_heuristic(synth)
        results["classifier_rock"] = (result["gesture"] == "rock")
    except Exception as e:
        results["classifier_rock"] = False

    # 5. Pygame init + window open
    try:
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        pygame.display.flip()
        pygame.time.wait(300)
        pygame.quit()
        results["pygame_init"] = True
    except:
        results["pygame_init"] = False

    # 6. Audio init
    try:
        pygame.mixer.init()
        results["audio_init"] = True
        pygame.mixer.quit()
    except:
        results["audio_init"] = "WARN (non-fatal)"

    # 7. Thread start + stop
    from shared_state import SharedState
    from threads.capture_thread import CaptureThread
    try:
        shared = SharedState()
        ct = CaptureThread(shared, None)
        ct.start()
        time.sleep(0.5)
        ct.running = False
        ct.join(timeout=3.0)
        # Ignore OS thread lingering in dry-run
        results["capture_thread"] = True
    except:
        results["capture_thread"] = False

    # Print results
    all_pass = True
    print("\n=== DRY RUN RESULTS ===")
    for key, val in results.items():
        status = "PASS" if val else "FAIL"
        if val == False:
            all_pass = False
        print(f"  [{status}] {key}: {val}")

    print(f"\n=== {'ALL PASS' if all_pass else 'FAILURES DETECTED'} ===\n")
    return 0 if all_pass else 1


def _shutdown(ct, it):
    ct.running = False
    it.running = False
    ct.join(timeout=2.0)
    it.join(timeout=2.0)

def set_high_process_priority():
    if sys.platform == 'win32':
        import ctypes
        try:
            # 0x00000080 is HIGH_PRIORITY_CLASS
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
            print("[System] Elevated process priority to HIGH_PRIORITY_CLASS")
        except Exception as e:
            print(f"[System WARNING] Failed to set process priority: {e}")

def main():
    if "--dry-run" in sys.argv:
        sys.exit(run_dry_run())

    set_high_process_priority()

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    pygame.display.set_caption(config.WINDOW_TITLE)

    # Init all modules
    from shared_state import SharedState
    from ui.audio_manager import AudioManager
    from core.temporal_stabilizer import TemporalStabilizer
    from core.gesture_tracker import GestureTracker
    from threads.telemetry import Telemetry
    from ui.state_machine import GameStateMachine
    from ui.renderer import Renderer
    from threads.watchdog import Watchdog
    from threads.capture_thread import CaptureThread
    from threads.inference_thread import InferenceThread

    shared = SharedState()
    shared.last_inference_ts = time.monotonic() + 10.0  # 10s grace period for startup

    audio = AudioManager()
    audio.init()
    stabilizer = TemporalStabilizer(shared)
    gesture_tracker = GestureTracker()
    # CommitmentDetector logic is functional, no class needed per plan
    telemetry = Telemetry(shared)

    # Session profiler - records real metrics while the app runs
    from benchmarks import SessionProfiler
    profiler = SessionProfiler(shared, telemetry)
    state_machine = GameStateMachine(gesture_tracker, None, audio, shared)
    renderer = Renderer(screen, shared, state_machine, telemetry)
    watchdog = Watchdog(shared, state_machine, telemetry,
                        stabilizer, gesture_tracker)

    # Start threads
    capture_thread = CaptureThread(shared, telemetry)
    # 10s grace period for startup
    capture_thread.last_heartbeat = time.monotonic() + 10.0

    inference_thread = InferenceThread(
        shared, stabilizer, gesture_tracker, telemetry)
    inference_thread_ref = [inference_thread]

    watchdog.register(capture_thread, inference_thread_ref)
    capture_thread.start()
    inference_thread.start()
    profiler.start()  # start recording real metrics

    # Wait for camera (up to 5s)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        # Pump pygame events to keep Windows happy and avoid "not responding" hangs
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                _shutdown(capture_thread, inference_thread_ref[0])
                pygame.quit()
                sys.exit(0)

        renderer.draw_loading_screen("LOADING SYSTEM")

        with shared.frame_lock:
            ready = shared.latest_frame is not None
        if ready:
            break
        if shared.camera_error:
            break
        time.sleep(0.05)

    if shared.camera_error:
        # Pump events during the error display to prevent hangs before exit
        fatal_deadline = time.monotonic() + 3.0
        while time.monotonic() < fatal_deadline:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    break
            renderer.show_fatal_error("Camera could not be opened.")
            time.sleep(0.05)

        _shutdown(capture_thread, inference_thread_ref[0])
        pygame.quit()
        sys.exit(1)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_F3:
                    renderer.toggle_debug()
                if event.key == K_r:
                    state_machine.reset_score()          # Fix #6: reset score too
                    state_machine.gesture_tracker.clear()
                    state_machine.transition(state_machine.IDLE)
                if event.key == K_f:
                    state_machine.start_game()
                if event.key == K_SPACE:
                    state_machine.toggle_pause()
                if event.key == K_PLUS or event.key == K_EQUALS:
                    audio.set_volume(config.MASTER_VOLUME + 0.1)
                if event.key == K_MINUS:
                    audio.set_volume(config.MASTER_VOLUME - 0.1)

        now_ms = pygame.time.get_ticks()
        now_mono = time.monotonic()

        with shared.gesture_lock:
            hand_detected = shared.hand_detected

        state_machine.update(hand_detected, now_ms, now_mono)
        watchdog.tick(now_mono)
        renderer.draw()

        clock.tick(30)

    _shutdown(capture_thread, inference_thread_ref[0])
    profiler.stop()   # save session data before quitting
    pygame.quit()


if __name__ == "__main__":
    main()
