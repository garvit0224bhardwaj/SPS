# Edge AI Rock-Paper-Scissors (SPS)

An interactive, multi-threaded Rock-Paper-Scissors game designed for **Edge AI / Embedded Systems**. It uses OpenCV, Pygame, and a custom NumPy-based MLP neural network running on top of MediaPipe hand tracking landmarks to classify gestures and beat players in real-time.

> The system has been hardened for high-noise, public environments — zero-jitter skeleton rendering, anti-cheat commitment detection, adaptive performance degradation, live watchdog recovery, and Windows process-priority elevation are all built-in.

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites-and-installation)
2. [Running the Application](#2-running-the-application)
3. [System Architecture](#3-system-architecture)
4. [Threading & Pipeline Model](#4-threading--pipeline-model)
5. [Core AI & Gesture Classifier](#5-core-ai--gesture-classifier)
6. [Smoothness & Jitter Elimination](#6-smoothness--jitter-elimination)
7. [Adaptive Performance Degradation](#7-adaptive-performance-degradation)
8. [Anti-Cheat & Commitment Detection](#8-anti-cheat--commitment-detection)
9. [Matchmaking & Win-Rate Control](#9-matchmaking--win-rate-control-rigged-deck-shuffle)
10. [Edge AI Benchmarking](#10-edge-ai-benchmarking)
11. [Technical Performance Results](#11-technical-performance-results)
12. [Project Files](#12-project-files)

---

## 1. Prerequisites and Installation

### Requirements
- **Python**: 3.11+ (enforced at startup)
- **OS**: Windows (recommended — uses DirectShow backend) or Linux/macOS
- **Hardware**: USB webcam (USB 3.0 recommended for MJPEG at 60fps)
- **Edge Target**: Tested on Windows x64; benchmarked for Raspberry Pi 5

### Installation

```powershell
# Clone / copy the project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

> **Note**: `mediapipe==0.10.11` is required. Newer versions removed the `mp.solutions` API. Pin this version in your environment.

---

## 2. Running the Application

### Self-Diagnostic Dry-Run
```powershell
python main.py --dry-run
```
Validates: Python version, camera open, frame brightness, MediaPipe init, MLP classifier, Pygame, audio, and thread startup. Prints `[PASS]` / `[FAIL]` for each check.

### Playing the Game
```powershell
python main.py
```

### Keybindings
| Key | Action |
|---|---|
| `ESC` | Exit |
| `SPACE` | Pause / Resume game (re-starts from IDLE) |
| `F` | Start game / next session |
| `F3` | Toggle live telemetry overlay (FPS, latency, degradation level) |
| `R` | Reset scoreboard and return to IDLE (Press F to Play screen) |
| `+` / `=` | Increase volume |
| `-` | Decrease volume |

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         SPS Architecture                         │
│                                                                  │
│  ┌─────────────┐  frame_lock  ┌──────────────────┐              │
│  │CaptureThread│ ──────────► │ InferenceThread   │              │
│  │  (OpenCV)   │             │ MediaPipe + MLP   │              │
│  └─────────────┘             │ 1€ Filter         │              │
│         │                    └────────┬─────────┘              │
│         │                             │ gesture_lock            │
│         │                    ┌────────▼─────────┐              │
│         │                    │   SharedState     │              │
│         │                    │  (landmark_ref)   │              │
│         │                    │  (gesture_events) │  ← queue.Queue│
│         │                    └────────┬─────────┘              │
│         │                             │                          │
│         │             ┌───────────────▼──────────────┐         │
│         │             │     Pygame Main Thread        │         │
│         │             │  Renderer (lerp interpolation)│         │
│         │             │  GameStateMachine             │         │
│         │             └───────────────────────────────┘         │
│         │                                                        │
│  ┌──────▼─────────────┐                                         │
│  │   Watchdog Thread  │  ← monitors both threads' heartbeats    │
│  └────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────┘
```
---
## 4. Threading & Pipeline Model

### Hybrid Pipeline Architecture
The system uses a carefully designed **two-tier pipeline** to eliminate the trade-off between latency and event reliability:

| Data Type | Transport | Rationale |
|---|---|---|
| Camera frames | Shared memory (overwrite) | Zero-latency. Dropped frames are acceptable — we only care about the *latest* frame. |
| Landmark coordinates | Shared memory (overwrite) | Same: always draw the freshest position. |
| Confirmed gesture events | `queue.Queue` | Discrete game actions (e.g., "rock confirmed") must **never** be dropped even if the UI thread is briefly busy. |

> **Why not queue everything?** A FIFO queue on a 60fps video stream creates a "lag balloon" — if the consumer slows down momentarily, the queue fills up and the user sees a 2-second delay between physical movement and UI response. Overwriting shared memory ensures the rendered state is always the *current* physical reality.

### Thread Summary
| Thread | Runs At | Responsibility |
|---|---|---|
| `CaptureThread` | Camera FPS (≤60) | OpenCV capture, warmup, reconnection |
| `InferenceThread` | 20–35 FPS | MediaPipe processing, 1€ landmark filtering, and MLP classification |
| `Main Thread (Pygame)` | Locked 60 FPS | Render, lerp interpolation, state machine |
| `Watchdog` | Polled each render frame | Heartbeat check, degradation, thread restart |

---

## 5. Core AI & Gesture Classifier

### A. Pose-Invariant Feature Engineering
The MLP classifier receives a 60-dimensional feature vector invariant to translation, rotation, and scale:

1. Translate all 21 landmarks relative to the wrist (origin).
2. Normalize by `hand_size = dist(wrist, middle_MCP)`.
3. Project into a local coordinate frame: `y_axis = wrist→middle_MCP`, `x_axis = index_MCP→pinky_MCP`, `z_axis = cross(x, y)`.

This makes classification work at 1–3m distances from the camera without retraining.

### B. MLP Neural Network
- **Architecture**: 2-layer ReLU MLP — 60 inputs → 128 hidden → 3 outputs (Rock, Paper, Scissors)
- **Implementation**: Pure NumPy (no TensorFlow/PyTorch — runs on any edge device)
- **Inference cost**: `<0.1 ms` (matrix multiply)
- **Fallback**: Heuristic rule-based classifier if `model_weights.json` is absent

### C. Temporal Stabilizer
- **Hysteresis**: Candidate gesture must appear for `3` consecutive inference frames before the stable label changes.
- **Confidence EMA**: $EMA_t = 0.20 \cdot c_{raw} + 0.80 \cdot EMA_{t-1}$ (slow decay = stable display brightness).
- **Gesture Event Queue**: On every confirmed gesture change, a discrete event is pushed to `shared.gesture_events` for the game state machine to consume losslessly.

---

## 6. Smoothness & Jitter Elimination

### Problem: Two Types of Jitter
| Type | Root Cause | Visible Effect |
|---|---|---|
| Skeleton trembling | Raw MediaPipe landmark noise (±2–5px per frame) | Skeleton wobbles even when hand is still |
| Skeleton snapping | 60fps renderer jumping to ~30fps inference updates | Skeleton "teleports" every 33ms |

### Fix A: One Euro (1€) Landmark Filter
Each of 21 MediaPipe landmarks is filtered by a `LandmarkOneEuroFilter` before being written to shared state:
- **Hovering still** → high smoothing → zero tremor
- **Moving fast** → alpha near 1.0 → zero tracking lag
- Parameters: `min_cutoff=0.1, beta=0.5`

### Fix B: Render-Side Lerp Interpolation
The renderer maintains `_display_lm` coordinates that interpolate 35% toward the latest landmark target each frame:
```python
d.x += (target.x - d.x) * 0.35   # runs at 60fps
```
Result: the skeleton glides fluidly rather than jumping between inference frames.

### Fix C: Hysteresis Stabilized Gesture Display
Gesture label changes are filtered by a 3-frame temporal hysteresis and smoothed using a confidence EMA. This prevents the label from flickering or jumping around between frames.

---

## 7. Adaptive Performance Degradation

When inference FPS drops below threshold, the Watchdog automatically reduces system load:

| Level | Trigger | Action |
|---|---|---|
| **L0 — Normal** | INF FPS ≥ 20 | Full 1280×720 resolution |
| **L1 — Half-Res** | INF FPS < 20 | Resize input to 640×360 (70% CPU reduction) |
| **L2 — Frame Skip** | INF FPS < 12 | Half-res + skip every other frame |

Self-recovers to L0 when FPS exceeds threshold +5 for a sustained period.

---

## 8. Gesture Capture & Anti-Cheat (Time-Based Majority Vote)

Instead of legacy locking-in mechanics based on hand stillness (velocity tracking), the game uses a clean **Sliding Window Majority Vote** tied to a classic countdown sequence.

### A. Sliding Window Majority Vote
- **Countdown Tick**: Words "READY", "ROCK", "PAPER", "SCISSORS" display sequentially (750ms ticks).
- **Grace Period (100ms)**: Allows the player's throw to settle.
- **Sampling Window (500ms)**: Captures hand gestures at the inference thread rate.
- **Majority Vote**: Calculates the statistical mode (most frequent gesture) of the collected frames.
- **Foul Check**: If the mode is "Unknown" or >50% of the window is "Unknown", the round is declared invalid and transitions directly to the `RESULT` state displaying "Hand obscured!" in red. Otherwise, the round transitions to the `RESULT` state with the winning/losing outcome instantly.

### B. Legacy Commitment Detection (Optional / Diagnostic)
*Note: The previous version included a legacy Shannon-entropy and transition-count commitment evaluation detector during a 'Locking In' state (designed to detect switches mid-throw). This was completely replaced by the more robust Sliding Window Majority Vote to simplify the game mechanics and prevent camera noise from blocking valid gestures.*

---

## 9. Matchmaking & Win-Rate Control (Rigged Deck Shuffle)

The game features a controlled win-rate strategy called the **Rigged Deck Shuffle** to guarantee a natural and engaging play experience.

### How it Works
1. **Pre-defined Pool**: The game state machine initializes a pool of 10 matches: `[1, 1, 1, 0, 1, 1, 0, 1, 1, 1]`.
   - `1` = The machine counters the player's gesture perfectly (machine wins).
   - `0` = The machine intentionally throws the match by choosing the gesture beaten by the player's throw (player wins).
2. **Shuffled Order**: The deck is shuffled on game startup using `random.shuffle`.
3. **Popping Moves**: Whenever a valid round resolves, the state machine pops the first value from the deck:
   - Popped `1` -> AI plays winning counter.
   - Popped `0` -> AI plays losing counter.
4. **Replenishment**: Once the deck size hits 0, the pool is replenished and reshuffled automatically.
5. **Exact Win-Rate**: This guarantees that the player wins exactly 20% of valid rounds, preventing extreme winning or losing streaks while keeping the interval between player wins completely natural.

---

## 10. Edge AI Benchmarking

A comprehensive benchmark suite is included to profile the system on any hardware — intended for Raspberry Pi vs. laptop before/after comparisons:

```powershell
# Basic run (10s timed tests)
python benchmarks.py

# Full run with output files
python benchmarks.py --duration 30 --output results.json --csv

# Raspberry Pi: skip camera (headless mode), run thermal stress test
python benchmarks.py --no-camera --stress --output rpi_baseline.json
```

### Metrics Captured

| Category | Metric |
|---|---|
| **CPU** | Mean/peak usage (%), per-core, frequency |
| **RAM** | RSS mean/peak (MB), virtual memory |
| **Inference** | Min/mean/median/P95/P99/max latency (ms), sustained FPS |
| **Camera** | Actual FPS, frame interval jitter (P99), bandwidth (MB/s) |
| **Pipeline** | End-to-end capture→inference latency, frame drop rate (%) |
| **Thermal** | CPU temperature (°C) idle/load/peak, throttling detection |
| **Components** | 1€ filter cost (µs), MLP forward pass (µs), frame copy bandwidth |
| **Model Load** | MediaPipe cold-start (ms), MLP weights load (ms/KB) |
| **Hardware** | CUDA, OpenCL, Hailo, Coral TPU, Raspberry Pi VideoCore detection |
| **System** | CPU model, core count, total RAM, OS, board model |

### Workflow: Before/After Report
```powershell
# Run on laptop BEFORE optimization
python benchmarks.py --output laptop_before.json --csv

# Run on Raspberry Pi AFTER optimization
python benchmarks.py --output rpi_after.json --csv --stress

# benchmark_log.csv now has one row per run — import into Excel/Sheets for report
```
---

## 11. Technical Performance Results

### On Development Hardware (Windows, mid-tier CPU)
| Metric | Value |
|---|---|
| Render FPS | 60 FPS (locked) |
| Inference FPS | 25–35 FPS (async) |
| Capture-to-display latency | 25–35 ms |
| MediaPipe inference time | ~12 ms (MODEL_COMPLEXITY=0) |
| MLP classification time | <0.1 ms |
| 1€ filter cost (21 landmarks) | ~8–15 µs |
| Gesture label flicker | Eliminated |
| Skeleton tremor | Eliminated |

### Watchdog Recovery
- Inference stall detected within: **3.0 seconds**
- Thread restart + MediaPipe re-init: **~1.5 seconds**
- Camera reconnect (30 failures → reopen): **~1.2 seconds**
---

## 12. Project Files

```
SPS/
├── main.py                     # Entry point, thread orchestration, priority elevation
├── config.py                   # All tunable parameters (thresholds, FPS, timing)
├── shared_state.py             # Thread-safe shared data + gesture_events queue
├── utilities.py                # Geometric helpers (palm_centre, dist, lm_xyz)
├── benchmarks.py               # Edge AI profiling suite (Raspberry Pi ready)
├── calibrate_velocity.py       # Velocity threshold calibration tool
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
├── HISTORY.md                  # Full development journey for academic reports
│
├── core/
│   ├── classifier.py           # MLP + heuristic gesture classifier
│   ├── extract_real_features.py # Real-world features extractor for hybrid dataset
│   ├── one_euro_filter.py      # 1€ dynamic smoothing filter
│   ├── temporal_stabilizer.py  # Hysteresis + EMA confidence + event queue push
│   ├── gesture_tracker.py      # Rolling gesture buffer (legacy)
│   ├── commitment_detector.py  # Anti-cheat: Shannon entropy + transition count (legacy/unused)
│   ├── session_logger.py       # CSV event logging per session
│   ├── model_weights.json      # Trained MLP weights (NumPy-loadable)
│   ├── collect_data.py         # Landmark data collection tool
│   ├── bootstrap_dataset.py    # Data augmentation utility (hybrid merge)
│   └── train_model.py          # MLP training script
│
├── threads/
│   ├── capture_thread.py       # Camera capture loop + reconnection
│   ├── inference_thread.py     # MediaPipe + 1€ filter + MLP inference loop
│   ├── watchdog.py             # Heartbeat monitor + adaptive degradation
│   └── telemetry.py            # FPS ring-buffer telemetry
│
├── ui/
│   ├── renderer.py             # Pygame rendering + lerp interpolation
│   ├── state_machine.py        # Game flow (IDLE→COUNTDOWN→SHOOT→RESULT→SCOREBOARD or back to COUNTDOWN)
│   ├── audio_manager.py        # Sound effects manager
│   └── gesture_comparison.py   # Result screen gesture comparison widget
│
├── tests/
│   ├── synthetic_hand.py       # Synthetic landmark generator for dry-run tests
│   └── test_state_machine.py   # Unit tests for rigged deck shuffle state machine logic
│
├── assets/                     # PNG icons (rock, paper, scissors, hand guide)
└── logs/                       # Session CSV logs (auto-generated)
```
