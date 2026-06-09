"""
commitment_detector.py
─────────────────────────────────────────────────────────────────────────────
Evaluates the gesture buffer at the moment of capture to detect whether the
player made a genuine, committed gesture or was still switching.

Two independent signals are combined:
  1. Shannon Entropy  — high entropy → player was randomly flailing
  2. Transition Count — too many switches → player tried to change last-moment

Both thresholds are configurable in config.py.
"""

import math
import config


def evaluate_commitment(snapshot: list) -> dict:
    """
    Analyse a snapshot of the gesture buffer (list of dicts with keys
    'gesture', 'confidence', 'timestamp') and return an evaluation dict:

        {
            "committed":        bool,   # True = accept the gesture
            "reason":           str,    # '' | 'too_many_switches' | 'high_entropy'
                                        #    | 'insufficient_data' | 'too_many_unknown_frames'
            "entropy":          float,
            "transitions":      int,
        }

    Rejection rules (evaluated in order):
      1. Total snapshot size < MIN_COMMITMENT_FRAMES  → insufficient_data
      2. More than half the frames are 'unknown'      → too_many_unknown_frames
      3. Fewer than MIN_COMMITMENT_FRAMES known frames → insufficient_data
      4. Transition count > COMMITMENT_MAX_TRANSITIONS → too_many_switches
      5. Shannon entropy > COMMITMENT_ENTROPY_THRESHOLD → high_entropy
    """
    # ── Pre-flight: check total snapshot size (including unknown frames) ─────
    # If the buffer has fewer than MIN_COMMITMENT_FRAMES total entries, the
    # player showed their hand too late. Reject rather than auto-accepting:
    # a single hallucinated "paper" frame would otherwise pass the entropy gate.
    if len(snapshot) < config.MIN_COMMITMENT_FRAMES:
        return {"committed": False, "reason": "insufficient_data", "entropy": 0.0, "transitions": 0}

    # Filter to only known gestures (exclude 'unknown' frames mid-throw)
    known = [e for e in snapshot if e["gesture"] not in ("unknown", None)]

    # ── Pre-flight: check unknown-frame dominance ─────────────────────────────
    # If more than half the snapshot is unknown frames (classifier hallucinating
    # during throw arc), the buffer does not represent a real deliberate gesture.
    unknown_count = len(snapshot) - len(known)
    if unknown_count > len(snapshot) / 2:
        return {"committed": False, "reason": "too_many_unknown_frames", "entropy": 0.0, "transitions": 0}

    if len(known) < config.MIN_COMMITMENT_FRAMES:
        return {"committed": False, "reason": "insufficient_data", "entropy": 0.0, "transitions": 0}

    gestures = [e["gesture"] for e in known]

    # ── Shannon Entropy ────────────────────────────────────────────────────────
    counts = {}
    for g in gestures:
        counts[g] = counts.get(g, 0) + 1
    total = len(gestures)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        if p > 0:
            entropy -= p * math.log2(p)

    # ── Transition Count ──────────────────────────────────────────────────────
    transitions = sum(1 for i in range(1, len(gestures)) if gestures[i] != gestures[i - 1])

    # ── Decision ─────────────────────────────────────────────────────────────
    if transitions > config.COMMITMENT_MAX_TRANSITIONS:
        return {
            "committed":   False,
            "reason":      "too_many_switches",
            "entropy":     entropy,
            "transitions": transitions,
        }

    if entropy > config.COMMITMENT_ENTROPY_THRESHOLD:
        return {
            "committed":   False,
            "reason":      "high_entropy",
            "entropy":     entropy,
            "transitions": transitions,
        }

    return {"committed": True, "reason": "", "entropy": entropy, "transitions": transitions}
