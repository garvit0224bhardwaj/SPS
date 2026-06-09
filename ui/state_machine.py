import pygame
import queue
import random
import time
import collections
import config
from core.session_logger import init_logger, log_round, log_event

COUNTER_MOVE = {"rock": "paper", "paper": "scissors", "scissors": "rock"}



def log(msg):
    print(f"[StateMachine] {msg}")

class GameStateMachine:
    # States
    IDLE              = "IDLE"
    COUNTDOWN         = "COUNTDOWN"
    SHOOT             = "SHOOT"
    RESULT            = "RESULT"
    PAUSED            = "PAUSED"
    SCOREBOARD        = "SCOREBOARD"
    ERROR_RECOVERY    = "ERROR_RECOVERY"

    def __init__(self, gesture_tracker, commitment_detector, audio, shared_state, renderer=None):
        self.gesture_tracker = gesture_tracker
        self.commitment_detector = commitment_detector
        self.audio = audio
        self.shared = shared_state
        self.renderer = renderer

        self.current       = self.IDLE
        self.enter_time    = 0
        self.tick_idx      = 0          # Index into COUNTDOWN_WORDS [0,1,2,3]
        self.result        = {}
        self.score         = {"wins": 0, "losses": 0, "streak": 0}
        self.shoot_start_time = 0.0     # Monotonic timestamp when SHOOT state was entered
        self.grace_start   = 0          # Timestamp hand was lost (for grace period)
        self.hand_was_lost = False

        # Sliding-window majority-vote buffer
        self.shoot_gesture_buffer = []

        init_logger()


    def reset_score(self):
        """Reset scores to zero. Called by R key in main.py."""
        self.score = {"wins": 0, "losses": 0, "streak": 0}
        log("Score reset by player (R key)")
        log_event("SCORE_RESET", "Player manually reset the score", value="R_KEY")

    def _drain_gesture_events(self):
        """Discard every pending item in the gesture_events queue."""
        drained = 0
        while True:
            try:
                self.shared.gesture_events.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            log(f"Drained {drained} stale gesture event(s) from queue")


    def transition(self, new_state):
        log(f"  {self.current} -> {new_state}")
        log_event(
            "STATE_TRANSITION",
            f"Game state changed",
            value=f"{self.current}->{new_state}",
        )
        self.current    = new_state
        self.enter_time = pygame.time.get_ticks()
        # Publish phase so inference_thread can gate the contour fail-safe
        with self.shared.gesture_lock:
            self.shared.game_phase = new_state


    def start_game(self):
        """Start the game countdown. Triggered by F key in IDLE state."""
        if self.current == self.IDLE:
            log("Game started via F key press")
            self.shoot_gesture_buffer = []
            self.tick_idx = 0
            self.hand_was_lost = False
            self.audio.play("locking_in")
            self.transition(self.COUNTDOWN)


    def toggle_pause(self):
        """Toggle pause state. Space key binding."""
        if self.current == self.PAUSED:
            log("Resuming game from pause...")
            if self.gesture_tracker:
                self.gesture_tracker.clear()
            self._drain_gesture_events()
            self.transition(self.IDLE)
        else:
            log("Game paused...")
            self.transition(self.PAUSED)


    def update(self, hand_detected: bool, now_ms: int, now_mono: float):
        elapsed = now_ms - self.enter_time

        # ── IDLE ─────────────────────────────────────────────────────────────
        if self.current == self.IDLE:
            pass  # Game starts strictly when F key is pressed

        # ── PAUSED ───────────────────────────────────────────────────────────
        elif self.current == self.PAUSED:
            return

        # ── COUNTDOWN ────────────────────────────────────────────────────────
        elif self.current == self.COUNTDOWN:
            self.shoot_gesture_buffer = []

            ticks_elapsed = (now_ms - self.enter_time) // config.COUNTDOWN_TICK_MS
            if ticks_elapsed > self.tick_idx:
                self.tick_idx = ticks_elapsed
                if self.tick_idx < len(config.COUNTDOWN_WORDS):
                    self.audio.play("countdown_tick")

            if elapsed >= len(config.COUNTDOWN_WORDS) * config.COUNTDOWN_TICK_MS:
                with self.shared.gesture_lock:
                    self.shared.reset_gesture_history = True
                    self.shared.fallback_to_rock      = False
                self.shoot_gesture_buffer = []
                self.shoot_start_time = time.monotonic()
                self.audio.play("shoot")
                self.transition(self.SHOOT)

        # ── SHOOT ─────────────────────────────────────────────────────────────
        elif self.current == self.SHOOT:
            elapsed_ms = (time.monotonic() - self.shoot_start_time) * 1000.0

            with self.shared.gesture_lock:
                stable_g         = self.shared.stable_gesture
                fallback_active  = self.shared.fallback_to_rock

            if fallback_active:
                log("Contour fail-safe triggered — assigning Rock via fallback")
                with self.shared.gesture_lock:
                    self.shared.fallback_to_rock = False
                self._resolve_shoot("rock", 1.0)
                return

            if elapsed_ms < config.SHOOT_GRACE_PERIOD_MS:
                return

            window_end_ms = config.SHOOT_GRACE_PERIOD_MS + config.SHOOT_WINDOW_DURATION_MS
            if elapsed_ms <= window_end_ms:
                if stable_g:
                    self.shoot_gesture_buffer.append(stable_g)
                return

            self._close_shoot_window(stable_g)

        # ── RESULT ────────────────────────────────────────────────────────────
        elif self.current == self.RESULT:
            if elapsed >= config.RESULT_DISPLAY_MS:
                if "error" in self.result:
                    if self.gesture_tracker:
                        self.gesture_tracker.clear()
                    self.hand_was_lost = False
                    self._drain_gesture_events()
                    self.tick_idx = 0
                    self.transition(self.COUNTDOWN)
                else:
                    self.transition(self.SCOREBOARD)

        # ── SCOREBOARD ────────────────────────────────────────────────────────
        elif self.current == self.SCOREBOARD:
            if elapsed >= config.SCOREBOARD_MS:
                if self.gesture_tracker:
                    self.gesture_tracker.clear()
                self.hand_was_lost = False
                self._drain_gesture_events()
                self.tick_idx = 0
                self.transition(self.COUNTDOWN)

        # ── ERROR_RECOVERY ────────────────────────────────────────────────────
        elif self.current == self.ERROR_RECOVERY:
            if elapsed >= 2000:
                self.gesture_tracker.clear()
                self.transition(self.IDLE)

    # ── Shoot resolution helpers ───────────────────────────────────────────────

    def _close_shoot_window(self, last_gesture: str):
        """Compute the majority vote from shoot_gesture_buffer and resolve the round."""
        buf = self.shoot_gesture_buffer

        if not buf:
            log("SHOOT window closed with empty buffer — foul")
            self.result = {"error": "Hand obscured!"}
            self._drain_gesture_events()
            self.transition(self.RESULT)
            return

        counter = collections.Counter(buf)
        mode_gesture, mode_count = counter.most_common(1)[0]
        total = len(buf)
        unknown_count = counter.get("unknown", 0) + counter.get("Unknown", 0)
        unknown_ratio = unknown_count / total

        log(f"SHOOT window: total={total}, distribution={dict(counter)}, mode={mode_gesture} ({mode_count}/{total})")

        if mode_gesture.lower() == "unknown" or unknown_ratio > 0.5:
            log("SHOOT foul: too many unknown readings")
            self.result = {"error": "Hand obscured!"}
            self._drain_gesture_events()
            self.transition(self.RESULT)
            return

        player = mode_gesture.lower()
        confidence = mode_count / total
        self._resolve_shoot(player, confidence)

    def _resolve_shoot(self, player: str, confidence: float):
        """Score the round and transition to RESULT state."""
        pre_robo   = self.score["losses"]
        pre_player = self.score["wins"]
        pre_streak = self.score["streak"]

        if player in ["rock", "paper", "scissors"]:
            BEATS   = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
            machine = COUNTER_MOVE[player]

            if BEATS.get(player) == machine:
                outcome = "player_win"
                self.score["wins"]   += 1
                self.score["streak"]  = 0
            elif BEATS.get(machine) == player:
                outcome = "machine_win"
                self.score["losses"] += 1
                self.score["streak"] += 1
            else:
                outcome = "tie"

            self.result = {
                "valid": True,
                "player_gesture": player,
                "counter_move": machine,
                "outcome": outcome,
                "confidence": confidence,
            }

            log_round(
                status="VALID",
                player_gesture=player,
                computer_gesture=machine,
                outcome=outcome,
                invalid_reason=None,
                pre_score_robo=pre_robo,
                pre_score_player=pre_player,
                pre_score_streak=pre_streak,
                post_score_robo=self.score["losses"],
                post_score_player=self.score["wins"],
                post_score_streak=self.score["streak"],
                palm_velocity=0.0,
                settle_timeout=False,
                confidence=confidence
            )
            self._drain_gesture_events()
            self.transition(self.RESULT)
        else:
            self.result = {"error": "Hand obscured!"}

            log_round(
                status="INVALID",
                player_gesture=None,
                computer_gesture=None,
                outcome="invalid",
                invalid_reason="low_confidence",
                pre_score_robo=pre_robo,
                pre_score_player=pre_player,
                pre_score_streak=pre_streak,
                post_score_robo=self.score["losses"],
                post_score_player=self.score["wins"],
                post_score_streak=self.score["streak"],
                palm_velocity=0.0,
                settle_timeout=False,
                confidence=0.0
            )
            if self.gesture_tracker:
                self.gesture_tracker.clear()
            self.transition(self.RESULT)

    def handle_camera_lost(self):
        self.transition(self.ERROR_RECOVERY)

    def handle_inference_stall(self):
        self.transition(self.ERROR_RECOVERY)

    def _reason_to_ui_text(self, reason: str) -> str:
        MAP = {
            "too_many_switches":       "Nice try \U0001f60f  \u2014 Pick one move and commit",
            "high_entropy":            "Gesture unclear \u2014 hold your move steady",
            "low_confidence":          "Couldn't read that \u2014 show your hand clearly",
            "insufficient_data":       "Show your hand earlier in the countdown",
            "too_many_unknown_frames": "Keep your hand visible during the countdown",
            "thumbs_up":               "That's a thumbs-up \u2014 try Rock, Paper, or Scissors!",
        }
        return MAP.get(reason, "Let's try again")
