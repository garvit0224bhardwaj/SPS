import pygame
import cv2
import math
import time
import mediapipe as mp
import numpy as np
import config
from ui.gesture_comparison import draw_gesture_comparison


def landmark_to_screen(lm, frame_shape, surface_size):
    # lm.x, lm.y are normalized [0, 1]
    # In full screen, the surface size is the monitor size.
    # The image is scaled to fit. We just multiply.
    return (int(lm.x * surface_size[0]), int(lm.y * surface_size[1]))


def lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, screen, shared_state, state_machine, telemetry):
        self.screen = screen
        self.shared = shared_state
        self.state_machine = state_machine
        self.telemetry = telemetry
        self.show_debug = config.DEBUG_OVERLAY_DEFAULT

        # Fix #11: Try to load bundled font first, fall back to SysFont with a logged warning
        self.font_large = self._load_font(
            config.COUNTDOWN_FONT_SIZE, bold=True)
        self.font_medium = self._load_font(72, bold=True)
        self.font_small = self._load_font(36, bold=True)
        self.font_hud = self._load_font(28, bold=True)
        self.font_debug = pygame.font.SysFont("Consolas", 20)

        # Pre-load image assets — Fix #12: log a warning when an asset is missing
        self.assets = {}
        _asset_names = {
            "rock":       "assets/rock.png",
            "paper":      "assets/paper.png",
            "scissors":   "assets/scissors.png",
            "hand_guide": "assets/hand_guide.png",
        }
        for key, path in _asset_names.items():
            try:
                self.assets[key] = pygame.image.load(path).convert_alpha()
            except Exception:
                pass  # Asset not found - renderer falls back to text-only for this element

        self.switch_banner_played = False

        # Pre-allocated image transformation buffers to save CPU and reduce allocations
        self._bgr_buf = None
        self._rgb_buf = None
        self._rot_buf = None

        # Anti-flicker: keep the last successfully rendered camera frame so we
        # never blit a black surface when latest_frame is momentarily None.
        self._last_valid_surf: pygame.Surface | None = None

        # Ghost-skeleton guard: only clear the overlay after this many
        # consecutive frames where hand_detected is False.
        self._hand_lost_frames: int = 0
        self._SKELETON_HIDE_THRESHOLD: int = 3

    # ── Font loading helper ────────────────────────────────────────────────────
    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Try assets/font.ttf first; fall back to Arial/SysFont with a logged warning."""
        try:
            font = pygame.font.Font("assets/font.ttf", size)
            return font
        except Exception:
            pass
        # SysFont fallback — may look different on machines without Arial
        font = pygame.font.SysFont("Arial", size, bold=bold)
        if font.get_height() < 10:
            # Arial wasn't found — pygame returned the default pixel font
            print(
                f"[Renderer WARNING] Arial not found for size {size} — using pygame default font")
        return font

    def toggle_debug(self):
        self.show_debug = not self.show_debug

    def draw(self):
        # 1. Read latest_frame
        frame = None
        current_frame_id = -1
        if self.shared.frame_lock.acquire(blocking=False):
            if self.shared.latest_frame is not None:
                if self._bgr_buf is None or self._bgr_buf.shape != self.shared.latest_frame.shape:
                    self._bgr_buf = np.empty_like(self.shared.latest_frame)
                np.copyto(self._bgr_buf, self.shared.latest_frame)
                frame = self._bgr_buf
                current_frame_id = self.shared.frame_id
            self.shared.frame_lock.release()

        # 2. Draw background
        screen_w, screen_h = self.screen.get_size()
        if frame is not None:
            if self._rgb_buf is None or self._rgb_buf.shape != frame.shape:
                self._rgb_buf = np.empty_like(frame)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
            rgb = self._rgb_buf

            # Pygame uses (width, height)
            h, w = rgb.shape[:2]
            if self._rot_buf is None or self._rot_buf.shape != (w, h, 3):
                self._rot_buf = np.empty((w, h, 3), dtype=np.uint8)
            cv2.transpose(rgb, dst=self._rot_buf)

            surf = pygame.surfarray.make_surface(self._rot_buf)
            surf = pygame.transform.scale(surf, (screen_w, screen_h))
            self._last_valid_surf = surf          # Cache for flicker prevention
            self.screen.blit(surf, (0, 0))
        elif self._last_valid_surf is not None:
            # Camera frame is temporarily unavailable — blit the last good frame
            # instead of a black surface to prevent black flicker.
            self.screen.blit(self._last_valid_surf, (0, 0))
        else:
            self.screen.fill((0, 0, 0))

        now_ms = pygame.time.get_ticks()
        state = self.state_machine.current
        elapsed = now_ms - self.state_machine.enter_time

        with self.shared.gesture_lock:
            hand_detected = self.shared.hand_detected
            lm = self.shared.landmark_ref
            lm_frame_id = self.shared.landmark_frame_id
            confidence = self.shared.stable_confidence
            gesture = self.shared.stable_gesture

        # Track consecutive frames where hand_detected is False.
        # The skeleton is only suppressed after _SKELETON_HIDE_THRESHOLD consecutive
        # misses, absorbing single noisy frames without flickering the overlay.
        if hand_detected:
            self._hand_lost_frames = 0
        else:
            self._hand_lost_frames += 1
        skeleton_visible = hand_detected or (self._hand_lost_frames <= self._SKELETON_HIDE_THRESHOLD)

        # 3. Layer 1: Hand guide overlay
        if config.SHOW_HAND_GUIDE and state == self.state_machine.IDLE:
            if "hand_guide" in self.assets:
                guide = self.assets["hand_guide"].copy()
                alpha = int(90 + 30 * math.sin(now_ms / 250.0))
                guide.set_alpha(alpha)
                r = guide.get_rect(center=(screen_w//2, screen_h//2))
                self.screen.blit(guide, r)

        # 4. Layer 2: Hand skeleton — only draw if landmarks match the current frame to prevent stale skeletons on black/frozen frames
        skeleton_is_fresh = (lm_frame_id == current_frame_id)
        if config.SHOW_SKELETON and skeleton_visible and lm is not None and skeleton_is_fresh:
            self._draw_skeleton(lm, (h, w) if frame is not None else (
                720, 1280), (screen_w, screen_h), confidence, state)

        # 5. Layer 3: Gesture Label
        if config.SHOW_GESTURE_LABEL and skeleton_visible:
            with self.shared.gesture_lock:
                gesture_label = self.shared.stable_gesture
                conf_label    = self.shared.stable_confidence
            label = f"{gesture_label.upper()} ({conf_label:.2f})"
            txt = self.font_small.render(label, True, (255, 255, 255))
            txt.set_alpha(180)
            self.screen.blit(txt, (20, screen_h - 120))

        # 6. Layer 4: State UI
        if state == self.state_machine.IDLE:
            self._draw_centered_text(
                "PRESS F TO PLAY", self.font_medium)


        elif state == self.state_machine.COUNTDOWN:
            idx = min((now_ms - self.state_machine.enter_time) //
                      config.COUNTDOWN_TICK_MS, len(config.COUNTDOWN_WORDS) - 1)
            word = config.COUNTDOWN_WORDS[idx]
            word_age_ms = (
                now_ms - self.state_machine.enter_time) % config.COUNTDOWN_TICK_MS
            scale = lerp(1.2, 1.0, min(word_age_ms / 150.0, 1.0))

            txt_surf = self.font_large.render(word, True, (255, 255, 255))
            new_size = (int(txt_surf.get_width() * scale),
                        int(txt_surf.get_height() * scale))
            txt_surf = pygame.transform.scale(txt_surf, new_size)
            r = txt_surf.get_rect(center=(screen_w//2, screen_h//2))

            # Optional icon
            if config.SHOW_COUNTDOWN_ICONS and word.lower() in self.assets:
                icon = self.assets[word.lower()]
                icon_r = icon.get_rect(midright=(r.left - 20, screen_h//2))
                self.screen.blit(icon, icon_r)

            self.screen.blit(txt_surf, r)

        elif state == self.state_machine.SHOOT:
            if elapsed < 50:
                s = pygame.Surface((screen_w, screen_h))
                s.set_alpha(40)
                s.fill((255, 255, 255))
                self.screen.blit(s, (0, 0))
            scale = lerp(1.0, 1.08, min(
                elapsed / config.SHOOT_DISPLAY_MS, 1.0))
            txt = self.font_large.render("SHOOT!", True, (255, 255, 255))
            txt = pygame.transform.scale(
                txt, (int(txt.get_width() * scale), int(txt.get_height() * scale)))
            r = txt.get_rect(center=(screen_w//2, screen_h//2))
            self.screen.blit(txt, r)

        elif state == self.state_machine.RESULT:
            if elapsed >= config.RESULT_REVEAL_DELAY_MS:
                result = self.state_machine.result
                if "error" in result:
                    self._draw_centered_text(
                        result["error"], self.font_medium, color=(255, 100, 100))
                else:
                    outcome = result.get("outcome", "machine_win")
                    if outcome == "player_win":
                        outcome_text = "YOU WIN!"
                        outcome_color = (80, 255, 120)    # green
                    elif outcome == "tie":
                        outcome_text = "TIE!"
                        outcome_color = (255, 220, 50)    # yellow
                    else:
                        outcome_text = "MACHINE WINS"
                        outcome_color = (255, 80, 80)     # red

                    # Draw outcome big on top
                    self._draw_text_at(outcome_text, self.font_large, outcome_color,
                                       y_offset=-80)
                    # Draw machine move smaller below
                    self._draw_text_at(f"Machine: {result['counter_move'].upper()}",
                                       self.font_small, (220, 220, 220), y_offset=60)
                    draw_gesture_comparison(
                        self.screen, result, self.font_small)

        elif state == self.state_machine.PAUSED:
            self._draw_centered_text("PAUSED", self.font_large, color=(255, 255, 255))

        elif state == self.state_machine.SCOREBOARD:
            score = self.state_machine.score
            txt = f"ROBO: {score['losses']}   YOU: {score['wins']}"
            self._draw_centered_text(
                txt, self.font_medium, color=(100, 255, 100))

        # 7. Fix #9: Persistent score HUD strip — always visible in top-right
        self._draw_score_hud(screen_w)

        # 7.5: On-screen FPS HUD — always visible (capture / inference / render)
        try:
            cap_f = self.telemetry.capture_fps()
            inf_f = self.telemetry.inference_fps()
            ren_f = self.telemetry.render_fps()
            fps_text = f"CAP:{cap_f:.0f} INF:{inf_f:.0f} REN:{ren_f:.0f}"
            fps_surf = self.font_hud.render(fps_text, True, (200, 255, 200))
            pad = 8
            fps_w = fps_surf.get_width() + pad * 2
            fps_h = fps_surf.get_height() + pad * 2
            fps_bg = pygame.Surface((fps_w, fps_h), pygame.SRCALPHA)
            pygame.draw.rect(fps_bg, (0, 0, 0, 160),
                             fps_bg.get_rect(), border_radius=8)
            self.screen.blit(fps_bg, (10, 10))
            self.screen.blit(fps_surf, (10 + pad, 10 + pad))
        except Exception:
            # Failsafe: don't break rendering if telemetry isn't ready yet
            pass

        # 8. Layer 6: Debug overlay
        if self.show_debug:
            dbg = self.telemetry.debug_string()
            txt = self.font_debug.render(dbg, True, (0, 255, 0))
            self.screen.blit(txt, (10, 10))
            state_txt = self.font_debug.render(
                f"STATE: {state}", True, (255, 255, 0))
            self.screen.blit(state_txt, (10, 30))

        # 9. Error overlay
        if state == self.state_machine.ERROR_RECOVERY:
            s = pygame.Surface((screen_w, screen_h))
            s.set_alpha(150)
            s.fill((200, 0, 0))
            self.screen.blit(s, (0, 0))
            self._draw_centered_text(
                "RECOVERING SYSTEM...", self.font_large, color=(255, 255, 255))

        pygame.display.flip()
        self.telemetry.record_render_frame(time.monotonic())

    # ── Score HUD ─────────────────────────────────────────────────────────────
    def _draw_score_hud(self, screen_w: int):
        """Fix #9: Persistent score strip in the top-right corner across all game states."""
        score = self.state_machine.score
        hud_text = f"ROBO  {score['losses']} : {score['wins']}  YOU"
        streak = score["streak"]

        # Semi-transparent pill background
        txt_surf = self.font_hud.render(hud_text, True, (255, 255, 255))
        pad_x, pad_y = 18, 10
        pill_w = txt_surf.get_width() + pad_x * 2
        pill_h = txt_surf.get_height() + pad_y * 2
        pill_x = screen_w - pill_w - 20
        pill_y = 16

        pill = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
        pygame.draw.rect(pill, (0, 0, 0, 160),
                         pill.get_rect(), border_radius=12)
        self.screen.blit(pill, (pill_x, pill_y))
        self.screen.blit(txt_surf, (pill_x + pad_x, pill_y + pad_y))

        # Streak indicator — shown when machine is on a winning streak
        if streak >= 2:
            streak_txt = self.font_hud.render(
                f"STREAK x{streak}", True, (255, 80, 80))
            s_pill_w = streak_txt.get_width() + pad_x * 2
            s_pill = pygame.Surface((s_pill_w, pill_h), pygame.SRCALPHA)
            pygame.draw.rect(s_pill, (180, 0, 0, 160),
                             s_pill.get_rect(), border_radius=12)
            self.screen.blit(
                s_pill, (screen_w - s_pill_w - 20, pill_y + pill_h + 6))
            self.screen.blit(streak_txt, (screen_w - s_pill_w -
                             20 + pad_x, pill_y + pill_h + 6 + pad_y))

    def _draw_skeleton(self, lm, frame_shape, surface_size, confidence, state=None):
        import types
        if getattr(self, '_display_lm', None) is None or len(self._display_lm) != len(lm):
            self._display_lm = [types.SimpleNamespace(
                x=(1.0 - l.x) if config.MIRROR_CAMERA else l.x,
                y=l.y
            ) for l in lm]
        else:
            alpha = 0.35
            for d, t in zip(self._display_lm, lm):
                tx = (1.0 - t.x) if config.MIRROR_CAMERA else t.x
                d.x += (tx - d.x) * alpha
                d.y += (t.y - d.y) * alpha
                
        disp_lm = self._display_lm
        
        sm = self.state_machine
        locked_in_states = (sm.COUNTDOWN, sm.SHOOT, sm.RESULT)
        if state in locked_in_states:
            color = (80, 230, 120)   # green — locked in
        else:
            color = (255, 255, 255)  # white — hand detected, not yet locked in

        # Draw lines
        for (start_idx, end_idx) in mp.solutions.hands.HAND_CONNECTIONS:
            pt1 = landmark_to_screen(disp_lm[start_idx], frame_shape, surface_size)
            pt2 = landmark_to_screen(disp_lm[end_idx],   frame_shape, surface_size)
            pygame.draw.line(self.screen, color, pt1, pt2, 4)

        # Draw joints
        for i in range(21):
            pt = landmark_to_screen(disp_lm[i], frame_shape, surface_size)
            pygame.draw.circle(self.screen, color, pt, 6)

    def _draw_centered_text(self, text, font, color=(255, 255, 255)):
        txt_surf = font.render(text, True, color)
        r = txt_surf.get_rect(
            center=(self.screen.get_width()//2, self.screen.get_height()//2))
        self.screen.blit(txt_surf, r)

    def _draw_text_at(self, text, font, color=(255, 255, 255), y_offset=0):
        """Render text centered horizontally, offset vertically from screen centre."""
        txt_surf = font.render(text, True, color)
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2 + y_offset
        r = txt_surf.get_rect(center=(cx, cy))
        self.screen.blit(txt_surf, r)

    def show_fatal_error(self, msg):
        self.screen.fill((100, 0, 0))
        self._draw_centered_text(msg, self.font_medium)
        pygame.display.flip()

    def draw_loading_screen(self, text):
        self.screen.fill((20, 20, 30))  # Sleek premium dark background
        screen_w, screen_h = self.screen.get_size()

        # Pulsing text color (soft blue/purple)
        now_ms = pygame.time.get_ticks()
        pulse = int(180 + 75 * math.sin(now_ms / 200.0))
        color = (pulse, pulse, 255)

        # Render main loading text
        txt_surf = self.font_medium.render(text, True, color)
        r = txt_surf.get_rect(center=(screen_w // 2, screen_h // 2 - 50))
        self.screen.blit(txt_surf, r)

        # Render subtitle
        sub_txt = self.font_small.render(
            "Initializing camera and neural networks...", True, (160, 160, 160))
        sub_r = sub_txt.get_rect(center=(screen_w // 2, screen_h // 2 + 30))
        self.screen.blit(sub_txt, sub_r)

        # Draw a high-end spinning trailing-dot loader
        radius = 45
        center = (screen_w // 2, screen_h // 2 + 130)
        angle = (now_ms / 150.0) % (2.0 * math.pi)
        for i in range(6):
            dot_angle = angle - (i * 0.22)
            x = int(center[0] + radius * math.cos(dot_angle))
            y = int(center[1] + radius * math.sin(dot_angle))
            # Fade out trailing dots
            intensity = int(255 * (1.0 - i * 0.15))
            dot_color = (0, intensity, 255)
            pygame.draw.circle(self.screen, dot_color, (x, y), 8 - i)

        pygame.display.flip()
