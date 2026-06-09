import pygame
import config

def log(msg):
    print(f"[Audio] {msg}")

def clamp(v, min_v, max_v):
    return max(min_v, min(v, max_v))

class AudioManager:
    def __init__(self):
        self.sounds = {}
        self.enabled = False

    def init(self):
        if not config.ENABLE_AUDIO:
            self.enabled = False
            return

        try:
            pygame.mixer.pre_init(
                frequency = config.AUDIO_FREQUENCY,
                size      = -16,
                channels  = config.AUDIO_CHANNELS,
                buffer    = config.AUDIO_BUFFER,
            )
            # Actually initialized in main.py, but safe to call here if needed or just use
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.enabled = True
        except Exception as e:
            log(f"Audio init failed ({e}) — running silent")
            self.enabled = False
            return

        # Pre-load all sounds at startup (no disk I/O during gameplay)
        SOUND_FILES = {
            "hand_detected":  "assets/sounds/hand_detected.wav",
            "locking_in":     "assets/sounds/locking_in.wav",
            "countdown_tick": "assets/sounds/countdown_tick.wav",
            "shoot":          "assets/sounds/shoot.wav",
            "result_reveal":  "assets/sounds/result_reveal.wav",
            "switch_caught":  "assets/sounds/switch_caught.wav",
        }

        for name, path in SOUND_FILES.items():
            try:
                s = pygame.mixer.Sound(path)
                s.set_volume(config.MASTER_VOLUME)
                self.sounds[name] = s
                log(f"Audio loaded: {name}")
            except FileNotFoundError:
                log(f"Audio file missing: {path} — that sound will be silent")
            except Exception as e:
                log(f"Audio load error ({name}): {e}")

    def play(self, name: str):
        if not self.enabled: return
        if name not in self.sounds: return
        try:
            self.sounds[name].play()   # Non-blocking — pygame plays on a channel
        except:
            pass    # Never let audio failure crash the game loop

    def stop(self, name: str):
        if not self.enabled: return
        if name not in self.sounds: return
        try:
            self.sounds[name].stop()
        except:
            pass

    def set_volume(self, v: float):
        v = clamp(v, 0.0, 1.0)
        config.MASTER_VOLUME = v
        for s in self.sounds.values():
            s.set_volume(v)
