# ui/gesture_comparison.py
# ─────────────────────────────────────────────────────────────────────────────
# Shows both gestures on the result screen so the player can see exactly
# what was captured vs what the machine played.
# ─────────────────────────────────────────────────────────────────────────────

import pygame

# ASCII-safe gesture symbols (avoids Windows cp1252 encoding issues with unicode emoji)
_GESTURE_SYMBOL = {
    "rock":     "[ROCK]",
    "paper":    "[PAPER]",
    "scissors": "[SCISSORS]",
}

# Pill background color (semi-transparent dark)
_BG_COLOR    = (30, 30, 40)
_BG_ALPHA    = 190

_YOU_COLOR     = (120, 220, 255)   # soft blue for player
_MACHINE_COLOR = (255, 140, 80)    # soft orange for machine
_VS_COLOR      = (180, 180, 180)   # neutral grey for "vs"


def draw_gesture_comparison(screen: pygame.Surface, result: dict, font: pygame.font.Font):
    """
    Draw a comparison pill:
        YOU: [ROCK]   vs   Machine: [PAPER]
    Positioned below screen centre. Called from renderer.py during RESULT_DISPLAY.

    Args:
        screen  – the pygame display surface
        result  – state_machine.result dict (must have player_gesture & counter_move)
        font    – the small font from Renderer (font_small)
    """
    player_gesture  = result.get("player_gesture", "?")
    machine_gesture = result.get("counter_move",   "?")

    player_label  = f"YOU:  {_GESTURE_SYMBOL.get(player_gesture,  player_gesture.upper())}"
    vs_label      = "vs"
    machine_label = f"Machine:  {_GESTURE_SYMBOL.get(machine_gesture, machine_gesture.upper())}"

    surf_you     = font.render(player_label,  True, _YOU_COLOR)
    surf_vs      = font.render(vs_label,      True, _VS_COLOR)
    surf_machine = font.render(machine_label, True, _MACHINE_COLOR)

    padding  = 22
    spacing  = 30   # gap between each text segment
    total_w  = surf_you.get_width() + spacing + surf_vs.get_width() + spacing + surf_machine.get_width() + padding * 2
    total_h  = max(surf_you.get_height(), surf_vs.get_height(), surf_machine.get_height()) + padding

    sw, sh = screen.get_size()
    # Position: centred horizontally, 160px below screen centre
    bg_x = sw // 2 - total_w // 2
    bg_y = sh // 2 + 160

    # Draw semi-transparent pill background
    bg_surf = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
    bg_surf.fill((*_BG_COLOR, _BG_ALPHA))
    # Rounded-rect feel via drawing a rect (pygame-ce supports border_radius)
    pygame.draw.rect(bg_surf, (*_BG_COLOR, _BG_ALPHA), bg_surf.get_rect(), border_radius=16)
    screen.blit(bg_surf, (bg_x, bg_y))

    # Draw text segments side by side, vertically centred in the pill
    cy = bg_y + total_h // 2

    x = bg_x + padding
    screen.blit(surf_you,     (x, cy - surf_you.get_height() // 2))

    x += surf_you.get_width() + spacing
    screen.blit(surf_vs,      (x, cy - surf_vs.get_height() // 2))

    x += surf_vs.get_width() + spacing
    screen.blit(surf_machine, (x, cy - surf_machine.get_height() // 2))
