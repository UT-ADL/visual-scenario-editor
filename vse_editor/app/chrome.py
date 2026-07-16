"""Chrome: window/start/loading screens, overlays, render_ui and the top-bar
family, dropdown bookkeeping, fps meter (duck-typed module functions, first arg
`editor`). Moved verbatim from vse.py (step-42, Phase 7). Frame-order contract
unchanged: render_ui is the sole re-nuller/writer of the click hit-rects and
must run before render_mode_toggle / render_fps_meter / deferred dropdown draw.
"""

import math
import os
import time

from typing import List, Optional, Tuple

import pygame

from vse_editor.constants import MODE_BUTTON_TOOLTIPS, TOP_UI_BUTTON_TOOLTIPS
from vse_editor.ui.menus import PlacementMode

_TEXT_CACHE: dict = {}
_TEXT_CACHE_MAX = 1024

def _render_text_cached(font, text, antialias, color):
    """Memoized Font.render (opt-06). Chrome re-renders the same label
    strings every frame; the surfaces are blit-only here (never mutated), so
    cache them keyed by (font, text, antialias, color). Same argument shape
    as Font.render so call sites are drop-in. Bounded: cleared when full
    (dynamic strings like timers cycle through a small working set).
    """
    key = (id(font), text, antialias, color)
    surface = _TEXT_CACHE.get(key)
    if surface is None:
        if len(_TEXT_CACHE) >= _TEXT_CACHE_MAX:
            _TEXT_CACHE.clear()
        surface = font.render(text, antialias, color)
        _TEXT_CACHE[key] = surface
    return surface


def _show_error_overlay(editor, message: str, duration: float = 4.0) -> None:
    """Show a temporary error overlay message in editor mode."""
    now = time.time()
    editor._error_overlay_active = True
    editor._error_overlay_message = message
    editor._error_overlay_until = now + duration

def _render_error_overlay(editor) -> None:
    """Render the error overlay if active."""
    if not getattr(editor, "_error_overlay_active", False):
        return
    if time.time() > editor._error_overlay_until:
        editor._error_overlay_active = False
        return
    # Dark semi-transparent overlay
    overlay = pygame.Surface((editor.screen_width, editor.screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    editor.screen.blit(overlay, (0, 0))
    # Error message in red
    msg = editor._error_overlay_message
    text = _render_text_cached(editor.font, msg, True, editor.colors['error'])
    rect = text.get_rect(center=(editor.screen_width // 2, editor.screen_height // 2))
    editor.screen.blit(text, rect)

def _render_server_lost_overlay(editor) -> None:
    """Persistent overlay while the CARLA server is unreachable (fix-05).

    Local/managed profiles get a "Restart server & reload" button (click
    handled in events._handle_left_click); remote profiles show
    wait-and-reconnect text — the editor cannot reboot a remote machine, and
    the watchdog auto-reattaches when the server answers again.
    """
    if not getattr(editor, "_server_lost", False):
        editor._server_restart_button_rect = None
        return
    overlay = pygame.Surface((editor.screen_width, editor.screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    editor.screen.blit(overlay, (0, 0))

    cx = editor.screen_width // 2
    cy = editor.screen_height // 2
    title = _render_text_cached(editor.font, "CARLA server connection lost", True, editor.colors['error'])
    editor.screen.blit(title, title.get_rect(center=(cx, cy - 60)))

    # Never tell the user to wait — always offer the action (user decision
    # 2026-07-07): local = restart the server & reload the scene; remote =
    # restart VSE & reload the scenario against the same remote profile.
    remote = editor._using_remote_server()
    restarting = bool(getattr(editor, "restart_in_progress", False))
    if restarting:
        stage = getattr(editor, "loading_stage", "") or "Restarting..."
        info = _render_text_cached(editor.small_font, stage, True, (230, 230, 230))
    elif remote:
        info = _render_text_cached(editor.small_font, 
            "The remote CARLA server is unreachable. Restart VSE below and "
            "reload the scenario once the server is back.",
            True, (230, 230, 230))
    else:
        info = _render_text_cached(editor.small_font, 
            "The local CARLA server crashed. Restart it below to reload the scene.",
            True, (230, 230, 230))
    editor.screen.blit(info, info.get_rect(center=(cx, cy - 15)))

    # Button acknowledges the click: it greys out and reads "Restarting..."
    # while the restart flow runs (clicks are ignored in that state).
    if restarting:
        label = "Restarting..."
    elif remote:
        label = "Restart VSE & reload scenario"
    else:
        label = "Restart server & reload scene"
    text = _render_text_cached(editor.font, label, True, (255, 255, 255))
    pad_x, pad_y = 24, 12
    btn = pygame.Rect(0, 0, text.get_width() + 2 * pad_x, text.get_height() + 2 * pad_y)
    btn.center = (cx, cy + 50)
    fill = (90, 90, 90) if restarting else (178, 34, 34)
    pygame.draw.rect(editor.screen, fill, btn, border_radius=6)
    pygame.draw.rect(editor.screen, (255, 255, 255), btn, width=2, border_radius=6)
    editor.screen.blit(text, text.get_rect(center=btn.center))
    editor._server_restart_button_rect = None if restarting else btn

def toggle_maximize(editor):
    """Toggle between maximized and normal window size"""
    was_maximized = editor.maximized

    if not was_maximized:
        surface = pygame.display.get_surface()
        if surface is not None:
            current_width, current_height = surface.get_size()
            if current_width > 0 and current_height > 0:
                editor.windowed_width = current_width
                editor.windowed_height = current_height
        else:
            editor.windowed_width = editor.screen_width
            editor.windowed_height = editor.screen_height

    editor.maximized = not editor.maximized
    
    if editor.maximized:
        info = pygame.display.Info()
        # Leave some space for taskbar/title bar (subtract ~100 pixels)
        editor.screen_width = info.current_w
        editor.screen_height = max(100, info.current_h - 100)
        editor.screen = pygame.display.set_mode((editor.screen_width, editor.screen_height), pygame.RESIZABLE)
        print(f"Window maximized: {editor.screen_width}x{editor.screen_height}")
    else:
        editor.screen_width = editor.windowed_width
        editor.screen_height = editor.windowed_height
        editor.screen = pygame.display.set_mode((editor.screen_width, editor.screen_height), pygame.RESIZABLE)
        print(f"Window restored to: {editor.screen_width}x{editor.screen_height}")

    if editor.ui_manager:
        editor.ui_manager.set_window_resolution((editor.screen_width, editor.screen_height))
    
    # Update camera processor with new screen size
    if editor.camera_processor:
        editor.camera_processor.update_screen_size(editor.screen_width, editor.screen_height)

def render_start_screen(editor):
    """Render the initial start screen shown on normal launches."""
    editor.screen.fill(editor.colors['background'])
    mouse_pos = pygame.mouse.get_pos()
    center_x = editor.screen_width // 2

    title_surface = _render_text_cached(editor.font, "Visual Scenario Editor", True, editor.colors['text'])
    title_rect = title_surface.get_rect(center=(center_x, max(60, editor.screen_height // 4)))
    editor.screen.blit(title_surface, title_rect)

    subtitle_surface = _render_text_cached(editor.small_font, "Choose what to do:", True, (210, 210, 210))
    subtitle_rect = subtitle_surface.get_rect(center=(center_x, title_rect.bottom + 22))
    editor.screen.blit(subtitle_surface, subtitle_rect)

    button_w = min(460, max(260, editor.screen_width - 120))
    button_h = 54
    spacing = 14
    start_y = subtitle_rect.bottom + 26

    def draw_button(rect: pygame.Rect, label: str, *, enabled: bool = True) -> None:
        hovered = rect.collidepoint(mouse_pos)
        if not enabled:
            fill = (80, 80, 80)
            border = (120, 120, 120)
            text_color = (170, 170, 170)
        else:
            fill = (90, 140, 90) if hovered else (70, 120, 70)
            border = (255, 255, 255) if hovered else (210, 210, 210)
            text_color = (255, 255, 255)
        pygame.draw.rect(editor.screen, fill, rect, border_radius=8)
        pygame.draw.rect(editor.screen, border, rect, 2, border_radius=8)
        label_surface = _render_text_cached(editor.font, label, True, text_color)
        label_rect = label_surface.get_rect(center=rect.center)
        editor.screen.blit(label_surface, label_rect)

    scenario_rect = pygame.Rect(center_x - button_w // 2, start_y, button_w, button_h)
    map_rect = pygame.Rect(center_x - button_w // 2, scenario_rect.bottom + spacing, button_w, button_h)
    remote_rect = pygame.Rect(center_x - button_w // 2, map_rect.bottom + spacing, button_w, button_h)

    editor.start_screen_open_scenario_rect = scenario_rect
    editor.start_screen_open_map_rect = map_rect
    editor.start_screen_connect_remote_rect = remote_rect

    draw_button(scenario_rect, "Open Scenario", enabled=True)
    draw_button(map_rect, "Open Map", enabled=True)
    draw_button(remote_rect, "Connect Remote", enabled=True)

    recent_top = remote_rect.bottom + 30
    editor.start_screen_recent_scenario_rects = []
    recent_entries = getattr(editor, "recent_scenarios", []) or []
    if recent_entries:
        header = _render_text_cached(editor.small_font, "Recent scenarios:", True, (210, 210, 210))
        header_rect = header.get_rect(center=(center_x, recent_top))
        editor.screen.blit(header, header_rect)
        item_y = header_rect.bottom + 10
        item_h = 34
        for entry in recent_entries[:3]:
            path = entry.get('path')
            name = entry.get('name') or (os.path.basename(path) if path else "Unknown")
            map_name = entry.get('map_name') or "Unknown"
            exists = bool(entry.get('exists')) if 'exists' in entry else bool(path and os.path.isfile(path))
            label = f"{name}  ({map_name})"

            row_rect = pygame.Rect(center_x - button_w // 2, item_y, button_w, item_h)
            editor.start_screen_recent_scenario_rects.append((row_rect, path))

            hovered = row_rect.collidepoint(mouse_pos)
            if exists and path:
                fill = (70, 70, 110) if hovered else (55, 55, 90)
                border = (200, 200, 255)
                text_color = (235, 235, 255)
            else:
                fill = (60, 60, 60)
                border = (120, 120, 120)
                text_color = (160, 160, 160)

            pygame.draw.rect(editor.screen, fill, row_rect, border_radius=6)
            pygame.draw.rect(editor.screen, border, row_rect, 1, border_radius=6)
            label_surface = _render_text_cached(editor.small_font, label, True, text_color)
            label_rect = label_surface.get_rect(midleft=(row_rect.left + 10, row_rect.centery))
            editor.screen.blit(label_surface, label_rect)

            item_y = row_rect.bottom + 8

    carla_root = os.environ.get('CARLA_ROOT') or "(CARLA_ROOT not set)"
    carla_surface = _render_text_cached(editor.small_font, f"CARLA_ROOT: {carla_root}", True, (180, 180, 180))
    editor.screen.blit(carla_surface, (10, editor.screen_height - 28))

    if editor.startup_error:
        err_surface = _render_text_cached(editor.small_font, str(editor.startup_error), True, editor.colors['error'])
        err_rect = err_surface.get_rect(center=(center_x, min(editor.screen_height - 60, (recent_top + 20))))
        editor.screen.blit(err_surface, err_rect)

    hint_surface = _render_text_cached(editor.small_font, "ESC to quit", True, (210, 210, 210))
    hint_rect = hint_surface.get_rect(bottomright=(editor.screen_width - 10, editor.screen_height - 10))
    editor.screen.blit(hint_surface, hint_rect)

def render_loading_screen(editor):
    """Render loading screen"""
    editor.screen.fill(editor.colors['background'])
    
    # Loading title
    title_text = _render_text_cached(editor.font, "Visual Scenario Editor", True, editor.colors['text'])
    title_rect = title_text.get_rect(center=(editor.screen_width//2, editor.screen_height//2 - 100))
    editor.screen.blit(title_text, title_rect)
    
    # Loading status
    color = editor.colors['error'] if editor.startup_error else editor.colors['loading']
    status_text = _render_text_cached(editor.font, editor.loading_stage, True, color)
    status_rect = status_text.get_rect(center=(editor.screen_width//2, editor.screen_height//2))
    editor.screen.blit(status_text, status_rect)
    
    if not editor.startup_error:
        # Progress animation (simple dots)
        dots = "." * (int(time.time() * 2) % 4)
        dots_text = _render_text_cached(editor.font, dots, True, editor.colors['loading'])
        dots_rect = dots_text.get_rect(center=(editor.screen_width//2, editor.screen_height//2 + 30))
        editor.screen.blit(dots_text, dots_rect)
    else:
        # Show error details
        error_text = _render_text_cached(editor.small_font, "Check console for details", True, editor.colors['error'])
        error_rect = error_text.get_rect(center=(editor.screen_width//2, editor.screen_height//2 + 30))
        editor.screen.blit(error_text, error_rect)
    
    # Instructions
    instruction_text = _render_text_cached(editor.small_font, "ESC to cancel", True, editor.colors['text'])
    instruction_rect = instruction_text.get_rect(center=(editor.screen_width//2, editor.screen_height//2 + 80))
    editor.screen.blit(instruction_text, instruction_rect)

def _render_external_swap_overlay(editor):
    """Render a non-blocking overlay while an external ego is being adopted."""
    if not getattr(editor, "_external_swap_overlay", False):
        return
    overlay = pygame.Surface((editor.screen_width, editor.screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    editor.screen.blit(overlay, (0, 0))

    message = editor._external_swap_overlay_message or "Adopting external ego..."
    try:
        elapsed = time.time() - editor._external_swap_overlay_started_at if editor._external_swap_overlay_started_at else 0.0
    except Exception:
        elapsed = 0.0
    dots = "." * (int(elapsed * 2) % 4)
    msg_surface = _render_text_cached(editor.font, f"{message}{dots}", True, editor.colors['text'])
    msg_rect = msg_surface.get_rect(center=(editor.screen_width // 2, editor.screen_height // 2))
    editor.screen.blit(msg_surface, msg_rect)

    sub_text = "Please wait; external ego is connecting."
    sub_surface = _render_text_cached(editor.small_font, sub_text, True, editor.colors['text'])
    sub_rect = sub_surface.get_rect(center=(editor.screen_width // 2, editor.screen_height // 2 + 28))
    editor.screen.blit(sub_surface, sub_rect)
    if editor._external_swap_overlay_keep_until and time.time() >= editor._external_swap_overlay_keep_until:
        editor._clear_external_swap_overlay()

def _render_external_swap_overlay_immediate(editor):
    """Render and display the overlay once, useful right as swap work begins."""
    try:
        editor._render_external_swap_overlay()
        pygame.display.flip()
    except Exception:
        pass

def _truncate_text_to_width(editor, font, text: str, max_width: int) -> str:
    """Return text truncated with ASCII ellipsis to fit within max_width."""
    if not text or max_width <= 0:
        return ""
    if font.size(text)[0] <= max_width:
        return text
    ellipsis = "..."
    ellipsis_width = font.size(ellipsis)[0]
    if ellipsis_width >= max_width:
        return ellipsis
    target_width = max_width - ellipsis_width
    trimmed = text
    while trimmed and font.size(trimmed)[0] > target_width:
        trimmed = trimmed[:-1]
    return f"{trimmed}{ellipsis}" if trimmed else ellipsis

def _draw_keycap(editor, surface, label, left, center_y, font):
    """Draw a single keyboard 'keycap' box with a centered gold label. Returns its rect."""
    text_surf = _render_text_cached(font, label, True, (255, 215, 0))
    pad_x = 8
    h = 24
    w = max(h, text_surf.get_width() + 2 * pad_x)
    rect = pygame.Rect(int(left), int(center_y - h / 2), int(w), h)
    pygame.draw.rect(surface, (70, 70, 70), rect, border_radius=4)
    pygame.draw.rect(surface, (150, 150, 150), rect, 1, border_radius=4)
    # Lighter top edge for a subtle 3D keycap feel
    pygame.draw.line(surface, (115, 115, 115), (rect.left + 3, rect.top + 1), (rect.right - 3, rect.top + 1), 1)
    surface.blit(text_surf, text_surf.get_rect(center=rect.center))
    return rect

def _draw_mouse_glyph(editor, surface, left, center_y, highlight=None):
    """Draw a small mouse icon. `highlight` in {'left','right','middle','wheel'} lights that part."""
    w, h = 16, 24
    rect = pygame.Rect(int(left), int(center_y - h / 2), w, h)
    accent = (100, 200, 255)
    outline = (150, 150, 150)
    cx = rect.centerx
    mid_y = rect.top + h // 2
    pygame.draw.rect(surface, (60, 60, 60), rect, border_radius=7)
    if highlight == 'left':
        pygame.draw.rect(surface, accent, (rect.left + 1, rect.top + 1, cx - rect.left - 1, mid_y - rect.top - 1),
                         border_top_left_radius=6)
    elif highlight == 'right':
        pygame.draw.rect(surface, accent, (cx, rect.top + 1, rect.right - cx - 1, mid_y - rect.top - 1),
                         border_top_right_radius=6)
    pygame.draw.rect(surface, outline, rect, 1, border_radius=7)
    pygame.draw.line(surface, outline, (rect.left + 1, mid_y), (rect.right - 1, mid_y), 1)
    pygame.draw.line(surface, outline, (cx, rect.top + 2), (cx, mid_y), 1)
    wheel = pygame.Rect(0, 0, 4, 7)
    wheel.center = (cx, rect.top + 7)
    pygame.draw.rect(surface, accent if highlight in ('wheel', 'middle') else outline, wheel, border_radius=2)
    return rect

def _draw_key_combo(editor, surface, tokens, left, center_y, font):
    """Render a sequence of keycap/mouse/separator tokens left-to-right. Returns total width."""
    x = left
    gap = 4
    for tok in tokens:
        kind = tok[0]
        if kind == 'cap':
            x = editor._draw_keycap(surface, tok[1], x, center_y, font).right + gap
        elif kind == 'mouse':
            x = editor._draw_mouse_glyph(surface, x, center_y, tok[1] if len(tok) > 1 else None).right + gap
        else:  # 'plus' or 'sep' -- plain connective text
            label = '+' if kind == 'plus' else tok[1]
            ts = _render_text_cached(font, label, True, (200, 200, 200))
            surface.blit(ts, ts.get_rect(midleft=(x, center_y)))
            x += ts.get_width() + gap
    return x - left

def render_keyboard_help_overlay(editor, screen):
    """Draw the centered 'Keyboard & Mouse Shortcuts' modal. Geometry is recomputed each frame
    from the current window size so it stays centered when the window is resized."""
    def cap(s):
        return ('cap', s)
    def mouse(b):
        return ('mouse', b)
    def sep(s):
        return ('sep', s)
    plus = ('plus',)

    col_a = [
        ('header', 'CAMERA & VIEW'),
        ('row', [cap('WASD'), sep('/'), cap('Arrows')], 'Pan camera'),
        ('row', [mouse('right'), sep('/'), mouse('middle'), sep('drag')], 'Pan camera'),
        ('row', [mouse('wheel')], 'Zoom in / out'),
        ('row', [cap('Tab')], 'Toggle 3D orbit view (editor)'),
        ('row', [mouse('right'), sep('drag')], 'Orbit camera (3D view)'),
        ('row', [cap('C')], 'Cycle play camera (during a run)'),
        ('header', 'PLACEMENT & EDIT'),
        ('row', [cap('Ctrl'), plus, mouse('left')], 'Spawn actor (snap to lane)'),
        ('row', [cap('Ctrl'), plus, cap('Shift'), plus, mouse('left')], 'Spawn actor (free, no snap)'),
        ('row', [mouse('left'), plus, sep('drag')], 'Move object (snap to lane)'),
        ('row', [cap('Shift'), plus, mouse('left'), plus, sep('drag')], 'Move / place without snap'),
        ('row', [mouse('right'), sep('tap')], 'Cancel / deselect (like Esc)'),
        ('row', [cap('Ctrl'), plus, mouse('left')], 'Split waypoint (on a waypoint)'),
        ('row', [cap('Delete')], 'Delete selected'),
        ('header', 'FILE & HISTORY'),
        ('row', [cap('Ctrl'), plus, cap('S')], 'Save scenario'),
        ('row', [cap('Ctrl'), plus, cap('L')], 'Load scenario'),
        ('row', [cap('Ctrl'), plus, cap('Z')], 'Undo'),
        ('row', [cap('Ctrl'), plus, cap('Y')], 'Redo (or Ctrl+Shift+Z)'),
    ]
    col_b = [
        ('header', 'SELECTION'),
        ('row', [mouse('left'), plus, sep('drag on empty')], 'Box-select (group)'),
        ('row', [cap('Shift'), plus, mouse('left'), sep('tap'), sep('/'), sep('box')], 'Toggle in / out of selection'),
        ('row', [cap('Esc'), sep('/'), mouse('right'), sep('tap')], 'Deselect (release Shift first)'),
        ('header', 'OVERLAYS & TOGGLES'),
        ('row', [cap('O')], 'Toggle OpenDRIVE lanes'),
        ('row', [cap('T')], 'Toggle traffic-light stop lines'),
        ('row', [cap('`')], 'Hide / show all UI (key left of 1)'),
        ('header', 'PLAYBACK & EXIT'),
        ('row', [cap('Esc')], 'Stop run / cancel / clear selection'),
        ('row', [cap('Alt'), plus, cap('F4')], 'Exit'),
        ('header', 'MANUAL EGO (playback, no route)'),
        ('row', [cap('Up'), sep('/'), cap('Down')], 'Accelerate / brake (hold Down = reverse)'),
        ('row', [cap('Left'), sep('/'), cap('Right')], 'Steer'),
        ('row', [cap('Space')], 'Handbrake'),
        ('row', [cap('Q'), sep('/'), cap('P')], 'Reverse / autopilot'),
        ('header', 'HELP'),
        ('row', [cap('F1'), sep('/'), cap('H')], 'Toggle this panel'),
    ]

    sw, sh = editor.screen_width, editor.screen_height
    panel_w = min(900, sw - 60)
    panel_h = min(600, sh - 40)
    panel_x = (sw - panel_w) // 2
    panel_y = (sh - panel_h) // 2

    # Dim the scene behind the modal
    shade = pygame.Surface((sw, sh), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 160))
    screen.blit(shade, (0, 0))

    # Panel background (translucent dark) + border, matching the tooltip palette
    panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surface.fill((50, 50, 50, 235))
    screen.blit(panel_surface, (panel_x, panel_y))
    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    pygame.draw.rect(screen, (150, 150, 150), panel_rect, 2)

    # Title
    title = _render_text_cached(editor.font, "Keyboard & Mouse Shortcuts", True, (255, 255, 255))
    screen.blit(title, (panel_x + 22, panel_y + 16))

    # Close button (X) -- top-right of the panel
    cb = pygame.Rect(panel_x + panel_w - 36, panel_y + 14, 22, 22)
    editor.help_close_button_rect = cb
    mouse_pos = pygame.mouse.get_pos()
    cb_hover = cb.collidepoint(mouse_pos)
    pygame.draw.rect(screen, (90, 60, 60) if cb_hover else (70, 70, 70), cb, border_radius=4)
    pygame.draw.rect(screen, (180, 180, 180), cb, 1, border_radius=4)
    x_color = (255, 140, 140) if cb_hover else (220, 220, 220)
    pygame.draw.line(screen, x_color, (cb.left + 6, cb.top + 6), (cb.right - 6, cb.bottom - 6), 2)
    pygame.draw.line(screen, x_color, (cb.right - 6, cb.top + 6), (cb.left + 6, cb.bottom - 6), 2)

    # Two columns of categorized rows
    row_h = 26
    header_gap = 10
    key_col_w = 158
    columns = [(panel_x + 24, col_a), (panel_x + panel_w // 2 + 14, col_b)]
    body_top = panel_y + 56
    for col_x, items in columns:
        y = body_top
        first_header = True
        for item in items:
            if item[0] == 'header':
                if not first_header:
                    y += header_gap
                first_header = False
                hs = _render_text_cached(editor.small_font, item[1], True, (120, 200, 255))
                screen.blit(hs, (col_x, y + (row_h - hs.get_height()) // 2))
                y += row_h
            else:
                _, tokens, desc = item
                editor._draw_key_combo(screen, tokens, col_x + 4, y + row_h // 2, editor.small_font)
                ds = _render_text_cached(editor.small_font, desc, True, (210, 210, 210))
                screen.blit(ds, ds.get_rect(midleft=(col_x + key_col_w, y + row_h // 2)))
                y += row_h

    # Footer hint
    footer = _render_text_cached(editor.small_font, "Press  F1 / H  or click  ✕  to close", True, (180, 180, 180))
    screen.blit(footer, footer.get_rect(center=(panel_x + panel_w // 2, panel_y + panel_h - 20)))

def render_ui(editor):
    """Render UI elements overlay"""
    # Semi-transparent background for UI (smaller height)
    ui_height = editor.top_ui_height
    ui_surface = pygame.Surface((editor.screen_width, ui_height))
    ui_surface.set_alpha(180)
    ui_surface.fill((30, 30, 30))
    editor.screen.blit(ui_surface, (0, 0))

    mouse_pos = pygame.mouse.get_pos()
    editor.connection_button_rect = None
    editor.gpu_button_rect = None
    editor.resolution_button_rect = None
    editor.fps_button_rect = None
    editor._culling_button_rect = None
    editor.agent_button_rect = None
    editor.agent_button_enabled = False
    editor.resolution_option_rects = []
    editor.fps_option_rects = []
    editor._culling_option_rects = []
    editor.play_camera_button_rect = None
    editor.play_camera_option_rects = []
    editor.view3d_button_rect = None
    dropdown_draw_ops: List[Tuple[pygame.Rect, Tuple[int, int, int], pygame.Surface]] = []
    row_y = 10
    button_height = 30
    button_spacing = 10
    next_x = 10

    # Map selector button
    map_display_name = editor._get_map_display_name()
    map_label_full = map_display_name if map_display_name != "Unknown" else "Open Map"
    map_label_full_surface = _render_text_cached(editor.small_font, map_label_full, True, editor.colors['text'])
    map_button_max_width = max(80, min(260, editor.screen_width - 20))
    map_button_width = max(140, map_label_full_surface.get_width() + 24)
    map_button_width = min(map_button_width, map_button_max_width)
    map_label_display = editor._truncate_text_to_width(editor.small_font, map_label_full, map_button_width - 24)
    map_label_surface = _render_text_cached(editor.small_font, map_label_display, True, editor.colors['text'])
    map_button_rect = pygame.Rect(next_x, row_y, map_button_width, button_height)
    is_map_hovered = map_button_rect.collidepoint(mouse_pos)
    map_button_color = (80, 120, 170) if is_map_hovered else (60, 90, 140)

    pygame.draw.rect(editor.screen, map_button_color, map_button_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], map_button_rect, 1, border_radius=6)
    map_label_rect = map_label_surface.get_rect(center=map_button_rect.center)
    prev_clip = editor.screen.get_clip()
    editor.screen.set_clip(map_button_rect)
    editor.screen.blit(map_label_surface, map_label_rect)
    editor.screen.set_clip(prev_clip)
    editor.open_map_button_rect = map_button_rect
    if is_map_hovered and editor.tooltip_manager:
        map_tooltip = TOP_UI_BUTTON_TOOLTIPS.get('map', '')
        if map_label_display != map_label_full:
            map_tooltip = f"{map_tooltip}: {map_label_full}" if map_tooltip else map_label_full
        editor.tooltip_manager.register_hover("map_button", map_button_rect, map_tooltip)

    next_x = map_button_rect.right + button_spacing

    # Play Cam selector (playback camera: Top-Down / Chase / Cockpit)
    play_camera_labels = {"topdown": "Top-Down", "chase": "Chase", "cockpit": "Cockpit"}
    play_camera_label = "Play Cam: " + play_camera_labels.get(editor.play_camera_mode, "Chase")
    play_camera_surface = _render_text_cached(editor.small_font, play_camera_label, True, editor.colors['text'])
    play_camera_button_width = max(150, play_camera_surface.get_width() + 24)
    play_camera_rect = pygame.Rect(next_x, row_y, play_camera_button_width, button_height)
    is_play_camera_hovered = play_camera_rect.collidepoint(mouse_pos)
    play_camera_color = (140, 110, 190) if is_play_camera_hovered else (110, 80, 160)

    pygame.draw.rect(editor.screen, play_camera_color, play_camera_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], play_camera_rect, 1, border_radius=6)
    play_camera_label_rect = play_camera_surface.get_rect(center=play_camera_rect.center)
    editor.screen.blit(play_camera_surface, play_camera_label_rect)
    editor.play_camera_button_rect = play_camera_rect
    if is_play_camera_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("play_camera_button", play_camera_rect, TOP_UI_BUTTON_TOOLTIPS.get('play_camera', ''))
    next_x = play_camera_rect.right + button_spacing

    if editor.play_camera_menu_open:
        option_height = button_height
        menu_top = play_camera_rect.bottom + 2
        for idx, opt_mode in enumerate(("topdown", "chase", "cockpit")):
            option_rect = pygame.Rect(
                play_camera_rect.left,
                menu_top + idx * option_height,
                play_camera_button_width,
                option_height,
            )
            is_selected = opt_mode == editor.play_camera_mode
            is_hovered = option_rect.collidepoint(mouse_pos)
            if is_selected:
                draw_color = (150, 120, 200)
            else:
                draw_color = (130, 100, 180) if is_hovered else (95, 70, 140)
            option_surface = _render_text_cached(editor.small_font, play_camera_labels[opt_mode], True, editor.colors['text'])
            editor.play_camera_option_rects.append((option_rect, opt_mode))
            dropdown_draw_ops.append((option_rect, draw_color, option_surface))

    editor.gpu_button_rect = None

    # Stream resolution selector (always visible)
    res_width, res_height = editor.stream_resolution
    resolution_label = f"{res_width}x{res_height}" if editor.camera_stream_enabled else "No Camera"
    resolution_surface = _render_text_cached(editor.small_font, resolution_label, True, editor.colors['text'])
    resolution_button_width = max(150, resolution_surface.get_width() + 24)
    resolution_rect = pygame.Rect(next_x, row_y, resolution_button_width, button_height)
    is_resolution_hovered = resolution_rect.collidepoint(mouse_pos)
    if editor._using_remote_server():
        resolution_color = (90, 120, 180) if is_resolution_hovered else (70, 95, 150)
    else:
        resolution_color = (110, 110, 110) if is_resolution_hovered else (85, 85, 85)

    pygame.draw.rect(editor.screen, resolution_color, resolution_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], resolution_rect, 1, border_radius=6)
    resolution_label_rect = resolution_surface.get_rect(center=resolution_rect.center)
    editor.screen.blit(resolution_surface, resolution_label_rect)
    editor.resolution_button_rect = resolution_rect
    if is_resolution_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("resolution_button", resolution_rect, TOP_UI_BUTTON_TOOLTIPS.get('resolution', ''))
    next_x = resolution_rect.right + button_spacing

    if editor.resolution_menu_open:
        option_height = button_height
        menu_top = resolution_rect.bottom + 2
        for idx, option in enumerate(editor.remote_resolution_options):
            option_rect = pygame.Rect(
                resolution_rect.left,
                menu_top + idx * option_height,
                resolution_button_width,
                option_height,
            )
            is_selected = (
                (option is None and not editor.camera_stream_enabled)
                or (
                    option is not None
                    and editor.camera_stream_enabled
                    and option == editor.stream_resolution
                )
            )
            is_hovered = option_rect.collidepoint(mouse_pos)
            if editor._using_remote_server():
                base_color = (90, 120, 180) if is_hovered else (60, 85, 130)
                selected_color = (120, 150, 200)
            else:
                base_color = (110, 110, 110) if is_hovered else (70, 70, 70)
                selected_color = (140, 140, 140)
            draw_color = selected_color if is_selected else base_color
            option_label = "No Camera" if option is None else f"{option[0]}x{option[1]}"
            option_surface = _render_text_cached(editor.small_font, option_label, True, editor.colors['text'])
            editor.resolution_option_rects.append((option_rect, option))
            dropdown_draw_ops.append((option_rect, draw_color, option_surface))

    # Culling distance selector (right after Resolution). Editor-only: dimmed during play or while
    # an external bridge/large map owns the sim, where apply_settings() is unsafe.
    cull_editable = (not editor.scenario_running) and editor._culling_apply_safe()
    if not cull_editable:
        editor._culling_menu_open = False
    cull_label = "Cull: Off" if editor.culling_distance_m <= 0 else f"Cull: {int(editor.culling_distance_m)} m"
    cull_surface = _render_text_cached(editor.small_font, cull_label, True, editor.colors['text'])
    cull_button_width = max(120, cull_surface.get_width() + 24)
    cull_rect = pygame.Rect(next_x, row_y, cull_button_width, button_height)
    is_cull_hovered = cull_editable and cull_rect.collidepoint(mouse_pos)
    if not cull_editable:
        cull_color = (60, 60, 60)
    elif is_cull_hovered:
        cull_color = (110, 110, 110)
    else:
        cull_color = (85, 85, 85)
    pygame.draw.rect(editor.screen, cull_color, cull_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], cull_rect, 1, border_radius=6)
    cull_label_rect = cull_surface.get_rect(center=cull_rect.center)
    editor.screen.blit(cull_surface, cull_label_rect)
    editor._culling_button_rect = cull_rect
    if is_cull_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("culling_button", cull_rect, TOP_UI_BUTTON_TOOLTIPS.get('culling', ''))
    next_x = cull_rect.right + button_spacing

    if cull_editable and editor._culling_menu_open:
        option_height = button_height
        menu_top = cull_rect.bottom + 2
        for idx, preset in enumerate(editor.CULLING_PRESETS):
            option_rect = pygame.Rect(
                cull_rect.left,
                menu_top + idx * option_height,
                cull_button_width,
                option_height,
            )
            is_selected = abs(preset - editor.culling_distance_m) < 1e-6
            is_hovered = option_rect.collidepoint(mouse_pos)
            base_color = (110, 110, 110) if is_hovered else (70, 70, 70)
            selected_color = (140, 140, 140)
            draw_color = selected_color if is_selected else base_color
            option_label = "Off" if preset <= 0 else f"{int(preset)} m"
            option_surface = _render_text_cached(editor.small_font, option_label, True, editor.colors['text'])
            editor._culling_option_rects.append((option_rect, preset))
            dropdown_draw_ops.append((option_rect, draw_color, option_surface))

    # FPS selector (remote only)
    if editor._using_remote_server():
        fps_label = f"{editor.stream_fps} FPS"
        fps_surface = _render_text_cached(editor.small_font, fps_label, True, editor.colors['text'])
        fps_button_width = max(120, fps_surface.get_width() + 24)
        fps_rect = pygame.Rect(next_x, row_y, fps_button_width, button_height)
        is_fps_hovered = fps_rect.collidepoint(mouse_pos)
        fps_color = (110, 110, 110) if is_fps_hovered else (80, 80, 80)

        pygame.draw.rect(editor.screen, fps_color, fps_rect, border_radius=6)
        pygame.draw.rect(editor.screen, editor.colors['text'], fps_rect, 1, border_radius=6)
        fps_label_rect = fps_surface.get_rect(center=fps_rect.center)
        editor.screen.blit(fps_surface, fps_label_rect)
        editor.fps_button_rect = fps_rect
        if is_fps_hovered and editor.tooltip_manager:
            editor.tooltip_manager.register_hover("fps_button", fps_rect, TOP_UI_BUTTON_TOOLTIPS.get('fps', ''))
        next_x = fps_rect.right + button_spacing

        if editor.fps_menu_open:
            option_height = button_height
            menu_top = fps_rect.bottom + 2
            for idx in range(editor.remote_fps_min, editor.remote_fps_max + 1):
                option_rect = pygame.Rect(
                    fps_rect.left,
                    menu_top + (idx - editor.remote_fps_min) * option_height,
                    fps_button_width,
                    option_height,
                )
                is_selected = idx == editor.stream_fps
                is_hovered = option_rect.collidepoint(mouse_pos)
                base_color = (110, 110, 110) if is_hovered else (70, 70, 70)
                selected_color = (140, 140, 140)
                draw_color = selected_color if is_selected else base_color
                option_surface = _render_text_cached(editor.small_font, f"{idx} FPS", True, editor.colors['text'])
                editor.fps_option_rects.append((option_rect, idx))
                dropdown_draw_ops.append((option_rect, draw_color, option_surface))
    else:
        editor.fps_button_rect = None
        editor.fps_option_rects = []
        editor.fps_menu_open = False

    # Connection toggle button (Local/Remote)
    is_remote = bool(editor.connection_profile and editor.connection_profile.is_remote)
    connection_label = "Remote" if is_remote else "Local"
    connection_label_surface = _render_text_cached(editor.small_font, connection_label, True, editor.colors['text'])
    connection_button_width = max(110, connection_label_surface.get_width() + 24)
    connection_rect = pygame.Rect(next_x, row_y, connection_button_width, button_height)
    is_connection_hovered = connection_rect.collidepoint(mouse_pos)
    if is_remote:
        connection_color = (150, 110, 60) if is_connection_hovered else (125, 90, 45)
    else:
        connection_color = (70, 140, 80) if is_connection_hovered else (55, 110, 60)

    pygame.draw.rect(editor.screen, connection_color, connection_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], connection_rect, 1, border_radius=6)
    connection_label_rect = connection_label_surface.get_rect(center=connection_rect.center)
    editor.screen.blit(connection_label_surface, connection_label_rect)
    editor.connection_button_rect = connection_rect
    if is_connection_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("connection_button", connection_rect, TOP_UI_BUTTON_TOOLTIPS.get('connection', ''))

    next_x = connection_rect.right + button_spacing

    # Drive Clock toggle (remote server only, last in the row).
    # On managed/local servers VSE drives sync/async automatically, so the
    # manual control is hidden; it only appears for remote servers VSE does
    # not own, where switching modes is the operator's call.
    if editor._using_remote_server():
        manual_label = "Drive Clock: ON" if editor.manual_tick_enabled else "Drive Clock: OFF"
        manual_label_surface = _render_text_cached(editor.small_font, manual_label, True, editor.colors['text'])
        manual_button_width = max(170, manual_label_surface.get_width() + 24)
        manual_rect = pygame.Rect(next_x, row_y, manual_button_width, button_height)
        is_manual_hovered = manual_rect.collidepoint(mouse_pos)
        if editor.manual_tick_enabled:
            manual_color = (70, 160, 80) if is_manual_hovered else (50, 130, 60)
        else:
            manual_color = (150, 100, 40) if is_manual_hovered else (120, 80, 30)

        pygame.draw.rect(editor.screen, manual_color, manual_rect, border_radius=6)
        pygame.draw.rect(editor.screen, editor.colors['text'], manual_rect, 1, border_radius=6)
        manual_label_rect = manual_label_surface.get_rect(center=manual_rect.center)
        editor.screen.blit(manual_label_surface, manual_label_rect)
        if is_manual_hovered and editor.tooltip_manager:
            editor.tooltip_manager.register_hover("manual_tick_button", manual_rect, TOP_UI_BUTTON_TOOLTIPS.get('manual_tick', ''))

        if editor.manual_tick_required and editor.manual_tick_recommendation and not editor.manual_tick_enabled:
            tip_text = "Enable Drive Clock for live RGB"
            tip_surface = _render_text_cached(editor.small_font, tip_text, True, (230, 200, 120))
            tip_rect = tip_surface.get_rect()
            tip_rect.left = manual_rect.right + 8
            tip_rect.centery = manual_rect.centery
            editor.screen.blit(tip_surface, tip_rect)

        editor.manual_tick_button_rect = manual_rect
        next_x = manual_rect.right + button_spacing
    else:
        # Hidden on local/managed servers; clear the rect so clicks are inert.
        editor.manual_tick_button_rect = None

    capture_rects: List[pygame.Rect] = []
    if editor.play_camera_menu_open:
        if editor.play_camera_button_rect:
            capture_rects.append(editor.play_camera_button_rect)
        capture_rects.extend(rect for rect, _ in editor.play_camera_option_rects)
    if editor.resolution_menu_open:
        if editor.resolution_button_rect:
            capture_rects.append(editor.resolution_button_rect)
        capture_rects.extend(rect for rect, _ in editor.resolution_option_rects)
    if editor.fps_menu_open:
        if editor.fps_button_rect:
            capture_rects.append(editor.fps_button_rect)
        capture_rects.extend(rect for rect, _ in editor.fps_option_rects)
    if getattr(editor, "agent_dropdown_open", False):
        if getattr(editor, "agent_button_rect", None):
            capture_rects.append(editor.agent_button_rect)
        capture_rects.extend(rect for rect, _ in getattr(editor, "_agent_dropdown_item_rects", []))
    if getattr(editor, "agent_behavior_menu_open", False):
        if getattr(editor, "agent_button_rect", None):
            capture_rects.append(editor.agent_button_rect)
        capture_rects.extend(getattr(editor, "_agent_behavior_item_rects", {}).values())
    if getattr(editor, "npc_dropdown_open", False):
        if getattr(editor, "npc_button_rect", None):
            capture_rects.append(editor.npc_button_rect)
        capture_rects.extend(rect for rect, _ in getattr(editor, "_npc_dropdown_item_rects", []))
    editor._dropdown_capture_rects = capture_rects
    editor._dropdown_mouse_captured = any(rect.collidepoint(mouse_pos) for rect in capture_rects)

    # Scenario button below the map selector
    scenario_label_full = editor.current_scenario_name if editor.current_scenario_name else "Scenario"
    scenario_button_width = map_button_width
    scenario_label_display = editor._truncate_text_to_width(editor.small_font, scenario_label_full, scenario_button_width - 24)
    scenario_label_surface = _render_text_cached(editor.small_font, scenario_label_display, True, editor.colors['text'])
    scenario_button_rect = pygame.Rect(10, row_y + button_height + 10, scenario_button_width, button_height)
    is_scenario_hovered = scenario_button_rect.collidepoint(mouse_pos)
    if editor._dropdown_mouse_captured and not scenario_button_rect.collidepoint(mouse_pos):
        is_scenario_hovered = False
    scenario_button_color = (80, 120, 170) if is_scenario_hovered else (60, 90, 140)

    pygame.draw.rect(editor.screen, scenario_button_color, scenario_button_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], scenario_button_rect, 1, border_radius=6)
    scenario_label_rect = scenario_label_surface.get_rect(center=scenario_button_rect.center)
    prev_clip = editor.screen.get_clip()
    editor.screen.set_clip(scenario_button_rect)
    editor.screen.blit(scenario_label_surface, scenario_label_rect)
    editor.screen.set_clip(prev_clip)
    editor.open_scenario_button_rect = scenario_button_rect
    if is_scenario_hovered and editor.tooltip_manager:
        scenario_tooltip = TOP_UI_BUTTON_TOOLTIPS.get('scenario', '')
        if scenario_label_display != scenario_label_full and scenario_label_full != "Scenario":
            scenario_tooltip = f"{scenario_tooltip}: {scenario_label_full}" if scenario_tooltip else scenario_label_full
        editor.tooltip_manager.register_hover("scenario_button", scenario_button_rect, scenario_tooltip)

    # 2. Play/Stop button (top-right corner, next to undo/redo)
    play_button_width = 80
    play_button_height = 30
    play_button_x = editor.screen_width - play_button_width - 10
    play_button_y = 10
    play_button_rect = pygame.Rect(play_button_x, play_button_y, play_button_width, play_button_height)
    is_play_hovered = play_button_rect.collidepoint(mouse_pos)
    scenario_ready = editor._scenario_has_playable_content()
    scenario_active = bool(editor.scenario_running or (editor.scenario_process and editor.scenario_process.poll() is None))

    # Yellow "Agent:" label + agent mode dropdown (to the left of Play/Stop)
    agent_mode = getattr(editor, "agent_mode", "autopilot")
    agent_behavior = getattr(editor, "agent_behavior", "normal")
    agent_mode_labels = {"autopilot": "Autopilot", "human": "Human", "custom": "Custom Agent"}
    agent_mode_label = agent_mode_labels.get(agent_mode, "Autopilot")
    if agent_mode == "autopilot":
        agent_mode_label += f" ({agent_behavior.capitalize()})"
    elif agent_mode == "custom" and getattr(editor, "agent_path", None):
        agent_mode_label = os.path.basename(editor.agent_path)
    agent_label_surface = _render_text_cached(editor.small_font, agent_mode_label, True, editor.colors['text'])
    agent_button_width = max(140, agent_label_surface.get_width() + 24)
    agent_button_height = play_button_height

    # Yellow "Agent:" prefix label
    agent_prefix_surface = _render_text_cached(editor.small_font, "Ego Agent:", True, (255, 255, 0))
    agent_prefix_width = agent_prefix_surface.get_width() + 6

    # "..." browse button width (shown only in custom mode)
    browse_button_width = 28
    show_browse = (agent_mode == "custom")
    browse_extra = (browse_button_width + 2) if show_browse else 0

    agent_button_x = play_button_x - agent_button_width - button_spacing - browse_extra
    agent_prefix_x = agent_button_x - agent_prefix_width
    agent_button_rect = pygame.Rect(agent_button_x, play_button_y, agent_button_width, agent_button_height)
    is_agent_hovered = agent_button_rect.collidepoint(mouse_pos)
    if editor._dropdown_mouse_captured and not agent_button_rect.collidepoint(mouse_pos):
        is_agent_hovered = False

    external_ego_connected = getattr(editor, '_external_ego_present_last_check', False)
    agent_button_enabled = not scenario_active and not external_ego_connected
    if agent_button_enabled:
        agent_button_color = (80, 120, 170) if is_agent_hovered else (60, 90, 140)
    else:
        agent_button_color = (80, 80, 80)

    # Draw yellow "Agent:" label
    editor.screen.blit(agent_prefix_surface, agent_prefix_surface.get_rect(
        midright=(agent_button_rect.left - 4, agent_button_rect.centery)))

    # Draw button
    pygame.draw.rect(editor.screen, agent_button_color, agent_button_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], agent_button_rect, 1, border_radius=6)
    agent_label_rect = agent_label_surface.get_rect(center=agent_button_rect.center)
    editor.screen.blit(agent_label_surface, agent_label_rect)

    editor.agent_button_rect = agent_button_rect
    editor.agent_button_enabled = bool(agent_button_enabled)
    if is_agent_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("agent_button", agent_button_rect, TOP_UI_BUTTON_TOOLTIPS.get('agent', ''))

    # "..." browse button (visible only in Custom Agent mode)
    if show_browse:
        browse_x = agent_button_rect.right + 2
        browse_rect = pygame.Rect(browse_x, play_button_y, browse_button_width, agent_button_height)
        is_browse_hovered = browse_rect.collidepoint(mouse_pos)
        agent_browse_enabled = not scenario_active
        if agent_browse_enabled:
            browse_color = (80, 120, 170) if is_browse_hovered else (60, 90, 140)
        else:
            browse_color = (80, 80, 80)
        pygame.draw.rect(editor.screen, browse_color, browse_rect, border_radius=6)
        pygame.draw.rect(editor.screen, editor.colors['text'], browse_rect, 1, border_radius=6)
        browse_label = _render_text_cached(editor.small_font, "...", True, editor.colors['text'])
        editor.screen.blit(browse_label, browse_label.get_rect(center=browse_rect.center))
        editor.agent_browse_button_rect = browse_rect
        if is_browse_hovered and editor.tooltip_manager:
            editor.tooltip_manager.register_hover("agent_browse_button", browse_rect, "Change custom agent script")
    else:
        editor.agent_browse_button_rect = None

    # ---- NPC Driving Mode dropdown (to the left of Agent) ----
    npc_mode_raw = editor.vehicle_control_mode
    npc_mode_display = "Simulated" if npc_mode_raw != "velocity" else "Scripted"
    npc_label_surface = _render_text_cached(editor.small_font, npc_mode_display, True, editor.colors['text'])
    npc_button_width = max(130, npc_label_surface.get_width() + 24)
    npc_button_height = play_button_height

    npc_prefix_surface = _render_text_cached(editor.small_font, "NPC Agent:", True, (255, 255, 0))
    npc_prefix_width = npc_prefix_surface.get_width() + 6

    npc_button_x = agent_prefix_x - npc_button_width - button_spacing
    npc_prefix_x = npc_button_x - npc_prefix_width
    npc_button_rect = pygame.Rect(npc_button_x, play_button_y, npc_button_width, npc_button_height)
    is_npc_hovered = npc_button_rect.collidepoint(mouse_pos)
    if editor._dropdown_mouse_captured and not npc_button_rect.collidepoint(mouse_pos):
        is_npc_hovered = False

    npc_button_enabled = not scenario_active
    if npc_button_enabled:
        npc_button_color = (80, 120, 170) if is_npc_hovered else (60, 90, 140)
    else:
        npc_button_color = (80, 80, 80)

    editor.screen.blit(npc_prefix_surface, npc_prefix_surface.get_rect(
        midright=(npc_button_rect.left - 4, npc_button_rect.centery)))

    pygame.draw.rect(editor.screen, npc_button_color, npc_button_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], npc_button_rect, 1, border_radius=6)
    npc_label_rect = npc_label_surface.get_rect(center=npc_button_rect.center)
    editor.screen.blit(npc_label_surface, npc_label_rect)

    editor.npc_button_rect = npc_button_rect
    editor.npc_button_enabled = bool(npc_button_enabled)
    if is_npc_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("npc_mode_button", npc_button_rect, TOP_UI_BUTTON_TOOLTIPS.get('npc_mode', ''))

    # NPC Driving Mode dropdown items (deferred via dropdown_draw_ops)
    npc_dropdown_open = getattr(editor, "npc_dropdown_open", False)
    editor._npc_dropdown_item_rects = []
    if npc_dropdown_open and npc_button_enabled:
        npc_dropdown_items = [
            ("basic_agent", "Simulated", "AI agent drives with realistic steering, throttle and braking"),
            ("velocity", "Scripted", "Follows waypoints at exact speed, bypassing vehicle physics"),
        ]
        item_height = play_button_height
        dropdown_width = npc_button_width
        dropdown_x = npc_button_rect.left
        dropdown_y = npc_button_rect.bottom + 2
        for idx, (mode_key, mode_text, mode_tooltip) in enumerate(npc_dropdown_items):
            item_rect = pygame.Rect(dropdown_x, dropdown_y + idx * item_height, dropdown_width, item_height)
            is_item_hovered = item_rect.collidepoint(mouse_pos)
            if is_item_hovered and editor.tooltip_manager:
                editor.tooltip_manager.register_hover(f"npc_dropdown_{mode_key}", item_rect, mode_tooltip)
            is_selected = (npc_mode_raw == mode_key)
            if is_selected:
                draw_color = (80, 140, 80)
            elif is_item_hovered:
                draw_color = (90, 120, 180)
            else:
                draw_color = (60, 85, 130)
            option_surface = _render_text_cached(editor.small_font, mode_text, True, editor.colors['text'])
            editor._npc_dropdown_item_rects.append((item_rect, mode_key))
            dropdown_draw_ops.append((item_rect, draw_color, option_surface))

    # Agent dropdown menu items (deferred via dropdown_draw_ops for overlay rendering)
    agent_dropdown_open = getattr(editor, "agent_dropdown_open", False)
    agent_behavior_menu_open = getattr(editor, "agent_behavior_menu_open", False)
    editor._agent_dropdown_item_rects = []
    editor._agent_behavior_item_rects = {}
    if agent_dropdown_open and agent_button_enabled:
        dropdown_items = [
            ("autopilot", "CARLA Autopilot", "CARLA BehaviorAgent drives the ego vehicle autonomously"),
            ("human", "Human Control", "Drive the ego vehicle manually with keyboard or steering wheel"),
            ("custom", "Custom Agent", "Use an external agent script to control the ego vehicle"),
        ]
        item_height = play_button_height
        dropdown_width = agent_button_width
        dropdown_x = agent_button_rect.left
        dropdown_y = agent_button_rect.bottom + 2
        for idx, (mode_key, mode_text, mode_tooltip) in enumerate(dropdown_items):
            item_rect = pygame.Rect(dropdown_x, dropdown_y + idx * item_height, dropdown_width, item_height)
            is_item_hovered = item_rect.collidepoint(mouse_pos)
            if is_item_hovered and editor.tooltip_manager:
                editor.tooltip_manager.register_hover(f"agent_dropdown_{mode_key}", item_rect, mode_tooltip)
            is_selected = (agent_mode == mode_key)
            if is_selected:
                draw_color = (120, 150, 200)
            elif is_item_hovered:
                draw_color = (90, 120, 180)
            else:
                draw_color = (60, 85, 130)
            option_surface = _render_text_cached(editor.small_font, mode_text, True, editor.colors['text'])
            editor._agent_dropdown_item_rects.append((item_rect, mode_key))
            dropdown_draw_ops.append((item_rect, draw_color, option_surface))

    # Agent behavior menu (shown after selecting Autopilot, replaces the dropdown)
    elif agent_behavior_menu_open and agent_button_enabled:
        behavior_options = [("cautious", "Cautious"), ("normal", "Normal"), ("aggressive", "Aggressive")]
        item_height = play_button_height
        dropdown_width = agent_button_width
        dropdown_x = agent_button_rect.left
        dropdown_y = agent_button_rect.bottom + 2
        for idx, (bkey, blabel) in enumerate(behavior_options):
            item_rect = pygame.Rect(dropdown_x, dropdown_y + idx * item_height, dropdown_width, item_height)
            is_item_hovered = item_rect.collidepoint(mouse_pos)
            is_selected = (agent_behavior == bkey)
            if is_selected:
                draw_color = (80, 140, 80)
            elif is_item_hovered:
                draw_color = (90, 120, 180)
            else:
                draw_color = (60, 85, 130)
            option_surface = _render_text_cached(editor.small_font, blabel, True, editor.colors['text'])
            editor._agent_behavior_item_rects[bkey] = item_rect
            dropdown_draw_ops.append((item_rect, draw_color, option_surface))


    # A just-finished run can still be tearing down in the background (ROS-agent subprocess
    # cancel/terminate/kill happens after results appear). The runner thread stays alive until
    # that teardown completes, so gate Play on it: don't let a new run start until it's gone.
    _runner = getattr(editor, "_mini_runner", None)
    _runner_thread = getattr(_runner, "_thread", None)
    runner_busy = bool(_runner and not scenario_active and _runner_thread and _runner_thread.is_alive())

    # Determine button state and appearance
    if scenario_active:
        play_button_text = "Stop"
        play_button_color = (170, 80, 80) if is_play_hovered else (140, 60, 60)  # Red for stop
        play_button_enabled = True
    elif runner_busy:
        # Previous run still shutting down (subprocess teardown); block Play until done.
        play_button_text = "Finishing…"
        play_button_color = (80, 80, 80)  # Gray, disabled
        play_button_enabled = False
    elif scenario_ready:
        custom_blocked = (agent_mode == "custom" and not getattr(editor, '_external_ego_present_last_check', False))
        if custom_blocked:
            play_button_text = "Play"
            play_button_color = (80, 80, 80)
            play_button_enabled = False
        else:
            play_button_text = "Play"
            play_button_color = (80, 170, 80) if is_play_hovered else (60, 140, 60)  # Green for play
            play_button_enabled = True
    else:
        play_button_text = "Play"
        play_button_color = (80, 80, 80)  # Gray when disabled
        play_button_enabled = False

    pygame.draw.rect(editor.screen, play_button_color, play_button_rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], play_button_rect, 1, border_radius=6)

    play_label_surface = _render_text_cached(editor.small_font, play_button_text, True, editor.colors['text'])
    play_label_rect = play_label_surface.get_rect(center=play_button_rect.center)
    editor.screen.blit(play_label_surface, play_label_rect)

    editor.play_button_rect = play_button_rect
    editor.play_button_enabled = play_button_enabled
    if is_play_hovered and editor.tooltip_manager:
        custom_blocked = (agent_mode == "custom" and not getattr(editor, '_external_ego_present_last_check', False))
        if custom_blocked and not scenario_active:
            editor.tooltip_manager.register_hover("play_button", play_button_rect,
                "No external ego \u2014 select another agent", text_color=(230, 120, 120))
        else:
            tooltip_key = 'stop' if scenario_active else 'play'
            editor.tooltip_manager.register_hover("play_button", play_button_rect, TOP_UI_BUTTON_TOOLTIPS.get(tooltip_key, ''))


    # 3. Help hint (centered slightly higher for spacing). The full controls/shortcuts
    #    list lives in the F1/H keyboard-help overlay; the "?" button (below) opens it too.
    controls = "Press F1 or H for help"
    controls_surface = _render_text_cached(editor.small_font, controls, True, (200, 200, 200))
    controls_center_y = ui_height - 35
    controls_rect = controls_surface.get_rect(center=(editor.screen_width // 2, controls_center_y))
    editor.screen.blit(controls_surface, controls_rect)

    # "?" help button just right of the help hint -- opens the shortcuts panel.
    # Lives in render_ui, so it auto-hides with the rest of the UI under hide_all_ui.
    help_btn_d = 22
    help_btn_cx = min(controls_rect.right + 18, editor.screen_width - help_btn_d // 2 - 6)
    help_btn_rect = pygame.Rect(0, 0, help_btn_d, help_btn_d)
    help_btn_rect.center = (help_btn_cx, controls_center_y)
    editor.help_button_rect = help_btn_rect
    hb_hover = help_btn_rect.collidepoint(pygame.mouse.get_pos())
    pygame.draw.circle(editor.screen, (70, 70, 70), help_btn_rect.center, help_btn_d // 2)
    pygame.draw.circle(editor.screen, (255, 215, 0) if hb_hover else (150, 150, 150),
                       help_btn_rect.center, help_btn_d // 2, 1)
    q_surf = _render_text_cached(editor.small_font, "?", True, (255, 215, 0) if hb_hover else (210, 210, 210))
    editor.screen.blit(q_surf, q_surf.get_rect(center=help_btn_rect.center))
    if hb_hover and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("help_button", help_btn_rect, "Keyboard shortcuts (F1 / H)")

    manual_message = None
    if editor.camera_processor:
        if editor.camera_processor.manual_control_enabled:
            state = getattr(editor.camera_processor, 'manual_control_state', {})
            autopilot_hint = ""
            if state.get('autopilot_enabled') or state.get('autopilot_active'):
                autopilot_hint = " (autopilot on)"
            manual_message = "Manual control: Up accel, Down brake (hold Down at stop for auto reverse) | Space hand brake | Q toggle reverse | P toggle autopilot" + autopilot_hint
        elif editor.scenario_running and editor.camera_processor.manual_control_pending:
            manual_message = "Manual ego control: waiting for ego vehicle to spawn..."

    manual_surface = None
    manual_rect = None
    if manual_message:
        manual_color = (200, 230, 150) if editor.camera_processor and editor.camera_processor.manual_control_enabled else (180, 180, 180)
        manual_surface = _render_text_cached(editor.small_font, manual_message, True, manual_color)
        manual_rect = manual_surface.get_rect(center=(editor.screen_width // 2, controls_rect.top - 18))
    if manual_surface:
        editor.screen.blit(manual_surface, manual_rect)
    elif (getattr(editor, "_status_hint_text", None)
            and time.time() < getattr(editor, "_status_hint_until", 0.0)):
        # Transient status hint (blocked-action feedback); shares the manual_message
        # slot, which is only populated during playback, so the two never collide.
        hint_surface = _render_text_cached(editor.small_font, editor._status_hint_text, True, (240, 200, 120))
        hint_rect = hint_surface.get_rect(center=(editor.screen_width // 2, controls_rect.top - 18))
        editor.screen.blit(hint_surface, hint_rect)

    editor._dropdown_draw_ops = dropdown_draw_ops

    # 4. Camera position centered beneath the control guide
    if editor.camera_controller:
        cam = editor.camera_controller
        if getattr(cam, "view_mode", "topdown") == "orbit":
            position_text = (f"Pivot: ({cam.center_x:.1f}, {cam.center_y:.1f}) "
                             f"Dist: {cam.orbit_distance:.1f}m Pitch: {cam.orbit_pitch:.0f}\N{DEGREE SIGN}")
        else:
            position_text = (f"Position: ({cam.center_x:.1f}, {cam.center_y:.1f}) "
                             f"Height: {cam.height:.1f}m")
        position_surface = _render_text_cached(editor.small_font,
            position_text,
            True,
            editor.colors['text']
        )
        position_rect = position_surface.get_rect(center=(editor.screen_width // 2, controls_rect.bottom + 18))
        editor.screen.blit(position_surface, position_rect)

def _close_all_dropdowns(editor, *, except_menu=None, keep_info_panel: bool = False) -> None:
    """Close all open dropdown-style menus in the UI."""
    editor.play_camera_menu_open = False
    editor.resolution_menu_open = False
    editor.fps_menu_open = False
    editor._culling_menu_open = False
    editor.map_menu_visible = False
    editor.scenario_menu_visible = False
    editor.agent_dropdown_open = False
    editor.agent_behavior_menu_open = False
    editor.npc_dropdown_open = False
    for menu in (editor.vehicle_menu, editor.pedestrian_menu, editor.ego_vehicle_menu):
        if menu and menu is not except_menu:
            menu.dropdown_open = False
    if editor.info_panel and not keep_info_panel:
        editor.info_panel.dropdown_open_field = None

def get_active_selection_menu(editor):
    """Return the menu matching the current placement mode."""
    if editor.placement_mode == PlacementMode.VEHICLE:
        return editor.vehicle_menu
    if editor.placement_mode == PlacementMode.PEDESTRIAN:
        return editor.pedestrian_menu
    if editor.placement_mode == PlacementMode.EGO:
        return editor.ego_vehicle_menu
    if editor.placement_mode == PlacementMode.TRIGGER:
        return getattr(editor, "traffic_light_group_menu", None)
    return None

def _get_mode_button_rects(editor):
    """Compute placement mode toggle button rectangles."""
    base_x = editor.mode_button_base_x
    base_y = editor.top_ui_height + 10
    button_width = editor.mode_button_width
    button_height = editor.mode_button_height
    spacing = editor.mode_button_spacing

    vehicle_rect = pygame.Rect(base_x, base_y, button_width, button_height)
    pedestrian_rect = pygame.Rect(base_x + button_width + spacing, base_y, button_width, button_height)
    ego_rect = pygame.Rect(base_x + 2 * (button_width + spacing), base_y, button_width, button_height)
    trigger_rect = pygame.Rect(base_x + 3 * (button_width + spacing), base_y, button_width, button_height)

    return {
        PlacementMode.VEHICLE: vehicle_rect,
        PlacementMode.PEDESTRIAN: pedestrian_rect,
        PlacementMode.EGO: ego_rect,
        PlacementMode.TRIGGER: trigger_rect,
    }

def _get_weather_button_rect(editor) -> pygame.Rect:
    """Return the rectangle for the weather control button."""
    base_x = editor.mode_button_base_x
    base_y = editor.top_ui_height + 10
    width = editor.mode_button_width
    height = editor.mode_button_height
    spacing = editor.mode_button_spacing
    return pygame.Rect(base_x + 4 * (width + spacing), base_y, width, height)

def handle_mode_toggle_click(editor, mouse_pos):
    """Handle clicks on the placement mode toggle."""
    rects = editor.mode_button_rects or editor._get_mode_button_rects()
    for mode, rect in rects.items():
        if rect.collidepoint(mouse_pos):
            editor.set_placement_mode(mode)
            return True
    return False

def set_placement_mode(editor, mode):
    """Update active placement mode and reset menus."""
    previous_mode = editor.placement_mode
    if mode == previous_mode:
        # Re-arm trigger placement if the user clicks the active trigger button while disarmed.
        if (mode == PlacementMode.TRIGGER and editor.camera_processor
                and not editor.camera_processor.placing_trigger):
            editor.camera_processor.start_trigger_placement()
        return

    editor.placement_mode = mode
    if editor.vehicle_menu:
        editor.vehicle_menu.dropdown_open = False
    if editor.pedestrian_menu:
        editor.pedestrian_menu.dropdown_open = False
    if editor.ego_vehicle_menu:
        editor.ego_vehicle_menu.dropdown_open = False

    if editor.camera_processor:
        if previous_mode == PlacementMode.TRIGGER and editor.camera_processor.placing_trigger:
            editor.camera_processor.stop_trigger_placement()
        editor.camera_processor.clear_vehicle_selection()
        editor.camera_processor.waypoint_display_vehicle_id = None
        editor.camera_processor.creating_waypoint = False
        if editor.info_panel:
            editor.info_panel.hide()
        if mode == PlacementMode.TRIGGER:
            editor.camera_processor.start_trigger_placement()

def render_mode_toggle(editor):
    """Render placement mode toggle buttons with active highlight."""
    rects = editor._get_mode_button_rects()
    editor.mode_button_rects = rects

    active_color = (90, 140, 90)
    inactive_color = (70, 70, 70)
    border_color = (120, 120, 120)
    hover_color = (110, 110, 110)

    mouse_pos = pygame.mouse.get_pos()
    if getattr(editor, "_dropdown_capture_rects", None):
        for rect in editor._dropdown_capture_rects:
            if rect.collidepoint(mouse_pos):
                mouse_pos = (-1, -1)
                break

    for mode, rect in rects.items():
        is_active = (mode == editor.placement_mode)
        is_hovered = rect.collidepoint(mouse_pos)

        fill_color = active_color if is_active else inactive_color
        if is_hovered and not is_active:
            fill_color = hover_color

        pygame.draw.rect(editor.screen, fill_color, rect, border_radius=6)
        pygame.draw.rect(editor.screen, border_color, rect, width=2, border_radius=6)

        if mode == PlacementMode.VEHICLE:
            label = "NPC Vehicles"
        elif mode == PlacementMode.PEDESTRIAN:
            label = "Pedestrians"
        elif mode == PlacementMode.EGO:
            label = "Ego Vehicle"
        else:
            label = "Triggers"

        text_surface = _render_text_cached(editor.small_font, label, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        editor.screen.blit(text_surface, text_rect)

        # Register hover for tooltip
        if is_hovered and editor.tooltip_manager:
            tooltip_text = MODE_BUTTON_TOOLTIPS.get(mode.name, '')
            if tooltip_text:
                editor.tooltip_manager.register_hover(f"mode_button_{mode.name}", rect, tooltip_text)

    weather_rect = editor._get_weather_button_rect()
    editor.weather_button_rect = weather_rect
    weather_enabled = editor.ready and (editor.world is not None)
    editor.weather_button_enabled = weather_enabled
    weather_active = weather_enabled and bool(editor.weather_window and editor.weather_window.alive())
    weather_hovered = weather_rect.collidepoint(mouse_pos)

    if weather_enabled:
        weather_fill = active_color if weather_active else inactive_color
        if weather_hovered and not weather_active:
            weather_fill = hover_color
        weather_text_color = (255, 255, 255)
    else:
        weather_fill = (55, 55, 55)
        weather_text_color = (170, 170, 170)

    pygame.draw.rect(editor.screen, weather_fill, weather_rect, border_radius=6)
    pygame.draw.rect(editor.screen, border_color, weather_rect, width=2, border_radius=6)

    weather_text_surface = _render_text_cached(editor.small_font, "Weather", True, weather_text_color)
    weather_text_rect = weather_text_surface.get_rect(center=weather_rect.center)
    editor.screen.blit(weather_text_surface, weather_text_rect)

    # Register hover for weather button tooltip
    if weather_hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("weather_button", weather_rect, TOP_UI_BUTTON_TOOLTIPS.get('weather', ''))

def render_view3d_button(editor):
    """Render the purple 3D-view toggle above the centered help hint."""
    if not editor.ready:
        return
    in_orbit = bool(editor.camera_controller
                    and getattr(editor.camera_controller, "view_mode", "topdown") == "orbit")
    label_surface = _render_text_cached(
        editor.small_font, "3D View (Tab)", True, editor.colors['text'])
    rect = pygame.Rect(0, 10, label_surface.get_width() + 24, 30)
    rect.centerx = editor.screen_width // 2
    hovered = rect.collidepoint(pygame.mouse.get_pos())
    if editor.scenario_running:
        color = (60, 60, 60)
    elif in_orbit:
        color = (180, 105, 230) if hovered else (155, 80, 210)
    else:
        color = (140, 90, 195) if hovered else (115, 70, 170)
    pygame.draw.rect(editor.screen, color, rect, border_radius=6)
    pygame.draw.rect(editor.screen, editor.colors['text'], rect, 1, border_radius=6)
    editor.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
    editor.view3d_button_rect = rect
    if hovered and editor.tooltip_manager:
        editor.tooltip_manager.register_hover("view3d_button", rect, TOP_UI_BUTTON_TOOLTIPS.get('view3d', ''))

def render_crosshair(editor):
    """Render red crosshair in the center of the screen"""
    center_x = editor.screen_width // 2
    center_y = editor.screen_height // 2
    crosshair_size = 20
    crosshair_thickness = 2
    crosshair_color = (255, 0, 0)  # Red
    
    # Horizontal line
    pygame.draw.line(editor.screen, crosshair_color, 
                    (center_x - crosshair_size, center_y), 
                    (center_x + crosshair_size, center_y), 
                    crosshair_thickness)
    
    # Vertical line
    pygame.draw.line(editor.screen, crosshair_color, 
                    (center_x, center_y - crosshair_size), 
                    (center_x, center_y + crosshair_size), 
                    crosshair_thickness)
    
    # Small center dot for precision
    pygame.draw.circle(editor.screen, crosshair_color,
                      (center_x, center_y), 2)

def _update_fps_meter(editor, raw_fps: float) -> None:
    """Smooth the incoming FPS readings for a stable on-screen display."""
    if not math.isfinite(raw_fps):
        return

    if raw_fps <= 0.0:
        # Gradually decay when the clock reports no measurable FPS.
        editor._fps_display = max(0.0, editor._fps_display * 0.95)
        return

    smoothing = editor._fps_smoothing
    if editor._fps_display <= 0.0:
        editor._fps_display = raw_fps
    else:
        editor._fps_display = max(
            0.0,
            (editor._fps_display * smoothing) + (raw_fps * (1.0 - smoothing)),
        )

def render_fps_meter(editor) -> Optional[pygame.Rect]:
    """Render UI, camera, and WallTime tick FPS on a single line near the top-right corner."""
    if not editor.screen or not editor.fps_font:
        return None

    ui_fps = max(0.0, editor._fps_display)
    try:
        camera_fps = max(0.0, float(editor.camera_processor.get_camera_fps())) if editor.camera_processor else 0.0
    except Exception:
        camera_fps = 0.0
    world_fps = max(0.0, getattr(editor, "_world_tick_fps", 0.0))

    segments = [
        f"UI: {ui_fps:0.1f} FPS" if ui_fps >= 0.05 else "UI: 0.0 FPS",
        f"Camera: {camera_fps:0.1f} FPS" if camera_fps >= 0.05 else "Camera: 0.0 FPS",
        f"WallTime: {world_fps:0.1f} FPS" if world_fps >= 0.05 else "WallTime: 0.0 FPS",
    ]
    # During a run (camera following an ego), show the current view mode + the C hotkey.
    cp = editor.camera_processor
    if cp and (getattr(cp, "playback_camera_follow_enabled", False)
               or getattr(cp, "manual_control_enabled", False)):
        view_label = {"chase": "Chase", "cockpit": "Cockpit"}.get(
            getattr(cp, "playback_camera_mode", "topdown"), "Top-Down")
        segments.append(f"View: {view_label} (C)")
    text_surface = _render_text_cached(editor.fps_font, "   |   ".join(segments), True, editor.colors['fps'])

    padding = 10
    text_rect = text_surface.get_rect()
    play_button_rect = getattr(editor, 'play_button_rect', None)
    screen_width = editor.screen.get_width()
    if play_button_rect and editor.ready:
        offset = 6
        text_rect.topright = (play_button_rect.right, play_button_rect.bottom + offset)
        text_rect.right = min(text_rect.right, screen_width - padding)
    else:
        text_rect.topright = (screen_width - padding, padding)

    editor.screen.blit(text_surface, text_rect)
    editor._fps_text_rect = text_rect
    return text_rect

def render_rendering_toggle(editor, anchor_rect: Optional[pygame.Rect]) -> None:
    """Render the scene rendering checkbox under the FPS counter."""
    if not editor.screen or not editor.small_font:
        return

    padding = 10
    right_edge = editor.screen.get_width() - padding
    top = (anchor_rect.bottom + 6) if anchor_rect else (padding + 8)

    label_text = "No Rendering Mode"
    label_surface = _render_text_cached(editor.small_font, label_text, True, editor.colors['text'])
    label_rect = label_surface.get_rect()
    label_rect.topright = (right_edge, top)

    checkbox_size = max(14, label_rect.height - 4)
    checkbox_rect = pygame.Rect(0, 0, checkbox_size, checkbox_size)
    checkbox_rect.top = label_rect.top + (label_rect.height - checkbox_size) // 2
    checkbox_rect.right = label_rect.left - 8

    toggle_rect = checkbox_rect.union(label_rect).inflate(6, 4)
    editor.rendering_toggle_rect = toggle_rect

    mouse_pos = pygame.mouse.get_pos()
    is_hovered = toggle_rect.collidepoint(mouse_pos)

    no_render_active = not editor.rendering_enabled
    base_color = editor.colors['toggle_on'] if no_render_active else editor.colors['toggle_off']
    if is_hovered:
        hover_boost = 20
        fill_color = tuple(min(c + hover_boost, 255) for c in base_color)
    else:
        fill_color = base_color

    pygame.draw.rect(editor.screen, fill_color, checkbox_rect, border_radius=3)
    pygame.draw.rect(editor.screen, editor.colors['text'], checkbox_rect, 1, border_radius=3)

    if no_render_active:
        # Simple check mark.
        check_points = [
            (checkbox_rect.left + 3, checkbox_rect.centery),
            (checkbox_rect.left + checkbox_rect.width // 2 - 1, checkbox_rect.bottom - 4),
            (checkbox_rect.right - 3, checkbox_rect.top + 3),
        ]
        pygame.draw.lines(editor.screen, editor.colors['text'], False, check_points, 2)

    editor.screen.blit(label_surface, label_rect)

    # Ego Collision checkbox, placed in front of (to the left of) the No Rendering Mode group.
    # When Ego Physics is off, collision is locked OFF and the control is dimmed/non-interactive.
    ego_locked = not editor.ego_physics_enabled
    ego_label_color = tuple(c // 2 for c in editor.colors['text']) if ego_locked else editor.colors['text']
    ego_label_surface = _render_text_cached(editor.small_font, "Ego Collision", True, ego_label_color)
    ego_label_rect = ego_label_surface.get_rect()
    ego_label_rect.topright = (checkbox_rect.left - 12, top)

    ego_checkbox_rect = pygame.Rect(0, 0, checkbox_size, checkbox_size)
    ego_checkbox_rect.top = ego_label_rect.top + (ego_label_rect.height - checkbox_size) // 2
    ego_checkbox_rect.right = ego_label_rect.left - 8

    ego_toggle_rect = ego_checkbox_rect.union(ego_label_rect).inflate(6, 4)
    editor.ego_collision_toggle_rect = ego_toggle_rect

    ego_is_hovered = (not ego_locked) and ego_toggle_rect.collidepoint(mouse_pos)
    ego_base_color = editor.colors['toggle_on'] if editor.ego_collision_enabled else editor.colors['toggle_off']
    if ego_locked:
        ego_fill_color = tuple(c // 2 for c in ego_base_color)
    elif ego_is_hovered:
        ego_fill_color = tuple(min(c + 20, 255) for c in ego_base_color)
    else:
        ego_fill_color = ego_base_color

    pygame.draw.rect(editor.screen, ego_fill_color, ego_checkbox_rect, border_radius=3)
    pygame.draw.rect(editor.screen, ego_label_color, ego_checkbox_rect, 1, border_radius=3)

    if editor.ego_collision_enabled:
        ego_check_points = [
            (ego_checkbox_rect.left + 3, ego_checkbox_rect.centery),
            (ego_checkbox_rect.left + ego_checkbox_rect.width // 2 - 1, ego_checkbox_rect.bottom - 4),
            (ego_checkbox_rect.right - 3, ego_checkbox_rect.top + 3),
        ]
        pygame.draw.lines(editor.screen, editor.colors['text'], False, ego_check_points, 2)

    editor.screen.blit(ego_label_surface, ego_label_rect)

    # Ego Physics checkbox, placed in front of (to the left of) the Ego Collision group.
    phys_label_surface = _render_text_cached(editor.small_font, "Ego Physics", True, editor.colors['text'])
    phys_label_rect = phys_label_surface.get_rect()
    phys_label_rect.topright = (ego_checkbox_rect.left - 12, top)

    phys_checkbox_rect = pygame.Rect(0, 0, checkbox_size, checkbox_size)
    phys_checkbox_rect.top = phys_label_rect.top + (phys_label_rect.height - checkbox_size) // 2
    phys_checkbox_rect.right = phys_label_rect.left - 8

    phys_toggle_rect = phys_checkbox_rect.union(phys_label_rect).inflate(6, 4)
    editor.ego_physics_toggle_rect = phys_toggle_rect

    phys_is_hovered = phys_toggle_rect.collidepoint(mouse_pos)
    phys_base_color = editor.colors['toggle_on'] if editor.ego_physics_enabled else editor.colors['toggle_off']
    if phys_is_hovered:
        phys_fill_color = tuple(min(c + 20, 255) for c in phys_base_color)
    else:
        phys_fill_color = phys_base_color

    pygame.draw.rect(editor.screen, phys_fill_color, phys_checkbox_rect, border_radius=3)
    pygame.draw.rect(editor.screen, editor.colors['text'], phys_checkbox_rect, 1, border_radius=3)

    if editor.ego_physics_enabled:
        phys_check_points = [
            (phys_checkbox_rect.left + 3, phys_checkbox_rect.centery),
            (phys_checkbox_rect.left + phys_checkbox_rect.width // 2 - 1, phys_checkbox_rect.bottom - 4),
            (phys_checkbox_rect.right - 3, phys_checkbox_rect.top + 3),
        ]
        pygame.draw.lines(editor.screen, editor.colors['text'], False, phys_check_points, 2)

    editor.screen.blit(phys_label_surface, phys_label_rect)
