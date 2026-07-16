"""Modal dialog helpers (nested blocking event loops on the editor).

Moved verbatim from vse.py (step-36, Phase 7).
"""

from typing import Dict, Optional, Union

import pygame

from pygame_gui._constants import (
    UI_WINDOW_CLOSE,
    UI_BUTTON_PRESSED,
    UI_TEXT_ENTRY_FINISHED,
)
from pygame_gui.elements.ui_window import UIWindow

from vse_editor.ui.dialogs import TextInputDialog


def _center_dialog_rect(editor, width: int, height: int) -> pygame.Rect:
    """Return a rectangle centered in the current window."""
    rect = pygame.Rect(0, 0, width, height)
    rect.center = (editor.screen_width // 2, editor.screen_height // 2)
    return rect

def _render_dialog_background(editor) -> None:
    """Re-render a clean background (with menus closed) for modal dialogs."""
    editor.screen.fill(editor.colors['background'])
    if not editor.ready:
        return

    if editor.camera_processor:
        latest_image = editor.camera_processor.get_latest_image()
        if editor.camera_stream_enabled and latest_image:
            editor.screen.blit(latest_image, (0, 0))
        elif not editor.camera_stream_enabled:
            editor.screen.fill((0, 0, 0))

    editor.render_crosshair()
    editor.render_ui()
    editor.render_mode_toggle()

    active_menu = editor.get_active_selection_menu()
    if active_menu and not editor.scenario_running:
        active_menu.render(editor.screen, tooltip_manager=editor.tooltip_manager)

    # Sync vehicle_control_mode from camera_processor (menu pills write there)
    if editor.camera_processor:
        editor.vehicle_control_mode = getattr(
            editor.camera_processor, 'vehicle_control_mode', 'basic_agent')

    if editor.camera_processor:
        editor.camera_processor.render_all_overlays(editor.screen)

    editor.info_panel.render(editor.screen, tooltip_manager=editor.tooltip_manager)

    fps_rect = editor.render_fps_meter()
    editor.render_rendering_toggle(fps_rect)

def _run_modal_window(editor, dialog: UIWindow, event_handler) -> Optional[Union[bool, str, Dict[str, Union[str, int]]]]:
    """
    Run a blocking modal loop around the supplied window.

    Args:
        dialog: The pygame_gui window to display.
        event_handler: Callback taking a pygame event and returning a tuple
            ``(should_stop, result)``. When ``should_stop`` is True the loop exits
            and ``result`` is returned.

    Returns:
        The result provided by the handler, or None if the dialog was cancelled.
    """
    editor._close_all_dropdowns(keep_info_panel=True)
    editor._render_dialog_background()

    clock = pygame.time.Clock()
    snapshot = editor.screen.copy()
    shade = pygame.Surface(editor.screen.get_size(), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 160))

    mouse_visible = pygame.mouse.get_visible()
    mouse_grabbed = pygame.event.get_grab()
    pygame.mouse.set_visible(True)
    if mouse_grabbed:
        pygame.event.set_grab(False)

    result = None
    running = True

    while running and dialog.alive():
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                dialog.kill()
                running = False
                result = None
                break

            if editor.ui_manager:
                editor.ui_manager.process_events(event)

            should_stop, handled_result = event_handler(event)
            if should_stop:
                result = handled_result
                if dialog.alive():
                    dialog.kill()
                running = False
                break

        if editor.ui_manager:
            editor.ui_manager.update(time_delta)

        editor.screen.blit(snapshot, (0, 0))
        editor.screen.blit(shade, (0, 0))
        if editor.ui_manager:
            editor.ui_manager.draw_ui(editor.screen)
        pygame.display.flip()

    if dialog.alive():
        dialog.kill()

    pygame.mouse.set_visible(mouse_visible)
    pygame.event.set_grab(mouse_grabbed)

    if editor.ui_manager:
        editor.ui_manager.update(0.0)

    return result

def _prompt_text_input(editor, title: str, prompt: str, default: str = "") -> Optional[str]:
    """Display a simple text input modal and return the user entry."""
    editor._close_all_dropdowns(keep_info_panel=True)
    dialog_rect = editor._center_dialog_rect(360, 200)
    dialog = TextInputDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        title=title,
        prompt=prompt,
        default_text=default,
    )

    def handler(event: pygame.event.Event):
        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element == dialog.ok_button:
                return True, dialog.text_entry.get_text()
            if event.ui_element == dialog.cancel_button:
                return True, None
        if event.type == UI_TEXT_ENTRY_FINISHED and event.ui_element == dialog.text_entry:
            return True, dialog.text_entry.get_text()
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, None
        return False, None

    result = editor._run_modal_window(dialog, handler)
    if result is None:
        return None
    result = str(result).strip()
    return result or None

def _ask_use_running_server(editor, port: int) -> bool:
    """Ask whether to connect to an already-running CARLA server.

    Returns True to reuse the running server, False to stop it and start
    a fresh one. Runs at startup before the main UI loop, so it uses a
    simple tkinter dialog (with a console fallback) rather than the
    pygame_gui modals used elsewhere.
    """
    message = (
        f"A CARLA server is already running on port {port}.\n\n"
        "Connect to it?\n\n"
        "Yes - connect to the running server (its current map is kept).\n"
        "No - stop it and start a fresh server."
    )
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        answer = messagebox.askyesno("CARLA server detected", message)
        root.destroy()
        return bool(answer)
    except Exception as exc:
        print(f"[Startup] GUI prompt unavailable ({exc}); falling back to console.")

    try:
        reply = input(
            f"CARLA server already running on port {port}. Connect to it? [Y/n] "
        ).strip().lower()
        return reply in ("", "y", "yes")
    except Exception:
        print("[Startup] No interactive console; defaulting to connecting to the running server.")
        return True
