"""pygame_gui dialogs (moved verbatim from vse.py): GPU selection, text
input, remote connection, file dialog with navigation history, and the
playback result window.
"""

from __future__ import annotations

import html
import locale
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui._constants import (
    UI_BUTTON_PRESSED,
    UI_CONFIRMATION_DIALOG_CONFIRMED,
    UI_FILE_DIALOG_PATH_PICKED,
    UI_HORIZONTAL_SLIDER_MOVED,
    UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION,
    UI_SELECTION_LIST_NEW_SELECTION,
    UI_TEXT_ENTRY_FINISHED,
    UI_WINDOW_CLOSE,
)
from pygame_gui.core import ObjectID
from pygame_gui.elements.ui_button import UIButton
from pygame_gui.elements.ui_label import UILabel
from pygame_gui.elements.ui_scrolling_container import UIScrollingContainer
from pygame_gui.elements.ui_selection_list import UISelectionList
from pygame_gui.elements.ui_text_box import UITextBox
from pygame_gui.elements.ui_text_entry_line import UITextEntryLine
from pygame_gui.elements.ui_window import UIWindow
from pygame_gui.windows.ui_file_dialog import UIFileDialog

class GPUSelectionDialog(UIWindow):
    """Modal window for selecting a GPU/port option."""

    def __init__(self, rect: pygame.Rect, manager: UIManager, options: List[Dict[str, Union[str, int]]], title: str = "Select GPU"):
        super().__init__(
            rect,
            manager,
            window_display_title=title,
            resizable=False,
            always_on_top=True,
        )
        self.set_blocking(True)
        self.options = options
        self.label_to_option = {option["label"]: option for option in options}
        self.selected_label: Optional[str] = None
        self.confirmed: bool = False

        container_width, container_height = self.get_container().get_size()
        list_rect = pygame.Rect(
            10,
            40,
            max(100, container_width - 20),
            max(60, container_height - 90),
        )

        self.selection_list = UISelectionList(
            relative_rect=list_rect,
            item_list=[option["label"] for option in options],
            manager=manager,
            container=self,
            allow_double_clicks=True,
            object_id="#gpu_selection_list",
        )

        button_width = 110
        button_height = 32
        button_y = container_height - button_height - 10

        self.cancel_button = UIButton(
            relative_rect=pygame.Rect(
                container_width - button_width - 10,
                button_y,
                button_width,
                button_height,
            ),
            text="Cancel",
            manager=manager,
            container=self,
            object_id="#gpu_cancel_button",
            anchors={
                "left": "right",
                "right": "right",
                "top": "bottom",
                "bottom": "bottom",
            },
        )

        self.ok_button = UIButton(
            relative_rect=pygame.Rect(
                container_width - (2 * button_width) - 20,
                button_y,
                button_width,
                button_height,
            ),
            text="Select",
            manager=manager,
            container=self,
            object_id="#gpu_select_button",
            anchors={
                "left": "right",
                "right": "right",
                "top": "bottom",
                "bottom": "bottom",
                "right_target": self.cancel_button,
            },
        )
        self.ok_button.disable()

    def process_event(self, event: pygame.event.Event) -> bool:
        handled = super().process_event(event)

        if event.type == UI_SELECTION_LIST_NEW_SELECTION and event.ui_element == self.selection_list:
            self.selected_label = event.text
            self.ok_button.enable()

        if event.type == UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION and event.ui_element == self.selection_list:
            self.selected_label = event.text
            self.confirm_selection()

        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element == self.ok_button and self.selected_label:
                self.confirm_selection()
            elif event.ui_element == self.cancel_button:
                self.selected_label = None
                self.confirmed = False
                self.kill()

        return handled

    def confirm_selection(self):
        if self.selected_label:
            self.confirmed = True
            self.kill()



class TextInputDialog(UIWindow):
    """Simple modal dialog for text input (used for new folder names)."""

    def __init__(self, rect: pygame.Rect, manager: UIManager, title: str, prompt: str, default_text: str = ""):
        super().__init__(
            rect,
            manager,
            window_display_title=title,
            resizable=False,
            always_on_top=True,
        )
        self.set_blocking(True)

        container_width, container_height = self.get_container().get_size()

        label_rect = pygame.Rect(10, 10, container_width - 20, 30)
        self.prompt_label = UILabel(
            relative_rect=label_rect,
            text=prompt,
            manager=manager,
            container=self,
            anchors={
                "left": "left",
                "right": "right",
                "top": "top",
                "bottom": "top",
            },
        )

        entry_rect = pygame.Rect(10, 50, container_width - 20, 30)
        self.text_entry = UITextEntryLine(
            relative_rect=entry_rect,
            manager=manager,
            container=self,
            anchors={
                "left": "left",
                "right": "right",
                "top": "top",
                "bottom": "top",
            },
        )
        self.text_entry.set_text(default_text)
        self.text_entry.select_range = [0, len(default_text)]
        self.text_entry.cursor_has_moved_recently = True
        try:
            self.text_entry.focus()
        except AttributeError:
            pass

        button_width = 72
        button_height = 28
        button_y = container_height - button_height - 12
        cancel_x = container_width - button_width - 10
        ok_x = cancel_x - button_width - 10

        self.cancel_button = UIButton(
            relative_rect=pygame.Rect(cancel_x, button_y, button_width, button_height),
            text="Cancel",
            manager=manager,
            container=self,
            object_id="#text_input_cancel",
            anchors={
                "left": "left",
                "right": "left",
                "top": "top",
                "bottom": "top",
            },
        )

        self.ok_button = UIButton(
            relative_rect=pygame.Rect(ok_x, button_y, button_width, button_height),
            text="OK",
            manager=manager,
            container=self,
            object_id="#text_input_ok",
            anchors={
                "left": "left",
                "right": "left",
                "top": "top",
                "bottom": "top",
            },
        )


class RemoteConnectionDialog(UIWindow):
    """Modal window for entering remote host and port."""

    def __init__(
        self,
        rect: pygame.Rect,
        manager: UIManager,
        default_host: str = "",
        default_port: str = "",
        show_local_button: bool = True,
    ):
        super().__init__(
            rect,
            manager,
            window_display_title="Connect to Remote Server",
            resizable=False,
            always_on_top=True,
        )
        self.set_blocking(True)
        self.show_local_button = show_local_button
        container_width, container_height = self.get_container().get_size()

        label_host_rect = pygame.Rect(10, 14, container_width - 20, 24)
        UILabel(
            relative_rect=label_host_rect,
            text="Host / IP",
            manager=manager,
            container=self,
        )

        entry_host_rect = pygame.Rect(10, 40, container_width - 20, 30)
        self.host_entry = UITextEntryLine(
            relative_rect=entry_host_rect,
            manager=manager,
            container=self,
        )
        self.host_entry.set_text(default_host or "")

        label_port_rect = pygame.Rect(10, 80, container_width - 20, 24)
        UILabel(
            relative_rect=label_port_rect,
            text="Port",
            manager=manager,
            container=self,
        )

        entry_port_rect = pygame.Rect(10, 106, container_width - 20, 30)
        self.port_entry = UITextEntryLine(
            relative_rect=entry_port_rect,
            manager=manager,
            container=self,
        )
        self.port_entry.set_text(default_port or "")

        button_width = 90
        button_height = 30
        button_y = container_height - button_height - 12

        self.local_button = UIButton(
            relative_rect=pygame.Rect(10, button_y, button_width, button_height),
            text="Local",
            manager=manager,
            container=self,
        )
        if not self.show_local_button:
            try:
                self.local_button.disable()
                self.local_button.hide()
            except Exception:
                pass

        self.cancel_button = UIButton(
            relative_rect=pygame.Rect(
                container_width - button_width - 10,
                button_y,
                button_width,
                button_height,
            ),
            text="Cancel",
            manager=manager,
            container=self,
        )
        self.ok_button = UIButton(
            relative_rect=pygame.Rect(
                container_width - (2 * button_width) - 20,
                button_y,
                button_width,
                button_height,
            ),
            text="OK",
            manager=manager,
            container=self,
        )
        self.result_action: Optional[str] = None

    def process_event(self, event: pygame.event.Event) -> bool:
        handled = super().process_event(event)

        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element == self.ok_button:
                self.result_action = "ok"
                self.kill()
            elif self.show_local_button and event.ui_element == self.local_button:
                self.result_action = "local"
                self.kill()
            elif event.ui_element == self.cancel_button:
                self.result_action = "cancel"
                self.kill()

        if event.type == UI_TEXT_ENTRY_FINISHED and event.ui_element in (self.host_entry, self.port_entry):
            self.result_action = "ok"
            self.kill()

        if event.type == UI_WINDOW_CLOSE and event.ui_element == self:
            self.result_action = "cancel"

        return handled

    def get_values(self) -> Dict[str, str]:
        return {
            "host": self.host_entry.get_text(),
            "port": self.port_entry.get_text(),
            "action": self.result_action or "cancel",
        }


class EnhancedFileDialog(UIFileDialog):
    """UIFileDialog with an additional 'new folder' button."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except Exception:
            pass

        # Position the button directly to the right of the existing refresh button.
        refresh_rect = self.refresh_button.relative_rect
        button_rect = pygame.Rect(
            refresh_rect.right + 4,
            refresh_rect.y,
            refresh_rect.width,
            refresh_rect.height,
        )
        self.new_folder_button = UIButton(
            relative_rect=button_rect,
            text="+",
            tool_tip_text="Create folder",
            manager=self.ui_manager,
            container=self,
            object_id="#new_folder_button",
            anchors={
                "left": "left",
                "right": "left",
                "top": "top",
            "bottom": "top",
        },
    )


class ResultWindow(UIWindow):
    """Modal window that displays scenario run results with copy/save/close controls."""

    _result_theme_loaded = False

    @classmethod
    def _ensure_result_theme(cls, manager: UIManager) -> None:
        """Remove the border/shadow frame on the result text area + scroll container (once)."""
        if cls._result_theme_loaded:
            return
        no_frame = {"misc": {"border_width": "0", "shadow_width": "0"}}
        try:
            manager.get_theme().load_theme({
                "#result_text_box": no_frame,
                "#result_scroll": no_frame,
            })
        except Exception as exc:
            print(f"[Results] Warning: failed to apply result theme overrides: {exc}")
        cls._result_theme_loaded = True

    def __init__(
        self,
        rect: pygame.Rect,
        manager: UIManager,
        text: str,
        on_copy: Callable[[], None],
        on_save: Callable[[], None],
        on_close: Callable[[], None],
        title: str = "Scenario Results",
    ):
        super().__init__(
            rect,
            manager,
            window_display_title=title,
            resizable=True,
            always_on_top=True,
        )
        self.set_blocking(True)
        # Drop the border/shadow "frame" on the text area and scroll container — the window now
        # hugs the content, so the frame is just visual noise.
        self._ensure_result_theme(manager)

        self._on_copy = on_copy
        self._on_save = on_save
        self._on_close = on_close

        margin = 10
        container_width, container_height = self.get_container().get_size()
        # Window chrome (title bar + borders) = outer window size minus inner container size; used
        # later to fit the window to its content.
        init_win_w, init_win_h = self.rect.size
        chrome_w = init_win_w - container_width
        chrome_h = init_win_h - container_height

        button_height = 32
        button_spacing = 10
        button_width = 80

        # Scrolling area fills everything above the button row and stretches with the window
        # (anchored on all four edges). It shows horizontal and/or vertical scrollbars only when the
        # fixed-size content below is larger than the visible area, and hides them when it fits.
        scroll_bottom_gap = button_height + (2 * margin)
        scroll_rect = pygame.Rect(
            margin,
            margin,
            container_width - (2 * margin),
            max(60, container_height - scroll_bottom_gap - margin),
        )
        self.scroll_container = UIScrollingContainer(
            relative_rect=scroll_rect,
            manager=manager,
            container=self,
            object_id="#result_scroll",
            anchors={"left": "left", "right": "right", "top": "top", "bottom": "bottom"},
            should_grow_automatically=False,
            allow_scroll_x=True,
            allow_scroll_y=True,
        )

        # The table is rendered at a fixed monospace size and sized to its true (unwrapped) extent;
        # the scroll container scrolls to reveal it. We can't trust a raw font measurement for the
        # width (the theme's rendered monospace metrics differ), so measure pygame_gui's own layout
        # with a throwaway box. The true text extent is the widest layout ROW — layout_rect just
        # echoes the box width, so it must not be used as a measurement. Wrapping is detected by
        # row count (a wrapped line yields more rows than the text has lines) and the probe doubles
        # until nothing wraps; the real box is then created just wide enough that it never wraps
        # and never shows its own (second) scrollbar.
        est_w, est_h = self._measure_content(manager, text)
        self._last_scroll_root_size: Optional[Tuple[int, int]] = None
        html_text = self._as_html(text)
        natural_w, natural_h = est_w, est_h
        try:
            n_lines = text.count("\n") + 1
            probe_w = max(est_w + 24, 1200)
            for _ in range(6):
                probe = UITextBox(
                    html_text,
                    relative_rect=pygame.Rect(0, 0, probe_w, est_h),
                    manager=manager,
                    container=self,
                    object_id="#result_text_box",
                )
                layout = probe.text_box_layout
                rows = list(layout.layout_rows)
                max_row_w = max((row.width for row in rows), default=est_w)
                pad_w = max(0, probe_w - layout.layout_rect.width)  # box border + padding
                layout_h = layout.layout_rect.height
                probe.kill()
                if len(rows) <= n_lines or probe_w >= 8000:
                    natural_w = max_row_w + pad_w + 2
                    natural_h = layout_h + 24
                    break
                probe_w = min(8000, probe_w * 2)
        except Exception:
            pass
        self.text_box = UITextBox(
            html_text,
            relative_rect=pygame.Rect(0, 0, natural_w, natural_h),
            manager=manager,
            container=self.scroll_container,
            object_id="#result_text_box",
        )
        self._natural_w = natural_w
        self._natural_h = natural_h
        # Enable only the axes the content actually overflows, so a scrollbar appears only when the
        # window is smaller than the table on that axis (and both vanish when it fits). pygame_gui
        # otherwise forces a spurious horizontal bar whenever a vertical bar is present.
        self._apply_scroll_axes()

        # Buttons stick to the bottom edge and stay horizontally centered on resize. With a centerx
        # anchor, the rect's x is the offset of the button's centre from the container centre.
        button_anchors = {"centerx": "centerx", "top": "bottom", "bottom": "bottom"}
        button_rel_y = -(button_height + margin)
        button_step = button_width + button_spacing
        self.copy_button = UIButton(
            relative_rect=pygame.Rect(-button_step, button_rel_y, button_width, button_height),
            text="Copy",
            manager=manager,
            container=self,
            object_id="#result_copy_button",
            anchors=button_anchors,
        )
        self.save_button = UIButton(
            relative_rect=pygame.Rect(0, button_rel_y, button_width, button_height),
            text="Save",
            manager=manager,
            container=self,
            object_id="#result_save_button",
            anchors=button_anchors,
        )
        self.close_button = UIButton(
            relative_rect=pygame.Rect(button_step, button_rel_y, button_width, button_height),
            text="Close",
            manager=manager,
            container=self,
            object_id="#result_close_button",
            anchors=button_anchors,
        )

        # Fit the window to its content on first open so the frame hugs the table (no large empty
        # area). Grown/shrunk to the content, capped at 90% of the screen so a very large table gets
        # scrollbars rather than an off-screen window. Anchored children reflow on resize.
        try:
            screen = pygame.display.get_surface()
            sw, sh = screen.get_size() if screen else (init_win_w, init_win_h)
            max_w, max_h = int(sw * 0.9), int(sh * 0.9)
            min_w = (3 * button_width) + (2 * button_spacing) + (2 * margin)
            bar = 20  # pygame_gui scrollbar thickness
            base_c_w = max(min_w, natural_w + (2 * margin) + 4)
            base_c_h = natural_h + scroll_bottom_gap + margin + 4
            # If the table is taller/wider than will fit (the cap kicks in on that axis), that
            # scrollbar will be shown and steals `bar` px from the cross axis — so widen/heighten to
            # leave room for it, otherwise the vertical bar forces a spurious horizontal one.
            v_needed = (base_c_h + chrome_h) > max_h
            h_needed = (base_c_w + chrome_w) > max_w
            need_c_w = base_c_w + (bar if v_needed else 0)
            need_c_h = base_c_h + (bar if h_needed else 0)
            desired_w = min(max_w, need_c_w + chrome_w)
            desired_h = min(max_h, need_c_h + chrome_h)
            if (desired_w, desired_h) != (init_win_w, init_win_h):
                self.set_dimensions((desired_w, desired_h))
                self.set_position((max(0, (sw - desired_w) // 2), max(0, (sh - desired_h) // 2)))
            # Re-evaluate scrollbars for the fitted size.
            self._last_scroll_root_size = None
            self._apply_scroll_axes()
        except Exception:
            pass

    @staticmethod
    def _as_html(text: str) -> str:
        """Render plain text with preserved whitespace in a monospace block."""
        safe = html.escape(text)
        safe = safe.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        safe = safe.replace(" ", "&nbsp;")
        safe = safe.replace("\n", "<br>")
        return f'<font face="monospace">{safe}</font>'

    @staticmethod
    def _measure_content(manager: UIManager, text: str, font_size: int = 14) -> Tuple[int, int]:
        """Natural pixel size of the monospace result text, so the text box never wraps.

        Measured with the actual monospace font pygame_gui would render (falling back to a system
        monospace font), plus padding to cover the text box's internal border/padding. Slightly
        over-measuring is safe — it only adds a little scroll slack; under-measuring would wrap.
        """
        lines = text.split("\n") or [""]
        font = None
        try:
            font = manager.get_theme().get_font_dictionary().find_font(
                font_size=font_size, font_name="monospace"
            )
        except Exception:
            font = None
        if font is None:
            try:
                font = pygame.font.SysFont("monospace", font_size)
            except Exception:
                font = pygame.font.Font(None, font_size)
        try:
            line_h = font.get_linesize()
        except Exception:
            line_h = font_size + 6
        max_w = 0
        for line in lines:
            try:
                w = font.size(line)[0]
            except Exception:
                w = len(line) * ((font_size // 2) + 1)
            if w > max_w:
                max_w = w
        # Padding: text box border/padding (width) and a little vertical breathing room.
        natural_w = max_w + 28
        natural_h = (line_h * max(1, len(lines))) + 16
        return natural_w, natural_h

    def _apply_scroll_axes(self) -> None:
        """Enable horizontal/vertical scrolling only on the axes the table overflows.

        Compared against the scroll container's full (root) area, accounting for the opposite
        scrollbar's width when present, so each bar shows only when the window is smaller than the
        table on that axis and both disappear once it fits. Driven from update() on resize.
        """
        sc = getattr(self, "scroll_container", None)
        if sc is None:
            return
        try:
            root = sc._root_container.rect  # full element area (excludes nothing)
            root_w, root_h = root.width, root.height
        except Exception:
            return
        bar = 20  # pygame_gui scrollbar thickness
        need_y = self._natural_h > root_h
        need_x = self._natural_w > (root_w - (bar if need_y else 0))
        if need_x:
            need_y = self._natural_h > (root_h - bar)
        try:
            sc.allow_scroll_x = need_x
            sc.allow_scroll_y = need_y
            sc.set_scrollable_area_dimensions((self._natural_w, self._natural_h))
        except Exception:
            pass

    def update(self, time_delta: float) -> None:
        super().update(time_delta)
        # Re-evaluate which scrollbars are needed whenever the scroll area's size changes (i.e. the
        # window was resized). Cheap no-op when the size is unchanged.
        sc = getattr(self, "scroll_container", None)
        if sc is None:
            return
        try:
            size = sc._root_container.rect.size
        except Exception:
            return
        if size != self._last_scroll_root_size:
            self._last_scroll_root_size = size
            self._apply_scroll_axes()

    def process_event(self, event: pygame.event.Event) -> bool:
        handled = super().process_event(event)
        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element == self.copy_button:
                if self._on_copy:
                    self._on_copy()
                return True
            if event.ui_element == self.save_button:
                if self._on_save:
                    self._on_save()
                return True
            if event.ui_element == self.close_button:
                if self._on_close:
                    self._on_close()
                self.kill()
                return True

        if event.type == UI_WINDOW_CLOSE and event.ui_element == self:
            if self._on_close:
                self._on_close()
            self.kill()
            return True

        return handled
