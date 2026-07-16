"""Weather control window (moved verbatim from vse.py): live weather
sliders + keyframe list editing over the editor's weather state.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import carla

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
from pygame_gui.elements.ui_horizontal_slider import UIHorizontalSlider
from pygame_gui.elements.ui_scrolling_container import UIScrollingContainer
from pygame_gui.elements.ui_selection_list import UISelectionList
from pygame_gui.elements.ui_text_box import UITextBox
from pygame_gui.elements.ui_text_entry_line import UITextEntryLine
from pygame_gui.elements.ui_window import UIWindow
from pygame_gui.windows.ui_file_dialog import UIFileDialog

from vse_editor.constants import WEATHER_PARAMETER_SPECS, WeatherParameterSpec

class WeatherControlWindow(UIWindow):
    """Floating window containing sliders to adjust CARLA weather parameters."""

    _theme_initialized: bool = False

    def __init__(
        self,
        rect: pygame.Rect,
        manager: UIManager,
        specs: Tuple[WeatherParameterSpec, ...],
        on_change_live: Callable[[str, float], None],
        on_change_commit: Optional[Callable[[str, float], None]] = None,
        *,
        keyframe_count: int = 1,
        active_index: int = 0,
        active_percentage: float = 0.0,
        percent_editable: bool = True,
        can_delete: bool = False,
        on_percentage_change: Optional[Callable[[float], None]] = None,
        on_add_keyframe: Optional[Callable[[], None]] = None,
        on_delete_keyframe: Optional[Callable[[], None]] = None,
        on_prev_keyframe: Optional[Callable[[], None]] = None,
        on_next_keyframe: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ):
        self._ensure_theme(manager)
        super().__init__(
            rect,
            manager,
            window_display_title="Weather Controls",
            resizable=False,
            always_on_top=True,
        )
        self.set_blocking(False)

        self._specs = specs
        self._spec_lookup: Dict[str, WeatherParameterSpec] = {spec.name: spec for spec in specs}
        self._on_change_live = on_change_live
        self._on_change_commit = on_change_commit
        self._on_close = on_close
        self._on_percentage_change = on_percentage_change
        self._on_add_keyframe = on_add_keyframe
        self._on_delete_keyframe = on_delete_keyframe
        self._on_prev_keyframe = on_prev_keyframe
        self._on_next_keyframe = on_next_keyframe
        self._suppress_callback = False
        self._keyframe_count = max(1, int(keyframe_count))
        self._active_index = max(0, min(int(active_index), self._keyframe_count - 1))
        self._active_percentage = float(active_percentage)
        self._percent_editable = bool(percent_editable)
        self._can_delete = bool(can_delete)

        self.sliders: Dict[str, UIHorizontalSlider] = {}
        self.value_labels: Dict[str, UILabel] = {}
        self._slider_to_param: Dict[UIHorizontalSlider, str] = {}
        self.percent_slider: Optional[UIHorizontalSlider] = None
        self.percent_value_label: Optional[UILabel] = None
        self.keyframe_label: Optional[UILabel] = None
        self.prev_button: Optional[UIButton] = None
        self.next_button: Optional[UIButton] = None
        self.add_button: Optional[UIButton] = None
        self.delete_button: Optional[UIButton] = None

        # Drag tracking for commit-on-release behavior
        self._active_drag_param: Optional[str] = None
        self._drag_last_value: Optional[float] = None

        container_width, _ = self.get_container().get_size()
        margin = 14
        top_margin = 16

        # Keyframe metadata
        label_rect = pygame.Rect(margin, top_margin, container_width - (2 * margin), 22)
        self.keyframe_label = UILabel(
            label_rect,
            "",
            manager,
            container=self,
            object_id="#weather_name_label",
        )
        top_margin = label_rect.bottom + 4

        slider_height = 18
        pct_value_width = 70
        percent_slider_rect = pygame.Rect(
            margin,
            top_margin,
            max(160, container_width - (2 * margin) - pct_value_width - 8),
            slider_height,
        )
        self.percent_slider = UIHorizontalSlider(
            relative_rect=percent_slider_rect,
            start_value=max(0.0, min(100.0, self._active_percentage)),
            value_range=(0.0, 100.0),
            manager=manager,
            container=self,
            object_id="#weather_slider",
            click_increment=1.0,
        )
        percent_value_rect = pygame.Rect(
            percent_slider_rect.right + 4,
            percent_slider_rect.y,
            pct_value_width,
            slider_height,
        )
        self.percent_value_label = UILabel(
            percent_value_rect,
            "",
            manager,
            container=self,
            object_id="#weather_value_label",
        )

        min_label_rect = pygame.Rect(percent_slider_rect.x, percent_slider_rect.bottom + 2, pct_value_width, 16)
        UILabel(
            min_label_rect,
            "0",
            manager,
            container=self,
            object_id="#weather_min_label",
        )
        max_label_rect = pygame.Rect(percent_slider_rect.right - pct_value_width, percent_slider_rect.bottom + 2, pct_value_width, 16)
        UILabel(
            max_label_rect,
            "100",
            manager,
            container=self,
            object_id="#weather_max_label",
        )

        buttons_y = max_label_rect.bottom + 6
        button_height = 26
        button_spacing = 6
        button_width = max(70, int((container_width - (2 * margin) - (3 * button_spacing)) / 4))
        self.prev_button = UIButton(
            relative_rect=pygame.Rect(margin, buttons_y, button_width, button_height),
            text="Prev",
            manager=manager,
            container=self,
            object_id="#weather_tab_button",
        )
        self.next_button = UIButton(
            relative_rect=pygame.Rect(margin + (button_width + button_spacing), buttons_y, button_width, button_height),
            text="Next",
            manager=manager,
            container=self,
            object_id="#weather_tab_button",
        )
        self.add_button = UIButton(
            relative_rect=pygame.Rect(margin + 2 * (button_width + button_spacing), buttons_y, button_width, button_height),
            text="Add",
            manager=manager,
            container=self,
            object_id="#weather_tab_button",
        )
        self.delete_button = UIButton(
            relative_rect=pygame.Rect(margin + 3 * (button_width + button_spacing), buttons_y, button_width, button_height),
            text="Delete",
            manager=manager,
            container=self,
            object_id="#weather_tab_button",
        )

        top_margin = buttons_y + button_height + 14
        slider_height = 18
        columns = 2 if container_width >= 340 and len(self._specs) > 6 else 1
        items_per_column = math.ceil(len(self._specs) / columns)
        column_width = max(170, int((container_width - (margin * (columns + 1))) / max(1, columns)))
        row_height = 58
        value_label_width = 58

        for index, spec in enumerate(self._specs):
            column_index = index // items_per_column
            row_index = index % items_per_column
            x = margin + column_index * (column_width + margin)
            y = top_margin + row_index * row_height

            label_rect = pygame.Rect(x, y, max(90, column_width - value_label_width - 8), 20)
            UILabel(
                label_rect,
                spec.display_name,
                manager,
                container=self,
                object_id="#weather_name_label",
            )

            value_label_rect = pygame.Rect(x + column_width - value_label_width, y, value_label_width, 18)
            value_label = UILabel(
                value_label_rect,
                "",
                manager,
                container=self,
                object_id="#weather_value_label",
            )

            slider_rect = pygame.Rect(x, y + 20, column_width, slider_height)
            slider = UIHorizontalSlider(
                relative_rect=slider_rect,
                start_value=spec.min_value,
                value_range=(spec.min_value, spec.max_value),
                manager=manager,
                container=self,
                object_id="#weather_slider",
                click_increment=spec.step,
            )

            min_label_rect = pygame.Rect(x, slider_rect.bottom + 2, value_label_width, 16)
            UILabel(
                min_label_rect,
                self._format_value(spec.min_value, spec.decimals),
                manager,
                container=self,
                object_id="#weather_min_label",
            )

            max_label_rect = pygame.Rect(x + column_width - value_label_width, slider_rect.bottom + 2, value_label_width, 16)
            UILabel(
                max_label_rect,
                self._format_value(spec.max_value, spec.decimals),
                manager,
                container=self,
                object_id="#weather_max_label",
            )

            self.sliders[spec.name] = slider
            self.value_labels[spec.name] = value_label
            self._slider_to_param[slider] = spec.name

        self.update_keyframe_metadata(
            count=self._keyframe_count,
            index=self._active_index,
            percentage=self._active_percentage,
            percent_editable=self._percent_editable,
            can_delete=self._can_delete,
        )

    def _format_value(self, value: float, decimals: int) -> str:
        if decimals <= 0:
            return f"{int(round(value))}"
        return f"{value:.{decimals}f}"

    def _update_value_label(self, name: str, value: float) -> None:
        spec = self._spec_lookup.get(name)
        label = self.value_labels.get(name)
        if not spec or not label:
            return
        label.set_text(self._format_value(value, spec.decimals))

    def apply_weather(self, weather: "carla.WeatherParameters") -> None:
        """Update sliders to match the provided weather object."""
        self._suppress_callback = True
        try:
            for spec in self._specs:
                slider = self.sliders.get(spec.name)
                if not slider:
                    continue
                raw_value = getattr(weather, spec.name, spec.min_value)
                clamped = max(spec.min_value, min(spec.max_value, float(raw_value)))
                slider.set_current_value(clamped)
                self._update_value_label(spec.name, slider.get_current_value())
        finally:
            self._suppress_callback = False

    def update_keyframe_metadata(
        self,
        *,
        count: int,
        index: int,
        percentage: float,
        percent_editable: bool = True,
        can_delete: bool = False,
        can_add: bool = True,
    ) -> None:
        """Refresh keyframe controls (label, slider, buttons)."""
        self._keyframe_count = max(1, int(count))
        self._active_index = max(0, min(int(index), self._keyframe_count - 1))
        self._active_percentage = max(0.0, min(100.0, float(percentage)))
        self._percent_editable = bool(percent_editable)
        self._can_delete = bool(can_delete)

        label_text = f"Keyframe {self._active_index + 1}/{self._keyframe_count} @ {self._active_percentage:.1f}%"
        if self.keyframe_label:
            self.keyframe_label.set_text(label_text)

        if self.percent_slider:
            self._suppress_callback = True
            try:
                self.percent_slider.set_current_value(self._active_percentage)
            finally:
                self._suppress_callback = False
        if self.percent_value_label:
            self.percent_value_label.set_text(self._format_value(self._active_percentage, 1))

        # Enable/disable controls based on selection
        if self.percent_slider:
            if self._percent_editable:
                self.percent_slider.enable()
            else:
                self.percent_slider.disable()
        if self.delete_button:
            if self._can_delete:
                self.delete_button.enable()
            else:
                self.delete_button.disable()
        if self.add_button:
            if can_add:
                self.add_button.enable()
            else:
                self.add_button.disable()
        if self.prev_button:
            if self._active_index <= 0:
                self.prev_button.disable()
            else:
                self.prev_button.enable()
        if self.next_button:
            if self._active_index >= (self._keyframe_count - 1):
                self.next_button.disable()
            else:
                self.next_button.enable()

    def process_event(self, event: pygame.event.Event) -> bool:
        handled = super().process_event(event)

        if event.type == UI_WINDOW_CLOSE and getattr(event, "ui_element", None) == self:
            if self._on_close:
                self._on_close()
            return True

        if event.type == UI_HORIZONTAL_SLIDER_MOVED and getattr(event, "ui_element", None) == self.percent_slider:
            slider = event.ui_element
            value = float(getattr(event, "value", slider.get_current_value()))
            if self.percent_value_label:
                self.percent_value_label.set_text(self._format_value(value, 1))
            if not self._suppress_callback and self._on_percentage_change:
                self._on_percentage_change(value)
            return True

        if event.type == UI_HORIZONTAL_SLIDER_MOVED and event.ui_element in self._slider_to_param:
            slider = event.ui_element
            param = self._slider_to_param.get(slider)
            if param:
                value = float(getattr(event, "value", slider.get_current_value()))
                self._update_value_label(param, value)
                if not self._suppress_callback:
                    self._active_drag_param = param
                    self._drag_last_value = value
                    if self._on_change_live:
                        self._on_change_live(param, value)
                return True

        if event.type == pygame.USEREVENT:
            user_type = getattr(event, "user_type", None)
            if user_type == UI_HORIZONTAL_SLIDER_MOVED and event.ui_element in self._slider_to_param:
                slider = event.ui_element
                param = self._slider_to_param.get(slider)
                if param:
                    value = float(getattr(event, "value", slider.get_current_value()))
                    self._update_value_label(param, value)
                    if not self._suppress_callback:
                        self._active_drag_param = param
                        self._drag_last_value = value
                        if self._on_change_live:
                            self._on_change_live(param, value)
                    return True
            if user_type == UI_HORIZONTAL_SLIDER_MOVED and getattr(event, "ui_element", None) == self.percent_slider:
                slider = event.ui_element
                value = float(getattr(event, "value", slider.get_current_value()))
                if self.percent_value_label:
                    self.percent_value_label.set_text(self._format_value(value, 1))
                if not self._suppress_callback and self._on_percentage_change:
                    self._on_percentage_change(value)
                return True
            if user_type == UI_BUTTON_PRESSED:
                ui_element = getattr(event, "ui_element", None)
                if ui_element == self.prev_button and self._on_prev_keyframe:
                    self._on_prev_keyframe()
                    return True
                if ui_element == self.next_button and self._on_next_keyframe:
                    self._on_next_keyframe()
                    return True
                if ui_element == self.add_button and self._on_add_keyframe:
                    self._on_add_keyframe()
                    return True
                if ui_element == self.delete_button and self._on_delete_keyframe:
                    self._on_delete_keyframe()
                    return True

        return handled

    def update(self, time_delta: float) -> None:
        """Ensure slider drags commit once the mouse is released."""
        super().update(time_delta)
        if self._suppress_callback:
            return
        if not self._active_drag_param or self._drag_last_value is None:
            return
        try:
            left_pressed = bool(pygame.mouse.get_pressed(num_buttons=3)[0])
        except TypeError:
            # Older pygame versions may not accept num_buttons kwarg
            left_pressed = bool(pygame.mouse.get_pressed()[0])
        if left_pressed:
            return
        self._commit_drag_if_needed()

    def _commit_drag_if_needed(self) -> bool:
        """Commit the last dragged slider value once."""
        if self._suppress_callback:
            return False
        if not self._active_drag_param or self._drag_last_value is None:
            return False
        if self._on_change_commit:
            self._on_change_commit(self._active_drag_param, float(self._drag_last_value))
        self._active_drag_param = None
        self._drag_last_value = None
        return True

    @classmethod
    def _ensure_theme(cls, manager: UIManager) -> None:
        """Load compact font/slider theme overrides once."""
        if cls._theme_initialized:
            return
        theme = {
            "#weather_name_label": {
                "font": {"size": "18"},
                "misc": {"text_horiz_alignment": "left"}
            },
            "#weather_value_label": {
                "font": {"size": "16"},
                "misc": {"text_horiz_alignment": "right"}
            },
            "#weather_min_label": {
                "font": {"size": "14"},
                "misc": {"text_horiz_alignment": "left"}
            },
            "#weather_max_label": {
                "font": {"size": "14"},
                "misc": {"text_horiz_alignment": "right"}
            },
            "#weather_slider": {
                "misc": {
                    "sliding_button_width": "14"
                }
            }
        }
        try:
            manager.get_theme().load_theme(theme)
        except Exception as exc:
            print(f"[Weather] Warning: failed to apply weather theme overrides: {exc}")
        cls._theme_initialized = True
