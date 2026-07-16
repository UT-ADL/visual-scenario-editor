"""Small pygame_gui widgets (moved verbatim from vse.py):
save-confirmation dialog and the hover tooltip system.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

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

class UISaveConfirmationDialog(UIWindow):
    """A three-button dialog for unsaved-changes prompts.

    Presents **Save**, **Don't Save**, and **Cancel** buttons so the user can
    choose to persist changes, discard them, or abort the pending action.
    """

    def __init__(self, rect, action_long_desc, manager, *,
                 window_title="Unsaved Changes", blocking=True):
        super().__init__(rect, manager,
                         window_display_title=window_title,
                         element_id=['save_confirmation_dialog'],
                         resizable=False,
                         always_on_top=True)

        self.set_minimum_dimensions((340, 200))

        # --- buttons (right-to-left along the bottom) ---
        self.cancel_button = UIButton(
            relative_rect=pygame.Rect(-10, -40, -1, 30),
            text="Cancel",
            manager=self.ui_manager,
            container=self,
            object_id='#cancel_button',
            anchors={'left': 'right', 'right': 'right',
                     'top': 'bottom', 'bottom': 'bottom'})

        self.dont_save_button = UIButton(
            relative_rect=pygame.Rect(-10, -40, -1, 30),
            text="Don't Save",
            manager=self.ui_manager,
            container=self,
            object_id='#dont_save_button',
            anchors={'left': 'right', 'right': 'right',
                     'top': 'bottom', 'bottom': 'bottom',
                     'left_target': self.cancel_button,
                     'right_target': self.cancel_button})

        self.save_button = UIButton(
            relative_rect=pygame.Rect(-10, -40, -1, 30),
            text="Save",
            manager=self.ui_manager,
            container=self,
            object_id='#save_button',
            anchors={'left': 'right', 'right': 'right',
                     'top': 'bottom', 'bottom': 'bottom',
                     'left_target': self.dont_save_button,
                     'right_target': self.dont_save_button})

        # --- description text ---
        text_width = self.get_container().get_size()[0] - 10
        text_height = self.get_container().get_size()[1] - 50
        self.confirmation_text = UITextBox(
            html_text=action_long_desc,
            relative_rect=pygame.Rect(5, 5, text_width, text_height),
            manager=self.ui_manager,
            container=self,
            anchors={'left': 'left', 'right': 'right',
                     'top': 'top', 'bottom': 'bottom'})

        self.set_blocking(blocking)


############################################################
# Tooltip System
############################################################


class TooltipManager:
    """
    Unified tooltip system for the Visual Scenario Editor.
    Tracks hover state across all UI elements and renders tooltips with configurable delay.
    """

    DELAY_MS = 500  # Milliseconds before tooltip appears
    BACKGROUND_COLOR = (50, 50, 50, 230)  # Semi-transparent dark gray
    BORDER_COLOR = (150, 150, 150)  # Light gray border
    TEXT_COLOR = (255, 255, 255)  # White text
    PADDING = 6  # Padding around text
    BORDER_WIDTH = 1  # Border thickness
    VERTICAL_OFFSET = 6  # Gap between element and tooltip

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font: Optional[pygame.font.Font] = None  # Initialized lazily

        # Hover tracking state
        self._current_element_id: Optional[str] = None
        self._hover_start_time: Optional[float] = None
        self._element_rect: Optional[pygame.Rect] = None
        self._tooltip_text: str = ""
        self._visible = False

        # Frame-based tracking to handle clear/register cycle
        self._frame_element_id: Optional[str] = None
        self._frame_element_rect: Optional[pygame.Rect] = None
        self._frame_tooltip_text: str = ""
        self._frame_text_color: Optional[Tuple[int, int, int]] = None
        self._text_color: Optional[Tuple[int, int, int]] = None

    def _get_font(self) -> pygame.font.Font:
        """Lazily initialize font (pygame must be initialized first)."""
        if self.font is None:
            self.font = pygame.font.Font(None, 18)
        return self.font

    def update_screen_size(self, width: int, height: int) -> None:
        """Update screen dimensions for boundary checking."""
        self.screen_width = width
        self.screen_height = height

    def register_hover(
        self,
        element_id: str,
        element_rect: pygame.Rect,
        tooltip_text: str,
        text_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Register that the mouse is hovering over an element.
        Call this for each hovered element during the render loop.
        """
        if not tooltip_text:
            return

        # Store what's being hovered THIS frame
        self._frame_element_id = element_id
        self._frame_element_rect = element_rect
        self._frame_tooltip_text = tooltip_text
        self._frame_text_color = text_color

    def clear_hover(self) -> None:
        """Reset frame state at the start of each frame."""
        self._frame_element_id = None
        self._frame_element_rect = None
        self._frame_tooltip_text = ""
        self._frame_text_color = None

    def update(self) -> None:
        """Update tooltip visibility based on hover duration."""
        # Compare this frame's hover to the tracked element
        if self._frame_element_id is None:
            # Nothing hovered this frame - clear state
            self._current_element_id = None
            self._hover_start_time = None
            self._element_rect = None
            self._tooltip_text = ""
            self._visible = False
            return

        if self._frame_element_id != self._current_element_id:
            # Switched to a new element - reset timer
            self._current_element_id = self._frame_element_id
            self._hover_start_time = time.time()
            self._visible = False

        # Update rect and text from this frame
        self._element_rect = self._frame_element_rect
        self._tooltip_text = self._frame_tooltip_text
        self._text_color = self._frame_text_color

        # Check if delay has passed
        if self._hover_start_time is not None:
            elapsed_ms = (time.time() - self._hover_start_time) * 1000
            self._visible = elapsed_ms >= self.DELAY_MS

    def render(self, screen: pygame.Surface) -> None:
        """Render the tooltip if visible."""
        if not self._visible or not self._tooltip_text or not self._element_rect:
            return

        font = self._get_font()
        text_surface = font.render(self._tooltip_text, True, self._text_color or self.TEXT_COLOR)
        text_width = text_surface.get_width()
        text_height = text_surface.get_height()

        # Calculate tooltip dimensions
        tooltip_width = text_width + self.PADDING * 2
        tooltip_height = text_height + self.PADDING * 2

        # Position tooltip ABOVE the element
        tooltip_x = self._element_rect.centerx - tooltip_width // 2
        tooltip_y = self._element_rect.top - tooltip_height - self.VERTICAL_OFFSET

        # Boundary checking - keep tooltip on screen
        tooltip_x = max(4, min(tooltip_x, self.screen_width - tooltip_width - 4))

        # If tooltip would go off top of screen, position below element instead
        if tooltip_y < 4:
            tooltip_y = self._element_rect.bottom + self.VERTICAL_OFFSET

        # Final vertical clamping
        tooltip_y = max(4, min(tooltip_y, self.screen_height - tooltip_height - 4))

        # Create tooltip rectangle
        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)

        # Draw background with alpha
        tooltip_surface = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
        tooltip_surface.fill(self.BACKGROUND_COLOR)
        screen.blit(tooltip_surface, tooltip_rect.topleft)

        # Draw border
        pygame.draw.rect(screen, self.BORDER_COLOR, tooltip_rect, self.BORDER_WIDTH)

        # Draw text
        text_x = tooltip_x + self.PADDING
        text_y = tooltip_y + self.PADDING
        screen.blit(text_surface, (text_x, text_y))


