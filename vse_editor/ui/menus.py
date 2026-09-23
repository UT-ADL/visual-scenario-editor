"""Selection menus (moved verbatim from vse.py): actor/vehicle/pedestrian/
ego blueprint menus, traffic-light group list, map menu and scenario menu.
Couple to the scene via set_camera_processor()/attribute injection.
"""

from __future__ import annotations

import math
import os
import time
import weakref
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast

import carla
import pygame

from vse_editor.scene_types import TrafficLightGroupData
from vse_editor.ui.widgets import TooltipManager

class PlacementMode(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    EGO = "ego"
    TRIGGER = "trigger"


class ActorSelectionMenu:
    """Generic dropdown menu for selecting CARLA actor blueprints."""

    def __init__(self, *, title, blueprint_filter, fallback_id, fallback_name, display_transform=None):
        self.title = title
        self.blueprint_filter = blueprint_filter
        self.fallback_id = fallback_id
        self.fallback_name = fallback_name
        self.display_transform = display_transform or (lambda actor_id: actor_id)

        self.menu_width = 300
        self.menu_height = 200  # Height when dropdown is closed
        self.base_x = 10
        self.base_y = 120
        self.dropdown_open = False
        self.on_dropdown_open: Optional[Callable[["ActorSelectionMenu"], None]] = None
        self.selected_index = 0
        self.available_ids = []
        self.display_names = []
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)

        # Scrolling support
        self.scroll_offset = 0
        self.max_visible_items = 10
        self.item_height = 25

        # UI colors
        self.bg_color = (40, 40, 40, 200)
        self.dropdown_color = (60, 60, 60)
        self.hover_color = (80, 80, 80)
        self.text_color = (255, 255, 255)
        self.border_color = (100, 100, 100)

    def initialize(self, world):
        """Populate menu options from CARLA blueprints."""
        try:
            blueprint_library = world.get_blueprint_library()
            blueprints = blueprint_library.filter(self.blueprint_filter)

            actor_entries = []
            for bp in blueprints:
                actor_id = bp.id
                display_name = self.display_transform(actor_id)
                actor_entries.append((actor_id, display_name))

            if not actor_entries:
                raise ValueError("No blueprints found")

            actor_entries.sort(key=lambda entry: entry[1].lower())

            self.available_ids = [entry[0] for entry in actor_entries]
            self.display_names = [entry[1] for entry in actor_entries]
            self.selected_index = min(self.selected_index, len(self.available_ids) - 1)
            self.scroll_offset = 0

            print(f"Found {len(self.available_ids)} {self.title.lower()} types")

        except Exception as e:
            print(f"Error initializing {self.title.lower()} menu: {e}")
            self.available_ids = [self.fallback_id]
            self.display_names = [self.fallback_name]
            self.selected_index = 0
            self.scroll_offset = 0

    def set_vertical_offset(self, top_offset):
        """Update the vertical offset for rendering and hit detection."""
        self.base_y = top_offset

    def handle_click(self, mouse_pos):
        """Handle mouse clicks on the menu."""
        menu_x = self.base_x
        menu_y = self.base_y
        dropdown_y = menu_y + 60
        dropdown_height = 30

        # Click on dropdown header toggles the list
        if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width - 20 and
            dropdown_y <= mouse_pos[1] <= dropdown_y + dropdown_height):
            new_state = not self.dropdown_open
            if new_state and callable(self.on_dropdown_open):
                try:
                    self.on_dropdown_open(self)
                except Exception as exc:
                    print(f"[UI] Failed to close other menus: {exc}")
            self.dropdown_open = new_state
            return True

        # Handle item selection
        if self.dropdown_open:
            visible_items = min(self.max_visible_items, len(self.display_names) - self.scroll_offset)
            for i in range(visible_items):
                actual_index = i + self.scroll_offset
                item_y = dropdown_y + dropdown_height + (i * self.item_height)
                if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width - 20 and
                    item_y <= mouse_pos[1] <= item_y + self.item_height):
                    self.selected_index = actual_index
                    self.dropdown_open = False
                    return True

        # Click outside closes dropdown
        if self.dropdown_open:
            if not (menu_x <= mouse_pos[0] <= menu_x + self.menu_width):
                self.dropdown_open = False
                return True

        return False

    def handle_scroll(self, mouse_pos, scroll_direction):
        """Handle mouse wheel scrolling in dropdown."""
        menu_x = self.base_x
        menu_y = self.base_y

        base_height = 120
        dropdown_list_height = self.max_visible_items * self.item_height if self.dropdown_open else 0
        total_menu_height = base_height + dropdown_list_height + 30

        if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width and
            menu_y <= mouse_pos[1] <= menu_y + total_menu_height):

            if self.dropdown_open and dropdown_list_height > 0:
                dropdown_y = menu_y + 60
                dropdown_height = 30

                if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width - 20 and
                    dropdown_y + dropdown_height <= mouse_pos[1] <= dropdown_y + dropdown_height + dropdown_list_height):

                    reversed_scroll = -scroll_direction

                    max_offset = max(0, len(self.display_names) - self.max_visible_items)
                    new_offset = max(0, min(
                        max_offset,
                        self.scroll_offset + reversed_scroll
                    ))

                    if new_offset != self.scroll_offset:
                        self.scroll_offset = new_offset

            return True

        return False

    def get_extra_height(self) -> int:
        """Return additional vertical space needed for subclass-specific UI."""
        return 0

    def render_extra_content(self, screen, menu_x: int, extra_top: int) -> None:
        """Render subclass-specific UI below the dropdown."""
        return

    def _get_spawn_tip_lines(self) -> List[str]:
        """Return context-specific tip lines to render above the dropdown."""
        return []

    def get_selected_actor(self):
        """Return the selected actor blueprint id."""
        if 0 <= self.selected_index < len(self.available_ids):
            return self.available_ids[self.selected_index]
        return None

    def get_blueprint_options(self) -> List[Tuple[str, str]]:
        """Return the currently loaded blueprint IDs and display names."""
        return list(zip(self.available_ids, self.display_names))

    def render(self, screen, tooltip_manager=None):
        """Render the selection menu."""
        self._tooltip_manager = tooltip_manager
        menu_x = self.base_x
        menu_y = self.base_y

        base_height = 120
        dropdown_list_height = 0
        dropdown_padding = 30 if self.dropdown_open else 0

        if self.dropdown_open:
            visible_items = min(self.max_visible_items, len(self.display_names) - self.scroll_offset)
            dropdown_list_height = visible_items * self.item_height

        content_offset = base_height + dropdown_list_height + dropdown_padding
        extra_height = self.get_extra_height()
        total_menu_height = content_offset + extra_height

        menu_surface = pygame.Surface((self.menu_width, total_menu_height), pygame.SRCALPHA)
        menu_surface.fill(self.bg_color)
        screen.blit(menu_surface, (menu_x, menu_y))

        pygame.draw.rect(screen, self.border_color,
                         (menu_x, menu_y, self.menu_width, total_menu_height), 2)

        title_text = self.font.render(self.title, True, self.text_color)
        screen.blit(title_text, (menu_x + 10, menu_y + 10))

        dropdown_y = menu_y + 60
        dropdown_height = 30

        tip_lines = self._get_spawn_tip_lines()
        if tip_lines:
            line_height = self.small_font.get_linesize()
            tip_top = dropdown_y - (line_height * len(tip_lines)) - 4
            for idx, tip in enumerate(tip_lines):
                tip_surface = self.small_font.render(tip, True, (200, 200, 200))
                screen.blit(tip_surface, (menu_x + 10, tip_top + idx * line_height))

        dropdown_rect = pygame.Rect(menu_x + 10, dropdown_y, self.menu_width - 40, dropdown_height)
        pygame.draw.rect(screen, self.dropdown_color, dropdown_rect)
        pygame.draw.rect(screen, self.border_color, dropdown_rect, 1)

        if self.selected_index < len(self.display_names):
            selected_text = self.display_names[self.selected_index]
            if len(selected_text) > 25:
                selected_text = selected_text[:22] + "..."

            text_surface = self.small_font.render(selected_text, True, self.text_color)
            screen.blit(text_surface, (menu_x + 15, dropdown_y + 8))

        arrow_x = menu_x + self.menu_width - 30
        arrow_y = dropdown_y + 15
        if self.dropdown_open:
            pygame.draw.polygon(screen, self.text_color, [
                (arrow_x, arrow_y + 5),
                (arrow_x + 10, arrow_y - 5),
                (arrow_x + 20, arrow_y + 5)
            ])
        else:
            pygame.draw.polygon(screen, self.text_color, [
                (arrow_x, arrow_y - 5),
                (arrow_x + 10, arrow_y + 5),
                (arrow_x + 20, arrow_y - 5)
            ])

        if self.dropdown_open:
            visible_items = min(self.max_visible_items, len(self.display_names) - self.scroll_offset)

            list_rect = pygame.Rect(menu_x + 10, dropdown_y + dropdown_height,
                                    self.menu_width - 40, dropdown_list_height)
            pygame.draw.rect(screen, self.dropdown_color, list_rect)
            pygame.draw.rect(screen, self.border_color, list_rect, 1)

            mouse_pos = pygame.mouse.get_pos()
            for i in range(visible_items):
                actual_index = i + self.scroll_offset
                item_y = dropdown_y + dropdown_height + (i * self.item_height)
                item_rect = pygame.Rect(menu_x + 10, item_y, self.menu_width - 40, self.item_height)

                if item_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, item_rect)

                if actual_index < len(self.display_names):
                    item_text = self.display_names[actual_index]
                    if len(item_text) > 25:
                        item_text = item_text[:22] + "..."

                    text_surface = self.small_font.render(item_text, True, self.text_color)
                    screen.blit(text_surface, (menu_x + 15, item_y + 5))

            if self.scroll_offset > 0:
                pygame.draw.polygon(screen, (150, 150, 150), [
                    (menu_x + self.menu_width - 15, dropdown_y + dropdown_height + 5),
                    (menu_x + self.menu_width - 10, dropdown_y + dropdown_height),
                    (menu_x + self.menu_width - 5, dropdown_y + dropdown_height + 5)
                ])

            if self.scroll_offset + self.max_visible_items < len(self.display_names):
                bottom_y = dropdown_y + dropdown_height + dropdown_list_height
                pygame.draw.polygon(screen, (150, 150, 150), [
                    (menu_x + self.menu_width - 15, bottom_y - 5),
                    (menu_x + self.menu_width - 10, bottom_y),
                    (menu_x + self.menu_width - 5, bottom_y - 5)
                ])

        if extra_height > 0:
            extra_origin = menu_y + content_offset
            self.render_extra_content(screen, menu_x, extra_origin)

    def cleanup(self):
        """Cleanup placeholder for future resources."""
        pass


class VehicleSelectionMenu(ActorSelectionMenu):
    """Vehicle-specific selection menu wrapper."""

    _EGO_MODEL_ID = "vehicle.lexus.utlexus"
    _SPAWNED_HEADER_MARGIN = 18
    _SPAWNED_HEADER_HEIGHT = 24
    _SPAWNED_BOTTOM_PADDING = 30

    def __init__(self):
        super().__init__(
            title="NPC Vehicles",
            blueprint_filter="vehicle.*",
            fallback_id="vehicle.tesla.model3",
            fallback_name="Tesla Model 3",
            display_transform=self._format_vehicle_name
        )
        self._camera_processor_ref: Optional["weakref.ReferenceType[CameraImageProcessor]"] = None
        self.spawned_scroll_offset = 0
        self._spawned_layout: Optional[Dict[str, object]] = None
        self._spawned_item_hitboxes: List[Tuple[int, pygame.Rect]] = []
        self._spawned_section_rect: Optional[pygame.Rect] = None
        self._last_spawned_click_id: Optional[int] = None
        self._last_spawned_click_time = 0
        self._double_click_ms = 350
        self._placeholder_text = "No vehicles added yet"

    @staticmethod
    def _format_vehicle_name(actor_id):
        return actor_id.replace('vehicle.', '').replace('_', ' ').title()

    def set_camera_processor(self, camera_processor: Optional["CameraImageProcessor"]) -> None:
        self._camera_processor_ref = weakref.ref(camera_processor) if camera_processor else None
        self._spawned_layout = None
        self._spawned_item_hitboxes = []
        self._spawned_section_rect = None

    def _get_camera_processor(self) -> Optional["CameraImageProcessor"]:
        if not self._camera_processor_ref:
            return None
        return self._camera_processor_ref()

    def initialize_vehicles(self, world):
        self.initialize(world)

        # Remove ego blueprint from NPC list if present
        if self._EGO_MODEL_ID in self.available_ids:
            idx = self.available_ids.index(self._EGO_MODEL_ID)
            del self.available_ids[idx]
            del self.display_names[idx]
        if self.available_ids:
            self.selected_index = min(self.selected_index, len(self.available_ids) - 1)
        else:
            self.available_ids = [self.fallback_id]
            self.display_names = [self.fallback_name]
            self.selected_index = 0
        self.scroll_offset = 0

    def get_selected_vehicle(self):
        return self.get_selected_actor()

    def _build_spawned_entries(self) -> List[Dict[str, Union[int, str]]]:
        processor = self._get_camera_processor()
        entries: List[Dict[str, Union[int, str]]] = []
        if not processor:
            return entries

        external_ego_id = getattr(processor, "external_ego_actor_id", None)

        for actor in getattr(processor, "spawned_vehicles", []):
            if not actor or not actor.is_alive:
                continue
            if actor.type_id.startswith('walker.'):
                continue
            if actor.type_id == self._EGO_MODEL_ID:
                continue
            if processor.is_ego_vehicle(actor.id):
                continue
            if external_ego_id and actor.id == external_ego_id:
                continue
            entries.append({
                'id': actor.id,
                'name': self._format_vehicle_name(actor.type_id),
            })
        entries.sort(key=lambda item: str(item['name']).lower())
        return entries

    def _get_selected_vehicle_id(self) -> Optional[int]:
        processor = self._get_camera_processor()
        if not processor or not processor.selected_vehicle or processor.selected_vehicle_is_pedestrian:
            return None
        return processor.selected_vehicle.id

    def _get_spawn_tip_lines(self) -> List[str]:
        return [
            "Tip: Ctrl+Left click to spawn snapped to lane",
            "Tip: Ctrl+Shift+Left click for free placement",
        ]

    def _ensure_spawned_layout(self, *, force: bool = False) -> Dict[str, object]:
        if self._spawned_layout is not None and not force:
            return self._spawned_layout

        entries = self._build_spawned_entries()
        max_scroll = max(0, len(entries) - self.max_visible_items)
        if self.spawned_scroll_offset > max_scroll:
            self.spawned_scroll_offset = max_scroll

        visible_entries = entries[self.spawned_scroll_offset:self.spawned_scroll_offset + self.max_visible_items]
        list_rows = max(len(visible_entries), 1)
        list_height = list_rows * self.item_height

        dropdown_list_height = 0
        dropdown_padding = 30 if self.dropdown_open else 0
        if self.dropdown_open:
            visible_dropdown = min(self.max_visible_items, len(self.display_names) - self.scroll_offset)
            dropdown_list_height = visible_dropdown * self.item_height

        extra_top = self.base_y + 120 + dropdown_list_height + dropdown_padding
        tip_height = self.small_font.get_height() + 2
        header_block_height = self._SPAWNED_HEADER_HEIGHT + tip_height
        section_height = (
            self._SPAWNED_HEADER_MARGIN +
            header_block_height +
            list_height +
            self._SPAWNED_BOTTOM_PADDING
        )
        section_rect = pygame.Rect(self.base_x, extra_top, self.menu_width, section_height)
        header_y = section_rect.y + self._SPAWNED_HEADER_MARGIN
        list_top = header_y + header_block_height + 6
        list_rect = pygame.Rect(self.base_x + 10, list_top, self.menu_width - 40, list_height)

        visible_rows: List[Tuple[Dict[str, Union[int, str]], pygame.Rect]] = []
        row_y = list_rect.y
        for entry in visible_entries:
            row_rect = pygame.Rect(list_rect.x, row_y, list_rect.width, self.item_height)
            visible_rows.append((entry, row_rect))
            row_y += self.item_height

        self._spawned_item_hitboxes = [(entry['id'], rect) for entry, rect in visible_rows]
        self._spawned_section_rect = section_rect
        layout: Dict[str, object] = {
            'entries': entries,
            'visible_rows': visible_rows,
            'section_rect': section_rect,
            'list_rect': list_rect,
            'header_y': header_y,
            'tip_y': header_y + self._SPAWNED_HEADER_HEIGHT,
            'total_height': section_height,
            'count': len(entries),
            'selected_vehicle_id': self._get_selected_vehicle_id(),
        }
        self._spawned_layout = layout
        return layout

    def get_extra_height(self) -> int:
        layout = self._ensure_spawned_layout(force=True)
        return int(layout.get('total_height', 0))

    def render_extra_content(self, screen, menu_x: int, extra_top: int) -> None:
        layout = self._ensure_spawned_layout()
        section_rect = layout.get('section_rect')
        list_rect = layout.get('list_rect')
        header_y = layout.get('header_y', extra_top)
        if not isinstance(section_rect, pygame.Rect) or not isinstance(list_rect, pygame.Rect):
            return

        header_text = f"Placed Vehicles ({layout.get('count', 0)})"
        header_surface = self.small_font.render(header_text, True, self.text_color)
        screen.blit(header_surface, (menu_x + 10, header_y))

        tip_y = layout.get('tip_y', header_y + self._SPAWNED_HEADER_HEIGHT)
        tip_surface = self.small_font.render("Tip: double-click to focus camera", True, (200, 200, 200))
        screen.blit(tip_surface, (menu_x + 10, tip_y))

        pygame.draw.rect(screen, self.dropdown_color, list_rect)
        pygame.draw.rect(screen, self.border_color, list_rect, 1)

        visible_rows = layout.get('visible_rows') or []
        mouse_pos = pygame.mouse.get_pos()
        selected_vehicle_id = layout.get('selected_vehicle_id')

        if visible_rows:
            for entry, row_rect in visible_rows:
                entry_id = entry['id']
                if selected_vehicle_id == entry_id:
                    pygame.draw.rect(screen, (90, 110, 150), row_rect)
                elif row_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, row_rect)

                name = str(entry['name'])
                if len(name) > 25:
                    name = name[:22] + "..."
                text_surface = self.small_font.render(name, True, self.text_color)
                screen.blit(text_surface, (row_rect.x + 5, row_rect.y + 5))
        else:
            placeholder_surface = self.small_font.render(self._placeholder_text, True, (180, 180, 180))
            text_rect = placeholder_surface.get_rect(center=list_rect.center)
            screen.blit(placeholder_surface, text_rect)

        total_entries = layout.get('count', 0)
        visible_count = len(visible_rows)
        if self.spawned_scroll_offset > 0:
            pygame.draw.polygon(screen, (150, 150, 150), [
                (list_rect.right - 10, list_rect.top + 6),
                (list_rect.right - 4, list_rect.top + 12),
                (list_rect.right - 16, list_rect.top + 12),
            ])
        if isinstance(total_entries, int) and total_entries > visible_count + self.spawned_scroll_offset:
            pygame.draw.polygon(screen, (150, 150, 150), [
                (list_rect.right - 10, list_rect.bottom - 6),
                (list_rect.right - 4, list_rect.bottom - 12),
                (list_rect.right - 16, list_rect.bottom - 12),
            ])

    def _process_spawned_entry_click(self, vehicle_id: int) -> bool:
        current_time = pygame.time.get_ticks()
        double_clicked = (
            self._last_spawned_click_id == vehicle_id and
            (current_time - self._last_spawned_click_time) <= self._double_click_ms
        )
        self._last_spawned_click_id = vehicle_id
        self._last_spawned_click_time = current_time

        actor = self._get_actor_by_id(vehicle_id)
        if not actor:
            print("Vehicle is no longer available.")
            self._spawned_layout = None
            return True

        processor = self._get_camera_processor()
        if processor:
            processor.select_vehicle_actor(actor, focus_camera=double_clicked)
        return True

    def _get_actor_by_id(self, vehicle_id: int):
        processor = self._get_camera_processor()
        if not processor:
            return None
        for actor in getattr(processor, "spawned_vehicles", []):
            if actor and actor.id == vehicle_id and actor.is_alive:
                return actor
        return None

    def _handle_spawned_section_click(self, mouse_pos) -> bool:
        layout = self._ensure_spawned_layout()
        section_rect = layout.get('section_rect')
        if not isinstance(section_rect, pygame.Rect) or not section_rect.collidepoint(mouse_pos):
            return False

        if not self._spawned_item_hitboxes:
            return True

        for vehicle_id, item_rect in self._spawned_item_hitboxes:
            if item_rect.collidepoint(mouse_pos):
                return self._process_spawned_entry_click(vehicle_id)
        return True

    def handle_click(self, mouse_pos):
        if self._handle_spawned_section_click(mouse_pos):
            return True
        return super().handle_click(mouse_pos)

    def handle_scroll(self, mouse_pos, scroll_direction):
        layout = self._ensure_spawned_layout()
        list_rect = layout.get('list_rect')
        section_rect = layout.get('section_rect')

        if ((isinstance(list_rect, pygame.Rect) and list_rect.collidepoint(mouse_pos)) or
                (isinstance(section_rect, pygame.Rect) and section_rect.collidepoint(mouse_pos))):
            entries = layout.get('entries', [])
            if isinstance(entries, list) and entries:
                reversed_scroll = -scroll_direction
                max_scroll = max(0, len(entries) - self.max_visible_items)
                new_offset = max(0, min(max_scroll, self.spawned_scroll_offset + reversed_scroll))
                if new_offset != self.spawned_scroll_offset:
                    self.spawned_scroll_offset = new_offset
                    self._spawned_layout = None
                    self._ensure_spawned_layout(force=True)
            return True

        return super().handle_scroll(mouse_pos, scroll_direction)


class EgoVehicleSelectionMenu(ActorSelectionMenu):
    """Selection menu for ego vehicle placement (single blueprint)."""

    _MODEL_ID = "vehicle.lexus.utlexus"
    _EGO_HEADER_MARGIN = 18
    _EGO_HEADER_HEIGHT = 24
    _EGO_ROW_HEIGHT = 28
    _EGO_BOTTOM_PADDING = 30

    def __init__(self):
        super().__init__(
            title="Ego Vehicle",
            blueprint_filter=self._MODEL_ID,
            fallback_id=self._MODEL_ID,
            fallback_name="Lexus Ego Vehicle",
            display_transform=self._format_vehicle_name
        )
        self._camera_processor_ref: Optional["weakref.ReferenceType[CameraImageProcessor]"] = None
        self._double_click_ms = 350
        self._last_row_click_time = 0
        self._ego_tip_height = self.small_font.get_linesize()
        self._ego_section_height = (
            self._EGO_HEADER_MARGIN +
            self._EGO_HEADER_HEIGHT +
            self._ego_tip_height +
            self._EGO_ROW_HEIGHT +
            self._EGO_BOTTOM_PADDING
        )

    def _get_spawn_tip_lines(self) -> List[str]:
        return [
            "Tip: Ctrl+Left click to spawn snapped to lane",
            "Tip: Ctrl+Shift+Left click for free placement",
        ]

    @staticmethod
    def _format_vehicle_name(actor_id):
        return actor_id.replace('vehicle.', '').replace('_', ' ').title()

    def set_camera_processor(self, camera_processor: Optional["CameraImageProcessor"]) -> None:
        self._camera_processor_ref = weakref.ref(camera_processor) if camera_processor else None

    def _get_camera_processor(self) -> Optional["CameraImageProcessor"]:
        if not self._camera_processor_ref:
            return None
        return self._camera_processor_ref()

    def initialize_ego_vehicle(self, world):
        try:
            blueprint_library = world.get_blueprint_library()
            blueprint_library.find(self._MODEL_ID)  # Ensure blueprint exists
            self.available_ids = [self._MODEL_ID]
            self.display_names = [self._format_vehicle_name(self._MODEL_ID)]
            self.selected_index = 0
            self.scroll_offset = 0
        except Exception as exc:
            print(f"Error initializing ego vehicle menu: {exc}")
            self.available_ids = [self.fallback_id]
            self.display_names = [self.fallback_name]
            self.selected_index = 0
            self.scroll_offset = 0

    def get_selected_ego_vehicle(self):
        return self.get_selected_actor()

    def _get_ego_actor(self):
        processor = self._get_camera_processor()
        if not processor:
            return None
        actor = processor.get_editor_ego_actor()
        if actor:
            return actor
        actor = processor.get_ego_vehicle_actor()
        if actor:
            return actor
        ego_id = processor.ego_vehicle_id
        if ego_id:
            for candidate in getattr(processor, "spawned_vehicles", []):
                if candidate and candidate.id == ego_id and candidate.is_alive:
                    return candidate
        return None

    def _compute_ego_layout(self) -> Dict[str, object]:
        base_height = 120
        dropdown_list_height = 0
        dropdown_padding = 30 if self.dropdown_open else 0
        if self.dropdown_open:
            visible_items = min(self.max_visible_items, len(self.display_names) - self.scroll_offset)
            dropdown_list_height = visible_items * self.item_height
        extra_top = self.base_y + base_height + dropdown_list_height + dropdown_padding
        section_rect = pygame.Rect(self.base_x, extra_top, self.menu_width, self._ego_section_height)
        header_y = section_rect.y + self._EGO_HEADER_MARGIN
        row_top = header_y + self._EGO_HEADER_HEIGHT + self._ego_tip_height + 6
        row_rect = pygame.Rect(self.base_x + 10, row_top, self.menu_width - 40, self._EGO_ROW_HEIGHT)
        return {
            'section_rect': section_rect,
            'header_y': header_y,
            'tip_y': header_y + self._EGO_HEADER_HEIGHT,
            'row_rect': row_rect,
        }

    def get_extra_height(self) -> int:
        return self._ego_section_height

    def render_extra_content(self, screen, menu_x: int, extra_top: int) -> None:
        layout = self._compute_ego_layout()
        section_rect = layout['section_rect']
        header_y = layout['header_y']
        tip_y = layout['tip_y']
        row_rect = layout['row_rect']

        processor = self._get_camera_processor()
        actor = self._get_ego_actor()
        header_surface = self.small_font.render("Ego Vehicle Instance", True, self.text_color)
        screen.blit(header_surface, (menu_x + 10, header_y))

        tip_surface = self.small_font.render("Tip: double-click to focus camera", True, (200, 200, 200))
        screen.blit(tip_surface, (menu_x + 10, tip_y))

        pygame.draw.rect(screen, self.dropdown_color, row_rect)
        pygame.draw.rect(screen, self.border_color, row_rect, 1)

        mouse_pos = pygame.mouse.get_pos()
        is_selected = False
        if processor and processor.selected_vehicle and not processor.selected_vehicle_is_pedestrian:
            is_selected = processor.is_ego_vehicle(processor.selected_vehicle.id)

        if actor:
            if is_selected:
                pygame.draw.rect(screen, (90, 110, 150), row_rect)
            elif row_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, self.hover_color, row_rect)
            name = self._format_vehicle_name(actor.type_id)
            label = f"{name} (ID {actor.id})"
        else:
            label = "No ego vehicle spawned"
            if row_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, self.hover_color, row_rect)

        text_surface = self.small_font.render(label, True, self.text_color)
        screen.blit(text_surface, (row_rect.x + 5, row_rect.y + 5))

    def _handle_ego_row_click(self, mouse_pos) -> bool:
        layout = self._compute_ego_layout()
        row_rect = layout['row_rect']
        if not row_rect.collidepoint(mouse_pos):
            return False

        actor = self._get_ego_actor()
        if not actor:
            return True

        current_time = pygame.time.get_ticks()
        double_clicked = (current_time - self._last_row_click_time) <= self._double_click_ms
        self._last_row_click_time = current_time

        processor = self._get_camera_processor()
        if processor:
            processor.select_vehicle_actor(actor, focus_camera=double_clicked)
        return True

    def handle_click(self, mouse_pos):
        if self._handle_ego_row_click(mouse_pos):
            return True
        return super().handle_click(mouse_pos)


class PedestrianSelectionMenu(ActorSelectionMenu):
    """Pedestrian-specific selection menu wrapper."""

    _SPAWNED_HEADER_MARGIN = 18
    _SPAWNED_HEADER_HEIGHT = 24
    _SPAWNED_BOTTOM_PADDING = 30

    def __init__(self):
        super().__init__(
            title="Pedestrian",
            blueprint_filter="walker.pedestrian.*",
            fallback_id="walker.pedestrian.0001",
            fallback_name="Pedestrian 0001",
            display_transform=self._format_pedestrian_name
        )
        self._camera_processor_ref: Optional["weakref.ReferenceType[CameraImageProcessor]"] = None
        self.spawned_scroll_offset = 0
        self._spawned_layout: Optional[Dict[str, object]] = None
        self._spawned_item_hitboxes: List[Tuple[int, pygame.Rect]] = []
        self._spawned_section_rect: Optional[pygame.Rect] = None
        self._last_spawned_click_id: Optional[int] = None
        self._last_spawned_click_time = 0
        self._double_click_ms = 350
        self._placeholder_text = "No pedestrians added yet"

    @staticmethod
    def _format_pedestrian_name(actor_id):
        return actor_id.replace('walker.pedestrian.', 'Pedestrian ').replace('_', ' ').title()

    def set_camera_processor(self, camera_processor: Optional["CameraImageProcessor"]) -> None:
        self._camera_processor_ref = weakref.ref(camera_processor) if camera_processor else None
        self._spawned_layout = None
        self._spawned_item_hitboxes = []
        self._spawned_section_rect = None

    def _get_camera_processor(self) -> Optional["CameraImageProcessor"]:
        if not self._camera_processor_ref:
            return None
        return self._camera_processor_ref()

    def initialize_pedestrians(self, world):
        self.initialize(world)

    def get_selected_pedestrian(self):
        return self.get_selected_actor()

    def _build_spawned_entries(self) -> List[Dict[str, Union[int, str]]]:
        processor = self._get_camera_processor()
        entries: List[Dict[str, Union[int, str]]] = []
        if not processor:
            return entries

        for actor in getattr(processor, "spawned_vehicles", []):
            if not actor or not actor.is_alive:
                continue
            if not actor.type_id.startswith('walker.'):
                continue
            entries.append({
                'id': actor.id,
                'name': self._format_pedestrian_name(actor.type_id),
            })
        entries.sort(key=lambda item: str(item['name']).lower())
        return entries

    def _get_selected_actor_id(self) -> Optional[int]:
        processor = self._get_camera_processor()
        if not processor or not processor.selected_vehicle or not processor.selected_vehicle_is_pedestrian:
            return None
        return processor.selected_vehicle.id

    def _get_spawn_tip_lines(self) -> List[str]:
        return [
            "Tip: Ctrl+Left click to spawn at mouse position",
        ]

    def _ensure_spawned_layout(self, *, force: bool = False) -> Dict[str, object]:
        if self._spawned_layout is not None and not force:
            return self._spawned_layout

        entries = self._build_spawned_entries()
        max_scroll = max(0, len(entries) - self.max_visible_items)
        if self.spawned_scroll_offset > max_scroll:
            self.spawned_scroll_offset = max_scroll

        visible_entries = entries[self.spawned_scroll_offset:self.spawned_scroll_offset + self.max_visible_items]
        list_rows = max(len(visible_entries), 1)
        list_height = list_rows * self.item_height

        dropdown_list_height = 0
        dropdown_padding = 30 if self.dropdown_open else 0
        if self.dropdown_open:
            visible_dropdown = min(self.max_visible_items, len(self.display_names) - self.scroll_offset)
            dropdown_list_height = visible_dropdown * self.item_height

        extra_top = self.base_y + 120 + dropdown_list_height + dropdown_padding
        tip_height = self.small_font.get_height() + 2
        header_block_height = self._SPAWNED_HEADER_HEIGHT + tip_height
        section_height = (
            self._SPAWNED_HEADER_MARGIN +
            header_block_height +
            list_height +
            self._SPAWNED_BOTTOM_PADDING
        )
        section_rect = pygame.Rect(self.base_x, extra_top, self.menu_width, section_height)
        header_y = section_rect.y + self._SPAWNED_HEADER_MARGIN
        list_top = header_y + header_block_height + 6
        list_rect = pygame.Rect(self.base_x + 10, list_top, self.menu_width - 40, list_height)

        visible_rows: List[Tuple[Dict[str, Union[int, str]], pygame.Rect]] = []
        row_y = list_rect.y
        for entry in visible_entries:
            row_rect = pygame.Rect(list_rect.x, row_y, list_rect.width, self.item_height)
            visible_rows.append((entry, row_rect))
            row_y += self.item_height

        self._spawned_item_hitboxes = [(entry['id'], rect) for entry, rect in visible_rows]
        self._spawned_section_rect = section_rect
        layout: Dict[str, object] = {
            'entries': entries,
            'visible_rows': visible_rows,
            'section_rect': section_rect,
            'list_rect': list_rect,
            'header_y': header_y,
            'tip_y': header_y + self._SPAWNED_HEADER_HEIGHT,
            'total_height': section_height,
            'count': len(entries),
            'selected_actor_id': self._get_selected_actor_id(),
        }
        self._spawned_layout = layout
        return layout

    def get_extra_height(self) -> int:
        layout = self._ensure_spawned_layout(force=True)
        return int(layout.get('total_height', 0))

    def render_extra_content(self, screen, menu_x: int, extra_top: int) -> None:
        layout = self._ensure_spawned_layout()
        section_rect = layout.get('section_rect')
        list_rect = layout.get('list_rect')
        header_y = layout.get('header_y', extra_top)
        if not isinstance(section_rect, pygame.Rect) or not isinstance(list_rect, pygame.Rect):
            return

        header_text = f"Placed Pedestrians ({layout.get('count', 0)})"
        header_surface = self.small_font.render(header_text, True, self.text_color)
        screen.blit(header_surface, (menu_x + 10, header_y))

        tip_y = layout.get('tip_y', header_y + self._SPAWNED_HEADER_HEIGHT)
        tip_surface = self.small_font.render("Tip: double-click to focus camera", True, (200, 200, 200))
        screen.blit(tip_surface, (menu_x + 10, tip_y))

        pygame.draw.rect(screen, self.dropdown_color, list_rect)
        pygame.draw.rect(screen, self.border_color, list_rect, 1)

        visible_rows = layout.get('visible_rows') or []
        mouse_pos = pygame.mouse.get_pos()
        selected_actor_id = layout.get('selected_actor_id')

        if visible_rows:
            for entry, row_rect in visible_rows:
                entry_id = entry['id']
                if selected_actor_id == entry_id:
                    pygame.draw.rect(screen, (90, 110, 150), row_rect)
                elif row_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, row_rect)

                name = str(entry['name'])
                if len(name) > 25:
                    name = name[:22] + "..."
                text_surface = self.small_font.render(name, True, self.text_color)
                screen.blit(text_surface, (row_rect.x + 5, row_rect.y + 5))
        else:
            placeholder_surface = self.small_font.render(self._placeholder_text, True, (180, 180, 180))
            text_rect = placeholder_surface.get_rect(center=list_rect.center)
            screen.blit(placeholder_surface, text_rect)

        total_entries = layout.get('count', 0)
        visible_count = len(visible_rows)
        if self.spawned_scroll_offset > 0:
            pygame.draw.polygon(screen, (150, 150, 150), [
                (list_rect.right - 10, list_rect.top + 6),
                (list_rect.right - 4, list_rect.top + 12),
                (list_rect.right - 16, list_rect.top + 12),
            ])
        if isinstance(total_entries, int) and total_entries > visible_count + self.spawned_scroll_offset:
            pygame.draw.polygon(screen, (150, 150, 150), [
                (list_rect.right - 10, list_rect.bottom - 6),
                (list_rect.right - 4, list_rect.bottom - 12),
                (list_rect.right - 16, list_rect.bottom - 12),
            ])

    def _process_spawned_entry_click(self, actor_id: int) -> bool:
        current_time = pygame.time.get_ticks()
        double_clicked = (
            self._last_spawned_click_id == actor_id and
            (current_time - self._last_spawned_click_time) <= self._double_click_ms
        )
        self._last_spawned_click_id = actor_id
        self._last_spawned_click_time = current_time

        actor = self._get_actor_by_id(actor_id)
        if not actor:
            print("Pedestrian is no longer available.")
            self._spawned_layout = None
            return True

        processor = self._get_camera_processor()
        if processor:
            processor.select_vehicle_actor(actor, focus_camera=double_clicked)
        return True

    def _get_actor_by_id(self, actor_id: int):
        processor = self._get_camera_processor()
        if not processor:
            return None
        for actor in getattr(processor, "spawned_vehicles", []):
            if actor and actor.id == actor_id and actor.is_alive and actor.type_id.startswith('walker.'):
                return actor
        return None

    def _handle_spawned_section_click(self, mouse_pos) -> bool:
        layout = self._ensure_spawned_layout()
        section_rect = layout.get('section_rect')
        if not isinstance(section_rect, pygame.Rect) or not section_rect.collidepoint(mouse_pos):
            return False

        if not self._spawned_item_hitboxes:
            return True

        for actor_id, item_rect in self._spawned_item_hitboxes:
            if item_rect.collidepoint(mouse_pos):
                return self._process_spawned_entry_click(actor_id)
        return True

    def handle_click(self, mouse_pos):
        if self._handle_spawned_section_click(mouse_pos):
            return True
        return super().handle_click(mouse_pos)

    def handle_scroll(self, mouse_pos, scroll_direction):
        layout = self._ensure_spawned_layout()
        list_rect = layout.get('list_rect')
        section_rect = layout.get('section_rect')

        if ((isinstance(list_rect, pygame.Rect) and list_rect.collidepoint(mouse_pos)) or
                (isinstance(section_rect, pygame.Rect) and section_rect.collidepoint(mouse_pos))):
            entries = layout.get('entries', [])
            if isinstance(entries, list) and entries:
                reversed_scroll = -scroll_direction
                max_scroll = max(0, len(entries) - self.max_visible_items)
                new_offset = max(0, min(max_scroll, self.spawned_scroll_offset + reversed_scroll))
                if new_offset != self.spawned_scroll_offset:
                    self.spawned_scroll_offset = new_offset
                    self._spawned_layout = None
                    self._ensure_spawned_layout(force=True)
            return True

        return super().handle_scroll(mouse_pos, scroll_direction)


class TrafficLightGroupSelectionMenu:
    """Selection list for traffic light groups (stop lines) that have triggers."""

    _SECTION_MARGIN = 18
    _SECTION_HEADER_HEIGHT = 24
    _SECTION_BOTTOM_PADDING = 30

    def __init__(self):
        self.title = "Triggers"
        self.menu_width = 300
        self.base_x = 10
        self.base_y = 120

        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)

        self.bg_color = (40, 40, 40, 200)
        self.dropdown_color = (60, 60, 60)
        self.hover_color = (80, 80, 80)
        self.text_color = (255, 255, 255)
        self.border_color = (100, 100, 100)

        self.scroll_offset = 0
        self.max_visible_items = 10
        self.item_height = 25

        self._camera_processor_ref: Optional["weakref.ReferenceType[CameraImageProcessor]"] = None
        self._layout: Optional[Dict[str, object]] = None
        self._item_hitboxes: List[Tuple[Tuple[str, Tuple], pygame.Rect]] = []
        self._section_rect: Optional[pygame.Rect] = None

        self._last_click_key: Optional[Tuple[str, Tuple]] = None
        self._last_click_time_ms = 0
        self._double_click_ms = 350

        self._empty_title = "No Traffic Light Group triggers yet"
        self._empty_tip = "Tip: Select a stop line, then click Add Trigger"

        # Global trigger section state
        self._global_trigger_hitbox: Optional[pygame.Rect] = None
        self._last_global_click_time_ms = 0
        self._global_empty_title = "No global trigger"
        self._global_empty_tip = "Tip: Use Place Trigger to add one"

    def set_vertical_offset(self, top_offset: int) -> None:
        self.base_y = int(top_offset)

    def set_camera_processor(self, camera_processor: Optional["CameraImageProcessor"]) -> None:
        self._camera_processor_ref = weakref.ref(camera_processor) if camera_processor else None
        self._layout = None
        self._item_hitboxes = []
        self._section_rect = None

    def _get_camera_processor(self) -> Optional["CameraImageProcessor"]:
        if not self._camera_processor_ref:
            return None
        return self._camera_processor_ref()

    def _stop_line_center_xy(
        self,
        processor: "CameraImageProcessor",
        group: TrafficLightGroupData,
    ) -> Optional[Tuple[float, float]]:
        try:
            center_location = processor._compute_traffic_light_group_trigger_center(group)
        except Exception:
            center_location = None
        if center_location is not None:
            try:
                return float(center_location.x), float(center_location.y)
            except Exception:
                return None

        center = getattr(group, "center_location", None)
        if center:
            try:
                return float(center[0]), float(center[1])
            except Exception:
                return None

        reference = getattr(group, "reference_light", None)
        if reference:
            try:
                location = reference.get_transform().location
                return float(location.x), float(location.y)
            except Exception:
                return None
        return None

    def _build_stop_line_number_map(
        self,
        processor: "CameraImageProcessor",
        groups: List[TrafficLightGroupData],
    ) -> Dict[Tuple[str, Tuple], int]:
        records: List[Tuple[Tuple[float, float], Tuple[str, Tuple]]] = []
        for group in groups:
            key = processor._traffic_light_trigger_key(group=group)
            if not key:
                continue
            center_xy = self._stop_line_center_xy(processor, group)
            if center_xy is None:
                sort_xy = (float("inf"), float("inf"))
            else:
                x, y = center_xy
                sort_xy = (round(y, 2), round(x, 2))
            records.append((sort_xy, key))

        records.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))
        return {key: idx + 1 for idx, (_center, key) in enumerate(records)}

    def _build_entries(self) -> List[Dict[str, object]]:
        processor = self._get_camera_processor()
        if not processor:
            return []

        groups = list(getattr(processor, "traffic_light_groups", []) or [])
        stop_line_number_map = self._build_stop_line_number_map(processor, groups)

        entries: List[Dict[str, object]] = []
        for group in groups:
            center_payload, radius_value, resolved_key = processor._get_traffic_light_trigger_data(group=group)
            if not center_payload or radius_value is None:
                continue
            if not resolved_key:
                resolved_key = processor._traffic_light_trigger_key(group=group)
            if not resolved_key:
                continue

            stop_line_number = stop_line_number_map.get(resolved_key)
            if stop_line_number is None:
                stop_line_number = 0

            light_count = getattr(group, "cached_size", 0) or len(getattr(group, "lights", []) or [])

            entries.append(
                {
                    "key": resolved_key,
                    "group": group,
                    "stop_line_number": int(stop_line_number),
                    "light_count": int(light_count),
                }
            )

        entries.sort(key=lambda entry: int(entry.get("stop_line_number", 0) or 0))
        return entries

    def _get_selected_key(self) -> Optional[Tuple[str, Tuple]]:
        processor = self._get_camera_processor()
        if not processor:
            return None
        group = getattr(processor, "selected_traffic_light_group", None)
        if not group:
            return None
        return processor._traffic_light_trigger_key(group=group)

    def _ensure_layout(self, *, force: bool = False) -> Dict[str, object]:
        if self._layout is not None and not force:
            return self._layout

        entries = self._build_entries()
        max_scroll = max(0, len(entries) - self.max_visible_items)
        if self.scroll_offset > max_scroll:
            self.scroll_offset = max_scroll

        visible_entries = entries[self.scroll_offset:self.scroll_offset + self.max_visible_items]
        # When empty, reserve 2 rows so the centered "No …" title + "Tip: …" line both fit.
        tl_list_rows = len(visible_entries) if visible_entries else 2
        tl_list_height = tl_list_rows * self.item_height

        title_height = self.font.get_height() + 12
        tip_height = self.small_font.get_height()
        header_block_height = self._SECTION_HEADER_HEIGHT + tip_height

        # --- Global trigger section geometry ---
        processor = self._get_camera_processor()
        has_global_trigger = bool(processor and getattr(processor, "triggers", None))
        # 1 row for the actual entry; 2 rows when empty so the title + tip both fit.
        global_list_height = (1 if has_global_trigger else 2) * self.item_height

        global_section_height = (
            header_block_height +
            6 +
            global_list_height +
            self._SECTION_BOTTOM_PADDING
        )

        # --- Traffic light section geometry ---
        tl_section_height = (
            header_block_height +
            6 +
            tl_list_height +
            self._SECTION_BOTTOM_PADDING
        )

        # --- Overall panel ---
        total_height = (
            title_height +
            self._SECTION_MARGIN +
            global_section_height +
            tl_section_height
        )

        section_rect = pygame.Rect(self.base_x, self.base_y, self.menu_width, total_height)

        # Global trigger sub-section positions
        global_header_y = section_rect.y + title_height + self._SECTION_MARGIN
        global_tip_y = global_header_y + self._SECTION_HEADER_HEIGHT
        global_list_top = global_tip_y + tip_height + 6
        global_list_rect = pygame.Rect(
            self.base_x + 10, global_list_top,
            self.menu_width - 40, global_list_height,
        )
        global_row_rect = pygame.Rect(
            global_list_rect.x, global_list_rect.y,
            global_list_rect.width, self.item_height,
        )
        self._global_trigger_hitbox = global_row_rect if has_global_trigger else None

        # Traffic light sub-section positions
        tl_header_y = global_list_top + global_list_height + self._SECTION_BOTTOM_PADDING
        tl_tip_y = tl_header_y + self._SECTION_HEADER_HEIGHT
        tl_list_top = tl_tip_y + tip_height + 6
        tl_list_rect = pygame.Rect(
            self.base_x + 10, tl_list_top,
            self.menu_width - 40, tl_list_height,
        )

        visible_rows: List[Tuple[Dict[str, object], pygame.Rect]] = []
        row_y = tl_list_rect.y
        for entry in visible_entries:
            row_rect = pygame.Rect(tl_list_rect.x, row_y, tl_list_rect.width, self.item_height)
            visible_rows.append((entry, row_rect))
            row_y += self.item_height

        self._item_hitboxes = [
            (cast(Tuple[str, Tuple], entry.get("key")), rect)
            for entry, rect in visible_rows
            if entry.get("key") is not None
        ]
        self._section_rect = section_rect
        layout = {
            "entries": entries,
            "visible_rows": visible_rows,
            "section_rect": section_rect,
            # Global trigger section
            "has_global_trigger": has_global_trigger,
            "global_header_y": global_header_y,
            "global_tip_y": global_tip_y,
            "global_list_rect": global_list_rect,
            "global_row_rect": global_row_rect,
            # Traffic light section
            "tl_list_rect": tl_list_rect,
            "header_y": tl_header_y,
            "tip_y": tl_tip_y,
            "list_rect": tl_list_rect,
            "count": len(entries),
            "selected_key": self._get_selected_key(),
        }
        self._layout = layout
        return layout

    def _resolve_group(self, key: Tuple[str, Tuple]) -> Optional[TrafficLightGroupData]:
        processor = self._get_camera_processor()
        if not processor:
            return None
        group = processor._find_traffic_light_group_by_key(key)
        if group:
            return group

        layout = self._ensure_layout()
        for entry in layout.get("entries", []) or []:
            if entry.get("key") == key:
                return entry.get("group")
        return None

    def _focus_camera_on_group(self, group: TrafficLightGroupData) -> None:
        processor = self._get_camera_processor()
        if not processor:
            return

        focus_location = None
        try:
            focus_location = processor._compute_traffic_light_group_trigger_center(
                group, stop_line=True)
        except Exception:
            focus_location = None

        if focus_location is None:
            center_payload, _radius_value, _key = processor._get_traffic_light_trigger_data(group=group)
            if center_payload:
                try:
                    focus_location = carla.Location(
                        float(center_payload.get("x", 0.0)),
                        float(center_payload.get("y", 0.0)),
                        float(center_payload.get("z", 0.0)),
                    )
                except Exception:
                    focus_location = None

        if focus_location is None:
            reference = getattr(group, "reference_light", None)
            if reference:
                try:
                    focus_location = reference.get_transform().location
                except Exception:
                    focus_location = None

        if focus_location is not None:
            processor.focus_camera_on_location(focus_location)

    def _process_global_trigger_click(self) -> bool:
        processor = self._get_camera_processor()
        if not processor:
            return True

        triggers = getattr(processor, "triggers", None)
        if not triggers:
            return True

        current_time = pygame.time.get_ticks()
        double_clicked = (current_time - self._last_global_click_time_ms) <= self._double_click_ms
        self._last_global_click_time_ms = current_time
        # Reset traffic-light double-click so they don't interfere
        self._last_click_key = None

        if getattr(processor, "placing_trigger", False):
            processor.stop_trigger_placement()

        trigger = triggers[0]

        # Select the global trigger
        processor.selected_trigger_index = 0
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            trigger['x'], trigger['y'], trigger['z']
        )
        if screen_pos.get('success'):
            processor.trigger_action_menu_position = (int(screen_pos['x']), int(screen_pos['y']))
        processor.trigger_menu_hidden_for_camera_pan = False
        # Clear other selections
        if getattr(processor, "selected_personal_trigger", None):
            processor.clear_personal_trigger_selection()
        if getattr(processor, "selected_vehicle", None) or getattr(processor, "vehicle_menu_position", None):
            processor.clear_vehicle_selection()
        if getattr(processor, "selected_traffic_light_group", None):
            processor.selected_traffic_light_group = None
            processor.traffic_light_menu_position = None

        if double_clicked:
            focus_location = carla.Location(
                float(trigger['x']),
                float(trigger['y']),
                float(trigger['z']),
            )
            processor.focus_camera_on_location(focus_location)

        return True

    def _process_entry_click(self, key: Tuple[str, Tuple]) -> bool:
        processor = self._get_camera_processor()
        if not processor:
            return True

        current_time = pygame.time.get_ticks()
        double_clicked = (
            self._last_click_key == key and
            (current_time - self._last_click_time_ms) <= self._double_click_ms
        )
        self._last_click_key = key
        self._last_click_time_ms = current_time

        group = self._resolve_group(key)
        if not group:
            print("Traffic light group is no longer available.")
            self._layout = None
            return True

        if getattr(processor, "placing_trigger", False):
            processor.stop_trigger_placement()

        processor.traffic_lights_visible = True
        if processor.select_traffic_light_group(group) and double_clicked:
            self._focus_camera_on_group(group)
        return True

    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        layout = self._ensure_layout(force=True)
        section_rect = layout.get("section_rect")
        if not isinstance(section_rect, pygame.Rect) or not section_rect.collidepoint(mouse_pos):
            return False

        processor = self._get_camera_processor()
        if processor and getattr(processor, "placing_trigger", False):
            processor.stop_trigger_placement()

        # Check global trigger hitbox first
        if self._global_trigger_hitbox and self._global_trigger_hitbox.collidepoint(mouse_pos):
            return self._process_global_trigger_click()

        for key, row_rect in self._item_hitboxes:
            if row_rect.collidepoint(mouse_pos):
                return self._process_entry_click(key)
        return True

    def handle_scroll(self, mouse_pos: Tuple[int, int], scroll_direction: int) -> bool:
        layout = self._ensure_layout(force=True)
        list_rect = layout.get("list_rect")
        section_rect = layout.get("section_rect")

        if ((isinstance(list_rect, pygame.Rect) and list_rect.collidepoint(mouse_pos)) or
                (isinstance(section_rect, pygame.Rect) and section_rect.collidepoint(mouse_pos))):
            entries = layout.get("entries") or []
            if isinstance(entries, list) and entries:
                reversed_scroll = -scroll_direction
                max_scroll = max(0, len(entries) - self.max_visible_items)
                new_offset = max(0, min(max_scroll, self.scroll_offset + reversed_scroll))
                if new_offset != self.scroll_offset:
                    self.scroll_offset = new_offset
                    self._layout = None
                    self._ensure_layout(force=True)
            return True
        return False

    def render(self, screen, tooltip_manager=None) -> None:
        layout = self._ensure_layout(force=True)
        section_rect = layout.get("section_rect")
        tl_list_rect = layout.get("tl_list_rect")
        tl_header_y = layout.get("header_y", self.base_y + self._SECTION_MARGIN)
        tl_tip_y = layout.get("tip_y", tl_header_y + self._SECTION_HEADER_HEIGHT)
        global_list_rect = layout.get("global_list_rect")
        global_header_y = layout.get("global_header_y", self.base_y + self._SECTION_MARGIN)
        global_tip_y = layout.get("global_tip_y", global_header_y + self._SECTION_HEADER_HEIGHT)
        global_row_rect = layout.get("global_row_rect")
        has_global_trigger = layout.get("has_global_trigger", False)
        if not isinstance(section_rect, pygame.Rect) or not isinstance(tl_list_rect, pygame.Rect):
            return

        # Panel background and border
        section_surface = pygame.Surface(section_rect.size, pygame.SRCALPHA)
        section_surface.fill(self.bg_color)
        screen.blit(section_surface, section_rect.topleft)
        pygame.draw.rect(screen, self.border_color, section_rect, 2)

        # Panel title
        title_surface = self.font.render(self.title, True, self.text_color)
        screen.blit(title_surface, (section_rect.x + 10, section_rect.y + 10))

        mouse_pos = pygame.mouse.get_pos()

        # ---- Global trigger section ----
        global_count = 1 if has_global_trigger else 0
        global_header_text = f"Global Trigger ({global_count})"
        global_header_surface = self.small_font.render(global_header_text, True, self.text_color)
        screen.blit(global_header_surface, (section_rect.x + 10, global_header_y))

        global_tip_text = "Tip: double-click to focus camera"
        global_tip_surface = self.small_font.render(global_tip_text, True, (200, 200, 200))
        screen.blit(global_tip_surface, (section_rect.x + 10, global_tip_y))

        if isinstance(global_list_rect, pygame.Rect):
            pygame.draw.rect(screen, self.dropdown_color, global_list_rect)
            pygame.draw.rect(screen, self.border_color, global_list_rect, 1)

            if has_global_trigger and isinstance(global_row_rect, pygame.Rect):
                processor = self._get_camera_processor()
                trigger = processor.triggers[0] if processor and processor.triggers else None
                is_selected = processor and getattr(processor, "selected_trigger_index", None) == 0

                if is_selected:
                    pygame.draw.rect(screen, (90, 110, 150), global_row_rect)
                elif global_row_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, global_row_rect)

                if trigger:
                    radius_str = f"{trigger.get('radius', 0):.1f}"
                    label = f"Global Trigger  (R: {radius_str}m)"
                else:
                    label = "Global Trigger"
                text_surface = self.small_font.render(label, True, self.text_color)
                screen.blit(text_surface, (global_row_rect.x + 5, global_row_rect.y + 5))
            else:
                empty_surface = self.small_font.render(self._global_empty_title, True, (180, 180, 180))
                tip_surface2 = self.small_font.render(self._global_empty_tip, True, (160, 160, 160))
                center_x = global_list_rect.centerx
                center_y = global_list_rect.centery
                empty_rect = empty_surface.get_rect(center=(center_x, center_y - 8))
                tip_rect2 = tip_surface2.get_rect(center=(center_x, center_y + 10))
                screen.blit(empty_surface, empty_rect)
                screen.blit(tip_surface2, tip_rect2)

        # ---- Traffic light groups section ----
        tl_header_text = f"Traffic Light Groups ({layout.get('count', 0)})"
        tl_header_surface = self.small_font.render(tl_header_text, True, self.text_color)
        screen.blit(tl_header_surface, (section_rect.x + 10, tl_header_y))

        tl_tip_surface = self.small_font.render("Tip: double-click to focus camera", True, (200, 200, 200))
        screen.blit(tl_tip_surface, (section_rect.x + 10, tl_tip_y))

        pygame.draw.rect(screen, self.dropdown_color, tl_list_rect)
        pygame.draw.rect(screen, self.border_color, tl_list_rect, 1)

        visible_rows = layout.get("visible_rows") or []
        selected_key = layout.get("selected_key")

        if visible_rows:
            for entry, row_rect in visible_rows:
                key = entry.get("key")
                if key == selected_key:
                    pygame.draw.rect(screen, (90, 110, 150), row_rect)
                elif row_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, row_rect)

                stop_line_number = entry.get("stop_line_number", 0)
                light_count = entry.get("light_count", 0)
                label = f"Stop Line {stop_line_number} ({light_count} lights)"
                text_surface = self.small_font.render(label, True, self.text_color)
                screen.blit(text_surface, (row_rect.x + 5, row_rect.y + 5))
        else:
            empty_surface = self.small_font.render(self._empty_title, True, (180, 180, 180))
            tip_surface = self.small_font.render(self._empty_tip, True, (160, 160, 160))
            center_x = tl_list_rect.centerx
            center_y = tl_list_rect.centery
            empty_rect = empty_surface.get_rect(center=(center_x, center_y - 8))
            tip_rect = tip_surface.get_rect(center=(center_x, center_y + 10))
            screen.blit(empty_surface, empty_rect)
            screen.blit(tip_surface, tip_rect)

        total_entries = int(layout.get("count", 0) or 0)
        visible_count = len(visible_rows)
        if self.scroll_offset > 0:
            pygame.draw.polygon(screen, (150, 150, 150), [
                (tl_list_rect.right - 10, tl_list_rect.top + 6),
                (tl_list_rect.right - 4, tl_list_rect.top + 12),
                (tl_list_rect.right - 16, tl_list_rect.top + 12),
            ])
        if total_entries > visible_count + self.scroll_offset:
            pygame.draw.polygon(screen, (150, 150, 150), [
                (tl_list_rect.right - 10, tl_list_rect.bottom - 6),
                (tl_list_rect.right - 4, tl_list_rect.bottom - 12),
                (tl_list_rect.right - 16, tl_list_rect.bottom - 12),
            ])



class MapSelectionMenu:
    """Dropdown menu for selecting and loading CARLA maps."""

    def __init__(self, world_handler):
        self.world_handler = world_handler
        self.title = "Open Map"
        self.entries = []
        self.selected_index = 0

        self.menu_width = 250
        self.item_height = 35
        self.max_visible_items = 15
        self.dropdown_open = True

        self.scroll_offset = 0

        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.bg_color = (40, 40, 40, 200)
        self.dropdown_color = (60, 60, 60)
        self.hover_color = (80, 80, 80)
        self.text_color = (255, 255, 255)
        self.border_color = (100, 100, 100)

    def initialize(self):
        """Fetch available maps from the currently connected server."""
        self.entries = []
        self.selected_index = 0
        self.scroll_offset = 0

        profile = getattr(self.world_handler, "connection_profile", None)
        client = getattr(self.world_handler, "client", None)

        target_host = '127.0.0.1'
        target_port = 2000
        entry_type = 'local_map'
        if profile is not None:
            target_host = profile.host
            target_port = profile.port
            if profile.is_remote:
                entry_type = 'remote_map'

        try:
            map_client = client or carla.Client(target_host, target_port)
            map_client.set_timeout(10.0)
            all_maps = map_client.get_available_maps()

            map_entries = []
            name_counts = {}
            for map_path in all_maps:
                map_name = map_path.split('/')[-1]
                base = map_name.split('_Tile_')[0] if '_Tile_' in map_name else map_name
                name_counts[base] = name_counts.get(base, 0) + 1

            for map_path in all_maps:
                map_name = map_path.split('/')[-1]
                base = map_name.split('_Tile_')[0] if '_Tile_' in map_name else map_name
                if '_Tile_' in map_name:
                    continue

                if name_counts[base] > 1:
                    parts = map_path.split('/')
                    if len(parts) >= 3:
                        parent = '/'.join(parts[-3:-1])
                        display = f"{base} ({parent})"
                    else:
                        display = f"{base} ({map_path})"
                else:
                    display = base

                map_entries.append((map_path, display))

            map_entries.sort(key=lambda entry: entry[1].lower())

            for map_path, display_name in map_entries:
                self.entries.append({
                    'type': entry_type,
                    'value': map_path,
                    'display': display_name
                })

            context = 'remote' if entry_type == 'remote_map' else 'local'
            print(f"Found {len(self.entries)} available maps ({context})")
        except Exception as exc:
            print(f"Error fetching available maps: {exc}")

    def set_vertical_offset(self, top_offset):
        self.base_y = top_offset

    def handle_click(self, mouse_pos):
        if not self.entries:
            return False

        button_rect = self.world_handler.open_map_button_rect
        if not button_rect:
            return False

        menu_x = button_rect.right + 5
        menu_y = button_rect.top
        visible_items = min(self.max_visible_items, len(self.entries) - self.scroll_offset)

        for i in range(visible_items):
            actual_index = i + self.scroll_offset
            item_y = menu_y + (i * self.item_height)

            if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width and
                    item_y <= mouse_pos[1] <= item_y + self.item_height):
                entry = self.entries[actual_index]
                entry_type = entry.get('type')
                map_name = entry.get('value')

                self.world_handler.map_menu_visible = False

                if entry_type == 'remote_map':
                    self.world_handler.request_remote_map_change(map_name)
                else:
                    if map_name:
                        print(f"Loading map: {entry.get('display', map_name)}")
                        self.world_handler.load_map(map_name)
                return True

        menu_height = visible_items * self.item_height
        if not (menu_x <= mouse_pos[0] <= menu_x + self.menu_width and
                menu_y <= mouse_pos[1] <= menu_y + menu_height):
            return False

        return False

    def handle_scroll(self, mouse_pos, direction):
        if not self.entries:
            return False

        button_rect = self.world_handler.open_map_button_rect
        if not button_rect:
            return False

        menu_x = button_rect.right + 5
        menu_y = button_rect.top
        visible_items = min(self.max_visible_items, len(self.entries))
        menu_height = visible_items * self.item_height

        if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width and
                menu_y <= mouse_pos[1] <= menu_y + menu_height):
            max_scroll = max(0, len(self.entries) - self.max_visible_items)
            self.scroll_offset = max(0, min(max_scroll, self.scroll_offset - direction))
            return True

        return False

    def render(self, screen, mouse_pos):
        if not self.entries:
            return

        button_rect = self.world_handler.open_map_button_rect
        if not button_rect:
            return

        menu_x = button_rect.right + 5
        menu_y = button_rect.top
        visible_items = min(self.max_visible_items, len(self.entries) - self.scroll_offset)
        menu_height = visible_items * self.item_height

        menu_surface = pygame.Surface((self.menu_width, menu_height), pygame.SRCALPHA)
        menu_surface.fill(self.bg_color)
        screen.blit(menu_surface, (menu_x, menu_y))

        menu_rect = pygame.Rect(menu_x, menu_y, self.menu_width, menu_height)
        pygame.draw.rect(screen, self.border_color, menu_rect, 2)

        current_map = self.world_handler._get_map_display_name()

        for i in range(visible_items):
            actual_index = i + self.scroll_offset
            entry = self.entries[actual_index]
            item_y = menu_y + (i * self.item_height)
            item_rect = pygame.Rect(menu_x, item_y, self.menu_width, self.item_height)

            is_hovered = item_rect.collidepoint(mouse_pos)
            if is_hovered:
                hover_surface = pygame.Surface((self.menu_width, self.item_height), pygame.SRCALPHA)
                hover_surface.fill(self.hover_color)
                screen.blit(hover_surface, (menu_x, item_y))

            if i > 0:
                pygame.draw.line(screen, self.border_color,
                                 (menu_x, item_y),
                                 (menu_x + self.menu_width, item_y), 1)

            item_text = entry.get('display', 'Unknown')
            entry_type = entry.get('type')
            entry_value = entry.get('value', '') or ''
            entry_short = entry_value.split('/')[-1] if entry_value else ''
            is_current = entry_short == current_map

            if entry_type == 'remote_map':
                base_color = (220, 200, 130) if is_current else (200, 200, 255)
            else:
                base_color = (255, 255, 120) if is_current else self.text_color

            text_color = (255, 255, 100) if is_hovered else base_color
            item_surface = self.font.render(item_text, True, text_color)
            text_x = menu_x + 10
            text_y = item_y + (self.item_height - item_surface.get_height()) // 2
            screen.blit(item_surface, (text_x, text_y))

        if len(self.entries) > self.max_visible_items:
            scroll_text = f"{self.scroll_offset + 1}-{self.scroll_offset + visible_items}/{len(self.entries)}"
            scroll_surface = self.small_font.render(scroll_text, True, (200, 200, 200))
            scroll_x = menu_x + self.menu_width - scroll_surface.get_width() - 5
            scroll_y = menu_y + menu_height - scroll_surface.get_height() - 5
            screen.blit(scroll_surface, (scroll_x, scroll_y))

class ScenarioMenu:
    """Menu for saving and loading scenarios."""

    def __init__(self, world_handler):
        self.world_handler = world_handler
        self.title = "Scenario"
        self.options = ["New", "Open", "Save As", "Export .xosc"]

        self.menu_width = 150
        self.item_height = 35

        # UI settings
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.bg_color = (40, 40, 40, 200)
        self.hover_color = (80, 80, 80)
        self.text_color = (255, 255, 255)
        self.border_color = (100, 100, 100)

    def _get_items(self) -> List[Tuple[str, str, bool, Optional[str]]]:
        """Return (action, label, enabled, path) tuples for menu rows."""
        items: List[Tuple[str, str, bool, Optional[str]]] = []
        recent_entries = getattr(self.world_handler, "recent_scenarios", []) or []
        for option in self.options:
            action = option.lower()
            items.append((action, option, True, None))
            if action == "open":
                for entry in recent_entries[:3]:
                    label = entry.get('name') or os.path.basename(entry.get('path', '') or '')
                    path = entry.get('path')
                    enabled = bool(path and os.path.isfile(path))
                    items.append(("recent", label, enabled, path))
        return items

    def handle_click(self, mouse_pos):
        """Handle mouse clicks on the menu."""
        # Get button position from world_handler
        button_rect = self.world_handler.open_scenario_button_rect
        if not button_rect:
            return False

        items = self._get_items()

        # Calculate menu position (to the right of button)
        menu_x = button_rect.right + 5
        menu_y = button_rect.top

        # Check each option for clicks
        for i, (action, label, enabled, path) in enumerate(items):
            item_y = menu_y + (i * self.item_height)

            # Check if click is within this item's bounds
            if (menu_x <= mouse_pos[0] <= menu_x + self.menu_width and
                item_y <= mouse_pos[1] <= item_y + self.item_height):

                # Execute the selected action
                if not enabled:
                    return True
                if action == "recent":
                    print("Opening recent scenario...")
                    self.world_handler.scenario_menu_visible = False
                    if path:
                        self.world_handler._load_scenario_from_path(path)
                    else:
                        self.world_handler._open_last_scenario_from_cache()
                elif action == "new":
                    print("Starting new scenario...")
                    self.world_handler.scenario_menu_visible = False
                    self.world_handler.reset_current_scenario()
                elif action == "open":
                    print("Opening scenario...")
                    self.world_handler.scenario_menu_visible = False
                    self.world_handler.load_scenario_with_dialog()
                elif action == "save as":
                    print("Save As scenario...")
                    self.world_handler.scenario_menu_visible = False
                    self.world_handler.save_scenario_with_dialog()
                elif action == "export .xosc":
                    print("Exporting to OpenSCENARIO...")
                    self.world_handler.scenario_menu_visible = False
                    self.world_handler.export_scenario_as_xosc_with_dialog()
                return True

        # Click outside menu - don't close it here (handled in main event loop)
        return False

    def render(self, screen, mouse_pos):
        """Render the scenario menu (horizontal layout to the right of button)."""
        # Get button position from world_handler
        button_rect = self.world_handler.open_scenario_button_rect
        if not button_rect:
            return

        items = self._get_items()

        # Position menu to the right of the button
        menu_x = button_rect.right + 5
        menu_y = button_rect.top

        # Calculate dimensions
        menu_height = len(items) * self.item_height

        # Draw semi-transparent background
        menu_surface = pygame.Surface((self.menu_width, menu_height), pygame.SRCALPHA)
        menu_surface.fill(self.bg_color)
        screen.blit(menu_surface, (menu_x, menu_y))

        # Draw border
        menu_rect = pygame.Rect(menu_x, menu_y, self.menu_width, menu_height)
        pygame.draw.rect(screen, self.border_color, menu_rect, 2)

        # Draw each option
        for i, (action, label, enabled, _path) in enumerate(items):
            item_y = menu_y + (i * self.item_height)
            item_rect = pygame.Rect(menu_x, item_y, self.menu_width, self.item_height)

            # Highlight hovered item
            is_hovered = item_rect.collidepoint(mouse_pos)
            if is_hovered:
                hover_surface = pygame.Surface((self.menu_width, self.item_height), pygame.SRCALPHA)
                hover_surface.fill(self.hover_color)
                screen.blit(hover_surface, (menu_x, item_y))

            # Draw separator line
            if i > 0:
                pygame.draw.line(screen, self.border_color,
                               (menu_x, item_y),
                               (menu_x + self.menu_width, item_y), 1)

            # Draw option text
            is_recent = action == "recent"
            text_color = (255, 255, 100) if (is_hovered and enabled) else ((180, 180, 180) if not enabled else self.text_color)
            display_label = label
            if len(display_label) > 30:
                display_label = display_label[:27] + "..."
            font = self.small_font if is_recent else self.font
            item_surface = font.render(display_label, True, text_color)
            text_x = menu_x + 10
            if is_recent:
                text_x += 10  # indent sub-row
            text_y = item_y + (self.item_height - item_surface.get_height()) // 2
            screen.blit(item_surface, (text_x, text_y))
