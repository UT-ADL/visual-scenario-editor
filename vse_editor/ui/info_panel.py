"""Info panel inspector (moved verbatim from vse.py): per-actor/waypoint
field editing; edits construct Commands through the camera_processor.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast

import carla
import pygame

from vse_editor.commands import (
    CompositeCommand,
    ReplaceActorBlueprintCommand,
    SetPersonalTriggerCommand,
    UpdateIgnoreFlagsCommand,
    UpdateTrafficLightSequenceCommand,
    UpdateVehiclePropertyCommand,
    UpdateWaypointPropertyCommand,
)
from vse_editor.constants import INFO_PANEL_BUTTON_TOOLTIPS, MIN_PERSONAL_TRIGGER_RADIUS
from vse_editor.scene_types import TrafficLightGroupData

@dataclass
class InfoPanelFieldSpec:
    name: str
    label: str
    kind: str  # 'text' or 'checkbox'
    value: Union[str, bool]
    editable: bool = True
    choices: Optional[List[str]] = None
    metadata: Optional[Dict[str, object]] = None


class InfoPanel:
    """
    UI panel for displaying and editing properties of waypoints, vehicles, and traffic lights.
    Supports property editing, scenario runner integration, and state restoration.
    """
    
    def __init__(self, camera_processor=None):
        self.camera_processor = camera_processor
        self.visible = False
        self.panel_width = 280
        self.panel_height = 400
        self.margin = 10
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.background_color = (40, 40, 40, 230)
        self.text_color = (255, 255, 255)
        self.field_color = (60, 60, 60)
        self.active_field_color = (80, 80, 120)
        self.button_color = (70, 70, 70)
        self.button_hover_color = (90, 90, 90)
        
        # Current selection
        self.selected_object = None
        self.object_type = None  # 'waypoint', 'vehicle', 'pedestrian', or 'traffic_light'
        self.waypoint_vehicle_id = None  # For waypoint references
        self.waypoint_index = None
        
        # Editable fields
        self.fields = {}
        self.field_specs: List[InfoPanelFieldSpec] = []
        self.active_field = None
        self.field_positions = {}
        self.checkbox_fields = set()  # Track which fields are checkboxes
        self.readonly_fields = set()
        self.field_kinds: Dict[str, str] = {}
        self.field_metadata: Dict[str, Dict[str, object]] = {}
        self.dropdown_fields: Dict[str, List[str]] = {}
        self.dropdown_open_field: Optional[str] = None
        self.dropdown_option_rects: List[Tuple[str, str, pygame.Rect]] = []
        self.dropdown_scroll_offsets: Dict[str, int] = {}
        self.dropdown_scrollbar_rects: Dict[str, pygame.Rect] = {}
        self.dropdown_scrollbar_thumb_rects: Dict[str, pygame.Rect] = {}
        self.extra_click_zones: List[Dict[str, object]] = []
        self._traffic_light_sequence_meta: Dict[int, Dict[str, str]] = {}
        self.field_spec_map: Dict[str, InfoPanelFieldSpec] = {}
        self._select_all_on_edit = False

        # Input handling
        self.text_input = ""
        self.editing = False

        # Buttons
        self.buttons = {}
        self.button_positions = {}

        # Scenario execution state is now in WorldHandler
        # (Play button has been moved to top-right corner)
        self.scenario_starting_point = None  # Legacy single marker reference
        self.scenario_starting_points = {}  # Mapping of vehicle_id -> marker data
    
    def show(self, obj, obj_type, screen_width, screen_height, vehicle_id=None, wp_index=None):
        """Show the info panel for a selected object"""
        # Stop any active editing when switching to a new object
        self._stop_editing()
        
        self.visible = True
        self.selected_object = obj
        self.object_type = obj_type
        self.waypoint_vehicle_id = vehicle_id
        self.waypoint_index = wp_index
        
        # Initialize fields based on object type
        self._initialize_fields()
        
        # Initialize buttons based on object type
        self._initialize_buttons()

        # Position panel relative to current window dimensions
        self._update_layout(screen_width, screen_height, recalc_height=True)
    
    def hide(self):
        """Hide the info panel"""
        self.visible = False
        self.selected_object = None
        self.object_type = None
        self.waypoint_vehicle_id = None
        self.waypoint_index = None
        self.active_field = None
        self.editing = False
        self.field_specs = []
        self.fields.clear()
        self.field_positions.clear()
        self.checkbox_fields.clear()
        self.readonly_fields.clear()
        self.buttons.clear()
        self.button_positions.clear()
        self._select_all_on_edit = False
        self.field_kinds.clear()
        self.field_metadata.clear()
        self.dropdown_fields.clear()
        self.dropdown_open_field = None
        self.dropdown_option_rects = []
        self.dropdown_scroll_offsets.clear()
        self.dropdown_scrollbar_rects.clear()
        self.dropdown_scrollbar_thumb_rects.clear()
        self.extra_click_zones = []
        self._traffic_light_sequence_meta = {}
    
    def _calculate_panel_height(self):
        """Calculate panel height based on content"""
        base_height = 100 + self._get_waypoint_tip_block_height()  # Title, padding, and tip (if shown)
        if not self.field_specs:
            self.panel_height = base_height
            return

        total_height = 0
        for spec in self.field_specs:
            if spec.kind == 'checkbox':
                total_height += 48
            else:
                total_height += 42

        button_block = 0
        if self.buttons:
            button_block = 10 + len(self.buttons) * 35

        dropdown_extra = 0
        if self.dropdown_open_field and self.dropdown_open_field in self.dropdown_fields:
            options = self._get_dropdown_visible_options(self.dropdown_open_field)
            dropdown_extra += max(0, len(options)) * 24 + (10 if options else 0)

        self.panel_height = base_height + total_height + button_block + dropdown_extra

    def _should_show_waypoint_split_tip(self) -> bool:
        """Return True if the waypoint split tip should be visible."""
        return self.object_type == 'waypoint'

    def _get_waypoint_tip_block_height(self) -> int:
        """Return the vertical space used by the waypoint tip block."""
        if not self._should_show_waypoint_split_tip():
            return 0
        line_height = self.small_font.get_linesize()
        return line_height + 10

    def _render_waypoint_split_tip(self, surface: pygame.Surface, y_offset: int) -> int:
        """Render the waypoint split hint and return consumed height."""
        if not self._should_show_waypoint_split_tip():
            return 0

        tip_text = "Tip: Ctrl+Left click to split this waypoint"
        tip_surface = self.small_font.render(tip_text, True, (200, 200, 200))
        surface.blit(tip_surface, (self.margin, y_offset))
        return tip_surface.get_height() + 10
    
    def _initialize_fields(self):
        """Initialize editable fields based on object type."""
        specs = self._build_field_specs()
        self._apply_field_specs(specs)
        self.field_positions.clear()

    def _group_waypoints(self):
        """(waypoints, sorted valid indices) for the waypoint-group selection."""
        group = self.selected_object or {}
        vehicle_id = group.get('vehicle_id')
        waypoints = []
        if self.camera_processor and vehicle_id is not None:
            waypoints = self.camera_processor.get_vehicle_waypoints(vehicle_id)
        indices = sorted(i for i in group.get('indices', ()) if 0 <= i < len(waypoints))
        return waypoints, indices

    @staticmethod
    def _uniform_or_multiple(waypoints, indices, key, default, fmt):
        """Shared field value across the group, or 'multiple' when it differs."""
        values = {round(float(waypoints[i].get(key, default)), 3) for i in indices}
        if len(values) == 1:
            return fmt(values.pop())
        return "multiple"

    def _get_actor_blueprint_choices(
        self,
        actor,
    ) -> Tuple[str, List[str], Dict[str, str]]:
        """Return the current actor label, labels, and label-to-ID mapping."""
        editor = getattr(self.camera_processor, 'editor', None)
        menu_name = 'pedestrian_menu' if actor.type_id.startswith('walker.') else 'vehicle_menu'
        menu = getattr(editor, menu_name, None)
        options: List[Tuple[str, str]] = []
        if menu and hasattr(menu, 'get_blueprint_options'):
            options = menu.get_blueprint_options()

        if not options:
            display_transform = getattr(menu, 'display_transform', None)
            if callable(display_transform):
                display_name = display_transform(actor.type_id)
            elif actor.type_id.startswith('walker.'):
                display_name = actor.type_id.replace('walker.pedestrian.', 'Pedestrian ')
            else:
                display_name = actor.type_id.replace('vehicle.', '').replace('_', ' ').title()
            options = [(actor.type_id, display_name)]
        elif actor.type_id not in {blueprint_id for blueprint_id, _ in options}:
            display_transform = getattr(menu, 'display_transform', None)
            if callable(display_transform):
                display_name = display_transform(actor.type_id)
            else:
                display_name = actor.type_id
            options.append((actor.type_id, display_name))

        labels: List[str] = []
        blueprint_by_label: Dict[str, str] = {}
        current_label = actor.type_id
        for blueprint_id, display_name in options:
            label = str(display_name)
            if label in blueprint_by_label and blueprint_by_label[label] != blueprint_id:
                label = f"{label} ({blueprint_id})"
            labels.append(label)
            blueprint_by_label[label] = blueprint_id
            if blueprint_id == actor.type_id:
                current_label = label
        return current_label, labels, blueprint_by_label

    def _get_dropdown_visible_options(self, field_name: str) -> List[str]:
        """Return the visible slice of an open dropdown's options."""
        options = self.dropdown_fields.get(field_name, [])
        metadata = self.field_metadata.get(field_name, {})
        max_visible = metadata.get('max_visible_options', len(options))
        try:
            max_visible = max(1, int(max_visible))
        except (TypeError, ValueError):
            max_visible = len(options)
        offset = self.dropdown_scroll_offsets.get(field_name, 0)
        offset = max(0, min(offset, max(0, len(options) - max_visible)))
        self.dropdown_scroll_offsets[field_name] = offset
        return options[offset:offset + max_visible]

    def _build_field_specs(self) -> List[InfoPanelFieldSpec]:
        specs: List[InfoPanelFieldSpec] = []
        if not self.selected_object:
            return specs

        if self.object_type == 'waypoint_group':
            waypoints, indices = self._group_waypoints()
            if not indices:
                return specs
            numbers = ", ".join(str(i + 1) for i in indices[:8])
            if len(indices) > 8:
                numbers += ", …"
            specs.append(
                InfoPanelFieldSpec(
                    'name', 'Name:', 'text',
                    f"Waypoints {numbers} ({len(indices)} selected)",
                    editable=False,
                )
            )
            # Positions differ per waypoint: shown greyed-out, not editable
            for coord in ('x', 'y', 'z'):
                specs.append(
                    InfoPanelFieldSpec(coord, f'{coord.upper()}:', 'text', 'multiple', editable=False)
                )
            specs.append(
                InfoPanelFieldSpec(
                    'speed_km_h', 'Speed (km/h):', 'text',
                    self._uniform_or_multiple(waypoints, indices, 'speed_km_h', 50, lambda v: f"{v:.0f}"),
                )
            )
            specs.append(
                InfoPanelFieldSpec(
                    'speed_deviation_km_h', 'Speed Deviation (km/h):', 'text',
                    self._uniform_or_multiple(waypoints, indices, 'speed_deviation_km_h', 0, lambda v: f"{v:.0f}"),
                )
            )
            specs.append(
                InfoPanelFieldSpec(
                    'idle_time_s', 'Idle Time (sim s):', 'text',
                    self._uniform_or_multiple(waypoints, indices, 'idle_time_s', 0.0, lambda v: f"{v:.1f}"),
                )
            )
            return specs

        if self.object_type == 'waypoint':
            waypoint = self.selected_object
            waypoint_index = waypoint.get(
                'index',
                self.waypoint_index if self.waypoint_index is not None else 0,
            )

            if waypoint_index == "destination":
                waypoint_name = "Waypoint (Destination)"
            else:
                try:
                    waypoint_name = f"Waypoint {int(waypoint_index) + 1}"
                except (ValueError, TypeError):
                    waypoint_name = f"Waypoint {waypoint_index}"

            speed_value = waypoint.get('speed_km_h', 50)
            if (
                waypoint.get('is_destination')
                and self.camera_processor
                and self.waypoint_vehicle_id is not None
            ):
                cached_speed = self.camera_processor.get_vehicle_destination_speed(
                    self.waypoint_vehicle_id
                )
                if cached_speed is not None:
                    speed_value = cached_speed

            specs.append(
                InfoPanelFieldSpec('name', 'Name:', 'text', waypoint_name, editable=False)
            )
            specs.append(
                InfoPanelFieldSpec('x', 'X:', 'text', f"{waypoint['x']:.2f}")
            )
            specs.append(
                InfoPanelFieldSpec('y', 'Y:', 'text', f"{waypoint['y']:.2f}")
            )
            specs.append(
                InfoPanelFieldSpec('z', 'Z:', 'text', f"{waypoint['z']:.2f}")
            )
            specs.append(
                InfoPanelFieldSpec(
                    'speed_km_h', 'Speed (km/h):', 'text', f"{speed_value:.0f}"
                )
            )
            deviation_value = waypoint.get('speed_deviation_km_h', 0)
            try:
                deviation_value = int(float(deviation_value or 0))
            except Exception:
                deviation_value = 0
            if deviation_value < 0:
                deviation_value = 0
            specs.append(
                InfoPanelFieldSpec(
                    'speed_deviation_km_h',
                    'Speed Deviation (km/h):',
                    'text',
                    str(deviation_value),
                )
            )
            specs.append(
                InfoPanelFieldSpec(
                    'idle_time_s', 'Idle Time (sim s):', 'text', f"{waypoint.get('idle_time_s', 0.0):.1f}"
                )
            )

        elif self.object_type in ('vehicle', 'pedestrian'):
            actor = self.selected_object
            if not actor or not actor.is_alive:
                return specs

            location = actor.get_location()
            rotation = actor.get_transform().rotation
            if self.camera_processor:
                default_actor_speed = 5 if actor.type_id.startswith('walker.') else 50
                actor_speed = self.camera_processor.get_vehicle_speed(actor.id, default_actor_speed)
                idle_time = self.camera_processor.get_actor_idle_time(actor.id, 0.0)
            else:
                actor_speed = 5 if actor.type_id.startswith('walker.') else 50
                idle_time = 0.0

            editor = getattr(self.camera_processor, 'editor', None)
            external_ego_id = getattr(editor, 'external_ego_actor_id', None)
            if external_ego_id is None:
                session = getattr(self.camera_processor, 'session', None)
                external_ego_id = getattr(session, 'external_ego_actor_id', None)
            replaceable = bool(
                self.camera_processor
                and not self.camera_processor.is_ego_vehicle(actor.id)
                and actor.id != external_ego_id
            )
            if replaceable:
                current_label, choices, blueprint_by_label = self._get_actor_blueprint_choices(actor)
                specs.append(
                    InfoPanelFieldSpec(
                        'name',
                        'Blueprint:',
                        'dropdown',
                        current_label,
                        choices=choices,
                        metadata={
                            'type': 'actor_blueprint_replacement',
                            'blueprint_by_label': blueprint_by_label,
                            'max_visible_options': 8,
                        },
                    )
                )
            else:
                actor_type = actor.type_id.split('.')[-1] if hasattr(actor, 'type_id') else "Unknown"
                label_prefix = "Pedestrian" if self.object_type == 'pedestrian' else "Vehicle"
                specs.append(
                    InfoPanelFieldSpec(
                        'name',
                        'Name:',
                        'text',
                        f"{label_prefix} ({actor_type})",
                        editable=False,
                    )
                )
            specs.append(InfoPanelFieldSpec('x', 'X:', 'text', f"{location.x:.2f}"))
            specs.append(InfoPanelFieldSpec('y', 'Y:', 'text', f"{location.y:.2f}"))
            specs.append(InfoPanelFieldSpec('z', 'Z:', 'text', f"{location.z:.2f}"))
            specs.append(InfoPanelFieldSpec('yaw', 'Yaw:', 'text', f"{rotation.yaw:.2f}"))
            specs.append(
                InfoPanelFieldSpec(
                    'speed_km_h', 'Speed (km/h):', 'text', f"{actor_speed:.0f}"
                )
            )
            specs.append(
                InfoPanelFieldSpec('idle_time_s', 'Idle Time (sim s):', 'text', f"{idle_time:.1f}")
            )

            if self.object_type == 'pedestrian':
                # Add trigger radius for pedestrians
                if self.camera_processor and actor.id in self.camera_processor.pedestrian_trigger_radii:
                    trigger_radius = self.camera_processor.pedestrian_trigger_radii[actor.id]
                    specs.append(
                        InfoPanelFieldSpec(
                            'trigger_radius',
                            'Trigger Radius (m):',
                            'text',
                            f"{float(trigger_radius):.2f}",
                            metadata={'type': 'pedestrian_trigger_radius'},
                        )
                    )

            if self.object_type == 'vehicle' and self.camera_processor:
                max_lat_acc_value = self.camera_processor.get_vehicle_max_lat_acc(actor.id, 3.0)
                specs.append(
                    InfoPanelFieldSpec(
                        'max_lat_acc',
                        'Maximum Lateral Acceleration (m/s^2):',
                        'text',
                        f"{float(max_lat_acc_value):.2f}",
                        metadata={'type': 'vehicle_max_lat_acc'},
                    )
                )
                if (actor.id in self.camera_processor.vehicle_trigger_radii and
                        not self.camera_processor.is_ego_vehicle(actor.id)):
                    trigger_radius = self.camera_processor.vehicle_trigger_radii[actor.id]
                    specs.append(
                        InfoPanelFieldSpec(
                            'trigger_radius',
                            'Trigger Radius (m):',
                            'text',
                            f"{float(trigger_radius):.2f}",
                            metadata={'type': 'vehicle_trigger_radius'},
                        )
                    )
                flags = self.camera_processor.get_vehicle_ignore_flags(actor.id)
                specs.append(
                    InfoPanelFieldSpec(
                        'ignore_traffic_lights',
                        'Ignore Traffic Lights:',
                        'checkbox',
                        flags['traffic_lights'],
                    )
                )
                specs.append(
                    InfoPanelFieldSpec(
                        'ignore_stop_signs',
                        'Ignore Stop Signs:',
                        'checkbox',
                        flags['stop_signs'],
                    )
                )
                specs.append(
                    InfoPanelFieldSpec(
                        'ignore_vehicles',
                        'Ignore Vehicles:',
                        'checkbox',
                        flags['vehicles'],
                    )
                )

                # Agent behavior selector (ego vehicle only)
                if self.camera_processor and self.camera_processor.is_ego_vehicle(actor.id):
                    editor = getattr(self.camera_processor, 'editor', None)
                    current_behavior = getattr(editor, 'agent_behavior', 'normal') if editor else 'normal'
                    specs.append(
                        InfoPanelFieldSpec(
                            'agent_behavior',
                            'Autopilot Behavior:',
                            'dropdown',
                            current_behavior.capitalize(),
                            choices=['Cautious', 'Normal', 'Aggressive'],
                            metadata={'type': 'agent_behavior'},
                        )
                    )

        elif self.object_type == 'traffic_light':
            group = self.selected_object  # TrafficLightGroupData
            if not group or not group.lights:
                return specs

            if group.sequence is None:
                group.sequence = []

            has_trigger = group.has_trigger()

            specs.append(
                InfoPanelFieldSpec(
                    'group_size',
                    'Lights in Group:',
                    'text',
                    f"{len(group.lights)}",
                    editable=False,
                )
            )
            specs.append(
                InfoPanelFieldSpec(
                    'light_ids',
                    'Light IDs:',
                    'text',
                    f"{', '.join(str(lid) for lid in sorted(group.ids))}",
                    editable=False,
                )
            )

            # Add trigger radius field if trigger exists
            if has_trigger and group.trigger_radius is not None:
                specs.append(
                    InfoPanelFieldSpec(
                        'trigger_radius',
                        'Trigger Radius (m):',
                        'text',
                        f"{float(group.trigger_radius):.2f}",
                        metadata={'type': 'traffic_light_trigger_radius'},
                    )
                )

            self._traffic_light_sequence_meta = {}
            if has_trigger:
                if group.sequence:
                    specs.append(
                        InfoPanelFieldSpec(
                            'sequence_header',
                            'Sequence Steps:',
                            'text',
                            'Configured order of light states',
                            editable=False,
                            metadata={'type': 'traffic_light_sequence_header'},
                        )
                    )
                else:
                    specs.append(
                        InfoPanelFieldSpec(
                            'sequence_empty',
                            'Sequence Steps:',
                            'text',
                            'No sequence defined. Use "Add Step" to begin.',
                            editable=False,
                            metadata={'type': 'traffic_light_sequence_header'},
                        )
                    )

                for idx, entry in enumerate(group.sequence):
                    color_field = f"seq_color_{idx}"
                    duration_field = f"seq_duration_{idx}"
                    self._traffic_light_sequence_meta[idx] = {
                        'color_field': color_field,
                        'duration_field': duration_field,
                    }

                    specs.append(
                        InfoPanelFieldSpec(
                            color_field,
                            f"Step {idx + 1} Color:",
                            'dropdown',
                            str(entry.get('color', 'Red')),
                            choices=['Red', 'Yellow', 'Green', 'Off'],
                            metadata={'type': 'traffic_light_sequence_color', 'index': idx, 'paired_field': duration_field},
                        )
                    )
                    specs.append(
                        InfoPanelFieldSpec(
                            duration_field,
                            f"Step {idx + 1} Duration (sim s):",
                            'text',
                            f"{float(entry.get('duration_s', 1.0)):.2f}",
                            metadata={'type': 'traffic_light_sequence_duration', 'index': idx, 'paired_color': color_field},
                        )
                    )

        return specs

    def _apply_field_specs(self, specs: List[InfoPanelFieldSpec]):
        self.field_specs = specs
        self.fields = {}
        self.checkbox_fields = set()
        self.readonly_fields = set()
        self.field_positions.clear()
        previous_dropdown = self.dropdown_open_field
        previous_scroll_offsets = self.dropdown_scroll_offsets
        self.field_kinds = {}
        self.field_metadata = {}
        self.dropdown_fields = {}
        self.dropdown_scroll_offsets = {}
        self.dropdown_option_rects = []
        self.dropdown_scrollbar_rects = {}
        self.dropdown_scrollbar_thumb_rects = {}
        self.extra_click_zones = []
        self._traffic_light_sequence_meta = {}

        for spec in specs:
            self.fields[spec.name] = spec.value
            self.field_kinds[spec.name] = spec.kind
            if spec.metadata:
                self.field_metadata[spec.name] = spec.metadata
            if spec.kind == 'checkbox':
                self.checkbox_fields.add(spec.name)
            elif not spec.editable:
                self.readonly_fields.add(spec.name)
            if spec.kind == 'dropdown' and spec.choices:
                self.dropdown_fields[spec.name] = list(spec.choices)
                self.dropdown_scroll_offsets[spec.name] = previous_scroll_offsets.get(spec.name, 0)

        self.field_spec_map = {spec.name: spec for spec in specs}

        if previous_dropdown not in self.dropdown_fields:
            self.dropdown_open_field = None
        else:
            self.dropdown_open_field = previous_dropdown

        self._calculate_panel_height()

    def _initialize_buttons(self):
        """Initialize buttons based on object type"""
        self.buttons.clear()
        self.button_positions.clear()

        if self.object_type == 'traffic_light':
            group = cast(Optional[TrafficLightGroupData], self.selected_object)
            if group and group.has_trigger():
                self.buttons['add_traffic_light_step'] = '+ Add Step'
                if getattr(group, 'sequence', None):
                    self.buttons['clear_traffic_light_steps'] = 'Clear Steps'

    def _update_layout(self, screen_width, screen_height, recalc_height=False):
        """Keep the panel anchored to the window edge during resizes."""
        if not self.visible:
            return

        if recalc_height:
            self._calculate_panel_height()

        # Anchor to the right, but keep a margin so it never floats off-screen
        self.panel_x = max(self.margin, screen_width - self.panel_width - self.margin)

        top_offset = 90
        panel_top = top_offset + self.margin
        if self.camera_processor and getattr(self.camera_processor, 'editor', None):
            editor = self.camera_processor.editor
            top_offset = getattr(editor, 'top_ui_height', top_offset)
            panel_top = getattr(editor, 'side_panel_top', top_offset + self.margin)

        max_panel_y = max(self.margin, screen_height - self.panel_height - self.margin)
        self.panel_y = max(self.margin, min(panel_top, max_panel_y))

    def _refresh_fields(self):
        """Refresh field values in real-time during movement/rotation"""
        if not self.visible or not self.selected_object or self.editing:
            return

        specs = self._build_field_specs()
        self._apply_field_specs(specs)

    # -- targeted refreshes dispatched from the editor's history on_change --
    # (step-21) Bodies moved from the three property commands' editor
    # reach-ins. Unlike _refresh_fields these deliberately run even while
    # self.editing is True: a committed in-panel edit must still update
    # snap_status and sibling fields immediately.

    def refresh_waypoint_property_fields(self, camera_processor, vehicle_id, waypoint_index):
        """Refresh the info panel if it's showing this waypoint"""
        if (self.visible and
                self.object_type == 'waypoint' and
                self.waypoint_vehicle_id == vehicle_id and
                self.waypoint_index == waypoint_index):

            # Get current waypoint data
            waypoints = camera_processor.get_vehicle_waypoints(vehicle_id)
            if waypoint_index >= len(waypoints):
                return
            waypoint = waypoints[waypoint_index]

            # Update all relevant fields
            self.fields['x'] = f"{waypoint['x']:.2f}"
            self.fields['y'] = f"{waypoint['y']:.2f}"
            self.fields['z'] = f"{waypoint['z']:.2f}"
            self.fields['speed_km_h'] = f"{waypoint.get('speed_km_h', 50):.0f}"
            if 'speed_deviation_km_h' in self.fields:
                self.fields['speed_deviation_km_h'] = f"{waypoint.get('speed_deviation_km_h', 0):.0f}"
            if 'idle_time_s' in self.fields:
                self.fields['idle_time_s'] = f"{waypoint.get('idle_time_s', 0.0):.1f}"

    def refresh_vehicle_property_fields(self, camera_processor, vehicle_id,
                                        property_name, value, transform=None):
        """Refresh the info panel if it's showing this vehicle.

        transform=None reads the live actor transform (execute/redo path,
        as the command's execute-side refresh always did); undo passes the
        command's old_transform instead — the live get_transform() can be
        one tick stale right after set_transform in async mode.
        """
        if not (self.visible and self.object_type in ('vehicle', 'pedestrian')):
            return
        vehicle = camera_processor.get_spawned_vehicle(vehicle_id)
        if not vehicle or self.selected_object != vehicle:
            return
        # Refresh the fields to show current values
        idle = camera_processor.get_actor_idle_time(vehicle.id, 0.0)

        if property_name == 'speed_km_h':
            self.fields['speed_km_h'] = str(int(value))
            if 'idle_time_s' in self.fields:
                self.fields['idle_time_s'] = f"{idle:.1f}"
        elif property_name == 'idle_time_s':
            if 'idle_time_s' in self.fields:
                self.fields['idle_time_s'] = f"{value:.1f}"
        elif property_name == 'max_lat_acc':
            if 'max_lat_acc' in self.fields:
                self.fields['max_lat_acc'] = f"{value:.2f}"
        else:
            # For coordinate updates, show the transform values
            if transform is None:
                transform = vehicle.get_transform()
            self.fields['x'] = f"{transform.location.x:.2f}"
            self.fields['y'] = f"{transform.location.y:.2f}"
            self.fields['z'] = f"{transform.location.z:.2f}"
            self.fields['yaw'] = f"{transform.rotation.yaw:.2f}"
            if 'idle_time_s' in self.fields:
                self.fields['idle_time_s'] = f"{idle:.1f}"

    def _apply_actor_blueprint_choice(
        self,
        choice_label: str,
        metadata: Dict[str, object],
    ) -> bool:
        """Apply an actor blueprint choice through the editor history."""
        actor = self.selected_object
        if not actor or not self.camera_processor:
            return False
        blueprint_by_label = metadata.get('blueprint_by_label')
        if not isinstance(blueprint_by_label, dict):
            return False
        new_type_id = blueprint_by_label.get(choice_label)
        if not isinstance(new_type_id, str) or new_type_id == actor.type_id:
            return False

        editor = getattr(self.camera_processor, 'editor', None)
        external_ego_id = getattr(editor, 'external_ego_actor_id', None)
        if external_ego_id is None:
            session = getattr(self.camera_processor, 'session', None)
            external_ego_id = getattr(session, 'external_ego_actor_id', None)
        if (self.camera_processor.is_ego_vehicle(actor.id)
                or actor.id == external_ego_id):
            return False

        command = ReplaceActorBlueprintCommand(
            self.camera_processor,
            actor.id,
            new_type_id,
        )
        if editor:
            success = editor.execute_command(command)
        else:
            success = command.execute()
        if success is False:
            return False

        replacement = self.camera_processor.get_spawned_vehicle(command.vehicle_id)
        if replacement:
            self.selected_object = replacement
        return True

    def set_ignore_flag_field(self, flag_key, value):
        """Sync an ignore-flag checkbox after an UpdateIgnoreFlagsCommand."""
        if not self.visible:
            return
        # Map flag_key -> checkbox field name used by InfoPanel
        field_map = {
            'traffic_lights': 'ignore_traffic_lights',
            'stop_signs': 'ignore_stop_signs',
            'vehicles': 'ignore_vehicles',
        }
        field_name = field_map.get(flag_key)
        if field_name and field_name in self.fields:
            self.fields[field_name] = value
            for spec in self.field_specs:
                if spec.name == field_name:
                    spec.value = value
                    break

    def handle_click(self, mouse_pos):
        """Handle mouse clicks on the info panel"""
        if not self.visible:
            return False
        
        # Check if click is within panel bounds
        if not self._is_point_in_panel(mouse_pos):
            self.dropdown_open_field = None
            return False
        
        # Check custom click zones (e.g., delete buttons)
        for zone in list(self.extra_click_zones):
            rect = zone.get('rect')
            if rect and rect.collidepoint(mouse_pos):
                zone_type = zone.get('type')
                if zone_type == 'traffic_light_delete_step':
                    idx = int(zone.get('index', -1))
                    self._delete_traffic_light_sequence_step(idx)
                    return True

        # Check dropdown option selections before buttons to prioritize list choices
        for field_name, option_value, rect in list(self.dropdown_option_rects):
            if rect.collidepoint(mouse_pos):
                self.dropdown_open_field = None
                metadata = self.field_metadata.get(field_name)
                if metadata and metadata.get('type') == 'actor_blueprint_replacement':
                    if self._apply_actor_blueprint_choice(option_value, metadata):
                        self._initialize_fields()
                        self._initialize_buttons()
                    else:
                        self._refresh_fields()
                else:
                    self.fields[field_name] = option_value
                if metadata and metadata.get('type') == 'traffic_light_sequence_color':
                    idx = int(metadata.get('index', -1))
                    self._update_traffic_light_sequence_color(idx, option_value)
                    self._initialize_fields()
                    self._initialize_buttons()
                elif metadata and metadata.get('type') == 'agent_behavior':
                    behavior = option_value.lower()
                    if behavior in ("cautious", "normal", "aggressive") and self.camera_processor:
                        editor = getattr(self.camera_processor, 'editor', None)
                        if editor:
                            editor.agent_behavior = behavior
                            if hasattr(editor, '_remember_last_agent'):
                                editor._remember_last_agent(getattr(editor, 'agent_path', None))
                            print(f"[Agent] Behavior set to: {behavior}")
                return True

        # Check button clicks first
        for button_name, rect in self.button_positions.items():
            if rect.collidepoint(mouse_pos):
                self._handle_button_click(button_name)
                return True
        
        # Check field clicks
        for field_name, rect in self.field_positions.items():
            if rect.collidepoint(mouse_pos):
                field_kind = self.field_kinds.get(field_name)
                if field_name in self.checkbox_fields:
                    # Toggle checkbox via the undo/redo command system.
                    new_value = not self.fields[field_name]
                    self.fields[field_name] = new_value
                    for spec in self.field_specs:
                        if spec.name == field_name:
                            spec.value = new_value
                            break
                    if self.camera_processor and self.selected_object:
                        vehicle_id = self.selected_object.id
                        # Map field name -> flag key used by the ignore-flags dict
                        flag_key_map = {
                            'ignore_traffic_lights': 'traffic_lights',
                            'ignore_stop_signs': 'stop_signs',
                            'ignore_vehicles': 'vehicles',
                        }
                        flag_key = flag_key_map.get(field_name)
                        if flag_key is not None:
                            old_value = not new_value
                            command = UpdateIgnoreFlagsCommand(
                                self.camera_processor, vehicle_id,
                                flag_key, old_value, new_value,
                            )
                            editor = getattr(self.camera_processor, 'editor', None)
                            if editor:
                                editor.execute_command(command)
                            else:
                                command.execute()
                    return True
                elif field_kind == 'dropdown':
                    if self.dropdown_open_field == field_name:
                        self.dropdown_open_field = None
                    else:
                        self.dropdown_open_field = field_name
                    return True
                elif field_name not in self.readonly_fields:
                    self._start_editing_field(field_name)
                    return True

        # Click elsewhere in panel - stop editing
        self._stop_editing()
        self.dropdown_open_field = None
        return True

    def handle_scroll(self, mouse_pos, scroll_direction) -> bool:
        """Scroll the open dropdown when the pointer is over the panel."""
        if not self.visible or not self._is_point_in_panel(mouse_pos):
            return False
        field_name = self.dropdown_open_field
        if field_name not in self.dropdown_fields:
            return True

        options = self.dropdown_fields[field_name]
        metadata = self.field_metadata.get(field_name, {})
        max_visible = metadata.get('max_visible_options', len(options))
        try:
            max_visible = max(1, int(max_visible))
        except (TypeError, ValueError):
            max_visible = len(options)
        if len(options) > max_visible:
            current_offset = self.dropdown_scroll_offsets.get(field_name, 0)
            self.dropdown_scroll_offsets[field_name] = max(
                0,
                min(
                    current_offset - int(scroll_direction),
                    len(options) - max_visible,
                ),
            )
        return True
    
    def handle_key_input(self, event):
        """Handle keyboard input for editing fields"""
        if not self.editing or not self.active_field:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                self._apply_field_change()
                return True
            elif event.key == pygame.K_ESCAPE:
                self._cancel_editing()
                return True
            elif event.key == pygame.K_BACKSPACE:
                if self._select_all_on_edit:
                    self.text_input = ""
                    self._select_all_on_edit = False
                else:
                    self.text_input = self.text_input[:-1]
                return True
            else:
                # Add character to input
                char = event.unicode
                if char.isprintable():
                    # Filter input based on field type
                    if self.active_field in ['x', 'y', 'z']:
                        # Allow numbers, decimal point, minus sign
                        if char.isdigit() or char in '.-':
                            self._append_char(char)
                    elif self.active_field == 'speed_km_h':
                        # Allow only positive numbers
                        if char.isdigit() or char == '.':
                            self._append_char(char)
                    elif self.active_field == 'speed_deviation_km_h':
                        # Allow only non-negative integers
                        if char.isdigit():
                            self._append_char(char)
                    elif self.active_field == 'idle_time_s':
                        if char.isdigit() or char == '.':
                            self._append_char(char)
                    elif self.active_field == 'max_lat_acc':
                        if char.isdigit() or char == '.':
                            self._append_char(char)
                    elif self.active_field == 'trigger_radius':
                        # Allow only positive numbers for trigger radius
                        if char.isdigit() or char == '.':
                            self._append_char(char)
                    else:
                        metadata = self.field_metadata.get(self.active_field)
                        if metadata and metadata.get('type') == 'traffic_light_sequence_duration':
                            if char.isdigit() or char == '.':
                                self._append_char(char)
                    return True

        return False

    def _append_char(self, char: str) -> None:
        if self._select_all_on_edit:
            self.text_input = char
            self._select_all_on_edit = False
        else:
            self.text_input += char
    
    def _start_editing_field(self, field_name):
        """Start editing a field"""
        self.active_field = field_name
        self.editing = True
        self.text_input = self.fields[field_name]
        self._select_all_on_edit = True
    
    def _stop_editing(self):
        """Stop editing without applying changes"""
        self.active_field = None
        self.editing = False
        self.text_input = ""
        self._select_all_on_edit = False
    
    def _cancel_editing(self):
        """Cancel editing and restore original value"""
        self._stop_editing()
    
    def _handle_button_click(self, button_name):
        """Handle button clicks"""
        if self.object_type == 'traffic_light':
            if button_name == 'add_traffic_light_step':
                self._add_traffic_light_sequence_step()
                return
            if button_name == 'clear_traffic_light_steps':
                self._clear_traffic_light_sequence()
                return

    def _apply_field_change(self):
        """Apply the field change"""
        if not self.active_field or not self.text_input:
            self._stop_editing()
            return
        
        try:
            # Validate and convert input
            if (self.object_type == 'waypoint_group' and
                    self.active_field in ('speed_km_h', 'speed_deviation_km_h', 'idle_time_s')):
                if self.active_field == 'speed_km_h':
                    new_value = max(0.0, float(self.text_input))
                elif self.active_field == 'speed_deviation_km_h':
                    new_value = max(0, int(float(self.text_input)))
                else:
                    new_value = max(0.0, float(self.text_input))
                self._update_group_property(self.active_field, new_value)
            elif self.active_field in ['x', 'y', 'z']:
                new_value = float(self.text_input)
                self._update_coordinate(self.active_field, new_value)
            elif self.active_field == 'speed_km_h':
                new_value = max(0, float(self.text_input))  # Ensure positive
                self._update_speed(new_value)
            elif self.active_field == 'speed_deviation_km_h':
                new_value = max(0, int(float(self.text_input)))  # Ensure non-negative
                self._update_speed_deviation(new_value)
                self.text_input = str(new_value)
            elif self.active_field == 'idle_time_s':
                new_value = max(0.0, float(self.text_input))
                self._update_idle_time(new_value)
            elif self.active_field == 'trigger_radius' and self.object_type == 'traffic_light':
                # Clamp trigger radius to reasonable values (2m - 100m) for traffic lights
                new_value = max(MIN_PERSONAL_TRIGGER_RADIUS, min(100.0, float(self.text_input)))
                self._update_traffic_light_trigger_radius(new_value)
                self.text_input = f"{new_value:.2f}"
            else:
                metadata = self.field_metadata.get(self.active_field)
                if metadata and metadata.get('type') == 'traffic_light_sequence_duration':
                    idx = int(metadata.get('index', -1))
                    new_value = max(0.0, float(self.text_input))
                    self._update_traffic_light_sequence_duration(idx, new_value)
                    self.text_input = f"{new_value:.2f}"
                elif metadata and metadata.get('type') == 'traffic_light_trigger_radius':
                    # Clamp trigger radius to reasonable values (2m - 100m)
                    new_value = max(MIN_PERSONAL_TRIGGER_RADIUS, min(100.0, float(self.text_input)))
                    self._update_traffic_light_trigger_radius(new_value)
                    self.text_input = f"{new_value:.2f}"
                elif metadata and metadata.get('type') == 'pedestrian_trigger_radius':
                    # Clamp trigger radius to minimum 2m (no maximum)
                    new_value = max(MIN_PERSONAL_TRIGGER_RADIUS, float(self.text_input))
                    self._update_pedestrian_trigger_radius(new_value)
                    self.text_input = f"{new_value:.2f}"
                elif metadata and metadata.get('type') == 'vehicle_trigger_radius':
                    new_value = max(MIN_PERSONAL_TRIGGER_RADIUS, float(self.text_input))
                    self._update_vehicle_trigger_radius(new_value)
                    self.text_input = f"{new_value:.2f}"
                elif metadata and metadata.get('type') == 'vehicle_max_lat_acc':
                    new_value = max(0.1, float(self.text_input))
                    self._update_vehicle_max_lat_acc(new_value)
                    self.text_input = f"{new_value:.2f}"
                else:
                    raise ValueError("Unsupported editable field")

            # Update field display
            self.fields[self.active_field] = self.text_input
            
        except ValueError:
            # Invalid input - ignore
            pass
        
        self._stop_editing()
    
    def _update_group_property(self, property_name, new_value):
        """Apply one property uniformly to every waypoint in the group
        selection — a single undoable step."""
        group = self.selected_object or {}
        vehicle_id = group.get('vehicle_id')
        waypoints, indices = self._group_waypoints()
        defaults = {'speed_km_h': 50, 'speed_deviation_km_h': 0, 'idle_time_s': 0.0}
        default = defaults.get(property_name, 0)

        commands = []
        for index in indices:
            old_value = waypoints[index].get(property_name, default)
            try:
                unchanged = abs(float(old_value) - float(new_value)) < 1e-6
            except (TypeError, ValueError):
                unchanged = False
            if unchanged:
                continue
            commands.append(
                UpdateWaypointPropertyCommand(
                    self.camera_processor, vehicle_id, index, property_name, old_value, new_value
                )
            )
        if not commands:
            return

        if len(commands) == 1:
            command = commands[0]
        else:
            command = CompositeCommand(
                commands, description=f"Set {property_name} on {len(commands)} waypoints"
            )
        if self.camera_processor and self.camera_processor.editor:
            self.camera_processor.editor.execute_command(command)
        else:
            command.execute()

        formats = {
            'speed_km_h': lambda v: f"{v:.0f}",
            'speed_deviation_km_h': lambda v: f"{v:.0f}",
            'idle_time_s': lambda v: f"{v:.1f}",
        }
        self.fields[property_name] = formats[property_name](float(new_value))
        print(f"Set {property_name} = {new_value} on {len(commands)} waypoint(s)")

    def _update_coordinate(self, coord_type, new_value):
        """Update waypoint or vehicle coordinates"""
        if self.object_type == 'waypoint':
            waypoint = self.selected_object
            old_value = waypoint[coord_type]

            # Only create command if value actually changed
            if abs(old_value - new_value) > 0.001:  # Avoid floating point precision issues
                command = UpdateWaypointPropertyCommand(
                    self.camera_processor,
                    self.waypoint_vehicle_id,
                    self.waypoint_index,
                    coord_type,
                    old_value,
                    new_value
                )

                # Execute command through editor if available
                if self.camera_processor.editor:
                    self.camera_processor.editor.execute_command(command)
                else:
                    # Fallback to direct execution
                    command.execute()

                # Update the field display
                self.fields[coord_type] = f"{new_value:.2f}"
                print(f"Updated waypoint {coord_type}: {old_value:.2f} -> {new_value:.2f}")
            
        elif self.object_type in ('vehicle', 'pedestrian'):
            vehicle = self.selected_object
            if vehicle and vehicle.is_alive:
                current_transform = vehicle.get_transform()
                current_loc = current_transform.location
                current_rot = current_transform.rotation
                
                # Get old value for comparison
                if coord_type == 'x':
                    old_value = current_loc.x
                elif coord_type == 'y':
                    old_value = current_loc.y
                elif coord_type == 'z':
                    old_value = current_loc.z
                elif coord_type == 'yaw':
                    old_value = current_rot.yaw
                else:
                    return  # Unknown coordinate type
                
                # Only create command if value actually changed
                if abs(old_value - new_value) > 0.001:  # Avoid floating point precision issues
                    # Create new transform with updated coordinate
                    if coord_type == 'x':
                        new_loc = carla.Location(new_value, current_loc.y, current_loc.z)
                        new_transform = carla.Transform(new_loc, current_rot)
                    elif coord_type == 'y':
                        new_loc = carla.Location(current_loc.x, new_value, current_loc.z)
                        new_transform = carla.Transform(new_loc, current_rot)
                    elif coord_type == 'z':
                        new_loc = carla.Location(current_loc.x, current_loc.y, new_value)
                        new_transform = carla.Transform(new_loc, current_rot)
                    elif coord_type == 'yaw':
                        new_rot = carla.Rotation(current_rot.pitch, new_value, current_rot.roll)
                        new_transform = carla.Transform(current_loc, new_rot)
                    
                    command = UpdateVehiclePropertyCommand(
                        self.camera_processor,
                        vehicle.id,
                        coord_type,
                        old_value,
                        new_value,
                        current_transform,
                        new_transform
                    )
                    
                    # Execute command through editor if available
                    if self.camera_processor.editor:
                        self.camera_processor.editor.execute_command(command)
                    else:
                        # Fallback to direct execution
                        command.execute()
                    
                    if coord_type == 'yaw':
                        print(f"Rotated vehicle yaw: {current_rot.yaw:.2f} -> {new_value:.2f}")
                    else:
                        print(f"Moved vehicle {coord_type}: {getattr(current_loc, coord_type):.2f} -> {new_value:.2f}")
    
    def _update_speed(self, new_speed):
        """Update speed for waypoint or vehicle"""
        if self.object_type == 'waypoint':
            waypoint = self.selected_object
            old_speed = waypoint.get('speed_km_h', 50)  # Default speed if not set
            
            # Only create command if value actually changed
            if abs(old_speed - new_speed) > 0.1:  # Avoid small floating point differences
                command = UpdateWaypointPropertyCommand(
                    self.camera_processor,
                    self.waypoint_vehicle_id,
                    self.waypoint_index,
                    'speed_km_h',
                    old_speed,
                    new_speed
                )
                
                # Execute command through editor if available
                if self.camera_processor.editor:
                    self.camera_processor.editor.execute_command(command)
                else:
                    # Fallback to direct execution
                    command.execute()
                
                print(f"Updated waypoint speed: {old_speed:.0f} -> {new_speed:.0f} km/h")
                
        elif self.object_type in ('vehicle', 'pedestrian'):
            actor = self.selected_object
            if actor and actor.is_alive and self.camera_processor:
                old_speed = self.camera_processor.get_vehicle_speed(actor.id, 50)  # Default speed

                if abs(old_speed - new_speed) > 0.1:  # Avoid small floating point differences
                    command = UpdateVehiclePropertyCommand(
                        self.camera_processor,
                        actor.id,
                        'speed_km_h',
                        old_speed,
                        new_speed
                    )

                    if self.camera_processor.editor:
                        self.camera_processor.editor.execute_command(command)
                    else:
                        command.execute()

                    label = "pedestrian" if self.object_type == 'pedestrian' else 'vehicle'
                    print(f"Updated {label} speed: {old_speed:.0f} -> {new_speed:.0f} km/h")

    def _update_speed_deviation(self, new_deviation_km_h: int) -> None:
        """Update speed deviation (km/h) for a waypoint."""
        if self.object_type != 'waypoint':
            return

        waypoint = self.selected_object
        old_deviation = waypoint.get('speed_deviation_km_h', 0)
        try:
            old_deviation = int(float(old_deviation or 0))
        except Exception:
            old_deviation = 0
        if old_deviation < 0:
            old_deviation = 0

        new_deviation_km_h = max(0, int(new_deviation_km_h))
        if old_deviation == new_deviation_km_h:
            return

        command = UpdateWaypointPropertyCommand(
            self.camera_processor,
            self.waypoint_vehicle_id,
            self.waypoint_index,
            'speed_deviation_km_h',
            old_deviation,
            new_deviation_km_h,
        )

        if self.camera_processor and self.camera_processor.editor:
            self.camera_processor.editor.execute_command(command)
        else:
            command.execute()

        if 'speed_deviation_km_h' in self.fields:
            self.fields['speed_deviation_km_h'] = str(new_deviation_km_h)
        print(
            f"Updated waypoint speed deviation: {old_deviation} -> {new_deviation_km_h} km/h"
        )

    def _update_idle_time(self, new_idle_time):
        """Update idle time for vehicles, pedestrians, and waypoints."""
        if self.object_type == 'waypoint':
            waypoint = self.selected_object
            old_idle = waypoint.get('idle_time_s', 0.0)

            if abs(old_idle - new_idle_time) > 0.01:
                command = UpdateWaypointPropertyCommand(
                    self.camera_processor,
                    self.waypoint_vehicle_id,
                    self.waypoint_index,
                    'idle_time_s',
                    old_idle,
                    new_idle_time
                )

                if self.camera_processor and self.camera_processor.editor:
                    self.camera_processor.editor.execute_command(command)
                else:
                    command.execute()

                self.fields['idle_time_s'] = f"{new_idle_time:.1f}"
                print(f"Updated waypoint idle time: {old_idle:.1f}s -> {new_idle_time:.1f}s")

        elif self.object_type in ('vehicle', 'pedestrian'):
            actor = self.selected_object
            if not actor or not actor.is_alive:
                return

            old_idle = 0.0
            if self.camera_processor and hasattr(self.camera_processor, 'actor_idle_times'):
                old_idle = self.camera_processor.actor_idle_times.get(actor.id, 0.0)

            if abs(old_idle - new_idle_time) > 0.01:
                command = UpdateVehiclePropertyCommand(
                    self.camera_processor,
                    actor.id,
                    'idle_time_s',
                    old_idle,
                    new_idle_time
                )

                if self.camera_processor and self.camera_processor.editor:
                    self.camera_processor.editor.execute_command(command)
                else:
                    command.execute()

                self.fields['idle_time_s'] = f"{new_idle_time:.1f}"
                actor_label = 'vehicle' if self.object_type == 'vehicle' else 'pedestrian'
                print(f"Updated {actor_label} idle time: {old_idle:.1f}s -> {new_idle_time:.1f}s")

    def _traffic_light_default_duration(self) -> float:
        return 5.0

    def _traffic_light_seconds_to_ticks(self, seconds: float) -> int:
        if seconds <= 0.0:
            return 0
        delta = 0.05
        try:
            if self.camera_processor and self.camera_processor.world:
                settings = self.camera_processor.world.get_settings()
                if settings.fixed_delta_seconds and settings.fixed_delta_seconds > 0:
                    delta = settings.fixed_delta_seconds
        except Exception:
            pass
        return max(1, int(round(seconds / delta)))

    def _ensure_traffic_light_sequence(self) -> Optional[TrafficLightGroupData]:
        if self.object_type != 'traffic_light':
            return None
        group = cast(Optional[TrafficLightGroupData], self.selected_object)
        if not group:
            return None
        if group.sequence is None:
            group.sequence = []
        return group

    def _add_traffic_light_sequence_step(self):
        group = self._ensure_traffic_light_sequence()
        if not group or not group.has_trigger():
            return
        duration_s = self._traffic_light_default_duration()
        old_seq = copy.deepcopy(group.sequence)
        new_seq = copy.deepcopy(group.sequence)
        new_seq.append({
            'color': 'Red',
            'duration_s': duration_s,
            'duration_ticks': self._traffic_light_seconds_to_ticks(duration_s),
        })
        command = UpdateTrafficLightSequenceCommand(
            self.camera_processor,
            group,
            old_seq,
            new_seq,
        )
        editor = getattr(self.camera_processor, "editor", None)
        success = editor.execute_command(command) if editor else command.execute()
        if success is False:
            print("Failed to add traffic light sequence step.")
            return
        self._initialize_fields()
        self._initialize_buttons()

    def _clear_traffic_light_sequence(self):
        group = self._ensure_traffic_light_sequence()
        if not group or not group.has_trigger() or not group.sequence:
            return
        old_seq = copy.deepcopy(group.sequence)
        new_seq = []
        command = UpdateTrafficLightSequenceCommand(
            self.camera_processor,
            group,
            old_seq,
            new_seq,
        )
        editor = getattr(self.camera_processor, "editor", None)
        success = editor.execute_command(command) if editor else command.execute()
        if success is False:
            print("Failed to clear traffic light sequence.")
            return
        self._initialize_fields()
        self._initialize_buttons()

    def _delete_traffic_light_sequence_step(self, index: int):
        group = self._ensure_traffic_light_sequence()
        if not group or not group.has_trigger() or not group.sequence:
            return
        if 0 <= index < len(group.sequence):
            old_seq = copy.deepcopy(group.sequence)
            new_seq = copy.deepcopy(group.sequence)
            del new_seq[index]
            command = UpdateTrafficLightSequenceCommand(
                self.camera_processor,
                group,
                old_seq,
                new_seq,
            )
            editor = getattr(self.camera_processor, "editor", None)
            success = editor.execute_command(command) if editor else command.execute()
            if success is False:
                print("Failed to delete traffic light sequence step.")
                return
            self._initialize_fields()
            self._initialize_buttons()

    def _update_traffic_light_sequence_color(self, index: int, color: str):
        group = self._ensure_traffic_light_sequence()
        if not group or not group.has_trigger() or not group.sequence:
            return
        if 0 <= index < len(group.sequence):
            old_seq = copy.deepcopy(group.sequence)
            new_seq = copy.deepcopy(group.sequence)
            new_seq[index]['color'] = color
            command = UpdateTrafficLightSequenceCommand(
                self.camera_processor,
                group,
                old_seq,
                new_seq,
            )
            editor = getattr(self.camera_processor, "editor", None)
            success = editor.execute_command(command) if editor else command.execute()
            if success is False:
                print("Failed to update traffic light sequence color.")
                return
            self.fields[f"seq_color_{index}"] = color
            if self.camera_processor:
                self.camera_processor._cache_traffic_light_sequence(group)

    def _update_traffic_light_sequence_duration(self, index: int, duration_s: float):
        group = self._ensure_traffic_light_sequence()
        if not group or not group.has_trigger() or not group.sequence:
            return
        duration_s = max(0.0, duration_s)
        if 0 <= index < len(group.sequence):
            old_seq = copy.deepcopy(group.sequence)
            new_seq = copy.deepcopy(group.sequence)
            new_seq[index]['duration_s'] = duration_s
            new_seq[index]['duration_ticks'] = self._traffic_light_seconds_to_ticks(duration_s)
            command = UpdateTrafficLightSequenceCommand(
                self.camera_processor,
                group,
                old_seq,
                new_seq,
            )
            editor = getattr(self.camera_processor, "editor", None)
            success = editor.execute_command(command) if editor else command.execute()
            if success is False:
                print("Failed to update traffic light sequence duration.")
                return
            self.fields[f"seq_duration_{index}"] = f"{duration_s:.2f}"
            if self.camera_processor:
                self.camera_processor._cache_traffic_light_sequence(group)

    def _update_traffic_light_trigger_radius(self, new_radius: float):
        """Update the trigger radius for the selected traffic light group."""
        if self.object_type != 'traffic_light' or not self.camera_processor:
            return

        # Get the current selected group from camera processor (source of truth)
        group = self.camera_processor.selected_traffic_light_group
        if not group or not group.has_trigger():
            return

        new_radius = max(MIN_PERSONAL_TRIGGER_RADIUS, min(100.0, new_radius))
        center_snapshot = group.trigger_center or {}
        cached_center, _, resolved_key = self.camera_processor._get_traffic_light_trigger_data(group=group)
        if not center_snapshot and cached_center:
            center_snapshot = dict(cached_center)
        key = self.camera_processor._traffic_light_trigger_key(group=group) or resolved_key
        old_radius = group.trigger_radius
        selection = {
            'kind': 'traffic_light',
            'group': group,
            'key': key,
        }
        command = SetPersonalTriggerCommand(
            self.camera_processor,
            selection,
            dict(center_snapshot) if center_snapshot else None,
            new_radius,
            old_center=copy.deepcopy(center_snapshot) if center_snapshot else None,
            old_radius=old_radius,
        )
        editor = getattr(self.camera_processor, "editor", None)
        if editor:
            success = editor.execute_command(command)
        else:
            success = command.execute()
        if success is False:
            print("Failed to update traffic light trigger radius.")
            return

        # Refresh from the authoritative data after command execution
        group_ids = group.ids
        applied_radius = group.trigger_radius if group.trigger_radius is not None else new_radius
        self.fields['trigger_radius'] = f"{applied_radius:.2f}"
        print(f"Updated traffic light trigger radius: {applied_radius:.2f} m for IDs {sorted(group_ids)}")
        current = self.camera_processor.selected_personal_trigger
        if current and current.get('kind') == 'traffic_light':
            self.camera_processor.update_personal_trigger_menu_position(force=True)

    def _update_pedestrian_trigger_radius(self, new_radius: float):
        """Update the trigger radius for the selected pedestrian."""
        if self.object_type != 'pedestrian' or not self.camera_processor:
            return

        # Get the selected pedestrian actor
        actor = self.selected_object
        if not actor or not actor.is_alive:
            return

        pedestrian_id = actor.id
        if not self.camera_processor._ensure_pedestrian_trigger(pedestrian_id):
            return

        # Clamp to minimum radius (no maximum)
        new_radius = max(MIN_PERSONAL_TRIGGER_RADIUS, new_radius)

        center_snapshot = self.camera_processor.pedestrian_trigger_centers.get(pedestrian_id)
        old_radius = self.camera_processor.pedestrian_trigger_radii.get(pedestrian_id)
        selection = {'kind': 'pedestrian', 'id': pedestrian_id}
        command = SetPersonalTriggerCommand(
            self.camera_processor,
            selection,
            dict(center_snapshot) if center_snapshot else None,
            new_radius,
            old_center=copy.deepcopy(center_snapshot) if center_snapshot else None,
            old_radius=old_radius,
        )
        editor = getattr(self.camera_processor, "editor", None)
        if editor:
            success = editor.execute_command(command)
        else:
            success = command.execute()
        if success is False:
            print("Failed to update pedestrian trigger radius.")
            return
        # Fetch the stored radius in case clamping changed it during execution.
        applied_radius = self.camera_processor.pedestrian_trigger_radii.get(pedestrian_id, new_radius)

        # Update the field display
        self.fields['trigger_radius'] = f"{applied_radius:.2f}"

        # Print confirmation
        print(f"Updated pedestrian trigger radius: {applied_radius:.2f} m for ID {pedestrian_id}")
        current = self.camera_processor.selected_personal_trigger
        if current and current.get('kind') == 'pedestrian' and current.get('id') == pedestrian_id:
            self.camera_processor.update_personal_trigger_menu_position(force=True)

    def _update_vehicle_max_lat_acc(self, new_value: float):
        """Update the lateral acceleration cap for the selected vehicle."""
        if self.object_type != 'vehicle' or not self.camera_processor:
            return

        actor = self.selected_object
        if not actor or not actor.is_alive:
            return

        vehicle_id = actor.id
        old_value = self.camera_processor.get_vehicle_max_lat_acc(vehicle_id, 3.0)
        if abs(old_value - new_value) < 1e-3:
            self.fields['max_lat_acc'] = f"{new_value:.2f}"
            return

        command = UpdateVehiclePropertyCommand(
            self.camera_processor,
            vehicle_id,
            'max_lat_acc',
            old_value,
            new_value,
        )

        if self.camera_processor and self.camera_processor.editor:
            self.camera_processor.editor.execute_command(command)
        else:
            command.execute()

        self.fields['max_lat_acc'] = f"{new_value:.2f}"
        print(f"Updated vehicle max lateral acceleration: {old_value:.2f} -> {new_value:.2f} m/s^2")

    def _update_vehicle_trigger_radius(self, new_radius: float):
        """Update the trigger radius for the selected vehicle."""
        if self.object_type != 'vehicle' or not self.camera_processor:
            return

        actor = self.selected_object
        if not actor or not actor.is_alive:
            return

        vehicle_id = actor.id
        if vehicle_id not in self.camera_processor.vehicle_trigger_radii:
            return

        if not self.camera_processor._ensure_vehicle_trigger(vehicle_id):
            return
        new_radius = max(MIN_PERSONAL_TRIGGER_RADIUS, new_radius)
        center_snapshot = self.camera_processor.vehicle_trigger_centers.get(vehicle_id)
        old_radius = self.camera_processor.vehicle_trigger_radii.get(vehicle_id)
        selection = {'kind': 'vehicle', 'id': vehicle_id}
        command = SetPersonalTriggerCommand(
            self.camera_processor,
            selection,
            dict(center_snapshot) if center_snapshot else None,
            new_radius,
            old_center=copy.deepcopy(center_snapshot) if center_snapshot else None,
            old_radius=old_radius,
        )
        editor = getattr(self.camera_processor, "editor", None)
        if editor:
            success = editor.execute_command(command)
        else:
            success = command.execute()
        if success is False:
            print("Failed to update vehicle trigger radius.")
            return

        applied_radius = self.camera_processor.vehicle_trigger_radii.get(vehicle_id, new_radius)
        self.fields['trigger_radius'] = f"{applied_radius:.2f}"
        print(f"Updated vehicle trigger radius: {applied_radius:.2f} m for ID {vehicle_id}")
        current = self.camera_processor.selected_personal_trigger
        if current and current.get('kind') == 'vehicle' and current.get('id') == vehicle_id:
            self.camera_processor.update_personal_trigger_menu_position(force=True)

    def _is_point_in_panel(self, point):
        """Check if a point is within the panel bounds"""
        return (self.panel_x <= point[0] <= self.panel_x + self.panel_width and
                self.panel_y <= point[1] <= self.panel_y + self.panel_height)
    
    def render(self, screen, tooltip_manager=None):
        """Render the info panel"""
        if not self.visible:
            return

        self._update_layout(screen.get_width(), screen.get_height(), recalc_height=True)
        
        # Create surface with alpha for transparency
        panel_surface = pygame.Surface((self.panel_width, self.panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.background_color)

        y_offset = 20

        # Title
        if self.object_type == 'waypoint_group':
            title = "Waypoints Info"
        elif self.object_type == 'waypoint':
            title = f"Waypoint Info"
        elif self.object_type == 'vehicle':
            title = f"Vehicle Info"
        elif self.object_type == 'pedestrian':
            title = f"Pedestrian Info"
        elif self.object_type == 'traffic_light':
            title = "Traffic Lights Info"
        else:
            title = "Object Info"

        title_text = self.font.render(title, True, self.text_color)
        panel_surface.blit(title_text, (self.margin, y_offset))
        y_offset += 40

        # Reset per-render collections
        self.field_positions.clear()
        self.dropdown_option_rects = []
        self.extra_click_zones = []

        # Waypoint-specific tip block
        tip_height = self._render_waypoint_split_tip(panel_surface, y_offset)
        if tip_height:
            y_offset += tip_height

        # Render fields
        processed_fields: Set[str] = set()

        for spec in self.field_specs:
            field_name = spec.name
            if field_name in processed_fields:
                continue

            field_value = self.fields.get(field_name, "")
            metadata = self.field_metadata.get(field_name)
            field_kind = self.field_kinds.get(field_name, spec.kind)

            if field_kind == 'checkbox':
                label_text = self.small_font.render(spec.label, True, self.text_color)
                panel_surface.blit(label_text, (self.margin, y_offset))

                checkbox_size = 20
                checkbox_rect = pygame.Rect(self.margin, y_offset + 20, checkbox_size, checkbox_size)
                absolute_rect = pygame.Rect(
                    self.panel_x + checkbox_rect.x,
                    self.panel_y + checkbox_rect.y,
                    checkbox_rect.width,
                    checkbox_rect.height,
                )
                self.field_positions[field_name] = absolute_rect

                pygame.draw.rect(panel_surface, self.field_color, checkbox_rect)
                pygame.draw.rect(panel_surface, self.text_color, checkbox_rect, 1)

                if field_value:
                    pygame.draw.line(
                        panel_surface,
                        self.text_color,
                        (checkbox_rect.left + 3, checkbox_rect.top + 3),
                        (checkbox_rect.right - 3, checkbox_rect.bottom - 3),
                        2,
                    )
                    pygame.draw.line(
                        panel_surface,
                        self.text_color,
                        (checkbox_rect.right - 3, checkbox_rect.top + 3),
                        (checkbox_rect.left + 3, checkbox_rect.bottom - 3),
                        2,
                    )

                y_offset += 48
                continue

            if field_kind == 'dropdown' and metadata and metadata.get('type') == 'traffic_light_sequence_color':
                paired_field = metadata.get('paired_field')
                processed_fields.add(field_name)
                paired_spec = self.field_spec_map.get(paired_field) if paired_field else None
                paired_value = self.fields.get(paired_field, "") if paired_field else ""

                delete_button_width = 28
                column_gap = 12
                available_width = self.panel_width - 2 * self.margin - delete_button_width - column_gap
                column_width = max(60, available_width // 2)

                color_label_text = spec.label.replace(' Color:', '').strip()
                color_label_surface = self.small_font.render(color_label_text, True, self.text_color)
                panel_surface.blit(color_label_surface, (self.margin, y_offset))

                duration_label_text = 'Duration (sim s)'
                if paired_spec:
                    raw_label = paired_spec.label
                    if 'Duration' in raw_label:
                        duration_label_text = raw_label.split('Duration', 1)[-1].strip(': ')
                        if not duration_label_text:
                            duration_label_text = 'Duration (sim s)'
                    elif raw_label:
                        duration_label_text = raw_label
                duration_label_surface = self.small_font.render(duration_label_text, True, self.text_color)
                duration_label_x = self.margin + column_width + column_gap
                panel_surface.blit(duration_label_surface, (duration_label_x, y_offset))

                color_rect = pygame.Rect(self.margin, y_offset + 18, column_width, 24)
                color_abs = pygame.Rect(
                    self.panel_x + color_rect.x,
                    self.panel_y + color_rect.y,
                    color_rect.width,
                    color_rect.height,
                )
                self.field_positions[field_name] = color_abs

                is_open = self.dropdown_open_field == field_name
                pygame.draw.rect(panel_surface, self.field_color, color_rect)
                pygame.draw.rect(panel_surface, self.text_color, color_rect, 1)

                display_text = str(field_value)
                text_surface = self.small_font.render(display_text, True, self.text_color)
                text_rect = text_surface.get_rect()
                text_rect.left = color_rect.left + 6
                text_rect.centery = color_rect.centery
                panel_surface.blit(text_surface, text_rect)

                if is_open:
                    arrow_points = [
                        (color_rect.right - 16, color_rect.centery + 4),
                        (color_rect.right - 8, color_rect.centery + 4),
                        (color_rect.right - 12, color_rect.centery - 4),
                    ]
                else:
                    arrow_points = [
                        (color_rect.right - 16, color_rect.centery - 4),
                        (color_rect.right - 8, color_rect.centery - 4),
                        (color_rect.right - 12, color_rect.centery + 4),
                    ]
                pygame.draw.polygon(panel_surface, self.text_color, arrow_points)

                option_block_height = 0
                options = self._get_dropdown_visible_options(field_name) if is_open else []
                if is_open and options:
                    option_height = 24
                    option_block_height = len(options) * option_height
                    option_rect = pygame.Rect(color_rect.left, color_rect.bottom, color_rect.width, option_block_height)
                    pygame.draw.rect(panel_surface, self.field_color, option_rect)
                    pygame.draw.rect(panel_surface, self.text_color, option_rect, 1)

                    for idx_opt, choice in enumerate(options):
                        opt_y = option_rect.top + idx_opt * option_height
                        choice_rect = pygame.Rect(option_rect.left + 2, opt_y + 2, option_rect.width - 4, option_height - 4)
                        choice_abs = pygame.Rect(
                            self.panel_x + choice_rect.x,
                            self.panel_y + choice_rect.y,
                            choice_rect.width,
                            choice_rect.height,
                        )
                        pygame.draw.rect(panel_surface, self.field_color, choice_rect)
                        if choice == field_value:
                            pygame.draw.rect(panel_surface, self.button_hover_color, choice_rect)
                        pygame.draw.rect(panel_surface, self.text_color, choice_rect, 1)
                        choice_text = self.small_font.render(choice, True, self.text_color)
                        panel_surface.blit(choice_text, (choice_rect.x + 4, choice_rect.y + 3))
                        self.dropdown_option_rects.append((field_name, choice, choice_abs))

                duration_rect = pygame.Rect(duration_label_x, y_offset + 18, column_width, 24)
                duration_abs = pygame.Rect(
                    self.panel_x + duration_rect.x,
                    self.panel_y + duration_rect.y,
                    duration_rect.width,
                    duration_rect.height,
                )
                if paired_field:
                    self.field_positions[paired_field] = duration_abs

                active_duration = paired_field and self.active_field == paired_field and self.editing
                duration_bg = self.active_field_color if active_duration else self.field_color
                pygame.draw.rect(panel_surface, duration_bg, duration_rect)
                pygame.draw.rect(panel_surface, self.text_color, duration_rect, 1)

                duration_display = str(paired_value)
                if active_duration:
                    duration_display = self.text_input if self._select_all_on_edit else self.text_input + "|"
                duration_text = self.small_font.render(duration_display, True, self.text_color)
                duration_text_rect = duration_text.get_rect()
                duration_text_rect.left = duration_rect.left + 5
                duration_text_rect.centery = duration_rect.centery
                panel_surface.blit(duration_text, duration_text_rect)

                delete_rect = pygame.Rect(
                    duration_rect.right + 6,
                    y_offset + 18,
                    delete_button_width - 6,
                    24,
                )
                delete_rect.x = min(delete_rect.x, self.panel_width - self.margin - delete_rect.width)
                delete_abs = pygame.Rect(
                    self.panel_x + delete_rect.x,
                    self.panel_y + delete_rect.y,
                    delete_rect.width,
                    delete_rect.height,
                )
                pygame.draw.rect(panel_surface, (120, 50, 50), delete_rect)
                pygame.draw.rect(panel_surface, self.text_color, delete_rect, 1)
                pygame.draw.line(
                    panel_surface,
                    self.text_color,
                    (delete_rect.left + 5, delete_rect.top + 5),
                    (delete_rect.right - 5, delete_rect.bottom - 5),
                    2,
                )
                pygame.draw.line(
                    panel_surface,
                    self.text_color,
                    (delete_rect.right - 5, delete_rect.top + 5),
                    (delete_rect.left + 5, delete_rect.bottom - 5),
                    2,
                )
                self.extra_click_zones.append(
                    {
                        'type': 'traffic_light_delete_step',
                        'index': metadata.get('index', -1),
                        'rect': delete_abs,
                    }
                )

                if paired_field:
                    processed_fields.add(paired_field)
                y_offset += 42 + option_block_height
                continue

            # Fallback for other dropdowns (if any)
            if field_kind == 'dropdown':
                label_text = self.small_font.render(spec.label, True, self.text_color)
                panel_surface.blit(label_text, (self.margin, y_offset))

                field_rect = pygame.Rect(self.margin, y_offset + 18, self.panel_width - 2 * self.margin, 24)
                absolute_rect = pygame.Rect(
                    self.panel_x + field_rect.x,
                    self.panel_y + field_rect.y,
                    field_rect.width,
                    field_rect.height,
                )
                self.field_positions[field_name] = absolute_rect

                is_open = self.dropdown_open_field == field_name
                pygame.draw.rect(panel_surface, self.field_color, field_rect)
                pygame.draw.rect(panel_surface, self.text_color, field_rect, 1)

                display_text = str(field_value)
                text_surface = self.small_font.render(display_text, True, self.text_color)
                text_rect = text_surface.get_rect()
                text_rect.left = field_rect.left + 6
                text_rect.centery = field_rect.centery
                panel_surface.blit(text_surface, text_rect)

                if is_open:
                    arrow_points = [
                        (field_rect.right - 16, field_rect.centery + 4),
                        (field_rect.right - 8, field_rect.centery + 4),
                        (field_rect.right - 12, field_rect.centery - 4),
                    ]
                else:
                    arrow_points = [
                        (field_rect.right - 16, field_rect.centery - 4),
                        (field_rect.right - 8, field_rect.centery - 4),
                        (field_rect.right - 12, field_rect.centery + 4),
                    ]
                pygame.draw.polygon(panel_surface, self.text_color, arrow_points)

                option_block_height = 0
                options = self._get_dropdown_visible_options(field_name) if is_open else []
                if is_open and options:
                    option_height = 24
                    option_block_height = len(options) * option_height
                    option_rect = pygame.Rect(field_rect.left, field_rect.bottom, field_rect.width, option_block_height)
                    pygame.draw.rect(panel_surface, self.field_color, option_rect)
                    pygame.draw.rect(panel_surface, self.text_color, option_rect, 1)

                    all_options = self.dropdown_fields.get(field_name, [])
                    scrollbar_width = 8
                    scrollbar_gap = 4
                    has_scrollbar = len(all_options) > len(options)
                    choice_width = option_rect.width - 4
                    if has_scrollbar:
                        track_rect = pygame.Rect(
                            option_rect.right - scrollbar_gap - scrollbar_width,
                            option_rect.top + 3,
                            scrollbar_width,
                            option_rect.height - 6,
                        )
                        pygame.draw.rect(panel_surface, (45, 45, 45), track_rect)
                        max_offset = len(all_options) - len(options)
                        thumb_height = max(
                            18,
                            int(track_rect.height * len(options) / len(all_options)),
                        )
                        max_thumb_offset = max(0, track_rect.height - thumb_height)
                        scroll_offset = self.dropdown_scroll_offsets.get(field_name, 0)
                        thumb_offset = 0
                        if max_offset:
                            thumb_offset = round(
                                max_thumb_offset * scroll_offset / max_offset
                            )
                        thumb_rect = pygame.Rect(
                            track_rect.left,
                            track_rect.top + thumb_offset,
                            track_rect.width,
                            thumb_height,
                        )
                        pygame.draw.rect(panel_surface, (170, 170, 170), thumb_rect)
                        self.dropdown_scrollbar_rects[field_name] = pygame.Rect(
                            self.panel_x + track_rect.x,
                            self.panel_y + track_rect.y,
                            track_rect.width,
                            track_rect.height,
                        )
                        self.dropdown_scrollbar_thumb_rects[field_name] = pygame.Rect(
                            self.panel_x + thumb_rect.x,
                            self.panel_y + thumb_rect.y,
                            thumb_rect.width,
                            thumb_rect.height,
                        )
                        choice_width -= scrollbar_width + scrollbar_gap

                    for idx_opt, choice in enumerate(options):
                        opt_y = option_rect.top + idx_opt * option_height
                        choice_rect = pygame.Rect(
                            option_rect.left + 2,
                            opt_y + 2,
                            choice_width,
                            option_height - 4,
                        )
                        choice_abs = pygame.Rect(
                            self.panel_x + choice_rect.x,
                            self.panel_y + choice_rect.y,
                            choice_rect.width,
                            choice_rect.height,
                        )
                        pygame.draw.rect(panel_surface, self.field_color, choice_rect)
                        if choice == field_value:
                            pygame.draw.rect(panel_surface, self.button_hover_color, choice_rect)
                        pygame.draw.rect(panel_surface, self.text_color, choice_rect, 1)
                        choice_text = self.small_font.render(choice, True, self.text_color)
                        panel_surface.blit(choice_text, (choice_rect.x + 4, choice_rect.y + 3))
                        self.dropdown_option_rects.append((field_name, choice, choice_abs))

                y_offset += 42 + option_block_height
                continue

            label_text = self.small_font.render(spec.label, True, self.text_color)
            panel_surface.blit(label_text, (self.margin, y_offset))

            if metadata and metadata.get('type') == 'traffic_light_sequence_header':
                message = str(field_value)
                message_surface = self.small_font.render(message, True, (200, 200, 200))
                panel_surface.blit(message_surface, (self.margin, y_offset + 20))
                y_offset += 36
                continue

            field_rect = pygame.Rect(
                self.margin, y_offset + 18, self.panel_width - 2 * self.margin, 24
            )
            absolute_rect = pygame.Rect(
                self.panel_x + field_rect.x,
                self.panel_y + field_rect.y,
                field_rect.width,
                field_rect.height,
            )
            self.field_positions[field_name] = absolute_rect

            if self.active_field == field_name and self.editing:
                field_bg_color = self.active_field_color
                display_text = self.text_input if self._select_all_on_edit else self.text_input + "|"
            else:
                field_bg_color = self.field_color
                display_text = str(field_value)

            pygame.draw.rect(panel_surface, field_bg_color, field_rect)
            pygame.draw.rect(panel_surface, self.text_color, field_rect, 1)

            text_color = self.text_color if spec.editable else (200, 200, 200)
            field_text = self.small_font.render(display_text, True, text_color)
            text_rect = field_text.get_rect()
            text_rect.left = field_rect.left + 5
            text_rect.centery = field_rect.centery
            panel_surface.blit(field_text, text_rect)

            y_offset += 42
        
        # Render buttons
        if self.buttons:
            y_offset += 10  # Add some spacing before buttons
            for button_name, button_text in self.buttons.items():
                button_rect = pygame.Rect(self.margin, y_offset, self.panel_width - 2 * self.margin, 30)
                
                # Store button position for click detection
                absolute_rect = pygame.Rect(
                    self.panel_x + button_rect.x,
                    self.panel_y + button_rect.y,
                    button_rect.width,
                    button_rect.height
                )
                self.button_positions[button_name] = absolute_rect
                
                # Check if mouse is hovering over button
                mouse_pos = pygame.mouse.get_pos()
                is_hovering = absolute_rect.collidepoint(mouse_pos)
                
                # Button background color
                button_bg_color = self.button_hover_color if is_hovering else self.button_color
                
                # Draw button background
                pygame.draw.rect(panel_surface, button_bg_color, button_rect)
                pygame.draw.rect(panel_surface, self.text_color, button_rect, 1)
                
                # Draw button text
                button_text_surface = self.small_font.render(button_text, True, self.text_color)
                text_rect = button_text_surface.get_rect()
                text_rect.center = button_rect.center
                panel_surface.blit(button_text_surface, text_rect)

                # Register hover for tooltip
                if tooltip_manager and is_hovering:
                    tooltip_text = INFO_PANEL_BUTTON_TOOLTIPS.get(button_name, "")
                    if tooltip_text:
                        tooltip_manager.register_hover(
                            f"info_panel_button_{button_name}",
                            absolute_rect,
                            tooltip_text,
                        )

                y_offset += 35
        
        # Draw panel border
        pygame.draw.rect(panel_surface, self.text_color, panel_surface.get_rect(), 2)
        
        # Blit panel to screen
        screen.blit(panel_surface, (self.panel_x, self.panel_y))
