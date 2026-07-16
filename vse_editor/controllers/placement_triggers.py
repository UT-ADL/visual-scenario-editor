"""Placement controller, part 2 — triggers (moved verbatim from
CameraImageProcessor, self -> processor rename only — step-30).

Global trigger placement/movement/scaling, personal trigger family
(place/select/move/scale/delete + hit testing), pedestrian/vehicle trigger
ensure/scale/delete, and the trigger click routing. Companion to
placement.py (part 1, step-29) — kept as its own module so each file stays
readable. Headless fallbacks (`editor is None -> command.execute()`) and lazy
getattr-defaults move verbatim; the processor
keeps one-line delegates for every function (commands.py and info_panel.py
call several via the processor).
"""

import copy
import math
import weakref
from typing import Optional

import carla

from vse_common.geometry import get_ground_height
from vse_editor.commands import (
    MovePersonalTriggerCommand,
    SetGlobalTriggerCommand,
    SetPersonalTriggerCommand,
)
from vse_editor.constants import (
    DEFAULT_PERSONAL_TRIGGER_RADIUS,
    MIN_PERSONAL_TRIGGER_RADIUS,
    PERSONAL_TRIGGER_MENU_ICONS,
    TRIGGER_MENU_ICONS,
)
from vse_editor.rendering.overlays import OverlayMenuRenderer
from vse_editor.scene_types import TrafficLightGroupData


def start_trigger_placement(processor):
    """Enter trigger placement mode"""
    if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
        # Placement is top-down only; the TRIGGER mode button re-arms this on click.
        editor = getattr(processor, 'editor', None)
        if editor is not None:
            editor.show_status_hint("3D view is view & select only - press Tab for top-down to edit")
        return
    processor.placing_trigger = True
    print("Trigger placement mode activated. Click on the map to place a trigger zone.")

def stop_trigger_placement(processor):
    """Exit trigger placement mode"""
    processor.placing_trigger = False
    print("Trigger placement mode cancelled.")

def start_personal_trigger_placement(processor, kind: str, *, actor=None, group=None) -> bool:
    """Enter placement mode for a personal trigger tied to an actor or group."""
    if kind not in ('vehicle', 'pedestrian', 'traffic_light'):
        return False

    if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
        # Placement is top-down only (belt-and-braces behind the menu icon filters).
        editor = getattr(processor, 'editor', None)
        if editor is not None:
            editor.show_status_hint("3D view is view & select only - press Tab for top-down to edit")
        return False

    if kind in ('vehicle', 'pedestrian'):
        if not actor or not actor.is_alive:
            print("Cannot start personal trigger placement: actor is unavailable.")
            return False
        if kind == 'vehicle' and processor.is_ego_vehicle(actor.id):
            print("Ego vehicles do not support personal triggers.")
            return False
        processor.pending_personal_trigger = {
            'kind': kind,
            'actor_id': actor.id,
            'actor_ref': weakref.ref(actor),
        }
        actor_label = "pedestrian" if kind == 'pedestrian' else "vehicle"
        print(f"{actor_label.title()} trigger placement mode activated. Left-click on the map to place the trigger.")
    elif kind == 'traffic_light':
        if not group:
            print("Cannot start personal trigger placement: traffic light group missing.")
            return False
        processor.pending_personal_trigger = {
            'kind': 'traffic_light',
            'group': group,
        }
        ids = sorted(group.ids) if getattr(group, 'ids', None) else []
        print(f"Traffic-light trigger placement mode activated for IDs {ids}. Left-click to place the trigger.")

    processor.personal_trigger_preview_radius = DEFAULT_PERSONAL_TRIGGER_RADIUS
    # Cancel global trigger placement to avoid conflicts
    if processor.placing_trigger:
        processor.stop_trigger_placement()
    return True

def cancel_personal_trigger_placement(processor) -> None:
    """Abort any pending personal trigger placement."""
    if processor.pending_personal_trigger:
        kind = processor.pending_personal_trigger.get('kind', 'unknown')
        print(f"Personal trigger placement cancelled ({kind})")
    processor.pending_personal_trigger = None

def place_trigger_at_click(processor, screen_x, screen_y):
    """Place a new trigger zone at the clicked location"""
    if not processor.placing_trigger:
        return

    # Convert screen coordinates to world coordinates
    world_coords = processor.coordinate_detector.screen_to_world_coordinates(
        screen_x, screen_y, processor.screen_width, processor.screen_height
    )

    if not world_coords['success']:
        print("Failed to determine world coordinates for trigger placement")
        return

    # Get ground height
    ground_z = get_ground_height(
        processor.world,
        carla.Location(world_coords['x'], world_coords['y'], world_coords['z']),
        cached_map=processor.cached_map,
        exclude_actors=processor.spawned_vehicles,
    )

    # Create new trigger with default radius
    trigger = {
        'x': world_coords['x'],
        'y': world_coords['y'],
        'z': ground_z,
        'radius': 2.5  # Default 2.5 meter radius
    }

    old_trigger = copy.deepcopy(processor.triggers[0]) if processor.triggers else None
    command = SetGlobalTriggerCommand(
        processor,
        old_trigger,
        trigger,
        preserve_selection=False,
        description="Place trigger",
    )
    if getattr(processor, "editor", None):
        processor.editor.execute_command(command)
    else:
        command.execute()

    print(f"Placed trigger at ({trigger['x']:.2f}, {trigger['y']:.2f}, {trigger['z']:.2f}) with radius {trigger['radius']:.2f}m")

    # Exit placement mode after placing one trigger
    processor.placing_trigger = False

def place_trigger_instantly(processor, screen_x, screen_y):
    """Place a trigger immediately, toggling placement mode on temporarily if needed."""
    was_active = processor.placing_trigger
    if not was_active:
        processor.start_trigger_placement()
    try:
        processor.place_trigger_at_click(screen_x, screen_y)
    finally:
        if not was_active and processor.placing_trigger:
            # If placement wasn't completed (e.g., coordinate failure), restore original state.
            processor.stop_trigger_placement()

def place_personal_trigger_at_click(processor, screen_x: int, screen_y: int) -> bool:
    """Place or reposition a personal trigger at the clicked location."""
    pending = processor.pending_personal_trigger
    if not pending:
        return False

    world_coords = processor.coordinate_detector.screen_to_world_coordinates(
        screen_x, screen_y, processor.screen_width, processor.screen_height
    )
    if not world_coords['success']:
        print("Failed to determine world coordinates for personal trigger placement.")
        return False

    ground_z = get_ground_height(
        processor.world,
        carla.Location(world_coords['x'], world_coords['y'], world_coords['z']),
        cached_map=processor.cached_map,
        exclude_actors=processor.spawned_vehicles,
    )

    center = {
        'x': float(world_coords['x']),
        'y': float(world_coords['y']),
        'z': float(ground_z),
    }
    radius = DEFAULT_PERSONAL_TRIGGER_RADIUS

    kind = pending.get('kind')
    selection = None
    old_center = None
    old_radius = None
    if kind == 'pedestrian':
        actor_id = pending.get('actor_id')
        if actor_id is None:
            print("Pending pedestrian trigger is missing an actor identifier.")
            return False
        selection = {'kind': 'pedestrian', 'id': actor_id}
        old_center, old_radius = processor._get_personal_trigger_payload(selection)
    elif kind == 'vehicle':
        actor_id = pending.get('actor_id')
        if actor_id is None:
            print("Pending vehicle trigger is missing an actor identifier.")
            return False
        selection = {'kind': 'vehicle', 'id': actor_id}
        old_center, old_radius = processor._get_personal_trigger_payload(selection)
    elif kind == 'traffic_light':
        group = pending.get('group')
        if not group:
            print("Pending traffic-light trigger is missing its group reference.")
            return False
        selection = {
            'kind': 'traffic_light',
            'group': group,
            'key': processor._traffic_light_trigger_key(group=group),
        }
        old_center, old_radius = processor._get_personal_trigger_payload(selection)
    else:
        print("Unsupported personal trigger kind.")
        return False

    command = SetPersonalTriggerCommand(
        processor,
        selection,
        center,
        radius,
        old_center=copy.deepcopy(old_center) if old_center else None,
        old_radius=old_radius,
    )
    if getattr(processor, "editor", None):
        success = processor.editor.execute_command(command)
    else:
        success = command.execute()
    if success is False:
        print("Failed to place personal trigger.")
        return False

    selection = command.selection or selection

    if kind == 'pedestrian':
        actor_id = pending.get('actor_id')
        print(f"Placed pedestrian trigger for ID {actor_id} at ({center['x']:.2f}, {center['y']:.2f}) radius {radius:.2f} m")
        panel = getattr(getattr(processor, "editor", None), "info_panel", None)
        if panel and panel.object_type == 'pedestrian' and panel.visible:
            current = getattr(panel, 'selected_object', None)
            if current and getattr(current, 'id', None) == actor_id:
                panel.show(current, 'pedestrian', processor.screen_width, processor.screen_height)
    elif kind == 'vehicle':
        actor_id = pending.get('actor_id')
        print(f"Placed vehicle trigger for ID {actor_id} at ({center['x']:.2f}, {center['y']:.2f}) radius {radius:.2f} m")
        panel = getattr(getattr(processor, "editor", None), "info_panel", None)
        if panel and panel.object_type == 'vehicle' and panel.visible:
            current = getattr(panel, 'selected_object', None)
            if current and getattr(current, 'id', None) == actor_id:
                panel.show(current, 'vehicle', processor.screen_width, processor.screen_height)
    elif kind == 'traffic_light':
        group = pending.get('group')
        key = selection.get('key')
        processor._cache_traffic_light_sequence(group)
        ids = sorted(group.ids) if getattr(group, 'ids', None) else []
        print(f"Placed traffic light trigger for IDs {ids} at ({center['x']:.2f}, {center['y']:.2f}) radius {radius:.2f} m")
        panel = getattr(getattr(processor, "editor", None), "info_panel", None)
        if panel and panel.object_type == 'traffic_light' and panel.visible:
            panel.show(group, 'traffic_light', processor.screen_width, processor.screen_height)

    if selection:
        processor.select_personal_trigger(selection)
        processor.update_personal_trigger_menu_position(force=True)
    processor.cancel_personal_trigger_placement()
    return True

def handle_trigger_click(processor, screen_x, screen_y):
    """Handle mouse click on triggers"""
    mouse_pos = (screen_x, screen_y)

    if processor.selected_personal_trigger:
        processor.clear_personal_trigger_selection()

    # First check if clicking on action menu icons
    if processor.selected_trigger_index is not None and processor.trigger_action_menu_position:
        action = processor.check_trigger_menu_icon_click(screen_x, screen_y)
        if action:
            # Keep trigger interactions exclusive with vehicle menus
            if processor.selected_vehicle or processor.vehicle_menu_position:
                processor.clear_vehicle_selection()
            if action == 'scale':
                # Scale is a press-drag gesture: it starts on mouse-down.
                processor.start_trigger_scaling(mouse_pos)
            else:
                # Click-type icons dispatch on mouse-UP (slide off to cancel).
                editor = getattr(processor, 'editor', None)
                if editor is not None:
                    editor.pending_menu_icon = {'menu': 'trigger', 'icon': action, 'pos': mouse_pos}
            return True

    # Check if clicking on a trigger
    for idx, trigger in enumerate(processor.triggers):
        # Project at the same height the circle is drawn (render_triggers uses
        # z + 0.3) so the clickable disc sits exactly on the visible one.
        draw_height = trigger['z'] + 0.3
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            trigger['x'], trigger['y'], draw_height
        )

        if not screen_pos['success']:
            continue

        trigger_screen_x = int(screen_pos['x'])
        trigger_screen_y = int(screen_pos['y'])

        # Calculate distance from mouse to trigger center
        distance = math.sqrt(
            (screen_x - trigger_screen_x) ** 2 +
            (screen_y - trigger_screen_y) ** 2
        )

        # Screen radius by projecting a rim point: exact at any zoom, screen
        # position, and resolution. (The old 500/camera_height approximation
        # undersized the clickable disc — most at high resolutions — making
        # the trigger feel unselectable.)
        rim_pos = processor.coordinate_detector.world_to_screen_coordinates(
            trigger['x'] + trigger['radius'], trigger['y'], draw_height
        )
        if rim_pos.get('success'):
            screen_radius = math.hypot(
                rim_pos['x'] - screen_pos['x'], rim_pos['y'] - screen_pos['y']
            )
        else:
            camera_height = processor.camera_controller.height if processor.camera_controller else 100
            screen_radius = trigger['radius'] * (500 / camera_height)  # Fallback approximation

        if distance <= screen_radius + 5:  # +5 for easier clicking
            # An actor's visible body or a waypoint marker under the cursor
            # wins over the trigger disc: fall through so the click hits it.
            if (processor.actor_under_click(screen_x, screen_y)
                    or processor._hit_test_displayed_waypoint(screen_x, screen_y)):
                return False
            # Select this trigger (selection kinds stay mutually exclusive)
            if processor.selected_vehicle or processor.vehicle_menu_position:
                processor.clear_vehicle_selection()
            processor.clear_traffic_light_selection()
            processor.selected_actor_ids = set()
            processor.selected_waypoint_group = None
            processor.selected_trigger_index = idx
            processor.trigger_action_menu_position = (trigger_screen_x, trigger_screen_y)
            processor.trigger_menu_hidden_for_camera_pan = False
            # Arm grab-to-move; update_trigger_movement promotes past the
            # drag threshold.
            processor.trigger_drag_armed = True
            processor.trigger_movement_start_pos = mouse_pos
            print(f"Selected trigger {idx} at ({trigger['x']:.2f}, {trigger['y']:.2f}, {trigger['z']:.2f})")
            return True

    return False

def trigger_targets_in_screen_rect(processor, rect):
    """All VISIBLE trigger-ish items whose center falls inside the screen rect
    (marquee tier 3): the selected actor's personal zone, the selected
    traffic-light group's trigger zone, the global trigger, and traffic-light
    stop-line groups (while the T overlay is on). Visibility rules match the
    click handlers — the box only selects what is on screen."""
    rect_min_x, rect_min_y = rect[0], rect[1]
    rect_max_x, rect_max_y = rect[0] + rect[2], rect[1] + rect[3]

    def in_rect(x, y):
        return rect_min_x <= x <= rect_max_x and rect_min_y <= y <= rect_max_y

    targets = []

    # Personal / traffic-light trigger zones (same visibility candidates as
    # _hit_test_personal_trigger: owner selected / overlay + group selected)
    zone_candidates = []
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        actor_id = processor.selected_vehicle.id
        if processor.selected_vehicle_is_pedestrian and actor_id in processor.pedestrian_trigger_radii:
            zone_candidates.append({'kind': 'pedestrian', 'id': actor_id})
        elif (not processor.selected_vehicle_is_pedestrian and not processor.is_ego_vehicle(actor_id)
                and actor_id in processor.vehicle_trigger_radii):
            zone_candidates.append({'kind': 'vehicle', 'id': actor_id})
    if processor.traffic_lights_visible and processor.selected_traffic_light_group:
        group = processor.selected_traffic_light_group
        center, radius, key = processor._get_traffic_light_trigger_data(group=group)
        if center and radius is not None:
            zone_candidates.append({'kind': 'traffic_light', 'group': group, 'key': key})
    for selection in zone_candidates:
        center, radius = processor._get_personal_trigger_payload(selection)
        if not center or radius is None:
            continue
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            center['x'], center['y'], center['z']
        )
        if screen_pos.get('success') and in_rect(screen_pos['x'], screen_pos['y']):
            targets.append({'kind': 'personal', 'selection': selection})

    # Global trigger (center projected at its draw height)
    for idx, trigger in enumerate(processor.triggers):
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            trigger['x'], trigger['y'], trigger['z'] + 0.3
        )
        if screen_pos.get('success') and in_rect(screen_pos['x'], screen_pos['y']):
            targets.append({'kind': 'global', 'index': idx,
                            'screen_pos': (int(screen_pos['x']), int(screen_pos['y']))})

    # Traffic-light stop-line groups (only while the overlay is shown)
    if processor.traffic_lights_visible and processor.traffic_light_groups:
        for group in processor.traffic_light_groups:
            if not group.lights:
                continue
            polygon_data = processor._get_traffic_light_group_screen_polygon(group)
            if not polygon_data:
                continue
            screen_points, center = polygon_data
            if center is None and screen_points:
                center = (
                    sum(point[0] for point in screen_points) / len(screen_points),
                    sum(point[1] for point in screen_points) / len(screen_points),
                )
            if center and in_rect(center[0], center[1]):
                targets.append({'kind': 'tl_group', 'group': group})

    return targets

def trigger_menu_icons(processor):
    """Icon row for the trigger menu — single source for hit-test AND render.
    The 3D orbit view drops the press-drag 'scale' icon (view + select only)."""
    icons = list(TRIGGER_MENU_ICONS)
    if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
        icons = [icon for icon in icons if icon != 'scale']
    return icons

def personal_trigger_menu_icons(processor):
    """Icon row for the personal-trigger menu; same 3D-view 'scale' filtering."""
    icons = list(PERSONAL_TRIGGER_MENU_ICONS)
    if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
        icons = [icon for icon in icons if icon != 'scale']
    return icons

def check_trigger_menu_icon_click(processor, mouse_x, mouse_y):
    """Check if a click hits any of the trigger menu icons"""
    if not (processor.selected_trigger_index is not None and processor.trigger_action_menu_position):
        return None

    return OverlayMenuRenderer.hit_test(
        processor.trigger_action_menu_position,
        trigger_menu_icons(processor),
        (processor.screen_width, processor.screen_height),
        (mouse_x, mouse_y),
    )

def check_personal_trigger_menu_icon_click(processor, mouse_x, mouse_y):
    """Check if a click hits any of the personal-trigger menu icons"""
    if not (processor.selected_personal_trigger and processor.personal_trigger_menu_position
            and not processor.personal_trigger_menu_hidden_for_camera_pan):
        return None

    return OverlayMenuRenderer.hit_test(
        processor.personal_trigger_menu_position,
        personal_trigger_menu_icons(processor),
        (processor.screen_width, processor.screen_height),
        (mouse_x, mouse_y),
    )

def start_trigger_movement(processor, mouse_pos):
    """Start moving the selected trigger"""
    if processor.selected_trigger_index is None:
        return

    processor.moving_trigger = True
    processor.trigger_movement_start_pos = mouse_pos
    trigger = processor.triggers[processor.selected_trigger_index]
    processor.trigger_movement_start_coords = {
        'x': trigger['x'],
        'y': trigger['y'],
        'z': trigger['z']
    }
    processor.trigger_original_height = trigger['z']  # Store original Z height
    processor._trigger_move_start_snapshot = copy.deepcopy(trigger)
    print(f"Started moving trigger {processor.selected_trigger_index}")

def stop_trigger_movement(processor):
    """Stop moving the trigger"""
    if not processor.moving_trigger:
        return

    processor.moving_trigger = False
    if processor.selected_trigger_index is not None:
        trigger = processor.triggers[processor.selected_trigger_index]
        ground_z = get_ground_height(
            processor.world,
            carla.Location(trigger['x'], trigger['y'], trigger['z']),
            cached_map=processor.cached_map,
            exclude_actors=processor.spawned_vehicles,
        )
        trigger['z'] = ground_z
        print(f"Stopped moving trigger {processor.selected_trigger_index}, adjusted Z to {ground_z:.2f}")

        start_snapshot = processor._trigger_move_start_snapshot
        processor._trigger_move_start_snapshot = None
        moved = True
        if start_snapshot:
            moved = (
                abs(start_snapshot.get('x', 0.0) - trigger['x']) > 1e-4
                or abs(start_snapshot.get('y', 0.0) - trigger['y']) > 1e-4
                or abs(start_snapshot.get('z', 0.0) - trigger['z']) > 1e-4
            )
        if moved:
            command = SetGlobalTriggerCommand(
                processor,
                start_snapshot,
                trigger,
                preserve_selection=True,
                description="Move trigger",
            )
            if getattr(processor, "editor", None):
                processor.editor.execute_command(command)
            else:
                command.execute()

def cancel_trigger_drag(processor):
    """Escape-cancel an in-progress global-trigger move: restore the start
    snapshot, keep the selection, push no undo command. Returns True when
    cancelled."""
    if not processor.moving_trigger:
        return False
    snapshot = processor._trigger_move_start_snapshot
    if (snapshot and processor.selected_trigger_index is not None
            and processor.selected_trigger_index < len(processor.triggers)):
        processor.triggers[processor.selected_trigger_index].update(copy.deepcopy(snapshot))
    processor.moving_trigger = False
    processor._trigger_move_start_snapshot = None
    processor.trigger_original_height = None
    return True

def cancel_trigger_scaling(processor):
    """Escape-cancel an in-progress global-trigger scale: restore the start
    radius, keep the selection, push no undo command. Returns True when
    cancelled."""
    if not processor.scaling_trigger:
        return False
    if (processor.selected_trigger_index is not None
            and processor.selected_trigger_index < len(processor.triggers)):
        processor.triggers[processor.selected_trigger_index]['radius'] = processor.trigger_scale_start_radius
    processor.scaling_trigger = False
    processor._trigger_scale_start_snapshot = None
    return True

def cancel_pedestrian_trigger_scaling(processor):
    """Escape-cancel an in-progress pedestrian-trigger scale (restore start
    radius, no undo command). Returns True when cancelled."""
    if not processor.scaling_pedestrian_trigger:
        return False
    target_id = processor._pedestrian_scaling_id
    if target_id is not None and target_id in processor.pedestrian_trigger_radii:
        processor.pedestrian_trigger_radii[target_id] = processor.pedestrian_trigger_scale_start_radius
    processor.scaling_pedestrian_trigger = False
    processor._pedestrian_scaling_id = None
    return True

def cancel_vehicle_trigger_scaling(processor):
    """Escape-cancel an in-progress vehicle-trigger scale (restore start
    radius, no undo command). Returns True when cancelled."""
    if not processor.scaling_vehicle_trigger:
        return False
    target_id = processor._vehicle_scaling_id
    if target_id is not None and target_id in processor.vehicle_trigger_radii:
        processor.vehicle_trigger_radii[target_id] = processor.vehicle_trigger_scale_start_radius
    processor.scaling_vehicle_trigger = False
    processor._vehicle_scaling_id = None
    return True

def cancel_personal_trigger_drag(processor):
    """Escape-cancel an in-progress personal-trigger move: restore the start
    center, keep the selection, push no undo command. Returns True when
    cancelled."""
    if not processor.moving_personal_trigger:
        return False
    selection = processor._personal_trigger_move_target or processor.selected_personal_trigger
    start_center = processor._personal_trigger_move_start_center
    if selection and start_center:
        center = dict(start_center)
        kind = selection.get('kind')
        if kind == 'pedestrian':
            actor_id = selection.get('id')
            if actor_id in processor.pedestrian_trigger_centers:
                processor.pedestrian_trigger_centers[actor_id] = center
        elif kind == 'vehicle':
            actor_id = selection.get('id')
            if actor_id in processor.vehicle_trigger_centers:
                processor.vehicle_trigger_centers[actor_id] = center
        elif kind == 'traffic_light':
            _, radius = processor._get_personal_trigger_payload(selection)
            processor._set_traffic_light_trigger_data(
                center,
                radius if radius is not None else DEFAULT_PERSONAL_TRIGGER_RADIUS,
                key=selection.get('key'),
                group=selection.get('group'),
            )
    processor.moving_personal_trigger = False
    processor._personal_trigger_move_target = None
    processor._personal_trigger_move_start_center = None
    processor.personal_trigger_original_height = None
    processor._personal_trigger_move_cache = None
    processor.update_personal_trigger_menu_position(force=True)
    return True

def _drag_grab_offset(processor, press_pos, center):
    """Screen-space offset between the press position and the object center.

    The trigger update functions place the center at the cursor's world
    position; subtracting this offset makes the grabbed point (not the
    center) track the cursor, so promotion does not teleport the disc.
    """
    if not center:
        return (0, 0)
    screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
        center['x'], center['y'], center['z']
    )
    if not screen_pos.get('success'):
        return (0, 0)
    return (press_pos[0] - screen_pos['x'], press_pos[1] - screen_pos['y'])

def update_trigger_movement(processor, mouse_pos):
    """Update trigger position during movement"""
    # Grab-to-move promotion: an armed click becomes a move once the mouse
    # travels past the threshold (a plain click below it only selects).
    if processor.trigger_drag_armed and not processor.moving_trigger:
        anchor = processor.trigger_movement_start_pos
        drag_distance = math.hypot(mouse_pos[0] - anchor[0], mouse_pos[1] - anchor[1])
        if drag_distance <= processor.object_drag_threshold:
            return
        processor.trigger_drag_armed = False
        trigger = None
        if processor.selected_trigger_index is not None and processor.selected_trigger_index < len(processor.triggers):
            trigger = processor.triggers[processor.selected_trigger_index]
        processor._trigger_drag_grab_offset = _drag_grab_offset(processor, anchor, trigger)
        processor.start_trigger_movement(anchor)

    if not processor.moving_trigger or processor.selected_trigger_index is None:
        return

    grab_dx, grab_dy = getattr(processor, '_trigger_drag_grab_offset', (0, 0))
    mouse_pos = (mouse_pos[0] - grab_dx, mouse_pos[1] - grab_dy)

    # Use fast coordinate conversion without raycast (like waypoints)
    world_coords = processor.coordinate_detector.screen_to_world_coordinates_no_raycast(
        mouse_pos[0], mouse_pos[1], processor.screen_width, processor.screen_height, processor.trigger_original_height
    )

    if not world_coords or not world_coords.get('success', False):
        return

    # Update trigger position (X and Y from mouse, keep original Z)
    trigger = processor.triggers[processor.selected_trigger_index]
    trigger['x'] = world_coords['x']
    trigger['y'] = world_coords['y']
    trigger['z'] = processor.trigger_original_height

    # Update menu position
    screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
        trigger['x'], trigger['y'], trigger['z']
    )
    if screen_pos['success']:
        processor.trigger_action_menu_position = (int(screen_pos['x']), int(screen_pos['y']))

def start_trigger_scaling(processor, mouse_pos):
    """Start scaling the selected trigger"""
    if processor.selected_trigger_index is None:
        return

    processor.scaling_trigger = True
    processor.trigger_scale_start_pos = mouse_pos
    trigger = processor.triggers[processor.selected_trigger_index]
    processor.trigger_scale_start_radius = trigger['radius']
    processor._trigger_scale_start_snapshot = copy.deepcopy(trigger)
    print(f"Started scaling trigger {processor.selected_trigger_index}")

def stop_trigger_scaling(processor):
    """Stop scaling the trigger"""
    if not processor.scaling_trigger:
        return

    processor.scaling_trigger = False
    if processor.selected_trigger_index is not None and processor.selected_trigger_index < len(processor.triggers):
        trigger = processor.triggers[processor.selected_trigger_index]
        print(f"Stopped scaling trigger {processor.selected_trigger_index}")
        start_snapshot = processor._trigger_scale_start_snapshot
        processor._trigger_scale_start_snapshot = None
        changed = True
        if start_snapshot:
            changed = abs(start_snapshot.get('radius', 0.0) - trigger.get('radius', 0.0)) > 1e-4
        if changed:
            command = SetGlobalTriggerCommand(
                processor,
                start_snapshot,
                trigger,
                preserve_selection=True,
                description="Scale trigger",
            )
            if getattr(processor, "editor", None):
                processor.editor.execute_command(command)
            else:
                command.execute()

def update_trigger_scaling(processor, mouse_pos):
    """Update trigger radius during scaling"""
    if not processor.scaling_trigger or processor.selected_trigger_index is None:
        return

    # Calculate vertical mouse movement
    delta_y = processor.trigger_scale_start_pos[1] - mouse_pos[1]  # Up is positive

    # Scale radius based on vertical movement (1 pixel = 0.1 meter)
    new_radius = processor.trigger_scale_start_radius + (delta_y * 0.1)

    # Clamp radius to reasonable values (minimum 2.5m, maximum 100m)
    new_radius = max(2.5, min(new_radius, 100.0))

    # Update trigger radius
    trigger = processor.triggers[processor.selected_trigger_index]
    trigger['radius'] = new_radius

def delete_selected_trigger(processor):
    """Delete the currently selected trigger"""
    if processor.selected_trigger_index is None:
        print("No trigger selected for deletion")
        return

    deleted_trigger = copy.deepcopy(processor.triggers[processor.selected_trigger_index])
    command = SetGlobalTriggerCommand(
        processor,
        deleted_trigger,
        None,
        preserve_selection=False,
        description="Delete trigger",
    )
    if getattr(processor, "editor", None):
        processor.editor.execute_command(command)
    else:
        command.execute()

    print(f"Deleted trigger at ({deleted_trigger['x']:.2f}, {deleted_trigger['y']:.2f}, {deleted_trigger['z']:.2f})")

def _get_personal_trigger_payload(processor, selection):
    """Return (center_dict, radius) for the provided personal trigger selection."""
    if not selection:
        return None, None
    kind = selection.get('kind')
    if kind == 'pedestrian':
        actor_id = selection.get('id')
        return (
            processor.pedestrian_trigger_centers.get(actor_id),
            processor.pedestrian_trigger_radii.get(actor_id),
        )
    if kind == 'vehicle':
        actor_id = selection.get('id')
        return (
            processor.vehicle_trigger_centers.get(actor_id),
            processor.vehicle_trigger_radii.get(actor_id),
        )
    if kind == 'traffic_light':
        group = selection.get('group')
        key = selection.get('key')
        center, radius, resolved_key = processor._get_traffic_light_trigger_data(key=key, group=group)
        if resolved_key and (key != resolved_key):
            selection['key'] = resolved_key
        if not group and resolved_key:
            selection['group'] = processor._find_traffic_light_group_by_key(resolved_key)
        return center, radius
    return None, None

def _personal_trigger_hit_radius(processor, trigger_radius: float) -> float:
    """Return the hover/click radius (screen space) for personal triggers."""
    base = max(14.0, min(64.0, float(trigger_radius) * 8.0))
    return base * 2.0

def _is_personal_trigger_hovered(processor, selection, mouse_pos) -> bool:
    """Determine whether the mouse cursor is hovering over a personal trigger marker."""
    if not selection or not mouse_pos:
        return False
    center, radius = processor._get_personal_trigger_payload(selection)
    if not center or radius is None:
        return False
    screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
        center['x'], center['y'], center['z']
    )
    if not screen_pos.get('success'):
        return False
    hit_radius = processor._personal_trigger_hit_radius(radius)
    distance = math.hypot(mouse_pos[0] - screen_pos['x'], mouse_pos[1] - screen_pos['y'])
    return distance <= hit_radius

def _ensure_traffic_light_trigger_context(
    processor,
    selection,
    *,
    auto_select: bool = True,
) -> Optional[TrafficLightGroupData]:
    """Resolve the traffic-light group for a trigger selection and ensure it is active."""
    if not selection or selection.get('kind') != 'traffic_light':
        return None

    group = selection.get('group')
    key = selection.get('key')
    if not group and key:
        group = processor._find_traffic_light_group_by_key(key)
        if group:
            selection['group'] = group

    if not group:
        return None

    if auto_select:
        current_group = processor.selected_traffic_light_group
        if current_group is not group:
            if not processor.select_traffic_light_group(group):
                return None
    return group

def select_personal_trigger(processor, selection):
    """Select a personal trigger and update its menu anchor."""
    if not selection:
        processor.clear_personal_trigger_selection()
        return
    if selection.get('kind') == 'traffic_light':
        group = processor._ensure_traffic_light_trigger_context(selection, auto_select=True)
        if not group:
            print("Select the traffic light group before editing its trigger.")
            return
        key = selection.get('key')
        if not key:
            key = processor._traffic_light_trigger_key(group=group)
            if key:
                selection['key'] = key
        processor._last_visible_traffic_light_trigger_key = selection.get(
            'key', processor._last_visible_traffic_light_trigger_key
        )
    processor.selected_personal_trigger = selection
    processor.selected_actor_ids = set()
    processor.selected_waypoint_group = None
    processor.personal_trigger_menu_position = None
    processor.personal_trigger_menu_hidden_for_camera_pan = False
    # Hide other action menus to keep UI exclusive
    processor.vehicle_menu_position = None
    processor.trigger_action_menu_position = None
    processor.traffic_light_menu_position = None
    processor.selected_trigger_index = None
    processor.update_personal_trigger_menu_position(force=True)

def clear_personal_trigger_selection(processor):
    """Clear any selected personal trigger."""
    processor.selected_personal_trigger = None
    processor.personal_trigger_drag_armed = False
    processor.personal_trigger_menu_position = None
    processor.personal_trigger_menu_hidden_for_camera_pan = False
    processor._last_visible_traffic_light_trigger_key = None
    if processor.moving_personal_trigger:
        processor.stop_personal_trigger_movement()
    if processor.scaling_pedestrian_trigger:
        processor.stop_pedestrian_trigger_scaling()
    if processor.scaling_vehicle_trigger:
        processor.stop_vehicle_trigger_scaling()

def handle_personal_trigger_click(processor, screen_x: int, screen_y: int) -> bool:
    """Handle clicks on personal triggers and their menus."""
    action = processor.check_personal_trigger_menu_icon_click(screen_x, screen_y)
    if action:
        if action == 'scale':
            # Scale is a press-drag gesture: it starts on mouse-down.
            processor.start_selected_personal_trigger_scaling((screen_x, screen_y))
        else:
            # Click-type icons dispatch on mouse-UP (slide off to cancel).
            editor = getattr(processor, 'editor', None)
            if editor is not None:
                editor.pending_menu_icon = {'menu': 'personal', 'icon': action, 'pos': (screen_x, screen_y)}
        return True

    hit_selection = processor._hit_test_personal_trigger(screen_x, screen_y)
    if hit_selection:
        # An actor's visible body or a waypoint marker under the cursor wins
        # over the zone disc (a zone centered on its owner would otherwise
        # make the owner un-grabbable while selected).
        if (processor.actor_under_click(screen_x, screen_y)
                or processor._hit_test_displayed_waypoint(screen_x, screen_y)):
            return False
        processor.select_personal_trigger(hit_selection)
        # Arm only if the selection actually stuck (select_personal_trigger
        # bails for traffic-light zones without group context).
        if processor.selected_personal_trigger is hit_selection:
            processor.personal_trigger_drag_armed = True
            processor.personal_trigger_movement_start_pos = (screen_x, screen_y)
        return True
    return False

def _hit_test_personal_trigger(processor, screen_x: int, screen_y: int):
    """Return the selection descriptor for a personal trigger under the cursor."""
    candidates = []
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        actor_id = processor.selected_vehicle.id
        if processor.selected_vehicle_is_pedestrian and actor_id in processor.pedestrian_trigger_radii:
            candidates.append({'kind': 'pedestrian', 'id': actor_id})
        elif (not processor.selected_vehicle_is_pedestrian and not processor.is_ego_vehicle(actor_id)
                and actor_id in processor.vehicle_trigger_radii):
            candidates.append({'kind': 'vehicle', 'id': actor_id})

    if processor.traffic_lights_visible:
        allowed_groups = []
        if processor.selected_traffic_light_group:
            allowed_groups.append(processor.selected_traffic_light_group)
        scaling_group = getattr(processor, "_traffic_light_scaling_group", None)
        if scaling_group and scaling_group not in allowed_groups:
            allowed_groups.append(scaling_group)

        for group in allowed_groups:
            center, radius, key = processor._get_traffic_light_trigger_data(group=group)
            if center and radius is not None:
                candidates.append({'kind': 'traffic_light', 'group': group, 'key': key})

    for selection in candidates:
        center, radius = processor._get_personal_trigger_payload(selection)
        if not center or radius is None:
            continue
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            center['x'], center['y'], center['z']
        )
        if not screen_pos['success']:
            continue
        distance = math.sqrt(
            (screen_x - screen_pos['x']) ** 2 +
            (screen_y - screen_pos['y']) ** 2
        )
        hit_radius = processor._personal_trigger_hit_radius(radius)
        if distance <= hit_radius:
            return selection
    return None

def start_personal_trigger_movement(processor, mouse_pos):
    """Begin drag-move for the currently selected personal trigger."""
    selection = processor.selected_personal_trigger
    if not selection:
        return
    center, _ = processor._get_personal_trigger_payload(selection)
    if not center:
        return

    processor.moving_personal_trigger = True
    processor._personal_trigger_move_target = dict(selection)
    processor.personal_trigger_movement_start_pos = mouse_pos
    processor.personal_trigger_original_height = center.get('z', 0.0)
    processor._personal_trigger_move_start_center = dict(center)
    processor._personal_trigger_move_cache = {}
    print("Started moving personal trigger")

def update_personal_trigger_movement(processor, mouse_pos):
    """Update personal trigger position while moving."""
    # Grab-to-move promotion: an armed click becomes a move once the mouse
    # travels past the threshold (a plain click below it only selects).
    if processor.personal_trigger_drag_armed and not processor.moving_personal_trigger:
        anchor = processor.personal_trigger_movement_start_pos
        drag_distance = math.hypot(mouse_pos[0] - anchor[0], mouse_pos[1] - anchor[1])
        if drag_distance <= processor.object_drag_threshold:
            return
        processor.personal_trigger_drag_armed = False
        armed_selection = processor.selected_personal_trigger
        center = None
        if armed_selection:
            center, _ = processor._get_personal_trigger_payload(armed_selection)
        processor._personal_trigger_drag_grab_offset = _drag_grab_offset(processor, anchor, center)
        processor.start_personal_trigger_movement(anchor)

    if not processor.moving_personal_trigger or not processor._personal_trigger_move_target:
        return

    grab_dx, grab_dy = getattr(processor, '_personal_trigger_drag_grab_offset', (0, 0))
    mouse_pos = (mouse_pos[0] - grab_dx, mouse_pos[1] - grab_dy)

    selection = processor._personal_trigger_move_target
    cache = processor._personal_trigger_move_cache
    coords = processor.coordinate_detector.screen_to_world_coordinates_no_raycast(
        mouse_pos[0],
        mouse_pos[1],
        processor.screen_width,
        processor.screen_height,
        processor.personal_trigger_original_height,
        cache=cache,
    )
    if (not coords) or (not coords.get('success', False)):
        coords = processor.coordinate_detector.screen_to_world_coordinates(
            mouse_pos[0],
            mouse_pos[1],
            processor.screen_width,
            processor.screen_height,
        )
    if not coords or not coords.get('success', False):
        return

    center = {
        'x': float(coords['x']),
        'y': float(coords['y']),
        'z': float(processor.personal_trigger_original_height),
    }
    _, current_radius = processor._get_personal_trigger_payload(selection)
    if selection.get('kind') == 'pedestrian':
        actor_id = selection.get('id')
        if actor_id in processor.pedestrian_trigger_centers:
            processor.pedestrian_trigger_centers[actor_id] = center
    elif selection.get('kind') == 'vehicle':
        actor_id = selection.get('id')
        if actor_id in processor.vehicle_trigger_centers:
            processor.vehicle_trigger_centers[actor_id] = center
    elif selection.get('kind') == 'traffic_light':
        group = selection.get('group')
        key = selection.get('key')
        resolved_key = processor._set_traffic_light_trigger_data(
            center,
            current_radius if current_radius is not None else DEFAULT_PERSONAL_TRIGGER_RADIUS,
            key=key,
            group=group,
        )
        if resolved_key and resolved_key != key:
            selection['key'] = resolved_key
        if group:
            processor._cache_traffic_light_sequence(group)
    processor.update_personal_trigger_menu_position(force=True)

def stop_personal_trigger_movement(processor):
    """Finish personal trigger movement."""
    if not processor.moving_personal_trigger:
        return

    selection = processor._personal_trigger_move_target or processor.selected_personal_trigger

    processor.moving_personal_trigger = False
    processor._personal_trigger_move_target = None
    processor.personal_trigger_original_height = None
    processor._personal_trigger_move_cache = None
    adjusted_height = None
    final_center = None
    radius_value = None

    if selection:
        center, radius = processor._get_personal_trigger_payload(selection)
        radius_value = radius
        if center:
            location = carla.Location(
                float(center.get('x', 0.0)),
                float(center.get('y', 0.0)),
                float(center.get('z', 0.0)),
            )
            ground_z = get_ground_height(
                processor.world,
                location,
                cached_map=processor.cached_map,
                exclude_actors=processor.spawned_vehicles,
            )
            adjusted_height = float(ground_z)
            updated_center = {
                'x': float(center.get('x', 0.0)),
                'y': float(center.get('y', 0.0)),
                'z': adjusted_height,
            }
            final_center = dict(updated_center)
            kind = selection.get('kind')
            if kind == 'pedestrian':
                actor_id = selection.get('id')
                if actor_id in processor.pedestrian_trigger_centers:
                    processor.pedestrian_trigger_centers[actor_id] = updated_center
            elif kind == 'vehicle':
                actor_id = selection.get('id')
                if actor_id in processor.vehicle_trigger_centers:
                    processor.vehicle_trigger_centers[actor_id] = updated_center
            elif kind == 'traffic_light':
                group = selection.get('group')
                key = selection.get('key')
                resolved_key = processor._set_traffic_light_trigger_data(
                    updated_center,
                    radius if radius is not None else DEFAULT_PERSONAL_TRIGGER_RADIUS,
                    key=key,
                    group=group,
                )
                if resolved_key and resolved_key != key:
                    selection['key'] = resolved_key
            processor.update_personal_trigger_menu_position(force=True)

    start_center = processor._personal_trigger_move_start_center
    if (processor.editor and selection and start_center and final_center):
        delta = (
            abs(start_center.get('x', 0.0) - final_center.get('x', 0.0)) +
            abs(start_center.get('y', 0.0) - final_center.get('y', 0.0)) +
            abs(start_center.get('z', 0.0) - final_center.get('z', 0.0))
        )
        if delta > 0.01:
            command = MovePersonalTriggerCommand(
                processor,
                dict(selection),
                dict(start_center),
                dict(final_center),
                radius_value,
            )
            processor.editor.execute_command(command)
    processor._personal_trigger_move_start_center = None

    if adjusted_height is not None:
        print(f"Stopped moving personal trigger, adjusted Z to {adjusted_height:.2f}")
    else:
        print("Stopped moving personal trigger")

def start_selected_personal_trigger_scaling(processor, mouse_pos):
    """Route scaling request for the selected personal trigger."""
    selection = processor.selected_personal_trigger
    if not selection:
        return
    kind = selection.get('kind')
    if kind == 'pedestrian':
        actor_id = selection.get('id')
        if actor_id is not None:
            processor.start_pedestrian_trigger_scaling(actor_id, mouse_pos)
    elif kind == 'vehicle':
        actor_id = selection.get('id')
        if actor_id is not None:
            processor.start_vehicle_trigger_scaling(actor_id, mouse_pos)
    elif kind == 'traffic_light':
        group = processor._ensure_traffic_light_trigger_context(selection, auto_select=True)
        if group:
            processor.start_traffic_light_trigger_scaling(mouse_pos)

def delete_selected_personal_trigger(processor):
    """Delete the currently selected personal trigger."""
    selection = processor.selected_personal_trigger
    if not selection:
        return
    kind = selection.get('kind')
    if kind == 'pedestrian':
        actor_id = selection.get('id')
        if actor_id is not None:
            processor.delete_pedestrian_trigger(actor_id)
    elif kind == 'vehicle':
        actor_id = selection.get('id')
        if actor_id is not None:
            processor.delete_vehicle_trigger(actor_id)
    elif kind == 'traffic_light':
        key = selection.get('key')
        group = processor._ensure_traffic_light_trigger_context(selection, auto_select=True)
        if group:
            processor.delete_traffic_light_trigger(group)
        elif key:
            print("Traffic light group missing; removing saved trigger data.")
            processor._delete_traffic_light_trigger_data(key=key)
    processor.clear_personal_trigger_selection()

def _ensure_pedestrian_trigger(processor, pedestrian_id: int) -> bool:
    """Create a trigger entry for the provided pedestrian if missing."""
    if pedestrian_id in processor.pedestrian_trigger_radii:
        return True

    # Get pedestrian actor
    pedestrian = None
    for actor in processor.spawned_vehicles:
        if actor.id == pedestrian_id and actor.is_alive:
            pedestrian = actor
            break

    if not pedestrian:
        print("Warning: Unable to find pedestrian actor.")
        return False

    # Get pedestrian location as trigger center
    location = pedestrian.get_location()
    processor.pedestrian_trigger_centers[pedestrian_id] = {
        'x': float(location.x),
        'y': float(location.y),
        'z': float(location.z),
    }
    processor.pedestrian_trigger_radii[pedestrian_id] = DEFAULT_PERSONAL_TRIGGER_RADIUS  # Default radius

    print(
        f"Created pedestrian trigger for ID {pedestrian_id} at "
        f"({location.x:.2f}, {location.y:.2f}, {location.z:.2f}) "
        f"with radius {DEFAULT_PERSONAL_TRIGGER_RADIUS:.1f} m"
    )

    # Refresh info panel if showing this pedestrian
    panel = getattr(getattr(processor, "editor", None), "info_panel", None)
    if panel and panel.object_type == 'pedestrian' and panel.visible:
        current = getattr(panel, 'selected_object', None)
        if current and getattr(current, 'id', None) == pedestrian_id:
            screen_width = getattr(processor, "screen_width", 1280)
            screen_height = getattr(processor, "screen_height", 720)
            panel.show(pedestrian, 'pedestrian', screen_width, screen_height)

    return True

def start_pedestrian_trigger_scaling(processor, pedestrian_id: int, mouse_pos: tuple) -> None:
    """Begin scaling operation for the selected pedestrian's trigger."""
    if pedestrian_id not in processor.pedestrian_trigger_radii:
        return

    processor.scaling_pedestrian_trigger = True
    processor._pedestrian_scaling_id = pedestrian_id
    processor.pedestrian_trigger_scale_start_pos = mouse_pos
    processor.pedestrian_trigger_scale_start_radius = float(processor.pedestrian_trigger_radii[pedestrian_id])
    print(f"Started scaling pedestrian trigger for ID {pedestrian_id}")

def update_pedestrian_trigger_scaling(processor, mouse_pos: tuple) -> None:
    """Update trigger radius while scaling a pedestrian trigger."""
    if not processor.scaling_pedestrian_trigger or processor._pedestrian_scaling_id is None:
        return

    pedestrian_id = processor._pedestrian_scaling_id
    if pedestrian_id not in processor.pedestrian_trigger_radii:
        processor.stop_pedestrian_trigger_scaling()
        return

    delta_y = processor.pedestrian_trigger_scale_start_pos[1] - mouse_pos[1]
    new_radius = processor.pedestrian_trigger_scale_start_radius + (delta_y * 0.1)
    new_radius = max(MIN_PERSONAL_TRIGGER_RADIUS, new_radius)
    processor.pedestrian_trigger_radii[pedestrian_id] = new_radius

def stop_pedestrian_trigger_scaling(processor) -> None:
    """End any active pedestrian trigger scaling."""
    if not processor.scaling_pedestrian_trigger:
        return

    target_id = processor._pedestrian_scaling_id
    radius = processor.pedestrian_trigger_radii.get(target_id) if target_id is not None else None
    if target_id is not None and radius is not None:
        print(
            f"Stopped scaling pedestrian trigger for ID "
            f"{target_id} "
            f"(radius {radius:.2f} m)"
        )
        start_radius = getattr(processor, "pedestrian_trigger_scale_start_radius", None)
        if start_radius is not None and abs(radius - start_radius) > 1e-4:
            center_snapshot = processor.pedestrian_trigger_centers.get(target_id)
            if center_snapshot is None:
                processor._ensure_pedestrian_trigger(target_id)
                center_snapshot = processor.pedestrian_trigger_centers.get(target_id)
            selection = {'kind': 'pedestrian', 'id': target_id}
            command = SetPersonalTriggerCommand(
                processor,
                selection,
                dict(center_snapshot) if center_snapshot else None,
                radius,
                old_center=copy.deepcopy(center_snapshot) if center_snapshot else None,
                old_radius=start_radius,
            )
            if getattr(processor, "editor", None):
                processor.editor.execute_command(command)
            else:
                command.execute()
    processor.scaling_pedestrian_trigger = False
    processor._pedestrian_scaling_id = None

def delete_pedestrian_trigger(processor, pedestrian_id: int) -> bool:
    """Remove the trigger associated with the given pedestrian."""
    if pedestrian_id not in processor.pedestrian_trigger_radii:
        print("No pedestrian trigger available to delete.")
        return False

    if processor.scaling_pedestrian_trigger and processor._pedestrian_scaling_id == pedestrian_id:
        processor.stop_pedestrian_trigger_scaling()

    old_center = processor.pedestrian_trigger_centers.get(pedestrian_id)
    old_radius = processor.pedestrian_trigger_radii.get(pedestrian_id)
    command = SetPersonalTriggerCommand(
        processor,
        {'kind': 'pedestrian', 'id': pedestrian_id},
        None,
        None,
        old_center=copy.deepcopy(old_center) if old_center is not None else None,
        old_radius=old_radius,
    )
    editor = getattr(processor, "editor", None)
    success = editor.execute_command(command) if editor else command.execute()
    if success is False:
        print(f"Failed to delete pedestrian trigger for ID {pedestrian_id}")
        return False

    print(f"Deleted pedestrian trigger for ID {pedestrian_id}")
    if (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'pedestrian'
            and processor.selected_personal_trigger.get('id') == pedestrian_id):
        processor.clear_personal_trigger_selection()
    return True

def _ensure_vehicle_trigger(processor, vehicle_id: int) -> bool:
    """Create a trigger entry for the provided NPC vehicle if missing."""
    if vehicle_id in processor.vehicle_trigger_radii:
        return True

    vehicle_actor = processor.get_spawned_vehicle(vehicle_id)
    if not vehicle_actor or not vehicle_actor.is_alive:
        print("Warning: Unable to find vehicle actor.")
        return False
    if vehicle_actor.type_id.startswith('walker.'):
        return False
    if processor.is_ego_vehicle(vehicle_id):
        return False

    location = vehicle_actor.get_location()
    processor.vehicle_trigger_centers[vehicle_id] = {
        'x': float(location.x),
        'y': float(location.y),
        'z': float(location.z),
    }
    processor.vehicle_trigger_radii[vehicle_id] = DEFAULT_PERSONAL_TRIGGER_RADIUS  # Default radius

    print(
        f"Created vehicle trigger for ID {vehicle_id} at "
        f"({location.x:.2f}, {location.y:.2f}, {location.z:.2f}) "
        f"with radius {DEFAULT_PERSONAL_TRIGGER_RADIUS:.1f} m"
    )

    panel = getattr(getattr(processor, "editor", None), "info_panel", None)
    if panel and panel.object_type == 'vehicle' and panel.visible:
        current = getattr(panel, 'selected_object', None)
        if current and getattr(current, 'id', None) == vehicle_id:
            screen_width = getattr(processor, "screen_width", 1280)
            screen_height = getattr(processor, "screen_height", 720)
            panel.show(vehicle_actor, 'vehicle', screen_width, screen_height)

    return True

def start_vehicle_trigger_scaling(processor, vehicle_id: int, mouse_pos: tuple) -> None:
    """Begin scaling operation for the selected vehicle's trigger."""
    if vehicle_id not in processor.vehicle_trigger_radii:
        return

    processor.scaling_vehicle_trigger = True
    processor._vehicle_scaling_id = vehicle_id
    processor.vehicle_trigger_scale_start_pos = mouse_pos
    processor.vehicle_trigger_scale_start_radius = float(processor.vehicle_trigger_radii[vehicle_id])
    print(f"Started scaling vehicle trigger for ID {vehicle_id}")

def update_vehicle_trigger_scaling(processor, mouse_pos: tuple) -> None:
    """Update trigger radius while scaling a vehicle trigger."""
    if not processor.scaling_vehicle_trigger or processor._vehicle_scaling_id is None:
        return

    vehicle_id = processor._vehicle_scaling_id
    if vehicle_id not in processor.vehicle_trigger_radii:
        processor.stop_vehicle_trigger_scaling()
        return

    delta_y = processor.vehicle_trigger_scale_start_pos[1] - mouse_pos[1]
    new_radius = processor.vehicle_trigger_scale_start_radius + (delta_y * 0.1)
    new_radius = max(MIN_PERSONAL_TRIGGER_RADIUS, new_radius)
    processor.vehicle_trigger_radii[vehicle_id] = new_radius

def stop_vehicle_trigger_scaling(processor) -> None:
    """End any active vehicle trigger scaling."""
    if not processor.scaling_vehicle_trigger:
        return

    target_id = processor._vehicle_scaling_id
    radius = processor.vehicle_trigger_radii.get(target_id) if target_id is not None else None
    if target_id is not None and radius is not None:
        print(
            f"Stopped scaling vehicle trigger for ID "
            f"{target_id} "
            f"(radius {radius:.2f} m)"
        )
        start_radius = getattr(processor, "vehicle_trigger_scale_start_radius", None)
        if start_radius is not None and abs(radius - start_radius) > 1e-4:
            center_snapshot = processor.vehicle_trigger_centers.get(target_id)
            if center_snapshot is None:
                processor._ensure_vehicle_trigger(target_id)
                center_snapshot = processor.vehicle_trigger_centers.get(target_id)
            selection = {'kind': 'vehicle', 'id': target_id}
            command = SetPersonalTriggerCommand(
                processor,
                selection,
                dict(center_snapshot) if center_snapshot else None,
                radius,
                old_center=copy.deepcopy(center_snapshot) if center_snapshot else None,
                old_radius=start_radius,
            )
            if getattr(processor, "editor", None):
                processor.editor.execute_command(command)
            else:
                command.execute()
    processor.scaling_vehicle_trigger = False
    processor._vehicle_scaling_id = None

def delete_vehicle_trigger(processor, vehicle_id: int) -> bool:
    """Remove the trigger associated with the given vehicle."""
    if vehicle_id not in processor.vehicle_trigger_radii:
        print("No vehicle trigger available to delete.")
        return False

    if processor.scaling_vehicle_trigger and processor._vehicle_scaling_id == vehicle_id:
        processor.stop_vehicle_trigger_scaling()

    old_center = processor.vehicle_trigger_centers.get(vehicle_id)
    old_radius = processor.vehicle_trigger_radii.get(vehicle_id)
    command = SetPersonalTriggerCommand(
        processor,
        {'kind': 'vehicle', 'id': vehicle_id},
        None,
        None,
        old_center=copy.deepcopy(old_center) if old_center is not None else None,
        old_radius=old_radius,
    )
    editor = getattr(processor, "editor", None)
    success = editor.execute_command(command) if editor else command.execute()
    if success is False:
        print(f"Failed to delete vehicle trigger for ID {vehicle_id}")
        return False

    print(f"Deleted vehicle trigger for ID {vehicle_id}")
    if (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'vehicle'
            and processor.selected_personal_trigger.get('id') == vehicle_id):
        processor.clear_personal_trigger_selection()
    return True
