"""In-world CARLA debug drawing (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-31).

world.debug drawing of waypoints/connections/finish lines, the pedestrian
highlight world-tick callback (subscription removed by stored int id — the
processor delegate keeps the registered bound-method name; the round-trip
worker getattr-strings _unregister_pedestrian_highlight_tick), and the three
render_*_triggers_overlay methods that draw via world.debug despite their
names (they also write scene selection caches — verbatim).
_WALKER_COLOR_PALETTE moves here with its only consumer
(_on_pedestrian_highlight_tick); vse_playback/scenario.py keeps its own
independent copy, as before.
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import carla
import pygame

from vse_editor.constants import WAYPOINT_MARKER_Z_OFFSET
from vse_editor.scene_types import TrafficLightGroupData

_WALKER_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 75, 0),      # Dark green
    (0, 0, 75),      # Dark blue
    (75, 75, 0),     # Dark yellow
    (75, 0, 75),     # Dark magenta
    (0, 75, 75),     # Dark cyan
    (75, 50, 0),     # Dark orange
    (75, 30, 55),    # Dark pink
]


def refresh_waypoints_carla_debug(processor):
    """Refresh waypoint visualization using CARLA debug drawing system"""
    # Hide waypoints during scenario playback
    if processor.session.scenario_running:
        return

    # Get hovered waypoint from mouse position
    hovered_waypoint = processor.get_hovered_waypoint()
    
    # Draw waypoints using CARLA debug drawing
    if hasattr(processor, 'world') and processor.world:
        # Use the existing marker system's debug drawing function
        if hasattr(processor, 'world') and hasattr(processor.world, 'debug'):
            # We need to create a temporary marker system instance to access the drawing function
            # Or add this functionality directly here
            processor.draw_waypoints_carla_debug_local()

def _get_start_marker_for_vehicle_id(processor, vehicle_id: Optional[int]) -> Optional[dict]:
    """Return stored spawn marker data for a vehicle ID while it is hidden during scenario playback."""
    if vehicle_id is None:
        return None

    info_panel = getattr(getattr(processor, "editor", None), "info_panel", None)
    if info_panel:
        markers = getattr(info_panel, "scenario_starting_points", {}) or {}
        marker = markers.get(vehicle_id)
        if marker:
            return marker
        legacy = getattr(info_panel, "scenario_starting_point", None)
        if legacy and legacy.get("vehicle_id") == vehicle_id:
            return legacy

    editor = getattr(processor, "editor", None)
    if editor:
        for entry in processor.session.saved_scene_vehicles:
            actor = entry.get("actor")
            if actor and actor.id == vehicle_id:
                transform = entry.get("transform")
                if transform:
                    loc = transform.location
                    return {
                        "x": loc.x,
                        "y": loc.y,
                        "z": loc.z,
                        "vehicle_id": vehicle_id,
                    }
    return None

def _get_start_marker_for_vehicle(processor, vehicle: Optional[carla.Actor]) -> Optional[dict]:
    """Backward-compatible helper that uses actor ID to resolve start markers."""
    vehicle_id = getattr(vehicle, "id", None) if vehicle else None
    return processor._get_start_marker_for_vehicle_id(vehicle_id)

def _register_pedestrian_highlight_tick(processor) -> None:
    """Subscribe to CARLA world ticks to draw pedestrian rings."""
    if not processor.world or processor._pedestrian_highlight_subscription is not None:
        return
    try:
        processor._pedestrian_highlight_subscription = processor.world.on_tick(processor._on_pedestrian_highlight_tick)
    except Exception:
        processor._pedestrian_highlight_subscription = None

def _unregister_pedestrian_highlight_tick(processor) -> None:
    """Remove the pedestrian highlight tick subscription."""
    if processor._pedestrian_highlight_subscription is None:
        return
    try:
        if processor.world:
            processor.world.remove_on_tick(processor._pedestrian_highlight_subscription)
    except Exception:
        pass
    finally:
        processor._pedestrian_highlight_subscription = None

def _on_pedestrian_highlight_tick(processor, _snapshot) -> None:
    """Draw a single world-space ring around each pedestrian (no center marker)."""
    if not processor.world or not processor.spawned_vehicles:
        return

    now = time.time()
    seen_ids: Set[int] = set()
    lifetime = 0.4
    radius = 1.0
    steps = 24
    # Scale line thickness based on camera height (thinner when zoomed in)
    line_scale = max(0.5, min(1.5, processor.camera_controller.height / 200.0))
    thickness = 0.1 * line_scale
    refresh_interval = 0.25  # seconds

    for actor in processor.spawned_vehicles:
        if not actor or not actor.is_alive:
            continue
        try:
            type_id = actor.type_id
        except Exception:
            continue
        if not type_id.startswith('walker.'):
            continue

        actor_id = actor.id
        seen_ids.add(actor_id)

        last_draw = processor._pedestrian_highlight_last_draw.get(actor_id, 0.0)
        if (now - last_draw) < refresh_interval:
            continue

        color = processor.pedestrian_colors.get(actor_id)
        if color is None:
            palette_index = len(processor.pedestrian_colors) % len(_WALKER_COLOR_PALETTE)
            r, g, b = _WALKER_COLOR_PALETTE[palette_index]
            color = carla.Color(r, g, b)
            processor.pedestrian_colors[actor_id] = color

        try:
            location = actor.get_location()
        except Exception:
            continue

        base_z = location.z + 0.25
        draw_height = base_z

        try:
            angle_step = 2.0 * math.pi / steps
            for i in range(steps):
                angle1 = angle_step * i
                angle2 = angle_step * (i + 1)
                point1 = carla.Location(
                    location.x + radius * math.cos(angle1),
                    location.y + radius * math.sin(angle1),
                    draw_height,
                )
                point2 = carla.Location(
                    location.x + radius * math.cos(angle2),
                    location.y + radius * math.sin(angle2),
                    draw_height,
                )
                processor.world.debug.draw_line(
                    point1,
                    point2,
                    thickness=thickness,
                    color=color,
                    life_time=lifetime,
                    persistent_lines=False,
                )
            processor._pedestrian_highlight_last_draw[actor_id] = now
        except Exception:
            continue

    if processor.pedestrian_colors:
        stale_ids = [pid for pid in processor.pedestrian_colors.keys() if pid not in seen_ids]
        for pid in stale_ids:
            processor.pedestrian_colors.pop(pid, None)
            processor._pedestrian_highlight_last_draw.pop(pid, None)

def draw_waypoints_carla_debug_local(processor):
    """Local version of waypoint debug drawing"""
    if not processor.waypoint_display_vehicle_id:
        return

    waypoints = processor.get_vehicle_waypoints(processor.waypoint_display_vehicle_id)
    if not waypoints:
        return

    if not waypoints:
        return
        
    # Find the vehicle for this waypoint set (may be missing during playback)
    vehicle = None
    for spawned_vehicle in processor.spawned_vehicles:
        if spawned_vehicle.id == processor.waypoint_display_vehicle_id:
            if spawned_vehicle.is_alive:
                vehicle = spawned_vehicle
            break
        
    # Get hovered waypoint
    hovered_waypoint = processor.get_hovered_waypoint()

    # Resolve any stored start marker (works even if vehicle is hidden)
    start_point = processor._get_start_marker_for_vehicle_id(processor.waypoint_display_vehicle_id)
    
    override_id = getattr(processor, "_waypoint_ego_override_id", None)
    is_ego_path = processor.is_ego_vehicle(processor.waypoint_display_vehicle_id) or (
        override_id is not None and processor.waypoint_display_vehicle_id == override_id
    )

    # Calculate line scale based on camera height (thinner when zoomed in close)
    # Camera height ranges from 40 (close) to 1000 (far), baseline 200
    line_scale = max(0.5, min(1.5, processor.camera_controller.height / 200.0))

    # Draw connection lines first (so they appear behind markers)
    processor._draw_waypoint_connections_carla_debug_local(vehicle, start_point, waypoints, is_ego_path=is_ego_path, line_scale=line_scale)
    
    # Draw starting point marker if vehicle is hidden for scenario
    if start_point:
        start_location = carla.Location(start_point['x'], start_point['y'], start_point['z'] + 0.5)
        
        # Draw starting point marker with distinct appearance
        processor.world.debug.draw_point(
            start_location,
            size=0.5,
            color=carla.Color(150, 0, 150),  # Muted magenta for starting point
            life_time=1.0
        )

        # Draw "START" text above the marker
        text_location = carla.Location(start_point['x'], start_point['y'], start_point['z'] + 2.0)
        processor.world.debug.draw_string(
            text_location,
            "START",
            draw_shadow=True,
            color=carla.Color(150, 0, 150),  # Muted magenta text
            life_time=1.0,
            persistent_lines=False
        )
    
    # Draw waypoint markers
    for i, waypoint in enumerate(waypoints):
        # Determine waypoint state and colors
        is_selected = (processor.selected_waypoint_vehicle_id == processor.waypoint_display_vehicle_id and 
                      processor.selected_waypoint_index == i)
        is_hovered = (hovered_waypoint and hovered_waypoint[0] == processor.waypoint_display_vehicle_id and 
                     hovered_waypoint[1] == i)
        is_finish_line = (i == len(waypoints) - 1)
        
        # Choose colors based on waypoint properties and state (muted colors for less glare)
        base_color = carla.Color(150, 130, 0) if is_ego_path else carla.Color(0, 150, 0)
        selected_color = carla.Color(150, 150, 70) if is_ego_path else carla.Color(150, 150, 0)
        hovered_color = carla.Color(150, 140, 90) if is_ego_path else carla.Color(150, 120, 0)
        finish_color = carla.Color(150, 130, 0) if is_ego_path else carla.Color(150, 30, 30)
            
        # Modify color based on state
        if is_finish_line and not is_selected:
            marker_color = finish_color
        elif is_selected:
            marker_color = selected_color
        elif is_hovered:
            marker_color = hovered_color
        else:
            marker_color = base_color
            
        # Create waypoint location
        location = carla.Location(waypoint['x'], waypoint['y'], waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET)
        
        # Draw different markers based on type
        if is_finish_line:
            processor._draw_finish_line_carla_debug_local(
                location, waypoints, i, is_selected, is_hovered, is_ego_path=is_ego_path, line_scale=line_scale
            )
        else:
            # Draw main waypoint marker (slightly bigger for better visibility)
            marker_size = 0.4 if is_selected or is_hovered else 0.3
            processor.world.debug.draw_point(
                location,
                size=marker_size,
                color=marker_color,
                life_time=processor.waypoint_debug_lifetime
            )
            
            # Draw waypoint number above the marker
            text_location = carla.Location(waypoint['x'], waypoint['y'], waypoint['z'] + 2.0)
            processor.world.debug.draw_string(
                text_location,
                str(i + 1),
                draw_shadow=True,
                color=carla.Color(180, 180, 180),  # Slightly dimmed white for readability
                life_time=processor.waypoint_debug_lifetime,
                persistent_lines=False
            )

def get_hovered_waypoint(processor):
    """Get the currently hovered waypoint based on screen coordinates"""
    if not hasattr(processor, 'waypoint_hover_index') or not processor.waypoint_hover_index:
        return None
    return processor.waypoint_hover_index

def _draw_waypoint_connections_carla_debug_local(processor, vehicle, start_point, waypoints, *, is_ego_path=False, line_scale=1.0):
    """Draw connection lines between waypoints using CARLA debug drawing"""
    line_color = carla.Color(75, 65, 0) if is_ego_path else carla.Color(0, 75, 0)  # Dark gold/green so markers stand out
    
    for i, waypoint in enumerate(waypoints):
        if i == 0:
            # First waypoint - connect to vehicle or starting point marker
            if start_point:
                # Use starting point marker
                start_location = carla.Location(
                    start_point['x'], 
                    start_point['y'], 
                    start_point['z'] + 0.5
                )
            elif vehicle:
                # Use actual vehicle location
                vehicle_location = vehicle.get_location()
                start_location = carla.Location(
                    vehicle_location.x, 
                    vehicle_location.y, 
                    vehicle_location.z + 0.5
                )
            else:
                start_location = None
        else:
            # Connect to previous waypoint
            prev_waypoint = waypoints[i - 1]
            start_location = carla.Location(
                prev_waypoint['x'],
                prev_waypoint['y'],
                prev_waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
            )

        # End point is current waypoint
        end_location = carla.Location(
            waypoint['x'],
            waypoint['y'],
            waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
        )
        
        # Draw connection line
        if start_location is not None:
            processor.world.debug.draw_line(
                start_location,
                end_location,
                thickness=0.1 * line_scale,
                color=line_color,
                life_time=processor.waypoint_debug_lifetime
            )

def _draw_finish_line_carla_debug_local(processor, location, waypoints, waypoint_index, is_selected, is_hovered, *, is_ego_path=False, line_scale=1.0):
    """Draw a finish line pattern for the last waypoint using CARLA debug drawing"""
    # Calculate trajectory direction for perpendicular orientation
    if waypoint_index > 0:
        # Use direction from previous waypoint to this one
        prev_waypoint = waypoints[waypoint_index - 1]
        current_waypoint = waypoints[waypoint_index]
        
        # Calculate trajectory angle
        dx = current_waypoint['x'] - prev_waypoint['x']
        dy = current_waypoint['y'] - prev_waypoint['y']
        trajectory_angle = math.atan2(dy, dx)
        
        # Perpendicular angle (90 degrees rotated)
        perpendicular_angle = trajectory_angle + math.pi / 2
    else:
        perpendicular_angle = 0  # Default horizontal for single waypoint
    
    # Finish line dimensions (wider + thicker than a plain waypoint marker so
    # the finish is easy to see and grab at any zoom)
    line_length = 8.0 if is_selected or is_hovered else 7.0
    line_thickness = (0.5 if is_selected or is_hovered else 0.4) * line_scale
    row_offset = 0.6  # meters between the two checkered rows (along the path)

    # Calculate finish line endpoints
    half_length = line_length / 2
    end1_x = location.x + half_length * math.cos(perpendicular_angle)
    end1_y = location.y + half_length * math.sin(perpendicular_angle)
    end2_x = location.x - half_length * math.cos(perpendicular_angle)
    end2_y = location.y - half_length * math.sin(perpendicular_angle)

    # Draw a two-row checkered flag (alternating segments, offset rows) so the
    # finish reads as a checkerboard instead of a thin line
    segments = 6
    if is_ego_path:
        segment_color_1 = carla.Color(150, 140, 90)   # Light gold
        segment_color_2 = carla.Color(110, 90, 0)     # Deep gold
    else:
        segment_color_1 = carla.Color(150, 150, 150)  # Light gray
        segment_color_2 = carla.Color(150, 25, 25)    # Red

    trajectory_dir = perpendicular_angle - math.pi / 2
    row_dx = (row_offset / 2.0) * math.cos(trajectory_dir)
    row_dy = (row_offset / 2.0) * math.sin(trajectory_dir)

    for row in (0, 1):
        row_sign = 1 if row == 0 else -1
        for i in range(segments):
            t1 = i / segments
            t2 = (i + 1) / segments

            seg_start_x = end1_x + t1 * (end2_x - end1_x) + row_sign * row_dx
            seg_start_y = end1_y + t1 * (end2_y - end1_y) + row_sign * row_dy
            seg_end_x = end1_x + t2 * (end2_x - end1_x) + row_sign * row_dx
            seg_end_y = end1_y + t2 * (end2_y - end1_y) + row_sign * row_dy

            seg_start = carla.Location(seg_start_x, seg_start_y, location.z)
            seg_end = carla.Location(seg_end_x, seg_end_y, location.z)

            # Alternate colors, offset by one per row (checkerboard)
            color = segment_color_1 if (i + row) % 2 == 0 else segment_color_2

            processor.world.debug.draw_line(
                seg_start,
                seg_end,
                thickness=line_thickness,
                color=color,
                life_time=processor.waypoint_debug_lifetime
            )

def render_traffic_light_triggers_overlay(processor):
    """Draw world-space debug circles for the selected traffic light trigger."""
    if not processor.world or not processor.traffic_lights_visible:
        return

    # Throttle to the waypoint-debug cadence (opt-02): these draws ran every
    # frame with 0.15 s lifetimes, stacking ~9 overlapping copies server-side.
    now = time.monotonic()
    if now - getattr(processor, "_tl_trigger_debug_last_t", 0.0) < processor.waypoint_debug_refresh_interval:
        return
    processor._tl_trigger_debug_last_t = now

    group = processor.selected_traffic_light_group
    selection_group: Optional[TrafficLightGroupData] = None
    center_payload: Optional[Dict[str, float]] = None
    radius_value: Optional[float] = None
    resolved_key: Optional[Tuple[str, Tuple]] = None

    editor_running = bool(processor.session.scenario_running)

    if group:
        center_payload, radius_value, resolved_key = processor._get_traffic_light_trigger_data(group=group)
        if center_payload:
            selection_group = group
    if (not center_payload and processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'traffic_light'):
        sel_group = processor.selected_personal_trigger.get('group')
        sel_key = processor.selected_personal_trigger.get('key')
        center_payload, radius_value, resolved_key = processor._get_traffic_light_trigger_data(
            key=sel_key,
            group=sel_group,
        )
        if center_payload:
            selection_group = sel_group or processor._find_traffic_light_group_by_key(resolved_key)
            if selection_group:
                processor.selected_personal_trigger['group'] = selection_group
            if resolved_key:
                processor.selected_personal_trigger['key'] = resolved_key
    if not center_payload and editor_running and processor._scenario_active_traffic_light_trigger:
        snapshot = processor._scenario_active_traffic_light_trigger
        center_payload = snapshot.get('center') if isinstance(snapshot, dict) else None
        radius_value = snapshot.get('radius') if isinstance(snapshot, dict) else None
    if (not center_payload and processor._last_visible_traffic_light_trigger_key
            and not processor.selected_personal_trigger):
        center_payload = processor.traffic_light_trigger_centers.get(processor._last_visible_traffic_light_trigger_key)
        radius_value = processor.traffic_light_trigger_radii.get(processor._last_visible_traffic_light_trigger_key)
        resolved_key = processor._last_visible_traffic_light_trigger_key
        selection_group = processor._find_traffic_light_group_by_key(resolved_key)
    if not center_payload:
        return

    if not center_payload or radius_value is None:
        return

    if processor.traffic_light_overlays_hidden_for_camera_pan and not selection_group:
        return

    center_location = carla.Location(
        float(center_payload.get('x', 0.0)),
        float(center_payload.get('y', 0.0)),
        float(center_payload.get('z', 0.0)),
    )
    draw_height = center_location.z + 0.3
    marker_location = carla.Location(center_location.x, center_location.y, draw_height)

    lifetime = processor.waypoint_debug_lifetime
    try:
        mouse_pos = pygame.mouse.get_pos()
    except pygame.error:
        mouse_pos = None
    hover_selection = {'kind': 'traffic_light', 'group': selection_group, 'key': resolved_key} if (selection_group or resolved_key) else None
    is_selected = bool(
        selection_group
        and processor.selected_personal_trigger
        and processor.selected_personal_trigger.get('kind') == 'traffic_light'
        and processor.selected_personal_trigger.get('group') is selection_group
    )
    is_hovered = bool(selection_group and processor._is_personal_trigger_hovered(hover_selection, mouse_pos))
    if is_selected:
        marker_color = carla.Color(180, 180, 0)
        circle_color = carla.Color(180, 140, 0)
    elif is_hovered:
        marker_color = carla.Color(240, 240, 240)
        circle_color = carla.Color(210, 210, 210)
    else:
        marker_color = carla.Color(0, 120, 220)
        circle_color = carla.Color(0, 100, 200)
    radius_float = float(radius_value)

    if resolved_key:
        processor._last_visible_traffic_light_trigger_key = resolved_key

    try:
        processor.world.debug.draw_point(
            marker_location,
            size=0.5,
            color=marker_color,
            life_time=lifetime,
        )
        processor.world.debug.draw_string(
            carla.Location(center_location.x, center_location.y, draw_height + 1.2),
            f"TL R: {radius_float:.1f}m",
            draw_shadow=True,
            color=carla.Color(180, 180, 180),
            life_time=lifetime,
            persistent_lines=False,
        )

        num_points = 32
        for i in range(num_points):
            angle1 = 2.0 * math.pi * i / num_points
            angle2 = 2.0 * math.pi * (i + 1) / num_points
            point1 = carla.Location(
                center_location.x + radius_float * math.cos(angle1),
                center_location.y + radius_float * math.sin(angle1),
                draw_height,
            )
            point2 = carla.Location(
                center_location.x + radius_float * math.cos(angle2),
                center_location.y + radius_float * math.sin(angle2),
                draw_height,
            )
            processor.world.debug.draw_line(
                point1,
                point2,
                thickness=0.1,
                color=circle_color,
                life_time=lifetime,
                persistent_lines=False,
            )
    except Exception as exc:
        print(f"[Traffic Light Trigger] Debug draw failed: {exc}")

def render_pedestrian_triggers_overlay(processor):
    """Draw world-space debug circles for pedestrian triggers."""
    if not processor.world:
        return

    now = time.monotonic()
    if now - getattr(processor, "_ped_trigger_debug_last_t", 0.0) < processor.waypoint_debug_refresh_interval:
        return
    processor._ped_trigger_debug_last_t = now

    pedestrian_id = None
    if (processor.selected_vehicle and processor.selected_vehicle_is_pedestrian
            and processor.selected_vehicle.is_alive
            and processor.selected_vehicle.id in processor.pedestrian_trigger_radii):
        pedestrian_id = processor.selected_vehicle.id
    elif (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'pedestrian'):
        candidate_id = processor.selected_personal_trigger.get('id')
        if candidate_id in processor.pedestrian_trigger_radii:
            pedestrian_id = candidate_id

    if pedestrian_id is None:
        return

    center_payload = processor.pedestrian_trigger_centers.get(pedestrian_id)
    radius_value = processor.pedestrian_trigger_radii.get(pedestrian_id)

    if not center_payload or radius_value is None:
        return

    center_location = carla.Location(
        float(center_payload.get('x', 0.0)),
        float(center_payload.get('y', 0.0)),
        float(center_payload.get('z', 0.0)),
    )

    draw_height = center_location.z + 0.3
    marker_location = carla.Location(center_location.x, center_location.y, draw_height)

    lifetime = processor.waypoint_debug_lifetime
    is_selected = (
        processor.selected_personal_trigger
        and processor.selected_personal_trigger.get('kind') == 'pedestrian'
        and processor.selected_personal_trigger.get('id') == pedestrian_id
    )
    mouse_pos = None
    try:
        mouse_pos = pygame.mouse.get_pos()
    except pygame.error:
        mouse_pos = None
    hover_selection = {'kind': 'pedestrian', 'id': pedestrian_id}
    is_hovered = bool(processor._is_personal_trigger_hovered(hover_selection, mouse_pos))
    if is_selected:
        marker_color = carla.Color(180, 180, 0)
        circle_color = carla.Color(90, 70, 0)      # Dimmer so marker stands out
    elif is_hovered:
        marker_color = carla.Color(240, 240, 240)
        circle_color = carla.Color(105, 105, 105)  # Dimmer so marker stands out
    else:
        marker_color = carla.Color(0, 120, 220)
        circle_color = carla.Color(0, 50, 100)     # Dimmer so marker stands out
    radius_float = float(radius_value)
    # Scale line thickness based on camera height
    line_scale = max(0.5, min(1.5, processor.camera_controller.height / 200.0))

    try:
        # Draw center marker
        processor.world.debug.draw_point(
            marker_location,
            size=0.5,
            color=marker_color,
            life_time=lifetime,
        )
        # Draw radius label
        processor.world.debug.draw_string(
            carla.Location(center_location.x, center_location.y, draw_height + 1.2),
            f"Ped R: {radius_float:.1f}m",
            draw_shadow=True,
            color=carla.Color(180, 180, 180),
            life_time=lifetime,
            persistent_lines=False,
        )

        # Draw circle with 32 line segments
        num_points = 32
        for i in range(num_points):
            angle1 = 2.0 * math.pi * i / num_points
            angle2 = 2.0 * math.pi * (i + 1) / num_points
            point1 = carla.Location(
                center_location.x + radius_float * math.cos(angle1),
                center_location.y + radius_float * math.sin(angle1),
                draw_height,
            )
            point2 = carla.Location(
                center_location.x + radius_float * math.cos(angle2),
                center_location.y + radius_float * math.sin(angle2),
                draw_height,
            )
            processor.world.debug.draw_line(
                point1,
                point2,
                thickness=0.1 * line_scale,
                color=circle_color,
                life_time=lifetime,
                persistent_lines=False,
            )
    except Exception as exc:
        print(f"[Pedestrian Trigger] Debug draw failed: {exc}")

def render_vehicle_triggers_overlay(processor):
    """Draw world-space debug circles for NPC vehicle triggers."""
    if not processor.world:
        return

    now = time.monotonic()
    if now - getattr(processor, "_veh_trigger_debug_last_t", 0.0) < processor.waypoint_debug_refresh_interval:
        return
    processor._veh_trigger_debug_last_t = now

    vehicle_id = None
    if (processor.selected_vehicle and not processor.selected_vehicle_is_pedestrian
            and processor.selected_vehicle.is_alive and not processor.is_ego_vehicle(processor.selected_vehicle.id)
            and processor.selected_vehicle.id in processor.vehicle_trigger_radii):
        vehicle_id = processor.selected_vehicle.id
    elif (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'vehicle'):
        candidate_id = processor.selected_personal_trigger.get('id')
        if (candidate_id is not None and not processor.is_ego_vehicle(candidate_id)
                and candidate_id in processor.vehicle_trigger_radii):
            vehicle_id = candidate_id

    if vehicle_id is None:
        return

    center_payload = processor.vehicle_trigger_centers.get(vehicle_id)
    radius_value = processor.vehicle_trigger_radii.get(vehicle_id)
    if not center_payload or radius_value is None:
        return

    center_location = carla.Location(
        float(center_payload.get('x', 0.0)),
        float(center_payload.get('y', 0.0)),
        float(center_payload.get('z', 0.0)),
    )

    draw_height = center_location.z + 0.3
    marker_location = carla.Location(center_location.x, center_location.y, draw_height)

    lifetime = processor.waypoint_debug_lifetime
    is_selected = (
        processor.selected_personal_trigger
        and processor.selected_personal_trigger.get('kind') == 'vehicle'
        and processor.selected_personal_trigger.get('id') == vehicle_id
    )
    mouse_pos = None
    try:
        mouse_pos = pygame.mouse.get_pos()
    except pygame.error:
        mouse_pos = None
    hover_selection = {'kind': 'vehicle', 'id': vehicle_id}
    is_hovered = bool(processor._is_personal_trigger_hovered(hover_selection, mouse_pos))
    if is_selected:
        marker_color = carla.Color(180, 180, 0)
        circle_color = carla.Color(180, 140, 0)
    elif is_hovered:
        marker_color = carla.Color(240, 240, 240)
        circle_color = carla.Color(210, 210, 210)
    else:
        marker_color = carla.Color(0, 120, 220)
        circle_color = carla.Color(0, 100, 200)
    radius_float = float(radius_value)

    try:
        processor.world.debug.draw_point(
            marker_location,
            size=0.5,
            color=marker_color,
            life_time=lifetime,
        )
        processor.world.debug.draw_string(
            carla.Location(center_location.x, center_location.y, draw_height + 1.2),
            f"Veh R: {radius_float:.1f}m",
            draw_shadow=True,
            color=carla.Color(180, 180, 180),
            life_time=lifetime,
            persistent_lines=False,
        )

        num_points = 32
        for i in range(num_points):
            angle1 = 2.0 * math.pi * i / num_points
            angle2 = 2.0 * math.pi * (i + 1) / num_points
            point1 = carla.Location(
                center_location.x + radius_float * math.cos(angle1),
                center_location.y + radius_float * math.sin(angle1),
                draw_height,
            )
            point2 = carla.Location(
                center_location.x + radius_float * math.cos(angle2),
                center_location.y + radius_float * math.sin(angle2),
                draw_height,
            )
            processor.world.debug.draw_line(
                point1,
                point2,
                thickness=0.1,
                color=circle_color,
                life_time=lifetime,
                persistent_lines=False,
            )
    except Exception as exc:
        print(f"[Vehicle Trigger] Debug draw failed: {exc}")
