"""Scene rendering (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-31).

Pygame drawing of action menus, selection highlights, traffic-light markers/
connectors, and the per-frame render_all_overlays entry point, plus the
pure screen-space geometry helpers (three of them were @staticmethods and are
plain module functions here, with staticmethod aliases on the processor —
the traffic-light engine and click handlers call them via processor.*).

Order contracts preserved verbatim: render_traffic_light_markers writes
group.screen_polygon/screen_center consumed by the click handlers; several
render methods mutate state (render_vehicle_action_menu clears a dead
selection; render_traffic_light_action_menu re-anchors the menu);
_traffic_light_connector_debug_signature is created by first write, never in
__init__ — setattr-on-processor semantics kept.
"""

import math
from typing import Dict, List, Optional, Tuple

import carla
import pygame

from vse_editor.constants import GROUP_MENU_ICONS
from vse_editor.rendering.overlays import OverlayMenuRenderer, WaypointOverlayRenderer
from vse_editor.scene_types import (
    TRAFFIC_LIGHT_STOP_LINE_MARKER_DEPTH,
    TrafficLightGroupData,
)


def render_trigger_action_menu(processor, screen):
    """Render the action menu for selected trigger"""
    if processor.session.scenario_running:
        return

    if processor.selected_personal_trigger:
        return

    if processor.trigger_menu_hidden_for_camera_pan:
        return

    # Hide while grab-dragging; the menu (re)appears on release.
    if processor.moving_trigger or processor.trigger_drag_armed:
        return

    if processor.selected_trigger_index is None or not processor.trigger_action_menu_position:
        return

    tooltip_mgr = getattr(processor.editor, 'tooltip_manager', None) if processor.editor else None
    OverlayMenuRenderer.draw_menu(
        screen,
        processor.trigger_action_menu_position,
        # Shared with the hit-test side (3D orbit view drops the 'scale' icon)
        processor.trigger_menu_icons(),
        (processor.screen_width, processor.screen_height),
        tooltip_manager=tooltip_mgr,
    )

def render_personal_trigger_action_menu(processor, screen):
    """Render contextual menu for selected personal trigger."""
    if processor.session.scenario_running or not processor.selected_personal_trigger:
        return
    if processor.personal_trigger_menu_hidden_for_camera_pan:
        return
    # Hide while grab-dragging; the menu (re)appears on release.
    if processor.moving_personal_trigger or processor.personal_trigger_drag_armed:
        return
    if not processor.personal_trigger_menu_position:
        return
    if (processor.selected_personal_trigger.get('kind') == 'traffic_light'
            and not processor.traffic_lights_visible):
        return

    tooltip_mgr = getattr(processor.editor, 'tooltip_manager', None) if processor.editor else None
    OverlayMenuRenderer.draw_menu(
        screen,
        processor.personal_trigger_menu_position,
        # Shared with the hit-test side (3D orbit view drops the 'scale' icon)
        processor.personal_trigger_menu_icons(),
        (processor.screen_width, processor.screen_height),
        tooltip_manager=tooltip_mgr,
    )

def render_personal_trigger_links(processor, screen):
    """Draw persistent link lines between actors/groups and their personal triggers."""
    if not screen:
        return

    link_entries: Dict[Tuple[str, object], Dict[str, object]] = {}

    def add_actor_link(kind: str, actor_id: Optional[int], actor_obj: Optional[carla.Actor]):
        if actor_id is None:
            return
        key = (kind, actor_id)
        if key in link_entries:
            return
        if kind == 'pedestrian':
            center = processor.pedestrian_trigger_centers.get(actor_id)
        else:
            center = processor.vehicle_trigger_centers.get(actor_id)
        if not center:
            return
        if not actor_obj or not actor_obj.is_alive:
            actor_obj = processor.get_spawned_vehicle(actor_id)
        if not actor_obj or not actor_obj.is_alive:
            return
        link_entries[key] = {
            'anchor_type': 'actor',
            'actor': actor_obj,
            'center': center,
        }

    def add_group_link(group: Optional[TrafficLightGroupData], key: Optional[Tuple[str, Tuple]] = None):
        resolved_key = key or processor._traffic_light_trigger_key(group=group)
        if not resolved_key:
            return
        center, _, final_key = processor._get_traffic_light_trigger_data(key=resolved_key, group=group)
        if not center:
            return
        group_ref = group or processor._find_traffic_light_group_by_key(final_key)
        if not group_ref:
            return
        entry_key = ('traffic_light', final_key)
        if entry_key in link_entries:
            return
        link_entries[entry_key] = {
            'anchor_type': 'traffic_light',
            'group': group_ref,
            'key': final_key,
            'center': center,
        }

    # Actor selections
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        if processor.selected_vehicle_is_pedestrian:
            actor_id = processor.selected_vehicle.id
            if actor_id in processor.pedestrian_trigger_centers:
                add_actor_link('pedestrian', actor_id, processor.selected_vehicle)
        else:
            actor_id = processor.selected_vehicle.id
            if (not processor.is_ego_vehicle(actor_id)
                    and actor_id in processor.vehicle_trigger_centers):
                add_actor_link('vehicle', actor_id, processor.selected_vehicle)

    # Personal trigger selection
    selection = processor.selected_personal_trigger
    if selection:
        kind = selection.get('kind')
        if kind in ('pedestrian', 'vehicle'):
            actor_id = selection.get('id')
            add_actor_link(kind, actor_id, processor.get_spawned_vehicle(actor_id))
        elif kind == 'traffic_light':
            add_group_link(selection.get('group'), selection.get('key'))

    # Traffic light group selection
    if processor.selected_traffic_light_group:
        add_group_link(processor.selected_traffic_light_group)

    color = (0, 140, 255)
    for entry in link_entries.values():
        if processor.actor_overlays_hidden_for_camera_pan and entry.get('anchor_type') == 'actor':
            continue
        center = entry.get('center')
        if not center:
            continue
        if (processor.traffic_light_overlays_hidden_for_camera_pan
                and entry.get('anchor_type') == 'traffic_light'):
            continue
        center_screen = processor.coordinate_detector.world_to_screen_coordinates(
            center['x'], center['y'], center['z']
        )
        if not center_screen.get('success'):
            continue
        target_pos = (int(center_screen['x']), int(center_screen['y']))

        anchor_pos = None
        if entry.get('anchor_type') == 'actor':
            actor = entry.get('actor')
            if actor and actor.is_alive:
                location = actor.get_location()
                anchor_result = processor.coordinate_detector.world_to_screen_coordinates(
                    location.x, location.y, location.z + 2.0
                )
                if anchor_result.get('success'):
                    anchor_pos = (int(anchor_result['x']), int(anchor_result['y']))
        elif entry.get('anchor_type') == 'traffic_light':
            group = entry.get('group')
            key = entry.get('key')
            if not group and key:
                group = processor._find_traffic_light_group_by_key(key)
            anchor = processor._get_traffic_light_group_menu_anchor(group) if group else None
            if not anchor and group and group.center_location:
                loc = group.center_location
                anchor_result = processor.coordinate_detector.world_to_screen_coordinates(
                    loc.x, loc.y, loc.z
                )
                if anchor_result.get('success'):
                    anchor = (anchor_result['x'], anchor_result['y'])
            if anchor:
                anchor_pos = (int(anchor[0]), int(anchor[1]))

        if anchor_pos:
            pygame.draw.line(screen, color, anchor_pos, target_pos, 2)

def render_selected_actor_highlight(processor, screen):
    """Render a green highlight circle around the selected actor.

    Pedestrians keep the original fixed-size ring at head height; vehicles (and
    the ego) get a ring sized from the projected bounding box, so it encloses
    the body and scales with both vehicle size and zoom.
    """
    if (not processor.selected_vehicle or
            not processor.selected_vehicle.is_alive or
            processor.session.scenario_running or
            processor.actor_overlays_hidden_for_camera_pan):
        return

    highlight_color = (100, 255, 100)
    thickness = 3

    try:
        if processor.selected_vehicle_is_pedestrian:
            location = processor._get_selected_vehicle_location_cached()
            if location is None:
                return
            screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
                location.x,
                location.y,
                location.z + 1.0
            )

            if screen_pos.get('success'):
                center = (int(screen_pos['x']), int(screen_pos['y']))
                radius = 25
                pygame.draw.circle(screen, highlight_color, center, radius, thickness)
        else:
            rect = processor.coordinate_detector._projected_actor_rect(
                processor.selected_vehicle)
            if rect is None:
                return
            center = (int((rect[0] + rect[2]) / 2), int((rect[1] + rect[3]) / 2))
            # Half the rect diagonal encloses the whole body; floor at the
            # pedestrian ring size so the marker stays visible zoomed out.
            radius = max(25, int(
                0.5 * math.hypot(rect[2] - rect[0], rect[3] - rect[1])) + 4)
            pygame.draw.circle(screen, highlight_color, center, radius, thickness)
    except Exception as e:
        print(f"Error rendering selection highlight: {e}")

def _get_traffic_light_rectangle_points(
    processor, traffic_light: carla.TrafficLight
) -> Optional[Tuple[List[carla.Location], Optional[carla.Location]]]:
    """Return world-space corners and center for the traffic light stop-line strip.

    The drawn/clickable rectangle is a strip of TRAFFIC_LIGHT_STOP_LINE_MARKER_DEPTH meters
    anchored at the trigger box's front edge (the side toward the actual stop line, i.e. the
    downstream edge along the road travel direction), keeping the full lateral width. The
    returned center stays at the full trigger-box center (it feeds default trigger placement,
    connectors and list focus). If the travel direction can't be determined from a road
    waypoint, the full box is returned (previous behavior).
    """
    if not traffic_light:
        return None
    cached = processor._traffic_light_rectangles.get(traffic_light.id)
    if cached:
        return cached

    trigger_volume = getattr(traffic_light, "trigger_volume", None)
    if trigger_volume is None:
        return None

    try:
        transform = traffic_light.get_transform()
    except Exception:
        return None

    extent = trigger_volume.extent
    if extent is None:
        return None

    trigger_transform = carla.Transform(trigger_volume.location, trigger_volume.rotation)

    x_lo, x_hi = -float(extent.x), +float(extent.x)
    y_lo, y_hi = -float(extent.y), +float(extent.y)
    try:
        yaw = float(transform.rotation.yaw) + float(trigger_volume.rotation.yaw)
        center = transform.transform(trigger_transform.transform(carla.Location()))
        travel_yaw = processor._get_travel_yaw_at(center)
    except Exception:
        travel_yaw = None
    if travel_yaw is not None:
        if processor._heading_diff_180(yaw, travel_yaw) <= processor._heading_diff_180(yaw + 90.0, travel_yaw):
            # Travel runs along local X; front edge is the X side pointing downstream
            depth = min(TRAFFIC_LIGHT_STOP_LINE_MARKER_DEPTH, x_hi - x_lo)
            if processor._heading_diff_360(yaw, travel_yaw) <= 90.0:
                x_lo = x_hi - depth
            else:
                x_hi = x_lo + depth
        else:
            # Travel runs along local Y (heading yaw + 90)
            depth = min(TRAFFIC_LIGHT_STOP_LINE_MARKER_DEPTH, y_hi - y_lo)
            if processor._heading_diff_360(yaw + 90.0, travel_yaw) <= 90.0:
                y_lo = y_hi - depth
            else:
                y_hi = y_lo + depth

    local_corners = [
        carla.Location(x_hi, y_hi, 0.0),
        carla.Location(x_lo, y_hi, 0.0),
        carla.Location(x_lo, y_lo, 0.0),
        carla.Location(x_hi, y_lo, 0.0),
    ]

    world_corners: List[carla.Location] = []
    for corner in local_corners:
        try:
            trigger_point = trigger_transform.transform(corner)
            world_point = transform.transform(trigger_point)
            world_corners.append(world_point)
        except Exception:
            return None

    try:
        trigger_center = transform.transform(trigger_transform.transform(carla.Location()))
    except Exception:
        trigger_center = None

    result = (world_corners, trigger_center)
    processor._traffic_light_rectangles[traffic_light.id] = result
    return result

def _is_point_inside_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Return True if the 2D point lies inside the given polygon (ray casting)."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        denominator = (py2 - py1)
        if ((py1 > y) != (py2 > y)) and (
            x < (px2 - px1) * (y - py1) / (denominator if abs(denominator) > 1e-8 else (1e-8)) + px1
        ):
            inside = not inside
        px1, py1 = px2, py2
    return inside

def _get_traffic_light_group_screen_polygon(
    processor, group: TrafficLightGroupData
) -> Optional[Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]]:
    """Return a convex hull polygon covering all lights in the group.

    Cached per group until the camera changes (opt-03): the lights are static
    props and the projection is fully determined by the camera-controller
    state + screen size (world_to_screen builds its transform from the
    controller, fov is fixed), so recomputing every frame only burned CPU.
    World-space corners were already cached per light (_traffic_light_
    rectangles); this adds the screen-space layer on top.
    """
    cam = processor.camera_controller
    detector = processor.coordinate_detector
    # step-12: view_state() includes the 3D-orbit rotation fields; a center/height
    # signature went stale when the orbit view rotated in place.
    sig = cam.view_state() + (detector.screen_width, detector.screen_height)
    if getattr(processor, "_tl_polygon_cache_sig", None) != sig:
        processor._tl_polygon_cache_sig = sig
        processor._tl_polygon_cache = {}
    cache = processor._tl_polygon_cache
    key = tuple(sorted(group.ids))
    if key in cache:
        return cache[key]
    result = _compute_traffic_light_group_screen_polygon(processor, group)
    cache[key] = result
    return result

def _compute_traffic_light_group_screen_polygon(
    processor, group: TrafficLightGroupData
) -> Optional[Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]]:
    all_points: List[Tuple[float, float]] = []

    for light in group.lights:
        rectangle_data = processor._get_traffic_light_rectangle_points(light)
        if not rectangle_data:
            continue
        corners, _ = rectangle_data
        for corner in corners:
            screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
                corner.x, corner.y, corner.z
            )
            if screen_pos.get('success'):
                all_points.append((screen_pos['x'], screen_pos['y']))

    if not all_points:
        return None

    hull = processor._compute_convex_hull(all_points)
    if not hull:
        return None

    center = (
        sum(pt[0] for pt in hull) / len(hull),
        sum(pt[1] for pt in hull) / len(hull),
    )
    return hull, center

def _compute_convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute convex hull (Monotonic chain) for the provided 2D points."""
    if not points:
        return []

    unique_points = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique_points) <= 2:
        return unique_points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return hull

def _is_point_near_segment(
    point: Tuple[float, float],
    segment_start: Tuple[float, float],
    segment_end: Tuple[float, float],
    tolerance: float = 6.0,
) -> bool:
    """Return True if the point is within tolerance of the line segment."""
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end

    dx = x2 - x1
    dy = y2 - y1
    segment_length_sq = dx * dx + dy * dy
    if segment_length_sq == 0:
        distance_sq = (px - x1) ** 2 + (py - y1) ** 2
        return distance_sq <= tolerance * tolerance

    t = ((px - x1) * dx + (py - y1) * dy) / segment_length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    distance_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
    return distance_sq <= tolerance * tolerance

def render_selected_traffic_light_connectors(processor, screen):
    """Draw connectors from the selected stopline group to its traffic lights."""
    group = processor.selected_traffic_light_group
    if not group:
        processor._traffic_light_connector_debug_signature = None
        return

    if processor.traffic_light_overlays_hidden_for_camera_pan:
        processor._traffic_light_connector_debug_signature = None
        return

    connector_color = (60, 150, 255)
    drawn_lines: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
    failure_messages: List[str] = []

    for light in group.lights:
        light_id = getattr(light, "id", None)
        try:
            rectangle_data = processor._get_traffic_light_rectangle_points(light)
        except Exception as exc:
            failure_messages.append(f"Light {light_id}: rectangle error {exc}")
            continue

        if not rectangle_data:
            failure_messages.append(f"Light {light_id}: no trigger rectangle data")
            continue

        corners, trigger_center = rectangle_data
        if trigger_center is not None:
            stop_pos = trigger_center
        elif corners:
            stop_pos = carla.Location(
                sum(corner.x for corner in corners) / len(corners),
                sum(corner.y for corner in corners) / len(corners),
                sum(corner.z for corner in corners) / len(corners),
            )
        else:
            failure_messages.append(f"Light {light_id}: trigger rectangle empty")
            continue

        stop_screen = processor.coordinate_detector.world_to_screen_coordinates(
            stop_pos.x, stop_pos.y, stop_pos.z
        )
        if not stop_screen.get("success"):
            failure_messages.append(f"Light {light_id}: stopline projection failed")
            continue

        try:
            light_location = light.get_transform().location
        except Exception as exc:
            failure_messages.append(f"Light {light_id}: transform error {exc}")
            continue

        light_screen = processor.coordinate_detector.world_to_screen_coordinates(
            light_location.x,
            light_location.y,
            light_location.z,
        )
        if not light_screen.get("success"):
            failure_messages.append(f"Light {light_id}: mesh projection failed")
            continue

        start = (int(stop_screen["x"]), int(stop_screen["y"]))
        end = (int(light_screen["x"]), int(light_screen["y"]))

        pygame.draw.line(screen, connector_color, start, end, 2)
        pygame.draw.circle(screen, connector_color, end, 4)
        drawn_lines.append((light_id, start, end))

    drawn_ids = tuple(sorted(light_id for light_id, _, _ in drawn_lines if light_id is not None))
    failures_sorted = tuple(sorted(failure_messages))
    debug_signature = (drawn_ids, failures_sorted)

    if debug_signature != processor._traffic_light_connector_debug_signature:
        if drawn_lines:
            print(
                f"[TrafficLines] Drawing {len(drawn_lines)} connector(s) for group {sorted(group.ids)}"
            )
            for light_id, start, end in drawn_lines:
                print(
                    f"  Light {light_id}: stopline screen {start} -> mesh screen {end}"
                )
        else:
            print(
                f"[TrafficLines] No connectors drawn for group {sorted(group.ids)}"
            )
        for message in failure_messages:
            print(f"  {message}")
        processor._traffic_light_connector_debug_signature = debug_signature

def render_traffic_light_markers(processor, screen):
    """Draw small markers above each traffic light to make them discoverable."""
    if processor.traffic_light_overlays_hidden_for_camera_pan:
        return

    if not processor.traffic_light_groups:
        if processor._traffic_light_debug_render:
            signature = (0, processor.traffic_lights_visible)
            if signature != processor._traffic_light_render_debug_signature:
                processor._traffic_light_render_debug_signature = signature
                processor._traffic_light_debug("Render markers skipped: no traffic light groups cached.")
        return

    marker_fill = (220, 50, 50)  # Red color for stop lines
    marker_outline = (20, 20, 20)
    drawn_groups = 0
    visibility_state = processor.traffic_lights_visible

    for group in processor.traffic_light_groups:
        polygon_data = processor._get_traffic_light_group_screen_polygon(group)
        if not polygon_data:
            group.screen_polygon = None
            group.screen_center = None
            continue

        screen_points, screen_center = polygon_data
        group.screen_polygon = screen_points
        group.screen_center = screen_center

        int_points = [(int(x), int(y)) for x, y in screen_points]

        try:
            if len(int_points) >= 3:
                pygame.draw.polygon(screen, marker_fill, int_points)
                pygame.draw.polygon(screen, marker_outline, int_points, 2)
            elif len(int_points) == 2:
                pygame.draw.line(screen, marker_outline, int_points[0], int_points[1], 4)
            elif len(int_points) == 1:
                pygame.draw.circle(screen, marker_fill, int_points[0], 6)
                pygame.draw.circle(screen, marker_outline, int_points[0], 6, 2)
        except Exception:
            continue

        if screen_center:
            center_int = (int(screen_center[0]), int(screen_center[1]))
            pygame.draw.circle(screen, marker_outline, center_int, 3)

            if len(group.lights) > 1 and processor._traffic_light_font:
                label = str(len(group.lights))
                try:
                    text_surface = processor._traffic_light_font.render(label, True, (10, 10, 10))
                    text_rect = text_surface.get_rect(center=center_int)
                    screen.blit(text_surface, text_rect)
                except Exception:
                    pass
        drawn_groups += 1

    if processor._traffic_light_debug_render:
        signature = (drawn_groups, visibility_state)
        if signature != processor._traffic_light_render_debug_signature:
            processor._traffic_light_render_debug_signature = signature
            if visibility_state:
                processor._traffic_light_debug(
                    f"Render markers: drew {drawn_groups} group marker(s) with overlay visible."
                )
            else:
                processor._traffic_light_debug(
                    f"Render markers suppressed: overlay hidden (groups available={drawn_groups})."
                )

def render_selected_traffic_light_highlight(processor, screen):
    """Render a blue highlight around the currently selected traffic light marker."""
    if processor.traffic_light_overlays_hidden_for_camera_pan:
        return

    group = processor.selected_traffic_light_group
    if not group:
        return

    polygon_data = None
    if group.screen_polygon:
        polygon_data = (group.screen_polygon, group.screen_center)
    else:
        polygon_data = processor._get_traffic_light_group_screen_polygon(group)
    if not polygon_data:
        return

    screen_points, screen_center = polygon_data
    if not group.screen_polygon:
        group.screen_polygon = screen_points
        group.screen_center = screen_center
    if not screen_points:
        return

    int_points = [(int(x), int(y)) for x, y in screen_points]
    highlight_color = (60, 150, 255)
    try:
        if len(int_points) >= 3:
            pygame.draw.polygon(screen, highlight_color, int_points, 3)
        elif len(int_points) == 2:
            pygame.draw.line(screen, highlight_color, int_points[0], int_points[1], 3)
        elif len(int_points) == 1:
            pygame.draw.circle(screen, highlight_color, int_points[0], 8, 3)
    except Exception:
        return

def render_traffic_light_action_menu(processor, screen):
    """Render the floating action menu for the selected traffic light group."""
    if processor.session.scenario_running or not processor.traffic_lights_visible:
        return

    if processor.selected_personal_trigger:
        return

    if processor.traffic_light_overlays_hidden_for_camera_pan:
        return

    if processor.traffic_light_menu_hidden_for_camera_pan:
        return

    group = processor.selected_traffic_light_group
    if not group:
        processor.traffic_light_menu_position = None
        return

    icons = processor._get_traffic_light_menu_icons(group)
    if not icons:
        processor.traffic_light_menu_position = None
        return

    processor._update_traffic_light_menu_anchor(group)
    if not processor.traffic_light_menu_position:
        return

    tooltip_mgr = getattr(processor.editor, 'tooltip_manager', None) if processor.editor else None
    OverlayMenuRenderer.draw_menu(
        screen,
        processor.traffic_light_menu_position,
        icons,
        (processor.screen_width, processor.screen_height),
        tooltip_manager=tooltip_mgr,
    )

def render_vehicle_action_menu(processor, screen):
    """Render the action menu for selected vehicle"""
    # Don't render menu if we're in waypoint creation/destination mode or scenario playback
    if (processor.creating_waypoint or processor.creating_destination or
            processor.session.scenario_running):
        return

    if processor.selected_personal_trigger:
        return

    if processor.vehicle_menu_hidden_for_camera_pan:
        return

    # Hide while grab-dragging; the menu (re)appears on release.
    if processor.moving_vehicle or processor.vehicle_drag_armed:
        return

    if not processor.selected_vehicle or not processor.vehicle_menu_position:
        return

    # Check if vehicle is still alive
    if not processor.selected_vehicle.is_alive:
        processor.clear_vehicle_selection()
        return

    # Shared with the hit-test side (placement.vehicle_menu_icon_order)
    icon_order = processor.get_vehicle_menu_icon_order()

    tooltip_mgr = getattr(processor.editor, 'tooltip_manager', None) if processor.editor else None
    OverlayMenuRenderer.draw_menu(
        screen,
        processor.vehicle_menu_position,
        icon_order,
        (processor.screen_width, processor.screen_height),
        tooltip_manager=tooltip_mgr,
    )

def render_waypoint_creation_overlay(processor, screen):
    """Delegate waypoint creation overlay rendering to helper."""
    WaypointOverlayRenderer.render_creation_overlay(processor, screen)

def render_destination_creation_overlay(processor, screen):
    """Delegate destination overlay rendering to helper."""
    WaypointOverlayRenderer.render_destination_overlay(processor, screen)

_GROUP_LABEL_FONT = None

def _group_label_font():
    global _GROUP_LABEL_FONT
    if _GROUP_LABEL_FONT is None:
        _GROUP_LABEL_FONT = pygame.font.Font(None, 22)
    return _GROUP_LABEL_FONT

def render_group_selection(processor, screen):
    """Highlight the group (marquee) selection and draw its delete menu."""
    if processor.session.scenario_running:
        return
    # Hidden while the camera pans, like the other actor overlays; the
    # suppress/restore pair in camera_stream drives this flag.
    if processor.actor_overlays_hidden_for_camera_pan:
        return

    ids = processor.selected_actor_ids
    waypoint_group = processor.selected_waypoint_group
    if not ids and not waypoint_group:
        return

    highlight_color = (255, 170, 60)
    count = 0
    if ids:
        for actor in processor.spawned_vehicles:
            if not (actor and actor.is_alive and actor.id in ids):
                continue
            rect = processor.coordinate_detector._projected_actor_rect(actor)
            if rect is None:
                continue
            count += 1
            pygame.draw.rect(
                screen,
                highlight_color,
                pygame.Rect(int(rect[0]) - 2, int(rect[1]) - 2,
                            int(rect[2] - rect[0]) + 4, int(rect[3] - rect[1]) + 4),
                2,
            )
        if not count:
            # Every selected actor is gone (e.g. undo/redo churn): drop the group.
            processor.selected_actor_ids = set()
            return
        label_text = f"{count} selected"
    else:
        for _, x, y in processor.group_waypoint_screen_positions():
            count += 1
            pygame.draw.circle(screen, highlight_color, (int(x), int(y)), 16, 2)
        if not count:
            processor.selected_waypoint_group = None
            return
        label_text = f"{count} waypoint(s) selected"

    anchor = processor.group_selection_menu_anchor()
    if anchor:
        tooltip_mgr = getattr(processor.editor, 'tooltip_manager', None) if processor.editor else None
        OverlayMenuRenderer.draw_menu(
            screen,
            anchor,
            list(GROUP_MENU_ICONS),
            (processor.screen_width, processor.screen_height),
            tooltip_manager=tooltip_mgr,
        )
        label = _group_label_font().render(label_text, True, (235, 235, 235))
        screen.blit(label, (anchor[0] + 40, anchor[1] - 44))

def render_marquee(processor, screen):
    """Draw the rubber-band rectangle while a marquee drag is active."""
    marquee_rect = getattr(processor.editor, 'marquee_rect', None) if processor.editor else None
    if not marquee_rect:
        return
    rect = pygame.Rect(int(marquee_rect[0]), int(marquee_rect[1]),
                       int(marquee_rect[2]), int(marquee_rect[3]))
    if rect.width > 0 and rect.height > 0:
        fill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        fill.fill((120, 180, 255, 40))
        screen.blit(fill, rect.topleft)
    pygame.draw.rect(screen, (120, 180, 255), rect, 1)

def render_action_menus(processor, screen):
    """Render action menus and selection indicators (vehicle, pedestrian, traffic light, triggers).
    Called early in render loop to appear behind UI panels."""
    # Traffic light markers and selection indicators
    if processor.traffic_lights_visible:
        processor.render_traffic_light_markers(screen)
        processor.render_selected_traffic_light_connectors(screen)
        processor.render_selected_traffic_light_highlight(screen)
    # Actor selection highlight
    processor.render_selected_actor_highlight(screen)
    # Group (marquee) selection
    processor.render_group_selection(screen)
    render_marquee(processor, screen)
    # Action menus
    processor.render_traffic_light_action_menu(screen)
    processor.render_vehicle_action_menu(screen)
    processor.render_trigger_action_menu(screen)
    processor.render_personal_trigger_action_menu(screen)

def render_all_overlays(processor, screen):
    """Render world-space overlays (waypoints, lanes, triggers, etc).
    Does NOT include action menus or selection indicators - those are rendered earlier via render_action_menus()."""
    # Chase and cockpit are perspective views (tilted / in-car); the editor's top-down
    # screen<->world projection would misplace these overlays, so skip them for a clean view.
    if getattr(processor, "playback_camera_mode", "topdown") in ("chase", "cockpit"):
        return
    processor.render_waypoint_creation_overlay(screen)
    processor.render_destination_creation_overlay(screen)
    processor.render_opendrive_lanes_overlay(screen)
    processor.render_waypoints_overlay(screen)
    processor.render_trigger_placement_overlay(screen)
    processor.render_triggers_overlay(screen)
    processor.render_personal_trigger_links(screen)
    processor.render_traffic_light_triggers_overlay()
    processor.render_pedestrian_triggers_overlay()
    processor.render_vehicle_triggers_overlay()
