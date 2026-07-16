"""Overlay renderers (moved verbatim from vse.py): waypoint paths, triggers,
floating action menus and OpenDRIVE lanes. Every renderer takes the
camera/scene *processor* as its first argument (duck-typed by design — do
not introduce an interface; the processor IS the contract).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import carla
import pygame

from vse_editor.constants import (
    DEFAULT_PERSONAL_TRIGGER_RADIUS,
    OVERLAY_ICON_TOOLTIPS,
    WAYPOINT_MARKER_Z_OFFSET,
)

class WaypointOverlayRenderer:
    """Render helper for waypoint overlays."""

    @staticmethod
    def _project_to_screen(cp, x, y, z, *, rounded=True):
        """Project a world point to screen coordinates via the shared detector."""
        result = cp.coordinate_detector.world_to_screen_coordinates(x, y, z)
        if not result.get('success'):
            return None
        if rounded:
            return int(result['x']), int(result['y'])
        return result['x'], result['y']

    @staticmethod
    def render_creation_overlay(processor, screen):
        """Render waypoint creation mode overlay with green circle and line."""
        cp = processor
        if not (cp.creating_waypoint and cp.waypoint_vehicle and cp.waypoint_vehicle.is_alive):
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        waypoints = cp.get_vehicle_waypoints(cp.waypoint_vehicle.id)

        if waypoints:
            last_waypoint = waypoints[-1]
            connection = WaypointOverlayRenderer._project_to_screen(
                cp, last_waypoint['x'], last_waypoint['y'],
                last_waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
            )
        else:
            vehicle_location = cp.waypoint_vehicle.get_location()
            connection = WaypointOverlayRenderer._project_to_screen(
                cp, vehicle_location.x, vehicle_location.y, vehicle_location.z + 1.0
            )

        if not connection:
            return

        connection_x, connection_y = connection
        waypoint_green = (255, 215, 0) if cp.waypoint_vehicle and cp.is_ego_vehicle(cp.waypoint_vehicle.id) else (0, 255, 0)

        pygame.draw.line(
            screen,
            waypoint_green,
            (connection_x, connection_y),
            (mouse_x, mouse_y),
            3,
        )
        pygame.draw.circle(screen, waypoint_green, (connection_x, connection_y), 8, 2)

    @staticmethod
    def render_destination_overlay(processor, screen):
        """Render destination creation mode overlay with yellow circle cursor."""
        cp = processor
        if not (cp.creating_destination and cp.selected_vehicle and cp.selected_vehicle.is_alive):
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        destination_yellow = (255, 215, 0)

        pygame.draw.circle(screen, destination_yellow, (mouse_x, mouse_y), 25, 3)
        pygame.draw.circle(screen, destination_yellow, (mouse_x, mouse_y), 5)

        vehicle_location = cp.selected_vehicle.get_location()
        vehicle_screen = WaypointOverlayRenderer._project_to_screen(
            cp, vehicle_location.x, vehicle_location.y, vehicle_location.z + 1.0
        )

        if not vehicle_screen:
            return

        vehicle_x, vehicle_y = vehicle_screen
        WaypointOverlayRenderer._draw_dashed_line(
            screen,
            destination_yellow,
            (vehicle_x, vehicle_y),
            (mouse_x, mouse_y),
            2,
            5,
        )
        pygame.draw.circle(screen, destination_yellow, (vehicle_x, vehicle_y), 8, 2)

    @staticmethod
    def render_waypoints_overlay(processor, screen):
        """
        Render hover feedback for waypoints while deferring primary drawing to CARLA debug helpers.
        """
        cp = processor
        if not cp.waypoint_display_vehicle_id:
            cp.waypoint_hover_index = None
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_y <= 80:
            cp.waypoint_hover_index = None
            return

        vehicle_id = cp.waypoint_display_vehicle_id
        waypoints = cp.get_vehicle_waypoints(vehicle_id)
        if not waypoints:
            return

        vehicle = next((v for v in cp.spawned_vehicles if v.id == vehicle_id and v.is_alive), None)
        if not vehicle:
            cp.waypoint_hover_index = None
            return

        cp.waypoint_hover_index = None

        for i, waypoint in enumerate(waypoints):
            screen_pos = WaypointOverlayRenderer._project_to_screen(
                cp, waypoint['x'], waypoint['y'],
                waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
            )
            if not screen_pos:
                continue
            marker_x, marker_y = screen_pos
            distance = ((marker_x - mouse_x) ** 2 + (marker_y - mouse_y) ** 2) ** 0.5
            if distance <= cp.waypoint_marker_radius + 5:
                cp.waypoint_hover_index = (vehicle_id, i)
                break

    @staticmethod
    def _draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=5):
        """Draw a dashed line between two points."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return

        dashes = int(distance / dash_length)
        for i in range(0, dashes, 2):
            start = i * dash_length / distance
            end = min((i + 1) * dash_length / distance, 1)
            start_x = int(x1 + (x2 - x1) * start)
            start_y = int(y1 + (y2 - y1) * start)
            end_x = int(x1 + (x2 - x1) * end)
            end_y = int(y1 + (y2 - y1) * end)
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)


class TriggerOverlayRenderer:
    """Render helper for trigger overlays."""

    @staticmethod
    def render_trigger_placement_overlay(processor, screen):
        """Render trigger placement cursor overlay."""
        cp = processor
        if not (cp.placing_trigger or getattr(cp, "pending_personal_trigger", None)):
            return

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            return

        circle_color = (0, 140, 255)
        circle_radius = max(12, int(getattr(cp, "personal_trigger_preview_radius", DEFAULT_PERSONAL_TRIGGER_RADIUS) * 6))
        pygame.draw.circle(screen, circle_color, mouse_pos, circle_radius, 2)

        pending_personal = getattr(cp, "pending_personal_trigger", None)
        if pending_personal:
            anchor = TriggerOverlayRenderer._get_personal_trigger_anchor(cp, pending_personal)
            if anchor:
                pygame.draw.line(screen, circle_color, anchor, mouse_pos, 2)

    @staticmethod
    def _get_personal_trigger_anchor(processor, pending):
        """Return the screen-space anchor for the pending personal trigger."""
        kind = pending.get('kind')
        if kind in ('vehicle', 'pedestrian'):
            actor_ref = pending.get('actor_ref')
            actor = actor_ref() if callable(actor_ref) else pending.get('actor')
            if actor and actor.is_alive:
                screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
                    actor.get_location().x,
                    actor.get_location().y,
                    actor.get_location().z + 2.0,
                )
                if screen_pos.get('success'):
                    return (int(screen_pos['x']), int(screen_pos['y']))
        elif kind == 'traffic_light':
            group = pending.get('group')
            if group:
                anchor = processor._get_traffic_light_group_menu_anchor(group)
                if anchor:
                    return (int(anchor[0]), int(anchor[1]))
        return None

    @staticmethod
    def render_triggers(processor, screen):
        """Render trigger zones using CARLA debug drawing."""
        cp = processor
        if not cp.triggers:
            return

        # Hide triggers during scenario playback
        if cp.editor and cp.editor.scenario_running:
            return

        for idx, trigger in enumerate(cp.triggers):
            is_selected = idx == cp.selected_trigger_index
            if is_selected:
                marker_color = carla.Color(180, 180, 0)
                circle_color = carla.Color(180, 140, 0)
            else:
                marker_color = carla.Color(0, 50, 150)
                circle_color = carla.Color(0, 50, 150)

            draw_height = trigger['z'] + 0.3
            center_location = carla.Location(trigger['x'], trigger['y'], draw_height)
            marker_size = 0.5 if is_selected else 0.4

            cp.world.debug.draw_point(
                center_location,
                size=marker_size,
                color=marker_color,
                life_time=cp.waypoint_debug_lifetime,
            )
            cp.world.debug.draw_string(
                carla.Location(trigger['x'], trigger['y'], draw_height + 2.2),
                "TRIGGER",
                draw_shadow=True,
                color=marker_color,
                life_time=cp.waypoint_debug_lifetime,
                persistent_lines=False,
            )
            cp.world.debug.draw_string(
                carla.Location(trigger['x'], trigger['y'], draw_height + 1.2),
                f"R: {trigger['radius']:.1f}m",
                draw_shadow=True,
                color=carla.Color(150, 150, 150),
                life_time=cp.waypoint_debug_lifetime,
                persistent_lines=False,
            )

            num_points = 32
            radius = trigger['radius']
            for i in range(num_points):
                angle1 = 2.0 * math.pi * i / num_points
                angle2 = 2.0 * math.pi * (i + 1) / num_points
                point1 = carla.Location(
                    trigger['x'] + radius * math.cos(angle1),
                    trigger['y'] + radius * math.sin(angle1),
                    draw_height,
                )
                point2 = carla.Location(
                    trigger['x'] + radius * math.cos(angle2),
                    trigger['y'] + radius * math.sin(angle2),
                    draw_height,
                )
                cp.world.debug.draw_line(
                    point1,
                    point2,
                    thickness=0.1,
                    color=circle_color,
                    life_time=cp.waypoint_debug_lifetime,
                    persistent_lines=False,
                )

            north = carla.Location(trigger['x'], trigger['y'] + radius, draw_height)
            south = carla.Location(trigger['x'], trigger['y'] - radius, draw_height)
            cp.world.debug.draw_line(
                north,
                south,
                thickness=0.05,
                color=circle_color,
                life_time=cp.waypoint_debug_lifetime,
                persistent_lines=False,
            )

            east = carla.Location(trigger['x'] + radius, trigger['y'], draw_height)
            west = carla.Location(trigger['x'] - radius, trigger['y'], draw_height)
            cp.world.debug.draw_line(
                east,
                west,
                thickness=0.05,
                color=circle_color,
                life_time=cp.waypoint_debug_lifetime,
                persistent_lines=False,
            )


class OverlayMenuRenderer:
    """Shared renderer for floating action menus anchored to world objects."""

    ICON_SIZE = 40
    ICON_SPACING = 50
    MENU_PADDING = 10
    SCREEN_MARGIN = 10

    @classmethod
    def _layout(cls, anchor, icons, screen_width, screen_height):
        if not anchor or not icons:
            return None, []
        anchor_x, anchor_y = anchor
        icon_count = len(icons)
        menu_width = icon_count * cls.ICON_SPACING + cls.MENU_PADDING * 2
        menu_height = cls.ICON_SIZE + cls.MENU_PADDING * 2

        menu_x = max(
            cls.SCREEN_MARGIN,
            min(int(anchor_x) - menu_width // 2, screen_width - menu_width - cls.SCREEN_MARGIN),
        )
        menu_y = max(
            cls.SCREEN_MARGIN,
            min(int(anchor_y) - menu_height - cls.SCREEN_MARGIN, screen_height - menu_height - cls.SCREEN_MARGIN),
        )

        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        icon_entries = []
        for idx, icon_name in enumerate(icons):
            center_x = menu_rect.x + cls.MENU_PADDING + idx * cls.ICON_SPACING + cls.ICON_SIZE // 2
            center_y = menu_rect.y + cls.MENU_PADDING + cls.ICON_SIZE // 2
            icon_rect = pygame.Rect(
                center_x - cls.ICON_SIZE // 2,
                center_y - cls.ICON_SIZE // 2,
                cls.ICON_SIZE,
                cls.ICON_SIZE,
            )
            icon_entries.append((icon_name, icon_rect, (center_x, center_y)))

        return menu_rect, icon_entries

    @classmethod
    def draw_menu(cls, screen, anchor, icons, screen_size, tooltip_manager=None):
        """Draw a floating action menu and return icon layout."""
        menu_rect, icon_entries = cls._layout(anchor, icons, screen_size[0], screen_size[1])
        if not icon_entries:
            return []

        menu_surface = pygame.Surface(menu_rect.size, pygame.SRCALPHA)
        menu_surface.fill((40, 40, 40, 220))
        screen.blit(menu_surface, menu_rect.topleft)
        pygame.draw.rect(screen, (100, 100, 100), menu_rect, 2)

        mouse_pos = pygame.mouse.get_pos()
        for icon_name, icon_rect, center in icon_entries:
            pygame.draw.rect(screen, (80, 80, 80), icon_rect)
            pygame.draw.rect(screen, (150, 150, 150), icon_rect, 2)
            cls._draw_icon(screen, icon_name, center)

            # Register hover for tooltip
            if tooltip_manager and icon_rect.collidepoint(mouse_pos):
                tooltip_text = OVERLAY_ICON_TOOLTIPS.get(icon_name, "")
                if tooltip_text:
                    tooltip_manager.register_hover(
                        f"overlay_icon_{icon_name}",
                        icon_rect,
                        tooltip_text,
                    )

        return icon_entries

    @classmethod
    def hit_test(cls, anchor, icons, screen_size, mouse_pos):
        """Return the icon name at the given mouse position, if any."""
        _, icon_entries = cls._layout(anchor, icons, screen_size[0], screen_size[1])
        for icon_name, icon_rect, _ in icon_entries:
            if icon_rect.collidepoint(mouse_pos):
                return icon_name
        return None

    @staticmethod
    def _draw_icon(screen, icon_name, center):
        center_x, center_y = center

        if icon_name == 'delete':
            x_size = 12
            pygame.draw.line(
                screen,
                (255, 100, 100),
                (center_x - x_size, center_y - x_size),
                (center_x + x_size, center_y + x_size),
                3,
            )
            pygame.draw.line(
                screen,
                (255, 100, 100),
                (center_x + x_size, center_y - x_size),
                (center_x - x_size, center_y + x_size),
                3,
            )
        elif icon_name == 'rotate':
            radius = 12
            pygame.draw.circle(screen, (200, 200, 255), (center_x, center_y), radius, 2)
            arrow_angle = math.radians(45)
            arrow_head = (
                center_x + radius * math.cos(arrow_angle),
                center_y - radius * math.sin(arrow_angle),
            )
            pygame.draw.polygon(
                screen,
                (200, 200, 255),
                [
                    arrow_head,
                    (arrow_head[0] - 6, arrow_head[1] - 2),
                    (arrow_head[0] - 2, arrow_head[1] - 6),
                ],
            )
            inner_radius = 8
            pygame.draw.arc(
                screen,
                (200, 200, 255),
                (
                    center_x - inner_radius,
                    center_y - inner_radius,
                    inner_radius * 2,
                    inner_radius * 2,
                ),
                math.radians(-20),
                math.radians(200),
                2,
            )
            arrow_end = (
                center_x + inner_radius * math.cos(math.radians(200)),
                center_y + inner_radius * math.sin(math.radians(200)),
            )
            pygame.draw.polygon(
                screen,
                (200, 200, 255),
                [
                    arrow_end,
                    (arrow_end[0] - 4, arrow_end[1] + 2),
                    (arrow_end[0], arrow_end[1] + 6),
                ],
            )
        elif icon_name == 'waypoint':
            radius = 12
            path_points = []
            for i in range(8):
                angle = 2 * math.pi * i / 8
                x = center_x + radius * 0.7 * math.cos(angle)
                y = center_y + radius * 0.5 * math.sin(angle)
                path_points.append((x, y))
            if len(path_points) > 1:
                pygame.draw.lines(screen, (100, 255, 100), False, path_points, 2)
            waypoint_positions = [
                (center_x - 8, center_y - 4),
                (center_x, center_y - 8),
                (center_x + 8, center_y - 4),
                (center_x + 6, center_y + 6),
                (center_x - 6, center_y + 6),
            ]
            for pos in waypoint_positions:
                pygame.draw.circle(screen, (100, 255, 100), pos, 2)
        elif icon_name in ('autoroute', 'ego_destination'):
            pin_center_x = center_x
            pin_center_y = center_y + 2
            pin_radius = 9
            pygame.draw.circle(screen, (255, 215, 0), (pin_center_x, pin_center_y - 4), pin_radius)
            pygame.draw.circle(screen, (255, 255, 255), (pin_center_x, pin_center_y - 4), 4)
            pygame.draw.polygon(
                screen,
                (255, 215, 0),
                [
                    (pin_center_x, pin_center_y + 8),
                    (pin_center_x - 7, pin_center_y),
                    (pin_center_x + 7, pin_center_y),
                ],
            )
        elif icon_name == 'scale':
            arrow_size = 10
            pygame.draw.polygon(
                screen,
                (255, 200, 100),
                [
                    (center_x, center_y - arrow_size),
                    (center_x - 5, center_y - arrow_size + 5),
                    (center_x + 5, center_y - arrow_size + 5),
                ],
            )
            pygame.draw.polygon(
                screen,
                (255, 200, 100),
                [
                    (center_x, center_y + arrow_size),
                    (center_x - 5, center_y + arrow_size - 5),
                    (center_x + 5, center_y + arrow_size - 5),
                ],
            )
            pygame.draw.line(
                screen,
                (255, 200, 100),
                (center_x, center_y - arrow_size + 5),
                (center_x, center_y + arrow_size - 5),
                2,
            )
        elif icon_name == 'remove_trigger':
            ring_color = (0, 140, 255)
            plus_color = (255, 255, 255)
            pygame.draw.circle(screen, ring_color, (center_x, center_y), 10, 2)
            pygame.draw.circle(screen, ring_color, (center_x, center_y), 3, 0)
            pygame.draw.line(screen, plus_color, (center_x - 6, center_y), (center_x + 6, center_y), 2)
            pygame.draw.line(screen, plus_color, (center_x, center_y - 6), (center_x, center_y + 6), 2)

            overlay_radius = 11
            pygame.draw.circle(screen, (200, 60, 60), (center_x, center_y), overlay_radius, 2)
            offset = 7
            pygame.draw.line(
                screen,
                (230, 70, 70),
                (center_x - offset, center_y - offset),
                (center_x + offset, center_y + offset),
                3,
            )
            pygame.draw.line(
                screen,
                (230, 70, 70),
                (center_x + offset, center_y - offset),
                (center_x - offset, center_y + offset),
                3,
            )
        elif icon_name == 'add_trigger':
            ring_color = (0, 140, 255)
            plus_color = (255, 255, 255)
            pygame.draw.circle(screen, ring_color, (center_x, center_y), 10, 2)
            pygame.draw.circle(screen, ring_color, (center_x, center_y), 3, 0)
            pygame.draw.line(screen, plus_color, (center_x - 6, center_y), (center_x + 6, center_y), 2)
            pygame.draw.line(screen, plus_color, (center_x, center_y - 6), (center_x, center_y + 6), 2)

class OpenDriveOverlayRenderer:
    """Render helper for OpenDRIVE lane overlays."""

    @staticmethod
    def invalidate_cache(processor, *, drop_surfaces=False, clear_projection=True):
        """Mark cached overlay surfaces for rebuild."""
        cp = processor
        if not cp:
            return
        if hasattr(cp, 'overlay_surface_dirty'):
            cp.overlay_surface_dirty = True
        if drop_surfaces:
            if hasattr(cp, 'overlay_surface'):
                cp.overlay_surface = None
            if hasattr(cp, '_lane_overlay_surface'):
                cp._lane_overlay_surface = None
        if clear_projection and hasattr(cp, 'projection_cache'):
            cp.projection_cache.clear()

    @staticmethod
    def disable_overlay(processor, *, silent=False):
        """Turn off the OpenDRIVE overlay and drop cached surfaces."""
        cp = processor
        if not cp:
            return
        if not getattr(cp, 'lane_overlay_enabled', False):
            return
        cp.lane_overlay_enabled = False
        OpenDriveOverlayRenderer.invalidate_cache(cp, drop_surfaces=True)
        if not silent:
            print("OpenDRIVE overlay disabled")

    @staticmethod
    def render(processor, screen):
        cp = processor
        if not getattr(cp, 'lane_overlay_enabled', False):
            return
        if getattr(cp, 'opendrive_overlay_hidden_for_camera_pan', False):
            return
        if not getattr(cp, 'opendrive_lane_data', None):
            return

        camera_moved = OpenDriveOverlayRenderer._has_camera_moved_significantly(cp)

        if not hasattr(cp, '_lane_overlay_surface') or cp._lane_overlay_surface is None:
            cp._lane_overlay_surface = pygame.Surface(
                (cp.screen_width, cp.screen_height), pygame.SRCALPHA
            )
            cp._lane_overlay_surface.fill((0, 0, 0, 0))
            screen.blit(cp._lane_overlay_surface, (0, 0))
        else:
            screen.blit(cp._lane_overlay_surface, (0, 0))

        if (
            cp.overlay_surface is None
            or cp.overlay_surface.get_size() != (cp.screen_width, cp.screen_height)
        ):
            cp.overlay_surface = pygame.Surface(
                (cp.screen_width, cp.screen_height), pygame.SRCALPHA
            )
            cp.overlay_surface_dirty = True

        if camera_moved or cp.overlay_surface_dirty:
            if camera_moved:
                cp.projection_cache.clear()
                OpenDriveOverlayRenderer._update_camera_state(cp)
            cp.overlay_surface.fill((0, 0, 0, 0))
            OpenDriveOverlayRenderer._render_lanes_to_surface(cp, cp.overlay_surface)
            cp.overlay_surface_dirty = False

        screen.blit(cp.overlay_surface, (0, 0))

    @staticmethod
    def precompute_lane_data(processor):
        cp = processor
        if not getattr(cp, 'world', None):
            return

        world_map = cp._get_cached_map(refresh=False)
        if not world_map:
            return

        start_time = time.time()
        try:
            cp.opendrive_lane_data = {
                'lane_segments': [],
                'waypoints': [],
                'spawn_points': [],
            }

            topology = world_map.get_topology()
            processed_segments = set()
            for start_wp, end_wp in topology:
                segment_key = (start_wp.road_id, start_wp.lane_id, int(start_wp.s / 10))
                if segment_key in processed_segments:
                    continue
                processed_segments.add(segment_key)

                lane_points = OpenDriveOverlayRenderer.trace_lane_segment(cp, start_wp, end_wp)
                for i in range(len(lane_points) - 1):
                    start_point = lane_points[i]
                    end_point = lane_points[i + 1]
                    start_z = start_point.z + 0.5
                    end_z = end_point.z + 0.5
                    cp.opendrive_lane_data['lane_segments'].append(
                        {
                            'start': {'x': start_point.x, 'y': start_point.y, 'z': start_z},
                            'end': {'x': end_point.x, 'y': end_point.y, 'z': end_z},
                        }
                    )

            OpenDriveOverlayRenderer._build_segment_grid(cp)
            print(f"[Lanes] Precomputed {len(cp.opendrive_lane_data['lane_segments'])} "
                  f"lane segments in {time.time() - start_time:.2f}s")
            OpenDriveOverlayRenderer.invalidate_cache(cp, drop_surfaces=True)
        except Exception:
            pass

    @staticmethod
    def trace_lane_segment(processor, start_wp, end_wp):
        cp = processor
        lane_points = []
        current = start_wp
        max_distance = start_wp.transform.location.distance(end_wp.transform.location)
        lane_points.append(current.transform.location)

        if max_distance < 10.0:
            lane_points.append(end_wp.transform.location)
            return lane_points

        step_size = min(2.0, max_distance / 20.0)
        total_distance = 0

        while total_distance < max_distance:
            try:
                next_waypoints = current.next(step_size)
                if not next_waypoints:
                    break
                next_wp = next_waypoints[0]
                next_pos = next_wp.transform.location
                distance_to_end = next_pos.distance(end_wp.transform.location)
                if distance_to_end < step_size * 1.5:
                    lane_points.append(end_wp.transform.location)
                    break

                lane_points.append(next_pos)
                current = next_wp
                total_distance += step_size

                if len(lane_points) > 200:
                    break
            except Exception:
                break

        return lane_points

    @staticmethod
    def toggle_overlay(processor):
        cp = processor
        if not hasattr(cp, 'lane_overlay_enabled'):
            cp.lane_overlay_enabled = False
        new_state = not cp.lane_overlay_enabled
        if new_state:
            cp.lane_overlay_enabled = True
            print("OpenDRIVE overlay enabled")
            if not getattr(cp, 'opendrive_lane_data', None):
                OpenDriveOverlayRenderer.precompute_lane_data(cp)
            OpenDriveOverlayRenderer.invalidate_cache(cp, drop_surfaces=True)
        else:
            OpenDriveOverlayRenderer.disable_overlay(cp)

    @staticmethod
    def _render_lanes_to_surface(processor, surface):
        cp = processor
        candidate_indices = None
        if cp.opendrive_segment_grid and cp.opendrive_segment_bboxes:
            viewport = OpenDriveOverlayRenderer._compute_viewport_bounds(cp)
            if viewport:
                minx, miny, maxx, maxy = viewport
                cell = cp.opendrive_grid_cell_size
                ix0 = int(math.floor(minx / cell))
                iy0 = int(math.floor(miny / cell))
                ix1 = int(math.floor(maxx / cell))
                iy1 = int(math.floor(maxy / cell))
                candidates = []
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        key = (ix, iy)
                        if key in cp.opendrive_segment_grid:
                            candidates.extend(cp.opendrive_segment_grid[key])
                candidate_indices = list(set(candidates))

        segments = cp.opendrive_lane_data.get('lane_segments', [])
        if candidate_indices is None:
            iterator = range(len(segments))
            viewport = OpenDriveOverlayRenderer._compute_viewport_bounds(cp)
        else:
            viewport = OpenDriveOverlayRenderer._compute_viewport_bounds(cp)
            if viewport is None:
                iterator = candidate_indices
            else:
                iterator = [
                    i
                    for i in candidate_indices
                    if OpenDriveOverlayRenderer._bbox_intersects(
                        cp.opendrive_segment_bboxes[i], viewport
                    )
                ]

        screen_width = cp.screen_width
        screen_height = cp.screen_height
        for i in iterator:
            segment = segments[i]
            start = segment['start']
            end = segment['end']
            start_screen = cp.coordinate_detector.world_to_screen_coordinates(
                start['x'], start['y'], start['z']
            )
            end_screen = cp.coordinate_detector.world_to_screen_coordinates(
                end['x'], end['y'], end['z']
            )
            if not (start_screen['success'] and end_screen['success']):
                continue

            start_x = int(start_screen['x'])
            start_y = int(start_screen['y'])
            end_x = int(end_screen['x'])
            end_y = int(end_screen['y'])
            if (
                (start_x < -50 and end_x < -50)
                or (start_x > screen_width + 50 and end_x > screen_width + 50)
                or (start_y < -50 and end_y < -50)
                or (start_y > screen_height + 50 and end_y > screen_height + 50)
            ):
                continue
            pygame.draw.line(surface, (255, 165, 0), (start_x, start_y), (end_x, end_y), 2)

    @staticmethod
    def _compute_viewport_bounds(processor):
        cp = processor
        cam = cp.camera_controller
        if getattr(cam, "view_mode", "topdown") == "orbit":
            # Screen corners near the horizon project to +-infinity (or fail); cull to a
            # pivot-centered box scaled by the zoom distance instead. Without this a
            # failed corner returns None and the lane renderer walks EVERY segment.
            r = max(100.0, min(3.0 * cam.orbit_distance, 2000.0))
            return (cam.center_x - r, cam.center_y - r, cam.center_x + r, cam.center_y + r)
        try:
            tl = cp.coordinate_detector.screen_to_world_coordinates_no_raycast(
                0, 0, cp.screen_width, cp.screen_height, 0.0
            )
            tr = cp.coordinate_detector.screen_to_world_coordinates_no_raycast(
                cp.screen_width, 0, cp.screen_width, cp.screen_height, 0.0
            )
            bl = cp.coordinate_detector.screen_to_world_coordinates_no_raycast(
                0, cp.screen_height, cp.screen_width, cp.screen_height, 0.0
            )
            br = cp.coordinate_detector.screen_to_world_coordinates_no_raycast(
                cp.screen_width, cp.screen_height, cp.screen_width, cp.screen_height, 0.0
            )
            if all(c['success'] for c in (tl, tr, bl, br)):
                x_coords = [tl['x'], tr['x'], bl['x'], br['x']]
                y_coords = [tl['y'], tr['y'], bl['y'], br['y']]
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        except Exception as exc:
            print(f"Error computing viewport bounds: {exc}")
        return None

    @staticmethod
    def _build_segment_grid(processor):
        cp = processor
        if not cp.opendrive_lane_data or not cp.opendrive_lane_data.get('lane_segments'):
            return

        segments = cp.opendrive_lane_data['lane_segments']
        cp.opendrive_segment_grid = {}
        cp.opendrive_segment_bboxes = []
        cell_size = cp.opendrive_grid_cell_size

        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            minx = min(start['x'], end['x'])
            miny = min(start['y'], end['y'])
            maxx = max(start['x'], end['x'])
            maxy = max(start['y'], end['y'])
            cp.opendrive_segment_bboxes.append((minx, miny, maxx, maxy))

            ix0 = int(math.floor(minx / cell_size))
            iy0 = int(math.floor(miny / cell_size))
            ix1 = int(math.floor(maxx / cell_size))
            iy1 = int(math.floor(maxy / cell_size))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    key = (ix, iy)
                    cp.opendrive_segment_grid.setdefault(key, []).append(i)

    @staticmethod
    def _bbox_intersects(bbox1, bbox2):
        minx1, miny1, maxx1, maxy1 = bbox1
        minx2, miny2, maxx2, maxy2 = bbox2
        return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)

    @staticmethod
    def _has_camera_moved_significantly(processor):
        cp = processor
        # step-12: any 3D-orbit pose change (including rotation in place) moves every
        # projected lane pixel, so orbit compares the full view_state(); top-down keeps
        # the legacy center/height tolerance math verbatim.
        pose = cp.camera_controller.view_state()
        last_pose = getattr(cp, "last_camera_pose", None)
        if last_pose is None or last_pose[0] != pose[0]:
            return True
        if pose[0] == "orbit":
            return pose != last_pose
        current_center = (cp.camera_controller.center_x, cp.camera_controller.center_y)
        current_height = cp.camera_controller.height

        if cp.last_camera_center is None or cp.last_camera_height is None:
            return True

        dx = abs(current_center[0] - cp.last_camera_center[0])
        dy = abs(current_center[1] - cp.last_camera_center[1])
        pos_change = (dx ** 2 + dy ** 2) ** 0.5
        height_change = abs(current_height - cp.last_camera_height)
        height_ratio = height_change / current_height if current_height > 0 else 0
        return pos_change > cp.cache_tolerance or height_ratio > 0.02

    @staticmethod
    def _update_camera_state(processor):
        cp = processor
        cp.last_camera_pose = cp.camera_controller.view_state()
        cp.last_camera_center = (cp.camera_controller.center_x, cp.camera_controller.center_y)
        cp.last_camera_height = cp.camera_controller.height
