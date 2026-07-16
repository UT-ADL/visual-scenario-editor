"""WorldCoordinateDetector (moved verbatim from vse.py): screen<->world
transforms and CARLA-debug waypoint drawing. Back-references (editor,
spawned_vehicles, traffic_lights) are assigned externally by the editor
after construction — kept as-is on purpose.
"""

from __future__ import annotations

import math
import time
import traceback
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np
import pygame

from vse_common.actor_cache import cached_bounding_box
from vse_common.geometry import cast_ray_with_tile_offset_compensation

# Upper bound on pick-ray plane intersections (m). Rejects near-horizon picks in the 3D
# orbit view that would otherwise land kilometers away. Provably inert in top-down: at
# 90-degree FOV the corner ray has |dir_z| >= 1/sqrt(3), and height <= 1000, so |t| <= ~1733.
MAX_PICK_RAY_T = 5000.0

class WorldCoordinateDetector:
    """
    Handles conversion between screen and world coordinates, lane snapping, and click detection for vehicles and waypoints.
    """

    def __init__(self, world, camera_controller):
        self.world = world
        self.world_map = None  # Lazy-load to avoid OpenDRIVE segfault after map changes
        self.camera_controller = camera_controller
        self.editor = None
        self.traffic_lights: List[carla.Actor] = []
        self._w2s_matrix_cache = None  # (signature, K, world_2_camera) -- opt-04

    def get_world_map(self):
        """Lazy-load world map with retry logic to avoid OpenDRIVE segfaults"""
        if self.world_map is not None:
            return self.world_map

        editor_cached = getattr(self, 'editor', None)
        if editor_cached is not None:
            cached_map = getattr(editor_cached, 'cached_map', None)
            if cached_map is not None:
                self.world_map = cached_map
                return self.world_map
            if getattr(editor_cached, '_map_refresh_disabled', False):
                return None

        # Try to get map with retries
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.world_map = self.world.get_map()
                return self.world_map
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Failed to get map (attempt {attempt + 1}/{max_attempts}): {e}")
                    time.sleep(2)
                else:
                    print(f"WARNING: Could not load world map after {max_attempts} attempts")
                    return None
        return None
        
    def build_projection_matrix(self, w, h, fov):
        """Build the camera intrinsic matrix K following CARLA's conventions"""
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    
    def get_image_point(self, loc, K, w2c, min_depth=None):
        """Project 3D world point to 2D image coordinates (for verification)"""
        # Format the input coordinate
        point = np.array([loc.x, loc.y, loc.z, 1])

        # Transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # CRITICAL: Change from UE4's coordinate system to standard camera coordinates
        # (x, y, z) -> (y, -z, x)
        point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])

        # Behind-camera guard (orbit view only; top-down callers pass min_depth=None to
        # stay byte-identical): points at or behind the camera plane have no projection.
        if min_depth is not None and point_camera[2] < min_depth:
            return None

        # Project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        
        # Normalize
        if point_img[2] != 0:
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]
        
        return point_img[0:2]
    
    def screen_to_world_coordinates(self, screen_x, screen_y, screen_width, screen_height):
        """Convert screen coordinates to world coordinates using proper CARLA transformations"""
        try:
            # Get camera transform
            camera_transform = self.camera_controller.get_carla_transform()
            camera_location = camera_transform.location
            debug_mode = bool(getattr(self, "debug_raycast", False))
            
            # Build the intrinsic matrix
            fov = 90.0  # Your camera FOV
            K = self.build_projection_matrix(screen_width, screen_height, fov)
            
            # Get world-to-camera transformation matrix
            world_2_camera = np.array(camera_transform.get_inverse_matrix())
            
            # For a top-down camera, we'll use raycasting with proper transformations
            # Convert screen coordinates to normalized image coordinates
            u = screen_x
            v = screen_y
            
            # Get inverse of intrinsic matrix
            K_inv = np.linalg.inv(K)
            
            # Convert pixel to normalized camera coordinates
            pixel_coords = np.array([u, v, 1.0])
            cam_coords = np.dot(K_inv, pixel_coords)
            
            # For CARLA's coordinate system, we need to reverse the UE4 transformation
            # Standard camera coords (x,y,z) to UE4 coords needs reverse of (y,-z,x)
            # So: cam(x,y,z) -> UE4(z, x, -y)
            ue4_ray_dir = np.array([cam_coords[2], cam_coords[0], -cam_coords[1], 0])
            
            # Get camera-to-world transformation (inverse of world-to-camera)
            camera_2_world = np.linalg.inv(world_2_camera)
            
            # Transform ray direction to world space
            world_ray = np.dot(camera_2_world, ue4_ray_dir)
            
            # Normalize the direction (ignore the homogeneous coordinate)
            ray_dir = world_ray[:3]
            ray_length = np.linalg.norm(ray_dir)
            if ray_length > 0:
                ray_dir = ray_dir / ray_length
            
            # Cast ray from camera position
            ray_start = camera_location
            ray_distance = abs(camera_location.z) + 100  # Ensure we reach the ground
            if getattr(self.camera_controller, "view_mode", "topdown") == "orbit":
                # Oblique orbit rays need dz-scaled reach to hit the ground; cap so a
                # near-horizon pixel can't fling the raycast kilometers away.
                ray_distance = min(2000.0, ray_distance / max(abs(ray_dir[2]), 0.05))
            
            ray_end = carla.Location(
                camera_location.x + ray_dir[0] * ray_distance,
                camera_location.y + ray_dir[1] * ray_distance,
                camera_location.z + ray_dir[2] * ray_distance
            )
            
            # Perform raycast
            hit_result, hit_meta = cast_ray_with_tile_offset_compensation(
                self.world,
                ray_start,
                ray_end,
                cached_map=self.get_world_map() if debug_mode else None,
                probe_on_miss=False,
                debug=debug_mode,
            )

            def _intersect_ray_with_z(target_z: float):
                if abs(ray_dir[2]) <= 0.001:
                    return None
                t = (target_z - camera_location.z) / ray_dir[2]
                if t <= 0 or t > MAX_PICK_RAY_T:  # behind camera / near-horizon fling
                    return None
                return (
                    camera_location.x + ray_dir[0] * t,
                    camera_location.y + ray_dir[1] * t,
                )

            if hit_result:
                hit_location = hit_result[0].location
                hit_z = float(hit_location.z)
                xy = _intersect_ray_with_z(hit_z)
                if xy is None:
                    # Fallback: use raw hit values if intersection math is degenerate.
                    xy = (float(hit_location.x), float(hit_location.y))
                if debug_mode:
                    corrected = bool(hit_meta.get('corrected', False))
                    offset_guess = hit_meta.get('offset_guess_xy', (0.0, 0.0))
                    correction_note = ""
                    if corrected:
                        correction_note = f" [tile+({float(offset_guess[0]):.0f},{float(offset_guess[1]):.0f})]"
                    print(
                        f"[Raycast] Hit: z={hit_z:.2f} "
                        f"(raw=({hit_location.x:.2f},{hit_location.y:.2f},{hit_location.z:.2f})) "
                        f"(proj=({xy[0]:.2f},{xy[1]:.2f},{hit_z:.2f})){correction_note}"
                    )
                return {
                    'x': xy[0],
                    'y': xy[1],
                    'z': hit_z,
                    'success': True,
                }

            # No ray hit: estimate ground Z (usually OpenDRIVE height), then intersect the ray
            # with that plane to avoid large-map "overshoot" when the map isn't raycastable.
            xy0 = _intersect_ray_with_z(0.0)
            if xy0 is None:
                return {'success': False, 'error': 'No intersection found'}

            estimated_z = 0.0
            world_map = self.get_world_map()
            if world_map is not None:
                try:
                    waypoint = world_map.get_waypoint(
                        carla.Location(float(xy0[0]), float(xy0[1]), 0.0),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk,
                    )
                except Exception:
                    waypoint = None
                if waypoint is not None:
                    estimated_z = float(waypoint.transform.location.z)

            xy1 = _intersect_ray_with_z(estimated_z)
            if xy1 is None:
                xy1 = xy0

            # Optional one-step refinement: re-sample height at the refined XY and re-intersect.
            if world_map is not None and estimated_z != 0.0:
                try:
                    waypoint2 = world_map.get_waypoint(
                        carla.Location(float(xy1[0]), float(xy1[1]), estimated_z),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk,
                    )
                except Exception:
                    waypoint2 = None
                if waypoint2 is not None:
                    refined_z = float(waypoint2.transform.location.z)
                    if abs(refined_z - estimated_z) > 0.05:
                        xy2 = _intersect_ray_with_z(refined_z)
                        if xy2 is not None:
                            xy1 = xy2
                            estimated_z = refined_z

            if debug_mode:
                print(
                    f"[Raycast] Miss: z0=0.00 -> z_est={estimated_z:.2f} "
                    f"(xy0=({xy0[0]:.2f},{xy0[1]:.2f})) "
                    f"(xy=({xy1[0]:.2f},{xy1[1]:.2f}))"
                )

            return {
                'x': float(xy1[0]),
                'y': float(xy1[1]),
                'z': float(estimated_z),
                'success': True,
                'estimated': True,
            }
                
        except Exception as e:
            print(f"Error in coordinate detection: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def screen_to_world_coordinates_no_raycast(self, screen_x, screen_y, screen_width, screen_height, fixed_z=0.0, cache=None):
        """
        Optimized: Use optional cache for camera transform and matrix calculations if called repeatedly within a frame.
        """
        try:
            if cache is None:
                cache = {}
            # Use cached camera transform and matrices if available
            if 'camera_transform' not in cache:
                cache['camera_transform'] = self.camera_controller.get_carla_transform()
            camera_transform = cache['camera_transform']
            camera_location = camera_transform.location

            if 'K' not in cache:
                fov = 90.0
                cache['K'] = self.build_projection_matrix(screen_width, screen_height, fov)
            K = cache['K']

            if 'world_2_camera' not in cache:
                cache['world_2_camera'] = np.array(camera_transform.get_inverse_matrix())
            world_2_camera = cache['world_2_camera']

            if 'K_inv' not in cache:
                cache['K_inv'] = np.linalg.inv(K)
            K_inv = cache['K_inv']

            u = screen_x
            v = screen_y
            pixel_coords = np.array([u, v, 1.0])
            cam_coords = np.dot(K_inv, pixel_coords)
            ue4_ray_dir = np.array([cam_coords[2], cam_coords[0], -cam_coords[1], 0])

            if 'camera_2_world' not in cache:
                cache['camera_2_world'] = np.linalg.inv(world_2_camera)
            camera_2_world = cache['camera_2_world']

            world_ray = np.dot(camera_2_world, ue4_ray_dir)
            ray_dir = world_ray[:3]
            ray_length = np.linalg.norm(ray_dir)
            if ray_length > 0:
                ray_dir = ray_dir / ray_length

            if abs(ray_dir[2]) > 0.001:
                t = (fixed_z - camera_location.z) / ray_dir[2]
                if 0 < t <= MAX_PICK_RAY_T:  # cap rejects near-horizon picks (3D view)
                    world_x = camera_location.x + ray_dir[0] * t
                    world_y = camera_location.y + ray_dir[1] * t
                    return {
                        'x': world_x,
                        'y': world_y,
                        'z': fixed_z,
                        'success': True
                    }
            return {'success': False, 'error': 'Invalid ray direction'}
        except Exception as e:
            print(f"Error in fast coordinate detection: {e}")
            return {'success': False, 'error': str(e)}
    
    def find_closest_lane_point(self, x, y, z):
        """Find the closest point on a lane to the given coordinates"""
        try:
            world_map = self.get_world_map()
            if not world_map:
                return None

            location = carla.Location(x, y, z)
            waypoint = world_map.get_waypoint(location,
                                              project_to_road=True,
                                              lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk)

            if waypoint:
                # Get lane direction from waypoint
                yaw = waypoint.transform.rotation.yaw

                # Use OpenDRIVE height directly (global coordinates, works correctly on large maps)
                # This matches the OpenDRIVE overlay and avoids tile-local coordinate issues
                ground_height = waypoint.transform.location.z

                return {
                    'x': waypoint.transform.location.x,
                    'y': waypoint.transform.location.y,
                    'z': ground_height,  # Use OpenDRIVE height (global coordinates)
                    'road_id': waypoint.road_id,
                    'lane_id': waypoint.lane_id,
                    'yaw': yaw,  # Lane direction
                    'lane_width': waypoint.lane_width,
                    'success': True
                }
            else:
                return {'success': False, 'error': 'No lane found'}
        except Exception as e:
            print(f"Error finding closest lane: {e}")
            return {'success': False, 'error': str(e)}
    
    def find_closest_lane_point_fast(self, x, y, z):
        """Find the closest point on a lane without using raycast (for fast movement updates)"""
        try:
            world_map = self.get_world_map()
            if not world_map:
                return None

            location = carla.Location(x, y, z)
            waypoint = world_map.get_waypoint(location,
                                              project_to_road=True,
                                              lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk)
            
            if waypoint:
                # Get lane direction from waypoint
                yaw = waypoint.transform.rotation.yaw
                
                # Use OpenDRIVE height (fast, no raycast)
                return {
                    'x': waypoint.transform.location.x,
                    'y': waypoint.transform.location.y,
                    'z': waypoint.transform.location.z,  # Use OpenDRIVE height for fast updates
                    'road_id': waypoint.road_id,
                    'lane_id': waypoint.lane_id,
                    'yaw': yaw,  # Lane direction
                    'lane_width': waypoint.lane_width,
                    'success': True
                }
            else:
                return {'success': False, 'error': 'No lane found'}
        except Exception as e:
            print(f"Error finding closest lane (fast): {e}")
            return {'success': False, 'error': str(e)}
    
    def move_camera_to_screen_position(self, screen_x, screen_y, screen_width, screen_height):
        """Move camera to the world position corresponding to screen click"""
        try:
            # Get the world coordinates at the click position
            coordinates = self.screen_to_world_coordinates_no_raycast(
                screen_x,
                screen_y,
                screen_width,
                screen_height,
                fixed_z=0.0,
            )
            
            if coordinates['success']:
                # Move camera to be above the clicked location
                self.camera_controller.center_x = coordinates['x']
                self.camera_controller.center_y = coordinates['y']
                
                print(f"Camera moved to: ({coordinates['x']:.2f}, {coordinates['y']:.2f})")
            else:
                print("Could not determine world position for camera movement")
                
        except Exception as e:
            print(f"Error moving camera: {e}")
    
    def world_to_screen_coordinates(self, world_x, world_y, world_z):
        """Convert world coordinates to screen coordinates"""
        try:
            # opt-04: K depends only on screen size (fov fixed) and the
            # world->camera matrix only on controller state (get_carla_transform
            # is pure math over the controller's pose), so both are cached under
            # the controller's view_state() pose signature instead of being
            # rebuilt for every projected point. view_state() includes the orbit
            # rotation fields (step-12) -- a center/height-only signature goes
            # stale the moment the 3D view rotates in place.
            cam = self.camera_controller
            sig = cam.view_state() + (self.screen_width, self.screen_height)
            cached = self._w2s_matrix_cache
            if cached is None or cached[0] != sig:
                camera_transform = cam.get_carla_transform()
                K = self.build_projection_matrix(self.screen_width, self.screen_height, 90.0)
                world_2_camera = np.array(camera_transform.get_inverse_matrix())
                cached = (sig, K, world_2_camera)
                self._w2s_matrix_cache = cached
            _, K, world_2_camera = cached

            # Create world point
            world_point = carla.Location(world_x, world_y, world_z)

            # Project to screen. In the 3D orbit view half the world can sit behind
            # the camera; those points must fail instead of projecting mirrored.
            min_depth = 0.1 if getattr(cam, "view_mode", "topdown") == "orbit" else None
            screen_point = self.get_image_point(world_point, K, world_2_camera, min_depth=min_depth)
            if screen_point is None:
                return {'success': False, 'error': 'behind camera'}

            return {
                'x': screen_point[0],
                'y': screen_point[1],
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _projected_actor_rect(self, actor):
        """Project the actor's 3D bounding-box corners to the screen; return
        their 2D extent as (min_x, min_y, max_x, max_y), or None."""
        try:
            # Cached box: this runs per frame for the selection highlight and per spawned
            # actor on click tests; a direct read is a blocking RPC on CARLA >= 0.9.16.
            corners = cached_bounding_box(actor).get_world_vertices(actor.get_transform())
        except Exception:
            return None

        xs = []
        ys = []
        for corner in corners:
            screen_pos = self.world_to_screen_coordinates(corner.x, corner.y, corner.z)
            if not screen_pos.get('success'):
                return None
            xs.append(screen_pos['x'])
            ys.append(screen_pos['y'])
        return (min(xs), min(ys), max(xs), max(ys))

    def check_vehicle_click(self, screen_x, screen_y):
        """Check if a click hits any spawned vehicle (screen-space test).

        Projects each actor's 3D bounding-box corners to the screen and tests
        the click against their 2D extent, so the clickable area matches the
        rendered body at every screen position. The previous ground-ray test
        missed tall actors near the screen edges: with the 90-degree FOV the
        oblique view ray lands height*tan(angle) behind the visible body.
        """
        padding = 5  # px of forgiveness around the projected box
        best_vehicle = None
        best_distance = None

        for vehicle in self.spawned_vehicles:
            if not (vehicle and vehicle.is_alive):
                continue
            rect = self._projected_actor_rect(vehicle)
            if rect is None:
                continue
            min_x, min_y, max_x, max_y = rect

            if (min_x - padding <= screen_x <= max_x + padding and
                    min_y - padding <= screen_y <= max_y + padding):
                # When boxes overlap, prefer the actor whose projected center
                # is closest to the click.
                center_dx = (min_x + max_x) / 2.0 - screen_x
                center_dy = (min_y + max_y) / 2.0 - screen_y
                distance = center_dx * center_dx + center_dy * center_dy
                if best_distance is None or distance < best_distance:
                    best_vehicle = vehicle
                    best_distance = distance

        return best_vehicle

    def actors_in_screen_rect(self, rect):
        """Return alive actors whose projected body box OVERLAPS the given
        screen rect (x, y, w, h) — marquee capture, touch/overlap rule."""
        rect_min_x, rect_min_y = rect[0], rect[1]
        rect_max_x, rect_max_y = rect[0] + rect[2], rect[1] + rect[3]
        hits = []
        for vehicle in self.spawned_vehicles:
            if not (vehicle and vehicle.is_alive):
                continue
            actor_rect = self._projected_actor_rect(vehicle)
            if actor_rect is None:
                continue
            if (actor_rect[0] <= rect_max_x and actor_rect[2] >= rect_min_x and
                    actor_rect[1] <= rect_max_y and actor_rect[3] >= rect_min_y):
                hits.append(vehicle)
        return hits
