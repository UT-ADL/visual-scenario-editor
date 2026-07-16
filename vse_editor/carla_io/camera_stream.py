"""Camera stream + processor residue: CameraImageProcessor.

Moved verbatim from vse.py (step-35, Phase 7). Holds the camera sensor
lifecycle, frame update loop, ego registry, menu-anchor/debounce family,
large-map machinery, scene accessors and the delegate table into
vse_editor/controllers/* and vse_editor/rendering/*.
"""

import math
import os
import queue
import time
import weakref

import numpy as np
import pygame

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import carla

from vse_common.actor_cache import clear_bounding_box_cache
from vse_common.env import env_bool, env_float
from vse_common.traffic_lights import (
    compute_traffic_light_fingerprint,
    normalize_traffic_light_fingerprint,
)
from vse_editor.scene_types import (
    TrafficLightGroupData,
    VehicleIgnoreFlags,
    clone_waypoint_data,
)
from vse_editor.carla_io.world_coords import WorldCoordinateDetector
from vse_editor.controllers import (
    manual_control,
    placement,
    placement_triggers,
    playback_camera,
    traffic_lights,
)
from vse_editor.rendering import carla_debug, scene_render
from vse_editor.rendering.overlays import (
    OpenDriveOverlayRenderer,
    TriggerOverlayRenderer,
    WaypointOverlayRenderer,
)
from vse_editor.scene import scenario_io, xosc_export
from vse_editor.scene.model import (
    SCENE_WEATHER_FIELDS,
    SceneModel,
    install_scene_forwarders,
)
from vse_editor.session import SessionState


class CameraImageProcessor:
    def start_auto_route_to_destination(self, vehicle):
        return placement.start_auto_route_to_destination(self, vehicle)
    """
    Manages the CARLA camera sensor, image acquisition, vehicle and waypoint management, and overlay rendering.
    Central logic for scenario editing, visualization, and interaction with CARLA world objects.
    """
    
    def __init__(self, world, camera_controller, screen_width, screen_height, editor=None, stream_resolution=None, stream_fps=None):
        self.world = world
        self.camera_controller = camera_controller
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.editor = editor
        self.large_map_active = bool(editor and getattr(editor, "large_map_active", False))
        try:
            self.large_map_ui_delay = float(getattr(editor, "large_map_ui_delay", 1.0)) if editor else 1.0
        except Exception:
            self.large_map_ui_delay = 1.0
        # Default to a short-but-tolerant timeout; values are clamped below.
        self._large_map_rpc_timeout_s = env_float("VSE_LARGE_MAP_RPC_TIMEOUT", 2.0)
        self._large_map_rpc_timeout_s = min(2.0, max(0.05, self._large_map_rpc_timeout_s))
        self._large_map_transform_retry_s = env_float("VSE_LARGE_MAP_TRANSFORM_RETRY", 0.25)
        self._large_map_transform_retry_s = min(5.0, max(0.05, self._large_map_transform_retry_s))
        self._large_map_focus_travel_speed = env_float("VSE_LARGE_MAP_FOCUS_TRAVEL_SPEED", 2000.0)
        self._large_map_focus_travel_speed = max(50.0, self._large_map_focus_travel_speed)
        self._large_map_focus_travel_threshold = env_float("VSE_LARGE_MAP_FOCUS_TRAVEL_THRESHOLD", 999999.0)
        self._large_map_focus_travel_threshold = max(0.0, self._large_map_focus_travel_threshold)
        if editor is not None and getattr(editor, "cached_map", None) is not None:
            self.cached_map = editor.cached_map
        else:
            self.cached_map = None
        if self.cached_map is None and self.world is not None:
            try:
                self.cached_map = self.world.get_map()
            except Exception:
                self.cached_map = None
        self.camera_sensor = None
        self.camera_active = True
        self._camera_stream_paused = False
        self.image_queue = queue.Queue()
        self.latest_image = None
        self.latest_frame_id = None
        self._scaled_image_cache = None
        self._scaled_image_frame = None
        self._scaled_image_size = None
        self.stream_fps = stream_fps
        if stream_resolution is not None:
            self.image_width, self.image_height = stream_resolution
        else:
            self.image_width = screen_width
            self.image_height = screen_height
        self.stream_resolution = (self.image_width, self.image_height)
        
        # Camera settings
        self.last_frame_time = time.time()
        self._camera_fps_display = 0.0
        self._camera_fps_smoothing = 0.8
        
        # Vehicle management
        # Scene document + selection state now live on SceneModel (editor-owned,
        # reset on every processor construction — same lifetime the moved
        # attributes had). @property forwarders keep self.<field> working.
        # getattr-guarded like every other editor read here, so duck-typed
        # editors (test harnesses) without a .scene still work.
        self.scene = getattr(editor, "scene", None) or SceneModel()
        self.scene.reset()
        # Session flags are read through a live view of the editor (or its
        # defaults when headless) — see vse_editor/session.py.
        self.session = getattr(editor, "session", None) or SessionState(editor)
        self.pedestrian_colors: Dict[int, carla.Color] = {}  # Stable per-pedestrian colors for highlights
        self._actor_location_cache: Dict[int, Tuple[carla.Location, float]] = {}
        self._pending_large_map_focus_actor_id: Optional[int] = None
        self._pending_large_map_focus_next_attempt = 0.0
        self._pending_large_map_focus_failures = 0
        self._pending_large_map_transform: Optional[carla.Transform] = None
        self._pending_large_map_transform_next_attempt = 0.0
        self._pending_large_map_transform_failures = 0
        self._fast_rpc_client: Optional[carla.Client] = None
        self._fast_rpc_world = None
        self._fast_rpc_host_port: Optional[Tuple[str, int]] = None
        self._fast_rpc_spectator_id: Optional[int] = None
        self._large_map_travel_target: Optional[Tuple[float, float]] = None
        self.manual_control_enabled = False
        self.manual_control_pending = False
        # Per-run latch: True for the whole local-ego manual run regardless of attach state.
        # Arrow-key routing gates on this (not the transient enabled/pending pair) so arrows
        # never leak to the camera while the ego is spawning or recovering from a transient drop.
        self.manual_control_armed = False
        self.manual_control_actor = None
        self.manual_control_state = {
            'control': None,
            'steer_cache': 0.0,
            'reverse': False,
            'auto_reverse_engaged': False,
            'hand_brake_pressed': False,
            'autopilot_enabled': False,
            'autopilot_active': False,
            'camera_follow_local_offset': (0.0, 0.0, 0.0),
        }
        self._manual_control_search_timer = 0.0
        self._manual_control_tick_subscription = None
        self._manual_control_target_transform: Optional[carla.Transform] = None
        self._manual_camera_free_look_sources: Set[str] = set()
        self._pending_free_look_restores: Dict[str, float] = {}
        self._manual_camera_restore_retry_window = 0.75

        # Playback camera follow (decoupled from manual control for ego with route)
        self.playback_camera_follow_enabled = False
        self.playback_camera_follow_pending = False
        self.playback_camera_follow_actor = None
        self._playback_camera_tick_subscription = None
        self._playback_camera_follow_search_timer = 0.0
        self._playback_camera_follow_local_offset = (0.0, 0.0, 0.0)
        self._playback_camera_target_transform: Optional[carla.Transform] = None
        # External-ego (VIL) camera-follow smoothing: ease the top-down camera toward the ego
        # instead of snapping each tick, so async ego teleports don't make the view shaky.
        # Exponential time constant in seconds: smaller = snappier (tighter), larger = floatier.
        # 0 = no smoothing (snap to the ego each frame).
        self._playback_camera_follow_tau = 0
        # Predictive lead (seconds): aim ahead by the ego's estimated velocity to cancel the
        # camera's one-tick set_transform lag, tightening the lock without the snap wobble.
        # 0 = no prediction. Try ~0.05-0.15 (optionally with a small tau).
        self._playback_camera_follow_lead = 0
        # Internal predictive-follow state (ego velocity estimated from observed target motion;
        # finite-difference, because get_velocity() can read 0 for a physics-off teleported ego).
        self._playback_camera_follow_vel = (0.0, 0.0)
        self._playback_camera_follow_prev_target: Optional[Tuple[float, float]] = None
        self._playback_camera_follow_prev_t = 0.0
        # Playback camera mode, cycled by C during a run: "topdown" (overhead, editor default),
        # "chase" (behind the vehicle, like awmini's spectator), or "cockpit" (driver-seat view,
        # rigid-attached to the ego). Chase params match awmini's carla.yaml spectator defaults;
        # tunable like tau/lead.
        self.playback_camera_mode = "topdown"
        self.chase_distance_behind = 8.0   # m behind the ego
        self.chase_height_above = 5.0      # m above the ego
        self.chase_pitch = 15            # degrees downward tilt
        # Cockpit (driver-seat) offset in the ego's own frame, looking forward. Matches the cockpit
        # presets in manual_control_steeringwheel.py / drive_racing.py (tuned for the Lexus ego).
        self.cockpit_x = 0.1    # m forward of the vehicle origin
        self.cockpit_y = -0.5   # m to the (left) driver seat
        self.cockpit_z = 1.4    # m up (head height)
        self.cockpit_pitch = 0.0  # degrees (0 = look straight ahead)
        # True while VSE's camera is rigid-attached to a CARLA actor: the spectator (external-ego
        # chase -> zero-lag, server-window-identical) or the ego itself (cockpit). While attached the
        # engine drives the camera, so the follow tick/step/apply are suspended (see
        # _is_camera_engine_attached). Local-ego chase stays compute-own (not attached).
        self._camera_attached_to_spectator = False
        self._camera_attached_to_ego = False

        # Weather scene state lives on SceneModel (single source of truth,
        # editor-lifetime — survives processor reconstruction). The old
        # per-processor shadow copy seeded from world weather is gone; the
        # save path sanitizes scene keyframes and falls back to live world
        # weather when nothing is populated yet (_sanitize_weather_keyframes).
        self._pedestrian_highlight_subscription = None
        self._pedestrian_highlight_last_draw: Dict[int, float] = {}

        self.vehicle_menu_position = None  # Position for vehicle action menu
        self.vehicle_menu_hidden_for_camera_pan = False  # Hide vehicle menu while user pans the camera
        self.trigger_menu_hidden_for_camera_pan = False  # Hide trigger menu while user pans the camera
        self.traffic_light_menu_hidden_for_camera_pan = False  # Hide traffic-light menu while user pans the camera
        self.traffic_light_overlays_hidden_for_camera_pan = False  # Hide traffic-light overlays during camera motion
        self.actor_overlays_hidden_for_camera_pan = False  # Hide actor selection overlays during camera motion
        self.camera_is_moving = False  # Track if camera is currently moving
        self.camera_movement_timer = 0  # Time since last camera movement
        self.menu_update_delay = 0.2  # Delay before updating menu after camera stops (200ms)
        if self.large_map_active:
            self.menu_update_delay = max(0.0, float(self.large_map_ui_delay))
        self._server_host = '127.0.0.1'
        self._server_port = 2000

        self.waypoint_drag_threshold = 5  # Minimum pixels to move before starting drag operation
        self.object_drag_threshold = 8  # Same, for actors/triggers (grab-to-move promotion)
        self._click_actor_hit_cache = None  # Per-press memo for actor_under_click

        # Waypoint CARLA debug drawing system (replaces overlay rendering)
        self.waypoint_debug_refresh_timer = 0  # Timer for refreshing debug drawings
        self.waypoint_debug_refresh_interval = 0.05  # Refresh every 50ms for very responsive updates
        self.waypoint_debug_lifetime = 0.15  # 100ms lifetime for quicker updates
        self.waypoint_marker_radius = 24  # Increased radius for hover detection (bigger markers)
        self.waypoint_split_offset = 1.5  # Desired offset distance when splitting points
        
        # OpenDRIVE lane overlay system (NEW)
        self.lane_overlay_enabled = False  # Start disabled, toggle with 'O'
        self.traffic_lights_visible = False  # Start disabled, toggle with 'T'
        self.opendrive_lane_data = None  # Precomputed lane data for overlay rendering
        # Spatial index for fast culling
        self.opendrive_grid_cell_size = 50.0  # meters
        self.opendrive_segment_grid = None  # dict[(ix,iy)] -> list[int]
        self.opendrive_segment_bboxes = None  # list of (minx, miny, maxx, maxy)
        # Camera state tracking for projection caching
        self.last_camera_center = None  # (x, y)
        self.last_camera_height = None
        self.last_camera_pose = None  # view_state() tuple (step-12: orbit-aware redraw gate)
        self.projection_cache = {}  # Cache projected coordinates by camera state
        self.cache_tolerance = 1.0  # Camera movement tolerance for cache invalidation (reduced for snappiness)
        # Offscreen surface caching for throttled redraw
        self.overlay_surface = None  # Cached overlay surface
        self.overlay_surface_dirty = True  # Whether overlay needs redraw
        self.opendrive_overlay_hidden_for_camera_pan = False  # Hide OpenDRIVE overlay during camera motion
        
        # Trigger system (NEW) - Only one trigger allowed per scenario
        self.trigger_action_menu_position = None  # Position of trigger action menu
        self.personal_trigger_menu_position = None  # Screen coords for selected personal trigger
        self.personal_trigger_menu_hidden_for_camera_pan = False
        self._personal_trigger_move_cache = None  # Reusable projection cache during trigger movement

        # Traffic light management
        self.traffic_lights: List[carla.TrafficLight] = []
        self.traffic_light_groups: List[TrafficLightGroupData] = []
        self._traffic_light_group_lookup: Dict[int, TrafficLightGroupData] = {}
        self._traffic_light_rectangles: Dict[int, Tuple[List[carla.Location], Optional[carla.Location]]] = {}
        self._traffic_light_volume_bounds: Dict[int, Tuple[float, float]] = {}
        self._traffic_light_visibility: Dict[int, bool] = {}
        self._traffic_light_screen_polygons: Dict[int, List[Tuple[float, float]]] = {}
        self._traffic_light_screen_centers: Dict[int, Tuple[float, float]] = {}
        self._traffic_light_ground_cache: Dict[int, Tuple[float, float]] = {}  # id -> (ground_z, timestamp)
        self._last_connector_debug_signature: Optional[Tuple[Tuple[int, ...], Tuple[str, ...]]] = None
        self._traffic_light_refresh_timer = 0.0
        self._traffic_light_refresh_interval = 2.0  # seconds
        self._traffic_light_refresh_error_logged = False
        self._traffic_light_debug_matching = env_bool("VSE_DEBUG_TRAFFIC_LIGHT_MATCH")
        self._traffic_light_debug_render = env_bool("VSE_DEBUG_TRAFFIC_LIGHT_DRAW")
        self._traffic_light_render_debug_signature: Optional[Tuple[int, bool]] = None
        self._traffic_light_group_snapshots: List[Dict[str, object]] = []
        self.traffic_light_menu_position: Optional[Tuple[int, int]] = None
        try:
            self._traffic_light_font = pygame.font.Font(None, 18)
        except Exception:
            self._traffic_light_font = None

        # Add coordinate detector
        self.coordinate_detector = WorldCoordinateDetector(world, camera_controller)
        self.coordinate_detector.editor = editor
        if editor is not None and getattr(editor, "cached_map", None) is not None:
            self.coordinate_detector.world_map = editor.cached_map
        self.coordinate_detector.screen_width = screen_width
        self.coordinate_detector.screen_height = screen_height
        self.coordinate_detector.spawned_vehicles = self.spawned_vehicles
        self.coordinate_detector.traffic_lights = self.traffic_lights

        # Debug flag for raycast coordinate testing (large maps)
        self.debug_raycast = False  # Set to True to enable debug logging
        self.setup_camera()
        self._refresh_traffic_lights()
        self._register_pedestrian_highlight_tick()

    def _get_cached_map(self, refresh=False):
        if self.cached_map is not None:
            return self.cached_map

        editor = getattr(self, "editor", None)
        refresh_allowed = refresh and not self.session._map_refresh_disabled

        map_obj = None
        if editor is not None:
            map_obj = editor._safe_get_world_map(refresh=refresh_allowed)
        elif refresh_allowed and self.world:
            try:
                map_obj = self.world.get_map()
            except Exception as exc:
                print(f"[Map] Unable to refresh cached map from camera processor: {exc}")

        if map_obj is not None:
            self.cached_map = map_obj
        return self.cached_map

    def setup_camera(
        self,
        attach_to: Optional[carla.Actor] = None,
        transform: Optional[carla.Transform] = None,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
    ):
        """Create and configure the camera sensor"""
        # Exactly one camera sensor at a time: destroy any existing one before spawning, so swapping
        # top-down <-> chase never leaves two cameras rendering (which would halve FPS).
        if self.camera_sensor is not None:
            try:
                self.camera_sensor.stop()   # stop client stream before destroy() (see cleanup())
            except Exception:
                pass
            try:
                self.camera_sensor.destroy()
            except Exception:
                pass
            self.camera_sensor = None
        # Get camera blueprint
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Configure camera
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', '90')
        if self.stream_fps and self.stream_fps > 0:
            tick_value = max(0.001, 1.0 / float(self.stream_fps))
        else:
            tick_value = 0.0
        camera_bp.set_attribute('sensor_tick', f"{tick_value:.3f}")
        
        # Spawn camera
        if transform is None:
            transform = self.camera_controller.get_carla_transform()
        if attach_to is not None:
            self.camera_sensor = self.world.spawn_actor(
                camera_bp,
                transform,
                attach_to=attach_to,
                attachment_type=attachment_type,
            )
        else:
            self.camera_sensor = self.world.spawn_actor(camera_bp, transform)
        
        # Set up image callback
        weak_self = weakref.ref(self)
        self.camera_sensor.listen(lambda image: CameraImageProcessor._on_image_received(weak_self, image))
        self._camera_stream_paused = False
        self.frames_received = 0
        
        print("Camera sensor created and listening")
    
    def update_screen_size(self, new_width, new_height):
        """Update screen size for camera processing without changing the stream resolution."""
        if abs(self.screen_width - new_width) < 5 and abs(self.screen_height - new_height) < 5:
            return

        self.screen_width = new_width
        self.screen_height = new_height
        self.coordinate_detector.screen_width = new_width
        self.coordinate_detector.screen_height = new_height
        self._scaled_image_cache = None
        self._scaled_image_frame = None
        self._scaled_image_size = None
        stream_w, stream_h = self.stream_resolution
        print(f"Display resized to: {new_width}x{new_height} (stream {stream_w}x{stream_h})")
    
    def apply_stream_settings(self, resolution=None, fps=None):
        """Apply updated stream settings and restart the camera sensor if needed."""
        need_restart = False

        if resolution is not None:
            new_w, new_h = int(resolution[0]), int(resolution[1])
            if (new_w, new_h) != self.stream_resolution:
                self.stream_resolution = (new_w, new_h)
                self.image_width = new_w
                self.image_height = new_h
                need_restart = True

        if fps is not None:
            new_fps = max(1, int(fps)) if fps > 0 else None
            if new_fps != self.stream_fps:
                self.stream_fps = new_fps
                need_restart = True

        if not self.camera_active:
            return

        if need_restart:
            self._scaled_image_cache = None
            self._scaled_image_frame = None
            self._scaled_image_size = None
            if self._is_camera_engine_attached():
                # Re-spawn the new-resolution sensor still attached to its actor (chase ->
                # spectator, cockpit -> ego) instead of free at the editor pose. Clear the
                # flags first so _apply_playback_camera_attach re-enters the attach.
                self._camera_attached_to_spectator = False
                self._camera_attached_to_ego = False
                self._apply_playback_camera_attach()
            else:
                self.restart_camera_sensor()
        else:
            self.stream_resolution = (int(self.stream_resolution[0]), int(self.stream_resolution[1]))

    def disable_camera_sensor(self):
        """Completely stop the RGB sensor and drop any buffered frames."""
        self.camera_active = False
        self._camera_stream_paused = False
        if self.camera_sensor:
            try:
                self.camera_sensor.stop()   # stop client stream before destroy() (see cleanup())
            except Exception:
                pass
            try:
                self.camera_sensor.destroy()
            except Exception as exc:
                print(f"[Camera] Warning: failed to destroy camera sensor cleanly: {exc}")
            finally:
                self.camera_sensor = None
        self.image_queue = queue.Queue()
        self.latest_image = None
        self.latest_frame_id = None
        self._scaled_image_cache = None
        self._scaled_image_frame = None
        self._scaled_image_size = None
        self.frames_received = 0

    def enable_camera_sensor(self):
        """Ensure the RGB sensor exists and is producing frames."""
        if self.camera_sensor and getattr(self, "_camera_stream_paused", False):
            self.camera_active = True
            self.resume_camera_stream()
            return
        if self.camera_active and self.camera_sensor:
            return
        self.camera_active = True
        self.restart_camera_sensor()

    def pause_camera_stream(self):
        """Stop receiving frames without destroying the sensor actor."""
        if getattr(self, "_camera_stream_paused", False):
            return
        if self.camera_sensor:
            try:
                self.camera_sensor.stop()
            except Exception as exc:
                print(f"[Camera] Warning: failed to stop camera sensor: {exc}")
        self._camera_stream_paused = True
        self.image_queue = queue.Queue()
        self.latest_image = None
        self.latest_frame_id = None
        self._scaled_image_cache = None
        self._scaled_image_frame = None
        self._scaled_image_size = None
        self.frames_received = 0

    def resume_camera_stream(self):
        """Resume receiving frames after pause_camera_stream()."""
        if not getattr(self, "_camera_stream_paused", False):
            return
        if not self.camera_sensor:
            self._camera_stream_paused = False
            self.camera_active = True
            self.restart_camera_sensor()
            return
        try:
            weak_self = weakref.ref(self)
            self.camera_sensor.listen(lambda image: CameraImageProcessor._on_image_received(weak_self, image))
        except Exception as exc:
            print(f"[Camera] Warning: failed to resume camera sensor: {exc}")
            return
        self._camera_stream_paused = False


    def restart_camera_sensor(
        self,
        attach_to: Optional[carla.Actor] = None,
        transform: Optional[carla.Transform] = None,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
    ):
        """Destroy and recreate the camera sensor with the current settings."""
        if not self.camera_active:
            return
        if self.camera_sensor:
            try:
                self.camera_sensor.stop()   # stop client stream before destroy() (see cleanup())
            except Exception:
                pass
            try:
                self.camera_sensor.destroy()
            except Exception as exc:
                print(f"[Camera] Warning: failed to destroy camera sensor cleanly: {exc}")
            finally:
                self.camera_sensor = None
        self.setup_camera(attach_to=attach_to, transform=transform, attachment_type=attachment_type)

    @staticmethod
    def _on_image_received(weak_self, carla_image):
        """Process received camera image"""
        self = weak_self()
        if not self:
            return
        
        # Convert CARLA image to a pygame surface (opt-07): CARLA's raw_data
        # is row-major BGRA, which pygame reads natively -- one memcpy
        # (bytes() -- the image buffer is recycled after this callback, so
        # the surface must own a copy) instead of the numpy reshape/slice/
        # BGR-reverse/swapaxes pipeline that copied via strided views.
        pygame_image = pygame.image.frombuffer(
            bytes(carla_image.raw_data),
            (carla_image.width, carla_image.height),
            'BGRA',
        )
        
        # Store latest image and update camera FPS estimate
        self.latest_image = pygame_image
        self.latest_frame_id = carla_image.frame
        now = time.time()
        if self.last_frame_time:
            frame_dt = now - self.last_frame_time
            if frame_dt > 0:
                instantaneous = 1.0 / frame_dt
                alpha = 1.0 - self._camera_fps_smoothing
                if self._camera_fps_display <= 0.0:
                    self._camera_fps_display = instantaneous
                else:
                    self._camera_fps_display = max(
                        0.0,
                        (self._camera_fps_display * self._camera_fps_smoothing) + (instantaneous * alpha),
                    )
        self.last_frame_time = now
        self._scaled_image_cache = None
        self._scaled_image_frame = None
        self._scaled_image_size = None
        if not hasattr(self, "_debug_printed_frames"):
            self._debug_printed_frames = 0
        self.frames_received = getattr(self, "frames_received", 0) + 1
        if self._debug_printed_frames < 5:
            print(f"[Camera] Received frame {carla_image.frame} ({carla_image.width}x{carla_image.height})")
            self._debug_printed_frames += 1

    def get_camera_fps(self) -> float:
        """Return smoothed camera FPS based on received sensor frames."""
        return max(0.0, getattr(self, "_camera_fps_display", 0.0))

    def _get_cached_actor_location(self, actor_id: int, *, max_age_s: float = 0.5) -> Optional[carla.Location]:
        """Return a cached actor location when available (used to avoid blocking RPC on large maps)."""
        cached = self._actor_location_cache.get(int(actor_id))
        if not cached:
            return None
        location, stamp = cached
        if (time.time() - float(stamp)) > float(max_age_s):
            return None
        return location

    def _cache_actor_location(self, actor_id: int, location: carla.Location) -> None:
        try:
            self._actor_location_cache[int(actor_id)] = (location, time.time())
        except Exception:
            pass

    def _get_fast_world(self) -> Optional["carla.World"]:
        """Return a short-timeout CARLA world handle for large-map safety operations."""
        host = getattr(self, "_server_host", "127.0.0.1")
        port = int(getattr(self, "_server_port", 2000))
        if not host or port <= 0:
            return None

        key = (str(host), int(port))
        if self._fast_rpc_client is None or self._fast_rpc_host_port != key:
            try:
                client = carla.Client(key[0], key[1])
                client.set_timeout(float(self._large_map_rpc_timeout_s))
            except Exception:
                self._fast_rpc_client = None
                self._fast_rpc_world = None
                self._fast_rpc_host_port = None
                self._fast_rpc_spectator_id = None
                return None
            self._fast_rpc_client = client
            self._fast_rpc_world = None
            self._fast_rpc_host_port = key
            self._fast_rpc_spectator_id = None

        if self._fast_rpc_world is None and self._fast_rpc_client is not None:
            try:
                self._fast_rpc_world = self._fast_rpc_client.get_world()
            except Exception:
                self._fast_rpc_world = None
                self._fast_rpc_spectator_id = None
                return None
        return self._fast_rpc_world

    def _get_actor_location_fast(self, actor_id: int, *, max_age_s: float = 0.5) -> Optional[carla.Location]:
        """Try to fetch an actor location with a short RPC timeout; returns cached value on failure."""
        cached = self._get_cached_actor_location(actor_id, max_age_s=max_age_s)
        if cached is not None:
            return cached

        world = self._get_fast_world()
        if world is None:
            return None

        actor = None
        try:
            actor = world.get_actor(int(actor_id))
        except Exception:
            actor = None
        if not actor:
            return None

        try:
            loc = actor.get_location()
        except Exception:
            return None
        self._cache_actor_location(actor_id, loc)
        return loc

    def _queue_large_map_transform(self, transform: carla.Transform) -> None:
        """Queue a transform apply for large maps to avoid blocking the UI thread on long RPC timeouts."""
        if self._pending_large_map_transform is None:
            self._pending_large_map_transform_failures = 0
            self._pending_large_map_transform_next_attempt = 0.0
        self._pending_large_map_transform = transform

    def _apply_large_map_transform_once(self) -> bool:
        """Attempt to apply the queued transform using a short-timeout CARLA client."""
        transform = self._pending_large_map_transform
        if not transform or not self.camera_sensor:
            self._pending_large_map_transform = None
            return True

        if not self.session.camera_stream_enabled:
            # Keep the transform queued; we apply when the stream is enabled again.
            return False

        skip_spectator = env_bool("VSE_LARGE_MAP_SKIP_SPECTATOR")

        world = self._get_fast_world()
        client = self._fast_rpc_client
        if world is None or client is None:
            return False

        if not skip_spectator and self._fast_rpc_spectator_id is None:
            try:
                spectator = world.get_spectator()
                self._fast_rpc_spectator_id = int(spectator.id) if spectator else None
            except Exception:
                self._fast_rpc_spectator_id = None

        batch = []
        try:
            batch.append(carla.command.ApplyTransform(int(self.camera_sensor.id), transform))
        except Exception:
            return False
        if not skip_spectator and self._fast_rpc_spectator_id is not None:
            try:
                batch.append(carla.command.ApplyTransform(int(self._fast_rpc_spectator_id), transform))
            except Exception:
                pass

        try:
            client.apply_batch_sync(batch, False)
        except Exception:
            return False
        return True

    def _process_large_map_transform_queue(self) -> None:
        """Retry queued camera/spectator transforms with backoff (large maps)."""
        if not self._pending_large_map_transform:
            return
        if not self.session.camera_stream_enabled:
            return
        now = time.time()
        if now < float(self._pending_large_map_transform_next_attempt or 0.0):
            return

        ok = self._apply_large_map_transform_once()
        if ok:
            self._pending_large_map_transform = None
            self._pending_large_map_transform_failures = 0
            self._pending_large_map_transform_next_attempt = 0.0
            return

        self._pending_large_map_transform_failures += 1
        backoff = float(self._large_map_transform_retry_s) * (2.0 ** min(4, self._pending_large_map_transform_failures))
        self._pending_large_map_transform_next_attempt = now + min(3.0, backoff)

    def _process_large_map_pending_focus(self) -> None:
        """Resolve pending focus requests without blocking (large maps)."""
        actor_id = self._pending_large_map_focus_actor_id
        if actor_id is None:
            return
        now = time.time()
        if now < float(self._pending_large_map_focus_next_attempt or 0.0):
            return

        loc = self._get_actor_location_fast(actor_id, max_age_s=0.25)
        if loc is None:
            self._pending_large_map_focus_failures += 1
            backoff = 0.25 * (2.0 ** min(5, self._pending_large_map_focus_failures))
            self._pending_large_map_focus_next_attempt = now + min(3.0, backoff)
            return

        self._pending_large_map_focus_actor_id = None
        self._pending_large_map_focus_failures = 0
        self._pending_large_map_focus_next_attempt = 0.0
        try:
            self.focus_camera_on_location(loc)
        except Exception:
            pass

    def _process_large_map_travel(self, dt: float) -> None:
        """Animate long-distance camera focus to reduce streaming hitches (large maps)."""
        target = self._large_map_travel_target
        if not target or not self.camera_controller:
            return
        try:
            tx, ty = float(target[0]), float(target[1])
        except Exception:
            self._large_map_travel_target = None
            return

        cx = float(getattr(self.camera_controller, "center_x", 0.0))
        cy = float(getattr(self.camera_controller, "center_y", 0.0))
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)
        if dist <= 1.0:
            self.camera_controller.center_x = tx
            self.camera_controller.center_y = ty
            self._large_map_travel_target = None
            self.update_camera_position()
            return

        step = float(self._large_map_focus_travel_speed) * max(0.0, float(dt))
        if step <= 0.0:
            return
        if step >= dist:
            nx, ny = tx, ty
        else:
            scale = step / dist
            nx = cx + dx * scale
            ny = cy + dy * scale

        self.camera_controller.center_x = nx
        self.camera_controller.center_y = ny
        self.update_camera_position()
    
    def update_camera_position(self):
        """Update camera sensor position"""
        if self.camera_sensor:
            transform = self.camera_controller.get_carla_transform()
            if self.large_map_active:
                self._queue_large_map_transform(transform)
            else:
                try:
                    self.camera_sensor.set_transform(transform)
                    # Spectator intentionally NOT moved on normal maps: the CARLA
                    # spectator is unhooked from the editor camera and left where
                    # it is. (Large maps anchor streaming via the queue path above.)
                except Exception:
                    pass

        # Mark camera as moving and reset timer
        self.camera_is_moving = True
        self.camera_movement_timer = time.time()
    
    def update_vehicle_menu_position(self):
        """Update the vehicle menu position to follow the selected vehicle"""
        if self.selected_personal_trigger:
            self.vehicle_menu_position = None
            return

        if not (self.selected_vehicle and self.selected_vehicle.is_alive):
            return
        
        # Only update menu position if camera has stopped moving for a while
        current_time = time.time()
        if self.camera_is_moving and (current_time - self.camera_movement_timer < self.menu_update_delay):
            return
        
        # Camera has stopped moving, update menu position
        self.camera_is_moving = False

        vehicle_location = None
        if self.large_map_active:
            vehicle_location = self._get_actor_location_fast(self.selected_vehicle.id, max_age_s=0.5)
        else:
            vehicle_location = self._get_selected_vehicle_location_cached()

        if vehicle_location is None:
            return
        screen_pos = self.coordinate_detector.world_to_screen_coordinates(
            vehicle_location.x, vehicle_location.y, vehicle_location.z + 2.0  # Slightly above vehicle
        )
        
        if screen_pos['success']:
            self.vehicle_menu_position = (int(screen_pos['x']), int(screen_pos['y']))
            self.vehicle_menu_hidden_for_camera_pan = False
        else:
            self.vehicle_menu_position = None

    def _get_selected_vehicle_location_cached(self, max_age_s=0.05):
        """One get_location RPC per refresh window for the selected actor.

        Shared by the menu anchor and the selection highlight, which each
        made their own per-frame RPC for the same actor (opt-05). Keyed by
        actor id, so a selection change refreshes immediately; a drag is at
        most max_age_s stale (one frame at 20 fps).
        """
        actor = self.selected_vehicle
        if not actor:
            return None
        now = time.monotonic()
        cached = getattr(self, "_selected_loc_cache", None)
        if cached and cached[0] == actor.id and (now - cached[1]) < max_age_s:
            return cached[2]
        location = actor.get_location()
        self._selected_loc_cache = (actor.id, now, location)
        return location

    def suppress_vehicle_menu_for_camera_pan(self) -> None:
        """Temporarily hide actor/trigger/traffic-light menus during camera motion."""
        if self.selected_vehicle and self.selected_vehicle.is_alive:
            self.vehicle_menu_hidden_for_camera_pan = True
        if (self.selected_trigger_index is not None
                and 0 <= self.selected_trigger_index < len(self.triggers)):
            self.trigger_menu_hidden_for_camera_pan = True
        if self.selected_personal_trigger:
            self.personal_trigger_menu_hidden_for_camera_pan = True
        if self.selected_traffic_light_group or getattr(self, '_scenario_active_traffic_light_trigger', None):
            self.traffic_light_menu_hidden_for_camera_pan = True
        self.traffic_light_overlays_hidden_for_camera_pan = True
        self.actor_overlays_hidden_for_camera_pan = True
        if self.lane_overlay_enabled:
            self.opendrive_overlay_hidden_for_camera_pan = True

    def restore_vehicle_menu_after_camera_pan(self) -> None:
        """Restore actor/trigger menus once manual camera panning stops."""
        self.vehicle_menu_hidden_for_camera_pan = False
        self.trigger_menu_hidden_for_camera_pan = False
        self.personal_trigger_menu_hidden_for_camera_pan = False
        self.traffic_light_menu_hidden_for_camera_pan = False
        self.traffic_light_overlays_hidden_for_camera_pan = False
        self.actor_overlays_hidden_for_camera_pan = False
        if self.lane_overlay_enabled:
            self.opendrive_overlay_hidden_for_camera_pan = False
            self.overlay_surface_dirty = True

        # Force an immediate position refresh so the menus snap to their actors.
        self.camera_is_moving = False
        self.camera_movement_timer = 0.0

        if self.selected_vehicle and self.selected_vehicle.is_alive:
            self.update_vehicle_menu_position()
        if (self.selected_trigger_index is not None
                and 0 <= self.selected_trigger_index < len(self.triggers)):
            self.update_trigger_menu_position()
        if self.selected_personal_trigger:
            self.update_personal_trigger_menu_position(force=True)
        if self.selected_traffic_light_group:
            self.update_traffic_light_menu_position()

    def update_trigger_menu_position(self) -> None:
        """Update the trigger action menu position to follow the selected trigger."""
        if self.selected_personal_trigger:
            self.trigger_action_menu_position = None
            return
        if self.selected_trigger_index is None:
            return
        if not (0 <= self.selected_trigger_index < len(self.triggers)):
            self.selected_trigger_index = None
            self.trigger_action_menu_position = None
            return

        current_time = time.time()
        if self.camera_is_moving and (current_time - self.camera_movement_timer < self.menu_update_delay):
            return

        self.camera_is_moving = False

        trigger = self.triggers[self.selected_trigger_index]
        screen_pos = self.coordinate_detector.world_to_screen_coordinates(
            trigger['x'], trigger['y'], trigger['z']
        )

        if screen_pos['success']:
            self.trigger_action_menu_position = (int(screen_pos['x']), int(screen_pos['y']))
            self.trigger_menu_hidden_for_camera_pan = False
        else:
            self.trigger_action_menu_position = None

    def update_traffic_light_menu_position(self) -> None:
        """Update the traffic-light action menu position after camera motion."""
        if self.selected_personal_trigger:
            self.traffic_light_menu_position = None
            return
        group = self.selected_traffic_light_group
        if not group:
            return

        current_time = time.time()
        if self.camera_is_moving and (current_time - self.camera_movement_timer < self.menu_update_delay):
            return

        self.camera_is_moving = False

        self._update_traffic_light_menu_anchor(group)
        if self.traffic_light_menu_position:
            self.traffic_light_menu_hidden_for_camera_pan = False
            self.traffic_light_overlays_hidden_for_camera_pan = False

    def refresh_selected_vehicle_ui(self):
        """Force-update menu positioning and info panel for the currently selected vehicle."""
        if not (self.selected_vehicle and self.selected_vehicle.is_alive):
            return

        # Allow immediate menu update
        self.camera_is_moving = False
        self.camera_movement_timer = 0.0
        self.update_vehicle_menu_position()

        panel = getattr(getattr(self, "editor", None), "info_panel", None)
        if (panel and panel.visible and panel.object_type in ('vehicle', 'pedestrian')
                and panel.selected_object is self.selected_vehicle):
            panel._refresh_fields()

    def clear_vehicle_selection(self, *, keep_waypoints=False, hide_info_panel=True, clear_traffic_light=True):
        """Deselect the current vehicle/pedestrian and optionally preserve waypoint display."""
        previous_personal_selection = self.selected_personal_trigger
        self.selected_vehicle = None
        self.selected_vehicle_is_pedestrian = False
        self.vehicle_drag_armed = False
        self.vehicle_menu_position = None
        if not keep_waypoints:
            self.waypoint_display_vehicle_id = None
        panel = getattr(getattr(self, "editor", None), "info_panel", None)
        if hide_info_panel and panel:
            panel.hide()
        if clear_traffic_light:
            self.clear_traffic_light_selection()
        if (previous_personal_selection
                and previous_personal_selection.get('kind') in ('vehicle', 'pedestrian')):
            self.clear_personal_trigger_selection()

    def restore_actor_selection(self, actor: Optional[carla.Actor]) -> bool:
        """Reapply selection state to a specific actor after scenario playback."""
        if not actor or not actor.is_alive:
            return False

        self.selected_vehicle = actor
        self.selected_vehicle_is_pedestrian = actor.type_id.startswith('walker.')
        self.vehicle_menu_position = None
        self.selected_waypoint_vehicle_id = None
        self.selected_waypoint_index = None
        self.waypoint_display_vehicle_id = actor.id

        try:
            self.update_vehicle_menu_position()
        except Exception:
            pass

        try:
            self.refresh_waypoints_carla_debug()
        except Exception:
            pass

        return True

    def focus_camera_on_location(self, location: Optional[carla.Location]):
        """Center the camera over a world location without changing its height."""
        if not location or not self.camera_controller:
            return

        # Preserve current height explicitly, as downstream updates may clamp values
        current_height = self.camera_controller.height

        if getattr(self.camera_controller, "view_mode", "topdown") == "orbit":
            # 3D orbit view: center_x/center_y ARE the pivot, so focus moves it. Skip the
            # large-map travel animation (it eases the center per frame without re-clamping
            # the orbit pose) and resample the ground fresh -- location.z is often a camera
            # height or an actor Z, never a trustworthy pivot ground height.
            self.camera_controller.center_x = float(location.x)
            self.camera_controller.center_y = float(location.y)
            self._large_map_travel_target = None  # focus overrides any pending travel ease
            self.camera_controller._cached_pivot_ground = None
            self.camera_controller._cached_eye_ground = None
            self.camera_controller._clamp_orbit_pose()  # refreshes pivot_z (miss-aware)
            self.update_camera_position()
            self.notify_manual_camera_adjustment()
            if hasattr(self.camera_controller, "stop_moving"):
                self.camera_controller.stop_moving()
            self.camera_is_moving = True
            self.camera_movement_timer = time.time()
            return

        if self.large_map_active:
            try:
                dx = float(location.x) - float(self.camera_controller.center_x)
                dy = float(location.y) - float(self.camera_controller.center_y)
                dist = math.hypot(dx, dy)
            except Exception:
                dist = 0.0
            if dist > float(self._large_map_focus_travel_threshold):
                self._large_map_travel_target = (float(location.x), float(location.y))
                self.camera_controller.height = current_height
                if hasattr(self.camera_controller, "stop_moving"):
                    self.camera_controller.stop_moving()
                self.camera_is_moving = True
                self.camera_movement_timer = time.time()
                return

        self.camera_controller.center_x = location.x
        self.camera_controller.center_y = location.y
        self.camera_controller.height = current_height

        # Apply the transform to the CARLA camera and spectator
        self.update_camera_position()
        self.notify_manual_camera_adjustment()

        # Treat focus as a camera movement so menu updates are delayed (important on large maps)
        if hasattr(self.camera_controller, "stop_moving"):
            self.camera_controller.stop_moving()
        self.camera_is_moving = True
        self.camera_movement_timer = time.time()
    
    def update(self, dt):
        """Update camera processor state (call this every frame)"""
        self._process_pending_free_look_restores()
        if self.large_map_active:
            # Large-map safety: avoid long blocking RPC in the render thread.
            self._process_large_map_pending_focus()
            self._process_large_map_travel(dt)
            self._process_large_map_transform_queue()

        current_time = time.time()
        if self.camera_is_moving and (current_time - self.camera_movement_timer >= self.menu_update_delay):
            self.camera_is_moving = False
            self.camera_movement_timer = current_time
            self.traffic_light_overlays_hidden_for_camera_pan = False
            if self.lane_overlay_enabled:
                self.opendrive_overlay_hidden_for_camera_pan = False
                self.overlay_surface_dirty = True

        # Check if camera movement has stopped and update menus if needed
        if self.selected_vehicle and self.selected_vehicle.is_alive:
            self.update_vehicle_menu_position()
        if (self.selected_trigger_index is not None
                and 0 <= self.selected_trigger_index < len(self.triggers)):
            self.update_trigger_menu_position()
        if self.selected_personal_trigger:
            self.update_personal_trigger_menu_position()
        if self.selected_traffic_light_group and self.traffic_lights_visible:
            self.update_traffic_light_menu_position()
        
        # Update waypoint debug drawing timer
        self.waypoint_debug_refresh_timer += dt
        if self.waypoint_debug_refresh_timer >= self.waypoint_debug_refresh_interval:
            self.waypoint_debug_refresh_timer = 0
            if not (self.large_map_active and self.camera_is_moving):
                try:
                    self.refresh_waypoints_carla_debug()
                except Exception:
                    pass
        self._manual_control_update(dt)
        self._playback_camera_follow_update(dt)

        # Periodically refresh traffic light actor list
        self._traffic_light_refresh_timer += dt
        if self._traffic_light_refresh_timer >= self._traffic_light_refresh_interval:
            self._traffic_light_refresh_timer = 0.0
            if not (self.large_map_active and self.camera_is_moving):
                self._refresh_traffic_lights()

    def _refresh_traffic_lights(self) -> None:
        return traffic_lights._refresh_traffic_lights(self)

    _bboxes_overlap = staticmethod(traffic_lights._bboxes_overlap)

    _yaws_aligned = staticmethod(traffic_lights._yaws_aligned)

    _heading_diff_180 = staticmethod(traffic_lights._heading_diff_180)

    _heading_diff_360 = staticmethod(traffic_lights._heading_diff_360)

    def _get_travel_yaw_at(self, location: carla.Location) -> Optional[float]:
        return traffic_lights._get_travel_yaw_at(self, location)

    def _get_traffic_light_grouping_geometry(
        self, traffic_light: carla.TrafficLight
    ) -> Optional[Tuple[Tuple[float, float, float, float, float, float], Optional[float]]]:
        return traffic_lights._get_traffic_light_grouping_geometry(self, traffic_light)

    # Delegates to the shared implementation in vse_common/traffic_lights.py
    _compute_traffic_light_fingerprint = staticmethod(compute_traffic_light_fingerprint)

    def _traffic_light_debug(self, message: str) -> None:
        return traffic_lights._traffic_light_debug(self, message)

    _compute_traffic_light_group_centroid = staticmethod(traffic_lights._compute_traffic_light_group_centroid)

    def _build_traffic_light_groups(self, grouping_meta: List[Dict[str, object]]) -> None:
        return traffic_lights._build_traffic_light_groups(self, grouping_meta)

    def _snapshot_traffic_light_groups(self, groups: List[TrafficLightGroupData]) -> None:
        return traffic_lights._snapshot_traffic_light_groups(self, groups)

    def _rebuild_traffic_light_group_lookup(self) -> None:
        return traffic_lights._rebuild_traffic_light_group_lookup(self)

    def _restore_selected_traffic_light_group(self, alive_ids: Set[int]) -> None:
        return traffic_lights._restore_selected_traffic_light_group(self, alive_ids)

    def _is_traffic_light_underground(self, light: carla.TrafficLight) -> bool:
        return traffic_lights._is_traffic_light_underground(self, light)

    def refresh_waypoints_carla_debug(self):
        return carla_debug.refresh_waypoints_carla_debug(self)

    def _get_start_marker_for_vehicle_id(self, vehicle_id: Optional[int]) -> Optional[dict]:
        return carla_debug._get_start_marker_for_vehicle_id(self, vehicle_id)

    def _get_start_marker_for_vehicle(self, vehicle: Optional[carla.Actor]) -> Optional[dict]:
        return carla_debug._get_start_marker_for_vehicle(self, vehicle)

    def register_ego_vehicle(
        self,
        actor: carla.Actor,
        transform: carla.Transform,
        color: Optional[str] = None,
        preserve_waypoints: bool = False,
    ) -> None:
        """Track the currently placed ego vehicle metadata."""
        if not actor:
            return
        self.ego_vehicle_id = actor.id
        self.ego_vehicle_transform = carla.Transform(transform.location, transform.rotation)
        self.vehicle_transforms[actor.id] = carla.Transform(transform.location, transform.rotation)
        self.ego_vehicle_blueprint = actor.type_id
        self.ego_vehicle_color = color
        if not preserve_waypoints:
            self.clear_vehicle_waypoints(actor.id)
        if self.manual_control_enabled and self.manual_control_actor and self.manual_control_actor.id != actor.id:
            self.disable_manual_control()

    def update_editor_ego_transform(self, transform: Optional[carla.Transform]) -> None:
        """Persist editor ego transform metadata when the actor is moved."""
        if transform is None:
            return
        self.ego_vehicle_transform = carla.Transform(transform.location, transform.rotation)
        if self.ego_vehicle_id is not None:
            self.vehicle_transforms[self.ego_vehicle_id] = carla.Transform(transform.location, transform.rotation)

    def clear_ego_vehicle(self, actor_id: Optional[int] = None) -> None:
        """Clear ego vehicle tracking data."""
        if actor_id is not None and self.ego_vehicle_id != actor_id:
            return
        if self.ego_vehicle_id:
            self.clear_vehicle_waypoints(self.ego_vehicle_id)
        self.ego_vehicle_id = None
        self.ego_vehicle_transform = None
        self.ego_vehicle_blueprint = None
        self.ego_vehicle_color = None

    # -- manual-control delegates (step-27) -----------------------------
    # State stays on the processor; bodies live in
    # vse_editor/controllers/manual_control.py (duck-typed module functions).
    # The two transform builders keep staticmethod aliases; the free-look
    # flag keeps its @property (modules can't host instance properties).

    def request_manual_control(self) -> None:
        return manual_control.request_manual_control(self)

    def disable_manual_control(self) -> None:
        return manual_control.disable_manual_control(self)

    def handle_manual_control_key(self, key: int, is_pressed: bool) -> bool:
        return manual_control.handle_manual_control_key(self, key, is_pressed)

    def _manual_control_update(self, dt: float) -> None:
        return manual_control._manual_control_update(self, dt)

    def _find_manual_control_actor(self) -> Optional[carla.Actor]:
        return manual_control._find_manual_control_actor(self)

    def _apply_manual_control(self, dt: float) -> None:
        return manual_control._apply_manual_control(self, dt)

    def _manual_control_compute_local_offset(
        self,
        actor_transform: carla.Transform,
        controller: Optional["TopDownCamera"],
    ) -> Tuple[float, float, float]:
        return manual_control._manual_control_compute_local_offset(self, actor_transform, controller)

    _manual_control_build_world_transform = staticmethod(manual_control._manual_control_build_world_transform)

    _build_chase_world_transform = staticmethod(manual_control._build_chase_world_transform)

    def _build_follow_target(self, actor_transform, local_offset):
        return manual_control._build_follow_target(self, actor_transform, local_offset)

    def _manual_control_apply_target_transform(self, *, force: bool = False) -> None:
        return manual_control._manual_control_apply_target_transform(self, force=force)

    def _manual_control_start_camera_follow(self, actor: Optional[carla.Actor]) -> None:
        return manual_control._manual_control_start_camera_follow(self, actor)

    def _manual_control_stop_camera_follow(self) -> None:
        return manual_control._manual_control_stop_camera_follow(self)

    @property
    def manual_camera_free_look_active(self) -> bool:
        return manual_control.manual_camera_free_look_active(self)

    def begin_manual_camera_free_look(self, source: str) -> None:
        return manual_control.begin_manual_camera_free_look(self, source)

    def end_manual_camera_free_look(self, source: str) -> None:
        return manual_control.end_manual_camera_free_look(self, source)

    def _restore_camera_follow_after_free_look(self) -> bool:
        return manual_control._restore_camera_follow_after_free_look(self)

    def notify_manual_camera_adjustment(self) -> None:
        return manual_control.notify_manual_camera_adjustment(self)

    def _process_pending_free_look_restores(self) -> None:
        return manual_control._process_pending_free_look_restores(self)

    def _manual_control_register_tick_callback(self) -> None:
        return manual_control._manual_control_register_tick_callback(self)

    def _manual_control_unregister_tick_callback(self) -> None:
        return manual_control._manual_control_unregister_tick_callback(self)

    def _manual_control_on_world_tick(self, _snapshot) -> None:
        return manual_control._manual_control_on_world_tick(self, _snapshot)

    # -------------------------------------------------------------------------
    # Playback Camera Follow (for local ego with route, decoupled from manual control)
    # -------------------------------------------------------------------------

    # -- playback-camera delegates (step-26) ----------------------------
    # State stays on the processor; bodies live in
    # vse_editor/controllers/playback_camera.py (duck-typed module functions).

    def request_playback_camera_follow(self) -> None:
        return playback_camera.request_playback_camera_follow(self)

    def toggle_playback_camera_mode(self) -> str:
        return playback_camera.toggle_playback_camera_mode(self)

    def set_playback_camera_mode(self, mode: str) -> str:
        return playback_camera.set_playback_camera_mode(self, mode)

    def _apply_playback_camera_attach(self) -> None:
        return playback_camera._apply_playback_camera_attach(self)

    def _attach_camera_to_spectator(self) -> None:
        return playback_camera._attach_camera_to_spectator(self)

    def _detach_camera_from_spectator(self) -> None:
        return playback_camera._detach_camera_from_spectator(self)

    def _attach_camera_to_ego_cockpit(self, actor: Optional[carla.Actor]) -> None:
        return playback_camera._attach_camera_to_ego_cockpit(self, actor)

    def _detach_camera_from_ego(self) -> None:
        return playback_camera._detach_camera_from_ego(self)

    def _is_camera_engine_attached(self) -> bool:
        return playback_camera._is_camera_engine_attached(self)

    def _stop_playback_camera_follow(self) -> None:
        return playback_camera._stop_playback_camera_follow(self)

    def _start_playback_camera_follow(self, actor: carla.Actor) -> None:
        return playback_camera._start_playback_camera_follow(self, actor)

    def _playback_camera_apply_target_transform(self, *, force: bool = False) -> None:
        return playback_camera._playback_camera_apply_target_transform(self, force=force)

    def _playback_camera_register_tick_callback(self) -> None:
        return playback_camera._playback_camera_register_tick_callback(self)

    def _playback_camera_unregister_tick_callback(self) -> None:
        return playback_camera._playback_camera_unregister_tick_callback(self)

    def _playback_camera_on_world_tick(self, snapshot) -> None:
        return playback_camera._playback_camera_on_world_tick(self, snapshot)

    def _playback_camera_follow_update(self, dt: float) -> None:
        return playback_camera._playback_camera_follow_update(self, dt)

    def _playback_camera_follow_step(self, dt: float) -> None:
        return playback_camera._playback_camera_follow_step(self, dt)

    def _register_pedestrian_highlight_tick(self) -> None:
        return carla_debug._register_pedestrian_highlight_tick(self)

    def _unregister_pedestrian_highlight_tick(self) -> None:
        return carla_debug._unregister_pedestrian_highlight_tick(self)

    def _on_pedestrian_highlight_tick(self, _snapshot) -> None:
        return carla_debug._on_pedestrian_highlight_tick(self, _snapshot)

    def is_ego_vehicle(self, vehicle_id: Optional[int]) -> bool:
        return self.scene.is_ego_vehicle(vehicle_id)

    def is_ego_vehicle_active(self) -> bool:
        if self.ego_vehicle_id is None:
            return False
        actor = next((veh for veh in self.spawned_vehicles if veh.id == self.ego_vehicle_id and veh.is_alive), None)
        if actor:
            return True
        self.clear_ego_vehicle(self.ego_vehicle_id)
        return False

    def get_editor_ego_actor(self) -> Optional[carla.Actor]:
        """Return the ego actor spawned via the editor, if it is still alive."""
        ego_id = self.ego_vehicle_id
        if ego_id is None:
            return None
        return next(
            (actor for actor in self.spawned_vehicles if actor and actor.is_alive and actor.id == ego_id),
            None,
        )

    def get_ego_vehicle_actor(self) -> Optional[carla.Actor]:
        """Find a live ego-designated actor in the world, even if not tracked."""
        if not self.world:
            return None
        try:
            actors = self.world.get_actors().filter('vehicle.*')
        except RuntimeError:
            return None
        tracked = {actor.id for actor in self.spawned_vehicles if actor and actor.is_alive}
        roles = {'hero', 'ego', 'ego_vehicle', 'player'}
        for actor in actors:
            if actor.id in tracked and self.is_ego_vehicle(actor.id):
                if actor.is_alive:
                    return actor
                continue
            role_name = actor.attributes.get('role_name', '').lower()
            if role_name in roles and actor.is_alive:
                return actor
        return None

    def get_ego_vehicle_data(self) -> Optional[dict]:
        """Return serialized ego vehicle data for saving, if present."""
        if not self.is_ego_vehicle_active():
            return None
        # Use stored authoritative transform — never read from live actor to avoid
        # position corruption when an external system (e.g. Autoware) moves the ego.
        transform = self.vehicle_transforms.get(self.ego_vehicle_id)
        if transform is None:
            transform = self.ego_vehicle_transform

        if not transform:
            return None

        ego_data = {
            "type": self.ego_vehicle_blueprint or "vehicle.lexus.utlexus",
            "location": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
            },
            "rotation": {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll,
            },
            "role": "ego_vehicle",
        }
        color = self.get_vehicle_color(self.ego_vehicle_id)
        if color is not None:
            ego_data["color"] = color
        ignore_flags = self.get_vehicle_ignore_flags(self.ego_vehicle_id)
        ego_data["ignore_traffic_lights"] = ignore_flags["traffic_lights"]
        ego_data["ignore_stop_signs"] = ignore_flags["stop_signs"]
        ego_data["ignore_vehicles"] = ignore_flags["vehicles"]
        ego_data["max_lat_acc"] = float(self.get_vehicle_max_lat_acc(self.ego_vehicle_id, 3.0))
        serialized_waypoints, destination_speed = self._serialize_waypoints_for_vehicle(
            self.ego_vehicle_id,
            start_location=transform,
        )
        if serialized_waypoints:
            ego_data["waypoints"] = serialized_waypoints
        if destination_speed is not None:
            ego_data["destination_speed_km_h"] = destination_speed
        return ego_data
    
    def draw_waypoints_carla_debug_local(self):
        return carla_debug.draw_waypoints_carla_debug_local(self)
                
            # Direction arrows removed for cleaner appearance
    
    def get_hovered_waypoint(self):
        return carla_debug.get_hovered_waypoint(self)
    
    def _draw_waypoint_connections_carla_debug_local(self, vehicle, start_point, waypoints, *, is_ego_path=False, line_scale=1.0):
        return carla_debug._draw_waypoint_connections_carla_debug_local(self, vehicle, start_point, waypoints, is_ego_path=is_ego_path, line_scale=line_scale)
    
    def _draw_finish_line_carla_debug_local(self, location, waypoints, waypoint_index, is_selected, is_hovered, *, is_ego_path=False, line_scale=1.0):
        return carla_debug._draw_finish_line_carla_debug_local(self, location, waypoints, waypoint_index, is_selected, is_hovered, is_ego_path=is_ego_path, line_scale=line_scale)
    
    
    
    def check_menu_icon_click(self, mouse_x, mouse_y):
        return placement.check_menu_icon_click(self, mouse_x, mouse_y)
    
    def get_vehicle_menu_icon_order(self):
        return placement.vehicle_menu_icon_order(self)

    def actor_under_click(self, screen_x, screen_y):
        return placement.actor_under_click(self, screen_x, screen_y)

    def group_selection_menu_anchor(self):
        return placement.group_selection_menu_anchor(self)

    def group_waypoint_screen_positions(self):
        return placement._group_waypoint_screen_positions(self)

    def check_group_menu_icon_click(self, mouse_x, mouse_y):
        return placement.check_group_menu_icon_click(self, mouse_x, mouse_y)

    def delete_selected_actors(self):
        return placement.delete_selected_actors(self)

    def waypoints_in_screen_rect(self, rect):
        return placement.waypoints_in_screen_rect(self, rect)

    def select_single_waypoint(self, vehicle_id, waypoint_index):
        return placement.select_single_waypoint(self, vehicle_id, waypoint_index)

    def select_and_arm_actor(self, actor, screen_x, screen_y):
        return placement._select_and_arm_actor(self, actor, screen_x, screen_y)

    def select_and_arm_waypoint(self, vehicle_id, waypoint_index, screen_x, screen_y):
        return placement._select_and_arm_waypoint(self, vehicle_id, waypoint_index, screen_x, screen_y)

    def toggle_actor_in_group(self, actor):
        return placement._toggle_actor_in_group(self, actor)

    def toggle_waypoint_in_group(self, vehicle_id, waypoint_index):
        return placement._toggle_waypoint_in_group(self, vehicle_id, waypoint_index)

    def trigger_targets_in_screen_rect(self, rect):
        return placement_triggers.trigger_targets_in_screen_rect(self, rect)

    def delete_selected_waypoint_group(self):
        return placement.delete_selected_waypoint_group(self)

    def delete_group_selection(self):
        """Delete whichever marquee group is active (actors or waypoints)."""
        if self.selected_actor_ids:
            return placement.delete_selected_actors(self)
        return placement.delete_selected_waypoint_group(self)

    def cancel_active_drag(self):
        """Escape-cancel whichever press-drag gesture is armed or in progress
        (moves, rotation, and every trigger-scaling family)."""
        cancelled = placement.cancel_vehicle_drag(self)
        cancelled = placement.cancel_vehicle_rotation(self) or cancelled
        cancelled = placement.cancel_waypoint_drag(self) or cancelled
        cancelled = placement_triggers.cancel_trigger_drag(self) or cancelled
        cancelled = placement_triggers.cancel_trigger_scaling(self) or cancelled
        cancelled = placement_triggers.cancel_personal_trigger_drag(self) or cancelled
        cancelled = placement_triggers.cancel_pedestrian_trigger_scaling(self) or cancelled
        cancelled = placement_triggers.cancel_vehicle_trigger_scaling(self) or cancelled
        cancelled = traffic_lights.cancel_traffic_light_trigger_scaling(self) or cancelled
        if self.vehicle_drag_armed or self.trigger_drag_armed or self.personal_trigger_drag_armed:
            self.vehicle_drag_armed = False
            self.trigger_drag_armed = False
            self.personal_trigger_drag_armed = False
            cancelled = True
        return cancelled

    def check_vehicle_collision(self, vehicle, new_transform):
        return placement.check_vehicle_collision(self, vehicle, new_transform)

    def _is_pedestrian_actor(self, actor_id):
        return placement._is_pedestrian_actor(self, actor_id)

    def _adjust_pedestrian_spawn_orientation(self, actor_id):
        return placement._adjust_pedestrian_spawn_orientation(self, actor_id)
    
    def start_vehicle_movement(self, mouse_pos, snap_to_lane=False):
        return placement.start_vehicle_movement(self, mouse_pos, snap_to_lane)
    
    def update_vehicle_movement(self, mouse_pos):
        return placement.update_vehicle_movement(self, mouse_pos)
        # If collision detected, don't move the vehicle
    
    def get_ground_height_at_location(self, x, y, reference_z=None, return_metadata=False, probe_on_miss=True):
        return placement.get_ground_height_at_location(self, x, y, reference_z, return_metadata, probe_on_miss)

    def stop_vehicle_movement(self):
        return placement.stop_vehicle_movement(self)
    

    def start_vehicle_rotation(self, mouse_y):
        return placement.start_vehicle_rotation(self, mouse_y)
    
    def update_vehicle_rotation(self, mouse_y):
        return placement.update_vehicle_rotation(self, mouse_y)
        # If collision detected, don't rotate the vehicle
    
    def stop_vehicle_rotation(self):
        return placement.stop_vehicle_rotation(self)
    
    def start_waypoint_creation(self, reset_existing=True):
        return placement.start_waypoint_creation(self, reset_existing)
    
    def stop_waypoint_creation(self, *, clear_waypoints=False):
        return placement.stop_waypoint_creation(self, clear_waypoints=clear_waypoints)

    def stop_destination_creation(self):
        return placement.stop_destination_creation(self)

    def place_waypoint_at_click(self, screen_x, screen_y):
        return placement.place_waypoint_at_click(self, screen_x, screen_y)

    def place_destination_at_click(self, screen_x, screen_y):
        return placement.place_destination_at_click(self, screen_x, screen_y)

    def _create_auto_waypoint_data(self, location: "carla.Location", waypoint_index: int,
                                    yaw: float, speed_km_h: float) -> dict:
        return placement._create_auto_waypoint_data(self, location, waypoint_index, yaw, speed_km_h)

    def auto_route_to_destination(self, vehicle, destination):
        return placement.auto_route_to_destination(self, vehicle, destination)

    # Old debug marker waypoint functions removed - now using overlay rendering only

    def delete_selected_vehicle(self):
        return placement.delete_selected_vehicle(self)

    ############################################################
    # Trigger Zone Management
    ############################################################

    def start_trigger_placement(self):
        return placement_triggers.start_trigger_placement(self)

    def stop_trigger_placement(self):
        return placement_triggers.stop_trigger_placement(self)

    def start_personal_trigger_placement(self, kind: str, *, actor=None, group=None) -> bool:
        return placement_triggers.start_personal_trigger_placement(self, kind, actor=actor, group=group)

    def cancel_personal_trigger_placement(self) -> None:
        return placement_triggers.cancel_personal_trigger_placement(self)

    def place_trigger_at_click(self, screen_x, screen_y):
        return placement_triggers.place_trigger_at_click(self, screen_x, screen_y)

    def place_trigger_instantly(self, screen_x, screen_y):
        return placement_triggers.place_trigger_instantly(self, screen_x, screen_y)

    def place_personal_trigger_at_click(self, screen_x: int, screen_y: int) -> bool:
        return placement_triggers.place_personal_trigger_at_click(self, screen_x, screen_y)

    def handle_trigger_click(self, screen_x, screen_y):
        return placement_triggers.handle_trigger_click(self, screen_x, screen_y)

    def check_trigger_menu_icon_click(self, mouse_x, mouse_y):
        return placement_triggers.check_trigger_menu_icon_click(self, mouse_x, mouse_y)

    def trigger_menu_icons(self):
        return placement_triggers.trigger_menu_icons(self)

    def personal_trigger_menu_icons(self):
        return placement_triggers.personal_trigger_menu_icons(self)

    def check_personal_trigger_menu_icon_click(self, mouse_x, mouse_y):
        return placement_triggers.check_personal_trigger_menu_icon_click(self, mouse_x, mouse_y)

    def start_trigger_movement(self, mouse_pos):
        return placement_triggers.start_trigger_movement(self, mouse_pos)

    def stop_trigger_movement(self):
        return placement_triggers.stop_trigger_movement(self)

    def update_trigger_movement(self, mouse_pos):
        return placement_triggers.update_trigger_movement(self, mouse_pos)

    def start_trigger_scaling(self, mouse_pos):
        return placement_triggers.start_trigger_scaling(self, mouse_pos)

    def stop_trigger_scaling(self):
        return placement_triggers.stop_trigger_scaling(self)

    def update_trigger_scaling(self, mouse_pos):
        return placement_triggers.update_trigger_scaling(self, mouse_pos)

    def delete_selected_trigger(self):
        return placement_triggers.delete_selected_trigger(self)

    def render_trigger_placement_overlay(self, screen):
        """Delegate trigger placement overlay to helper."""
        TriggerOverlayRenderer.render_trigger_placement_overlay(self, screen)

    def render_triggers_overlay(self, screen):
        """Delegate trigger debug overlay to helper."""
        TriggerOverlayRenderer.render_triggers(self, screen)

    def render_trigger_action_menu(self, screen):
        return scene_render.render_trigger_action_menu(self, screen)

    ############################################################
    # Personal Trigger Interaction (Vehicles, Pedestrians, Lights)
    ############################################################

    def _get_personal_trigger_payload(self, selection):
        return placement_triggers._get_personal_trigger_payload(self, selection)

    def _personal_trigger_hit_radius(self, trigger_radius: float) -> float:
        return placement_triggers._personal_trigger_hit_radius(self, trigger_radius)

    def _is_personal_trigger_hovered(self, selection, mouse_pos) -> bool:
        return placement_triggers._is_personal_trigger_hovered(self, selection, mouse_pos)

    def _ensure_traffic_light_trigger_context(
        self,
        selection,
        *,
        auto_select: bool = True,
    ) -> Optional[TrafficLightGroupData]:
        return placement_triggers._ensure_traffic_light_trigger_context(self, selection, auto_select=auto_select)

    def select_personal_trigger(self, selection):
        return placement_triggers.select_personal_trigger(self, selection)

    def clear_personal_trigger_selection(self):
        return placement_triggers.clear_personal_trigger_selection(self)

    def update_personal_trigger_menu_position(self, force: bool = False) -> None:
        """Update floating menu anchor for the selected personal trigger."""
        selection = self.selected_personal_trigger
        if not selection:
            self.personal_trigger_menu_position = None
            return

        if not force:
            current_time = time.time()
            if self.camera_is_moving and (current_time - self.camera_movement_timer < self.menu_update_delay):
                return

        center, _ = self._get_personal_trigger_payload(selection)
        if not center:
            self.clear_personal_trigger_selection()
            return

        screen_pos = self.coordinate_detector.world_to_screen_coordinates(
            center['x'], center['y'], center['z']
        )
        if screen_pos['success']:
            self.personal_trigger_menu_position = (int(screen_pos['x']), int(screen_pos['y']))
            self.personal_trigger_menu_hidden_for_camera_pan = False
        else:
            self.personal_trigger_menu_position = None

    def handle_personal_trigger_click(self, screen_x: int, screen_y: int) -> bool:
        return placement_triggers.handle_personal_trigger_click(self, screen_x, screen_y)

    def _hit_test_personal_trigger(self, screen_x: int, screen_y: int):
        return placement_triggers._hit_test_personal_trigger(self, screen_x, screen_y)

    def start_personal_trigger_movement(self, mouse_pos):
        return placement_triggers.start_personal_trigger_movement(self, mouse_pos)

    def update_personal_trigger_movement(self, mouse_pos):
        return placement_triggers.update_personal_trigger_movement(self, mouse_pos)

    def stop_personal_trigger_movement(self):
        return placement_triggers.stop_personal_trigger_movement(self)

    def start_selected_personal_trigger_scaling(self, mouse_pos):
        return placement_triggers.start_selected_personal_trigger_scaling(self, mouse_pos)

    def delete_selected_personal_trigger(self):
        return placement_triggers.delete_selected_personal_trigger(self)

    def render_personal_trigger_action_menu(self, screen):
        return scene_render.render_personal_trigger_action_menu(self, screen)

    def render_personal_trigger_links(self, screen):
        return scene_render.render_personal_trigger_links(self, screen)

    ############################################################
    # Scenario Persistence
    ############################################################

    def get_vehicle_waypoints(self, vehicle_id):
        return self.scene.get_vehicle_waypoints(vehicle_id)

    def set_vehicle_waypoints(self, vehicle_id, waypoints):
        return self.scene.set_vehicle_waypoints(vehicle_id, waypoints)

    def clear_vehicle_waypoints(self, vehicle_id):
        return self.scene.clear_vehicle_waypoints(vehicle_id)

    def get_vehicle_speed(self, vehicle_id, default=50):
        return self.scene.get_vehicle_speed(vehicle_id, default)

    def set_vehicle_speed(self, vehicle_id, value):
        return self.scene.set_vehicle_speed(vehicle_id, value)

    def clear_vehicle_speed(self, vehicle_id):
        return self.scene.clear_vehicle_speed(vehicle_id)

    def get_vehicle_destination_speed(self, vehicle_id):
        return self.scene.get_vehicle_destination_speed(vehicle_id)

    def set_vehicle_destination_speed(self, vehicle_id, value):
        return self.scene.set_vehicle_destination_speed(vehicle_id, value)

    def clear_vehicle_destination_speed(self, vehicle_id):
        return self.scene.clear_vehicle_destination_speed(vehicle_id)

    def get_vehicle_max_lat_acc(self, vehicle_id, default=3.0):
        return self.scene.get_vehicle_max_lat_acc(vehicle_id, default)

    def set_vehicle_max_lat_acc(self, vehicle_id, value):
        return self.scene.set_vehicle_max_lat_acc(vehicle_id, value)

    def clear_vehicle_max_lat_acc(self, vehicle_id):
        return self.scene.clear_vehicle_max_lat_acc(vehicle_id)

    def get_actor_idle_time(self, actor_id, default=0.0):
        return self.scene.get_actor_idle_time(actor_id, default)

    def set_actor_idle_time(self, actor_id, value):
        return self.scene.set_actor_idle_time(actor_id, value)

    def clear_actor_idle_time(self, actor_id):
        return self.scene.clear_actor_idle_time(actor_id)

    def get_vehicle_color(self, vehicle_id):
        return self.scene.get_vehicle_color(vehicle_id)

    def set_vehicle_color(self, vehicle_id, color):
        return self.scene.set_vehicle_color(vehicle_id, color)

    def clear_vehicle_color(self, vehicle_id):
        return self.scene.clear_vehicle_color(vehicle_id)

    def get_vehicle_ignore_flags(self, vehicle_id) -> VehicleIgnoreFlags:
        return self.scene.get_vehicle_ignore_flags(vehicle_id)

    def set_vehicle_ignore_flags(self, vehicle_id, flags):
        return self.scene.set_vehicle_ignore_flags(vehicle_id, flags)

    def clear_vehicle_ignore_flags(self, vehicle_id):
        return self.scene.clear_vehicle_ignore_flags(vehicle_id)

    def _ensure_waypoint_container(self, vehicle_id):
        return self.scene._ensure_waypoint_container(vehicle_id)

    def append_waypoint_data(self, vehicle_id, waypoint):
        return self.scene.append_waypoint_data(vehicle_id, waypoint)

    def insert_waypoint_data(self, vehicle_id, index, waypoint):
        waypoints = self._ensure_waypoint_container(vehicle_id)
        if index < 0 or index > len(waypoints):
            return None
        waypoint_copy = clone_waypoint_data(waypoint)
        waypoints.insert(index, waypoint_copy)
        # If we inserted at the front for a pedestrian, reorient to the new first waypoint
        if index == 0 and self._is_pedestrian_actor(vehicle_id):
            self._adjust_pedestrian_spawn_orientation(vehicle_id)
        return waypoint_copy

    def remove_waypoint_data(self, vehicle_id, index):
        waypoints = self.waypoint_list.get(vehicle_id)
        if not waypoints or index < 0 or index >= len(waypoints):
            return None
        removed = clone_waypoint_data(waypoints.pop(index))
        if not waypoints:
            self.clear_vehicle_waypoints(vehicle_id)
        else:
            # If the first waypoint changed (e.g., deleted index 0), reorient pedestrians
            if index == 0 and self._is_pedestrian_actor(vehicle_id):
                self._adjust_pedestrian_spawn_orientation(vehicle_id)
        return removed

    def update_waypoint_fields(self, vehicle_id, index, updates: Dict[str, Union[float, bool, str, None]]):
        return self.scene.update_waypoint_fields(vehicle_id, index, updates)

    def get_spawned_vehicle(self, vehicle_id: Optional[int]):
        """Return the live CARLA actor for a tracked vehicle id (None if missing or dead)."""
        if vehicle_id is None:
            return None
        for actor in self.spawned_vehicles:
            if actor and actor.is_alive and actor.id == vehicle_id:
                return actor
        return None

    def assert_spawned_vehicle(self, vehicle_id: Optional[int], context: str = "") -> bool:
        """Log a warning (and return False) when a command targets a missing actor."""
        vehicle = self.get_spawned_vehicle(vehicle_id)
        if vehicle:
            return True
        if context:
            print(f"[WARN] {context}: vehicle {vehicle_id} no longer available.")
        else:
            print(f"[WARN] Vehicle {vehicle_id} no longer available.")
        return False

    def initialize_vehicle_metadata(
        self,
        vehicle_id,
        *,
        speed,
        destination_speed=None,
        idle_time=0.0,
        color=None,
        ignore_flags=None,
        max_lat_acc=3.0,
    ):
        return self.scene.initialize_vehicle_metadata(vehicle_id, speed=speed, destination_speed=destination_speed, idle_time=idle_time, color=color, ignore_flags=ignore_flags, max_lat_acc=max_lat_acc)

    def clear_vehicle_metadata(self, vehicle_id, *, clear_waypoints=False):
        return self.scene.clear_vehicle_metadata(vehicle_id, clear_waypoints=clear_waypoints)

    def clear_all_vehicle_metadata(self):
        return self.scene.clear_all_vehicle_metadata()

    def _cleanup_leftover_ego_actor(self):
        """Destroy any ego-designated actor the editor no longer tracks."""
        if not self.world:
            return
        try:
            actors = self.world.get_actors().filter('vehicle.*')
        except RuntimeError:
            return

        tracked_ids = {actor.id for actor in self.spawned_vehicles if actor and actor.is_alive}
        roles = {'hero', 'ego', 'ego_vehicle', 'player'}
        preserved_id = None
        editor = getattr(self, 'editor', None)
        if editor:
            preserved_id = self.session.external_ego_actor_id

        for actor in actors:
            if actor.id in tracked_ids:
                continue
            role_name = actor.attributes.get('role_name', '').lower()
            if role_name in roles:
                if preserved_id and actor.id == preserved_id:
                    print(f"Preserving external ego actor {actor.id} (role '{role_name}') during cleanup.")
                    continue
                try:
                    print(f"Destroying leftover ego actor {actor.id} before reload")
                    actor.destroy()
                except Exception as exc:
                    print(f"Failed to destroy leftover ego actor {actor.id}: {exc}")
                break

    def is_manual_control_actor(self, actor_id: Optional[int]) -> bool:
        return manual_control.is_manual_control_actor(self, actor_id)

    def _serialize_waypoints_for_vehicle(self, vehicle_id, *, start_location: Optional[carla.Transform] = None):
        return scenario_io._serialize_waypoints_for_vehicle(self, vehicle_id, start_location=start_location)






    def _collect_scenario_snapshot(self, map_name):
        return scenario_io._collect_scenario_snapshot(self, map_name)



    # -- scenario_io seam delegates (step-22, thinned step-25) ---------
    # Only externally-called seams remain (editor / CIP body / regress
    # harness callers); intra-package calls were retargeted to direct
    # module calls in step-25.

    def save_waypoint_data_to_file(self, filename):
        return scenario_io.save_waypoint_data_to_file(self, filename)

    def export_to_openscenario(self, filename):
        return xosc_export.export_to_openscenario(self, filename)

    def _reset_scenario_state(self):
        """Clear currently loaded scenario actors, waypoints, and triggers."""
        editor = getattr(self, 'editor', None)
        self.session.scene_preview_destroyed = False
        clear_bounding_box_cache()  # scenario actors are going away; drop their cached boxes
        if editor:
            try:
                editor._detect_external_ego_vehicle()
                editor._refresh_external_ego_actor_reference()
            except Exception as exc:
                print(f"[Scenario] Warning: failed to probe external ego vehicle before reset ({exc})")
        self.cleanup_all_vehicles()
        self._cleanup_leftover_ego_actor()
        self.waypoint_list.clear()
        self.triggers.clear()
        self.traffic_light_trigger_centers.clear()
        self.traffic_light_trigger_radii.clear()
        self.traffic_light_sequences.clear()
        self._traffic_light_group_snapshots.clear()
        self._scenario_active_traffic_light_trigger = None
        self._last_visible_traffic_light_trigger_key = None
        self.scaling_traffic_light_trigger = False
        self._traffic_light_scaling_group = None
        self.traffic_light_menu_position = None
        for group in self.traffic_light_groups:
            group.trigger_center = None
            group.trigger_radius = None
            group.sequence = []
        self.clear_traffic_light_selection()







    def load_waypoint_data_from_file(
        self,
        filename,
        *,
        preserve_camera: bool = False,
        skip_ego_spawn: bool = False,
        apply_to_actor_only: bool = False,
        external_ego_actor: Optional[carla.Actor] = None,
        preserved_actor_id: Optional[int] = None,
    ):
        return scenario_io.load_waypoint_data_from_file(self, filename, preserve_camera=preserve_camera, skip_ego_spawn=skip_ego_spawn, apply_to_actor_only=apply_to_actor_only, external_ego_actor=external_ego_actor, preserved_actor_id=preserved_actor_id)

    def get_actor_original_json_position(self, actor_id, actor_type):
        """Get the original position of an actor from the loaded JSON data"""
        if not hasattr(self, 'loaded_scenario_data') or not self.loaded_scenario_data:
            return None

        # Search for the actor in the loaded JSON data by type
        for vehicle_data in self.loaded_scenario_data.get('vehicles', []):
            if vehicle_data.get('type') == actor_type:
                location_data = vehicle_data.get('location', {})
                rotation_data = vehicle_data.get('rotation', {})

                # Create CARLA location and rotation objects
                location = carla.Location(
                    location_data.get('x', 0),
                    location_data.get('y', 0),
                    location_data.get('z', 0)
                )
                rotation = carla.Rotation(
                    rotation_data.get('pitch', 0),
                    rotation_data.get('yaw', 0),
                    rotation_data.get('roll', 0)
                )

                print(f"Found original JSON position for {actor_type}: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")
                return carla.Transform(location, rotation)

        ego_data = self.loaded_scenario_data.get('ego_vehicle')
        if ego_data and ego_data.get('type') == actor_type:
            location = carla.Location(
                ego_data.get('location', {}).get('x', 0),
                ego_data.get('location', {}).get('y', 0),
                ego_data.get('location', {}).get('z', 0)
            )
            rotation = carla.Rotation(
                ego_data.get('rotation', {}).get('pitch', 0),
                ego_data.get('rotation', {}).get('yaw', 0),
                ego_data.get('rotation', {}).get('roll', 0)
            )
            print(f"Found original JSON position for ego {actor_type}: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")
            return carla.Transform(location, rotation)

        print(f"No JSON position found for actor type: {actor_type}")
        return None

    def get_latest_image(self):
        """Get the latest camera image scaled to the current screen size if needed."""
        if not self.latest_image:
            return None

        target_size = (int(self.screen_width), int(self.screen_height))
        if target_size[0] <= 0 or target_size[1] <= 0:
            return self.latest_image

        if self.latest_image.get_size() == target_size:
            return self.latest_image

        if (
            self._scaled_image_cache is not None
            and self._scaled_image_size == target_size
            and self._scaled_image_frame == self.latest_frame_id
        ):
            return self._scaled_image_cache

        scaled = pygame.transform.smoothscale(self.latest_image, target_size)
        self._scaled_image_cache = scaled
        self._scaled_image_size = target_size
        self._scaled_image_frame = self.latest_frame_id
        return scaled
    
    def handle_mouse_click(self, screen_x, screen_y, move_camera=False):
        return placement.handle_mouse_click(self, screen_x, screen_y, move_camera)
    
    def _hit_test_displayed_waypoint(self, screen_x: int, screen_y: int):
        return placement._hit_test_displayed_waypoint(self, screen_x, screen_y)

    def handle_waypoint_click(self, screen_x, screen_y):
        return placement.handle_waypoint_click(self, screen_x, screen_y)

    def split_waypoint_at_click(self, screen_x: int, screen_y: int) -> bool:
        return placement.split_waypoint_at_click(self, screen_x, screen_y)
    
    def handle_waypoint_mouse_movement(self, screen_x, screen_y):
        return placement.handle_waypoint_mouse_movement(self, screen_x, screen_y)
    
    def handle_waypoint_mouse_release(self):
        return placement.handle_waypoint_mouse_release(self)
        # Keep the waypoint selected (don't clear selected_waypoint_vehicle_id and selected_waypoint_index)
        # This allows for quick re-selection and maintains selection for other operations like deletion

    def _cache_destination_speed(self, vehicle_id):
        return placement._cache_destination_speed(self, vehicle_id)

    def _start_waypoint_extension_mode(self, vehicle_id):
        return placement._start_waypoint_extension_mode(self, vehicle_id)

    def _is_last_waypoint(self, vehicle_id, waypoint_index):
        return placement._is_last_waypoint(self, vehicle_id, waypoint_index)

    def update_waypoint_movement(self, mouse_pos):
        """Update waypoint position based on mouse movement (used by main event loop)"""
        self.handle_waypoint_mouse_movement(mouse_pos[0], mouse_pos[1])
    
    def stop_waypoint_movement(self):
        """Stop waypoint movement (used by main event loop)"""
        self.handle_waypoint_mouse_release()
    
    def delete_selected_waypoint(self):
        return placement.delete_selected_waypoint(self)
    
    def select_vehicle_actor(
        self,
        actor: Optional[carla.Actor],
        *,
        focus_camera: bool = False,
        fallback_screen_pos: Optional[Tuple[int, int]] = None,
    ) -> bool:
        return placement.select_vehicle_actor(self, actor, focus_camera=focus_camera, fallback_screen_pos=fallback_screen_pos)

    def handle_vehicle_click(self, screen_x, screen_y):
        return placement.handle_vehicle_click(self, screen_x, screen_y)

    def handle_traffic_light_click(self, screen_x: int, screen_y: int) -> bool:
        return traffic_lights.handle_traffic_light_click(self, screen_x, screen_y)

    def handle_traffic_light_action_click(self, screen_x: int, screen_y: int) -> bool:
        return traffic_lights.handle_traffic_light_action_click(self, screen_x, screen_y)

    def _get_traffic_light_menu_icons(self, group: Optional[TrafficLightGroupData]) -> List[str]:
        return traffic_lights._get_traffic_light_menu_icons(self, group)

    def _get_traffic_light_group_menu_anchor(
        self, group: Optional[TrafficLightGroupData]
    ) -> Optional[Tuple[float, float]]:
        return traffic_lights._get_traffic_light_group_menu_anchor(self, group)

    def _update_traffic_light_menu_anchor(self, group: Optional[TrafficLightGroupData]) -> None:
        return traffic_lights._update_traffic_light_menu_anchor(self, group)

    def _compute_traffic_light_group_trigger_center(
        self, group: TrafficLightGroupData
    ) -> Optional[carla.Location]:
        return traffic_lights._compute_traffic_light_group_trigger_center(self, group)


    def start_traffic_light_trigger_scaling(self, mouse_pos: Tuple[int, int]) -> None:
        return traffic_lights.start_traffic_light_trigger_scaling(self, mouse_pos)

    def update_traffic_light_trigger_scaling(self, mouse_pos: Tuple[int, int]) -> None:
        return traffic_lights.update_traffic_light_trigger_scaling(self, mouse_pos)

    def stop_traffic_light_trigger_scaling(self) -> None:
        return traffic_lights.stop_traffic_light_trigger_scaling(self)

    def delete_traffic_light_trigger(
        self, group: Optional[TrafficLightGroupData] = None
    ) -> bool:
        return traffic_lights.delete_traffic_light_trigger(self, group)

    def _ensure_pedestrian_trigger(self, pedestrian_id: int) -> bool:
        return placement_triggers._ensure_pedestrian_trigger(self, pedestrian_id)

    def start_pedestrian_trigger_scaling(self, pedestrian_id: int, mouse_pos: tuple) -> None:
        return placement_triggers.start_pedestrian_trigger_scaling(self, pedestrian_id, mouse_pos)

    def update_pedestrian_trigger_scaling(self, mouse_pos: tuple) -> None:
        return placement_triggers.update_pedestrian_trigger_scaling(self, mouse_pos)

    def stop_pedestrian_trigger_scaling(self) -> None:
        return placement_triggers.stop_pedestrian_trigger_scaling(self)

    def delete_pedestrian_trigger(self, pedestrian_id: int) -> bool:
        return placement_triggers.delete_pedestrian_trigger(self, pedestrian_id)

    def _ensure_vehicle_trigger(self, vehicle_id: int) -> bool:
        return placement_triggers._ensure_vehicle_trigger(self, vehicle_id)

    def start_vehicle_trigger_scaling(self, vehicle_id: int, mouse_pos: tuple) -> None:
        return placement_triggers.start_vehicle_trigger_scaling(self, vehicle_id, mouse_pos)

    def update_vehicle_trigger_scaling(self, mouse_pos: tuple) -> None:
        return placement_triggers.update_vehicle_trigger_scaling(self, mouse_pos)

    def stop_vehicle_trigger_scaling(self) -> None:
        return placement_triggers.stop_vehicle_trigger_scaling(self)

    def delete_vehicle_trigger(self, vehicle_id: int) -> bool:
        return placement_triggers.delete_vehicle_trigger(self, vehicle_id)

    def _cache_traffic_light_sequence(self, group: Optional[TrafficLightGroupData]) -> None:
        return traffic_lights._cache_traffic_light_sequence(self, group)

    def _normalize_traffic_light_sequence(
        self,
        sequence: Optional[Iterable[Dict[str, Union[str, float, int]]]],
        *,
        coerce_color: bool = False,
    ) -> List[Dict[str, Union[str, float, int]]]:
        return traffic_lights._normalize_traffic_light_sequence(self, sequence, coerce_color=coerce_color)


    def _normalize_traffic_light_fingerprint(self, fingerprint):
        # Delegates to the shared implementation in vse_common/traffic_lights.py
        return normalize_traffic_light_fingerprint(fingerprint)

    def _traffic_light_trigger_key(
        self,
        group: Optional[TrafficLightGroupData] = None,
        *,
        fingerprint: Optional[Iterable[Tuple[int, int, int]]] = None,
        ids: Optional[Iterable[int]] = None,
    ) -> Optional[Tuple[str, Tuple]]:
        return traffic_lights._traffic_light_trigger_key(self, group, fingerprint=fingerprint, ids=ids)

    def _get_traffic_light_trigger_data(
        self,
        *,
        key: Optional[Tuple[str, Tuple]] = None,
        group: Optional[TrafficLightGroupData] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[Tuple[str, Tuple]]]:
        return traffic_lights._get_traffic_light_trigger_data(self, key=key, group=group)

    def _set_traffic_light_trigger_data(
        self,
        center: Dict[str, float],
        radius: float,
        *,
        key: Optional[Tuple[str, Tuple]] = None,
        group: Optional[TrafficLightGroupData] = None,
        mark_visible: bool = True,
    ) -> Optional[Tuple[str, Tuple]]:
        return traffic_lights._set_traffic_light_trigger_data(self, center, radius, key=key, group=group, mark_visible=mark_visible)

    def _delete_traffic_light_trigger_data(
        self,
        *,
        key: Optional[Tuple[str, Tuple]] = None,
        group: Optional[TrafficLightGroupData] = None,
    ) -> bool:
        return traffic_lights._delete_traffic_light_trigger_data(self, key=key, group=group)

    def _find_traffic_light_group_by_key(
        self,
        key: Optional[Tuple[str, Tuple]],
    ) -> Optional[TrafficLightGroupData]:
        return traffic_lights._find_traffic_light_group_by_key(self, key)

    # Backwards-compatible wrappers (legacy calls still reference these)
    def _cache_traffic_light_trigger_payload(self, group: Optional[TrafficLightGroupData]) -> None:
        return traffic_lights._cache_traffic_light_trigger_payload(self, group)


    def _mark_last_visible_traffic_light_trigger(self, group: Optional[TrafficLightGroupData], *, key=None) -> None:
        return traffic_lights._mark_last_visible_traffic_light_trigger(self, group, key=key)

    def select_traffic_light_group(self, group: TrafficLightGroupData) -> bool:
        return traffic_lights.select_traffic_light_group(self, group)

    def clear_traffic_light_selection(self) -> None:
        return traffic_lights.clear_traffic_light_selection(self)
    
    def spawn_vehicle_at_marker(self, vehicle_type, coordinates, *, role="npc"):
        return placement.spawn_vehicle_at_marker(self, vehicle_type, coordinates, role=role)

    def render_selected_actor_highlight(self, screen):
        return scene_render.render_selected_actor_highlight(self, screen)

    def _get_traffic_light_rectangle_points(
        self, traffic_light: carla.TrafficLight
    ) -> Optional[Tuple[List[carla.Location], Optional[carla.Location]]]:
        return scene_render._get_traffic_light_rectangle_points(self, traffic_light)


    _is_point_inside_polygon = staticmethod(scene_render._is_point_inside_polygon)

    def _get_traffic_light_group_screen_polygon(
        self, group: TrafficLightGroupData
    ) -> Optional[Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]]:
        return scene_render._get_traffic_light_group_screen_polygon(self, group)

    _compute_convex_hull = staticmethod(scene_render._compute_convex_hull)

    _is_point_near_segment = staticmethod(scene_render._is_point_near_segment)

    def render_selected_traffic_light_connectors(self, screen):
        return scene_render.render_selected_traffic_light_connectors(self, screen)


    def render_traffic_light_markers(self, screen):
        return scene_render.render_traffic_light_markers(self, screen)

    def render_selected_traffic_light_highlight(self, screen):
        return scene_render.render_selected_traffic_light_highlight(self, screen)

    def render_traffic_light_action_menu(self, screen):
        return scene_render.render_traffic_light_action_menu(self, screen)

    def render_traffic_light_triggers_overlay(self):
        return carla_debug.render_traffic_light_triggers_overlay(self)

    def render_pedestrian_triggers_overlay(self):
        return carla_debug.render_pedestrian_triggers_overlay(self)

    def render_vehicle_triggers_overlay(self):
        return carla_debug.render_vehicle_triggers_overlay(self)

    def render_vehicle_action_menu(self, screen):
        return scene_render.render_vehicle_action_menu(self, screen)
    
    def render_waypoint_creation_overlay(self, screen):
        return scene_render.render_waypoint_creation_overlay(self, screen)

    def render_destination_creation_overlay(self, screen):
        return scene_render.render_destination_creation_overlay(self, screen)


    def render_group_selection(self, screen):
        return scene_render.render_group_selection(self, screen)

    def render_action_menus(self, screen):
        return scene_render.render_action_menus(self, screen)

    def render_all_overlays(self, screen):
        return scene_render.render_all_overlays(self, screen)


    def render_waypoints_overlay(self, screen):
        """Delegate waypoint hover overlay rendering to helper."""
        WaypointOverlayRenderer.render_waypoints_overlay(self, screen)


    def render_opendrive_lanes_overlay(self, screen):
        """Delegate OpenDRIVE overlay rendering to helper."""
        if self.opendrive_overlay_hidden_for_camera_pan:
            return
        OpenDriveOverlayRenderer.render(self, screen)
    
    def precompute_opendrive_lane_data(self):
        """Delegate lane data precomputation to helper."""
        OpenDriveOverlayRenderer.precompute_lane_data(self)
    
    
    def toggle_opendrive_overlay(self):
        """Toggle OpenDRIVE overlay via helper."""
        OpenDriveOverlayRenderer.toggle_overlay(self)
    
    
    
    def update_vehicle_id_references(self, old_id, new_id, new_actor=None):
        """Update vehicle ID references in commands after vehicle respawn"""
        if new_actor is None:
            new_actor = next(
                (actor for actor in self.spawned_vehicles
                 if actor and actor.id == new_id and actor.is_alive),
                None,
            )
        waypoints = self.get_vehicle_waypoints(old_id)
        if waypoints:
            self.clear_vehicle_waypoints(old_id)
            self.set_vehicle_waypoints(new_id, waypoints)

        if old_id in self.vehicle_speeds:
            speed = self.vehicle_speeds[old_id]
            self.clear_vehicle_speed(old_id)
            self.set_vehicle_speed(new_id, speed)

        if old_id in self.vehicle_destination_speeds:
            dest_speed = self.vehicle_destination_speeds[old_id]
            self.clear_vehicle_destination_speed(old_id)
            self.set_vehicle_destination_speed(new_id, dest_speed)

        if old_id in self.vehicle_max_lat_acc:
            lat_acc = self.vehicle_max_lat_acc.pop(old_id)
            self.vehicle_max_lat_acc[new_id] = lat_acc

        if old_id in self.vehicle_colors:
            color = self.vehicle_colors[old_id]
            self.clear_vehicle_color(old_id)
            self.set_vehicle_color(new_id, color)

        if old_id in self.actor_idle_times:
            idle = self.actor_idle_times[old_id]
            self.clear_actor_idle_time(old_id)
            self.set_actor_idle_time(new_id, idle)

        if old_id in self.vehicle_trigger_centers:
            center = self.vehicle_trigger_centers.pop(old_id)
            self.vehicle_trigger_centers[new_id] = center
        if old_id in self.vehicle_trigger_radii:
            radius = self.vehicle_trigger_radii.pop(old_id)
            self.vehicle_trigger_radii[new_id] = radius
        if old_id in self.pedestrian_trigger_centers:
            center = self.pedestrian_trigger_centers.pop(old_id)
            self.pedestrian_trigger_centers[new_id] = center
        if old_id in self.pedestrian_trigger_radii:
            radius = self.pedestrian_trigger_radii.pop(old_id)
            self.pedestrian_trigger_radii[new_id] = radius
        if old_id in self.vehicle_transforms:
            transform = self.vehicle_transforms.pop(old_id)
            self.vehicle_transforms[new_id] = transform
        if old_id in self._actor_location_cache:
            cached_location = self._actor_location_cache.pop(old_id)
            self._actor_location_cache[new_id] = cached_location

        flags = self.get_vehicle_ignore_flags(old_id)
        self.clear_vehicle_ignore_flags(old_id)
        if any(flags.values()):
            self.set_vehicle_ignore_flags(new_id, flags)

        if old_id in self.selected_actor_ids:
            self.selected_actor_ids.discard(old_id)
            self.selected_actor_ids.add(new_id)

        for selection_name in ('selected_personal_trigger', '_personal_trigger_move_target'):
            selection = getattr(self, selection_name, None)
            if isinstance(selection, dict) and selection.get('id') == old_id:
                selection['id'] = new_id

        if self._pedestrian_scaling_id == old_id:
            self._pedestrian_scaling_id = new_id
        if self._vehicle_scaling_id == old_id:
            self._vehicle_scaling_id = new_id
        if getattr(self, '_pending_large_map_focus_actor_id', None) == old_id:
            self._pending_large_map_focus_actor_id = new_id
        if self.waypoint_vehicle and self.waypoint_vehicle.id == old_id and new_actor:
            self.waypoint_vehicle = new_actor
        if self.selected_waypoint_group and self.selected_waypoint_group.get('vehicle_id') == old_id:
            self.selected_waypoint_group['vehicle_id'] = new_id

        if self.ego_vehicle_id == old_id:
            self.ego_vehicle_id = new_id
            if self.manual_control_actor and self.manual_control_actor.id == old_id:
                # The ego respawned under a new id; re-pend so the finder re-attaches to it
                # (it checks ego_vehicle_id first) instead of dropping manual control for good.
                self.request_manual_control()

        # Update selection and UI references
        if self.selected_vehicle and self.selected_vehicle.id == old_id:
            self.selected_vehicle = new_actor
            self.selected_vehicle_is_pedestrian = bool(new_actor and new_actor.type_id.startswith('walker.'))
            if new_actor:
                self.refresh_selected_vehicle_ui()
            else:
                self.vehicle_menu_position = None

        if self.waypoint_display_vehicle_id == old_id:
            self.waypoint_display_vehicle_id = new_id

        if self.selected_waypoint_vehicle_id == old_id:
            self.selected_waypoint_vehicle_id = new_id

        panel = getattr(getattr(self, "editor", None), "info_panel", None)
        if (panel and panel.visible and panel.object_type in ('vehicle', 'pedestrian')
            and panel.selected_object
            and getattr(panel.selected_object, 'id', None) == old_id):
            if new_actor:
                panel.show(new_actor,
                           'pedestrian' if new_actor.type_id.startswith('walker.') else 'vehicle',
                           self.editor.screen_width,
                           self.editor.screen_height)
            else:
                panel.hide()
    
        # Update commands in both stacks from the editor
        if self.editor:
            self.editor.history.remap_vehicle_id(old_id, new_id, new_actor)
    
    def cleanup(self):
        """Cleanup camera sensor and visualization"""
        self._unregister_pedestrian_highlight_tick()
        if self.camera_sensor:
            # stop() before destroy(): CARLA 0.9.16 rewrote sensor streaming so a listener still
            # live when its stream is invalidated -- e.g. an external world reload (this cleanup()
            # runs on the [World Reset] recovery path, by which point the reload already destroyed
            # the actor server-side) -- spins reconnecting to the dead stream and floods the server
            # with "Invalid session" errors (~12k/s, measured). destroy() alone does NOT reclaim the
            # orphaned client reader once the actor is gone; stop() does. Harmless if still alive.
            try:
                self.camera_sensor.stop()
            except Exception:
                pass
            # Guard destroy(): against a dead/crashed server this RPC blocks on
            # the client timeout (F7). The sensor dies with the server anyway.
            try:
                self.camera_sensor.destroy()
            except Exception as exc:
                print(f"[Cleanup] camera sensor destroy skipped: {exc}")
            self.camera_sensor = None

        # Clear coordinate detector's cached world map to avoid stale references
        if hasattr(self, 'coordinate_detector') and self.coordinate_detector:
            self.coordinate_detector.world_map = None

        # Clear traffic light references
        if self.traffic_lights:
            self.traffic_lights.clear()
        if self.traffic_light_groups:
            self.traffic_light_groups.clear()
        self._traffic_light_group_lookup.clear()
        self._traffic_light_rectangles.clear()
        self.clear_traffic_light_selection()

        # Clean up all spawned vehicles
        self.cleanup_all_vehicles()
    
    def cleanup_all_vehicles(self, preserve_ids=None, preserve_ego=False):
        """Remove all editor-spawned vehicles, optionally preserving specific actor ids."""
        preserve_ids = set(preserve_ids or [])
        editor = getattr(self, "editor", None)
        if editor:
            external_id = self.session.external_ego_actor_id
            swap_id = self.session._external_swap_current_id
            if swap_id:
                external_id = swap_id
            if external_id is not None:
                preserve_ids.add(external_id)
        preserved_actors = []
        vehicles_destroyed = 0

        if self.spawned_vehicles:
            print(f"Cleaning up {len(self.spawned_vehicles)} spawned vehicles...")

            # Work on a copy so we can mutate the original list in place.
            current_spawned = list(self.spawned_vehicles)
            preserved_actors = []
            destroy_pairs = []
            destroy_fallback = []

            for vehicle in current_spawned:
                actor_id = None
                try:
                    actor_id = vehicle.id if vehicle else None
                except Exception:
                    actor_id = None

                keep_actor = actor_id is not None and actor_id in preserve_ids

                if keep_actor:
                    preserved_actors.append(vehicle)
                    continue

                if actor_id is not None:
                    try:
                        self.clear_vehicle_metadata(int(actor_id), clear_waypoints=True)
                    except Exception:
                        pass

                if vehicle and getattr(vehicle, "is_alive", False):
                    if actor_id is not None:
                        try:
                            destroy_pairs.append((int(actor_id), vehicle))
                        except Exception:
                            destroy_fallback.append(vehicle)
                    else:
                        destroy_fallback.append(vehicle)

            if destroy_pairs:
                client = self.session.client
                if client is not None and hasattr(client, "apply_batch_sync"):
                    try:
                        batch = [carla.command.DestroyActor(actor_id) for actor_id, _actor in destroy_pairs]
                        responses = client.apply_batch_sync(batch, False)
                        if responses:
                            for response in responses:
                                try:
                                    if not getattr(response, "error", None):
                                        vehicles_destroyed += 1
                                except Exception:
                                    vehicles_destroyed += 1
                        else:
                            vehicles_destroyed += len(destroy_pairs)
                    except Exception as exc:
                        print(f"Error destroying vehicles via batch: {exc}")
                        destroy_fallback.extend([actor for _actor_id, actor in destroy_pairs])
                else:
                    destroy_fallback.extend([actor for _actor_id, actor in destroy_pairs])

            for actor in destroy_fallback:
                try:
                    if actor and getattr(actor, "is_alive", False):
                        actor.destroy()
                        vehicles_destroyed += 1
                except Exception as exc:
                    print(f"Error destroying vehicle: {exc}")
            # Keep the same list object that other systems (like the coordinate detector)
            # hold on to, so in-place operations remain visible everywhere.
            self.spawned_vehicles.clear()
            self.spawned_vehicles.extend(preserved_actors)

            if vehicles_destroyed > 0:
                print(f"Successfully destroyed {vehicles_destroyed} vehicles")
            else:
                if preserved_actors:
                    print("No vehicles destroyed; preserving requested actors.")
                else:
                    print("No vehicles to clean up")
        else:
            print("No vehicles to clean up")

        # Reset selection state if preserved actors do not include the selected vehicle
        if (
            self.selected_vehicle
            and self.selected_vehicle.id not in preserve_ids
        ):
            self.clear_vehicle_selection()
        elif not self.selected_vehicle:
            self.clear_vehicle_selection()

        if (
            self.waypoint_display_vehicle_id is not None
            and self.waypoint_display_vehicle_id not in preserve_ids
        ):
            self.waypoint_display_vehicle_id = None

        if not preserve_ids:
            self.clear_all_vehicle_metadata()

        if (
            self.ego_vehicle_id
            and (self.ego_vehicle_id not in preserve_ids or not preserve_ego)
        ):
            self.clear_ego_vehicle()

        self.disable_manual_control()
        self.pedestrian_colors.clear()

        # Ensure the coordinate detector always references the live list object.
        if hasattr(self, "coordinate_detector") and self.coordinate_detector:
            self.coordinate_detector.spawned_vehicles = self.spawned_vehicles


install_scene_forwarders(CameraImageProcessor)
install_scene_forwarders(CameraImageProcessor, SCENE_WEATHER_FIELDS)
