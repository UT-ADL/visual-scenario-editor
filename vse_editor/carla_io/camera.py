"""TopDownCamera (moved verbatim from vse.py): pan/zoom/height state with
ground-height-aware minimum altitude.

step-12 adds an editor-only 3D orbit view (view_mode == "orbit"): the camera circles a
ground pivot that reuses center_x/center_y, so every legacy code path that pokes the
top-down center (focus, large-map travel, restore) moves the pivot with no branches.
All top-down paths are byte-identical while view_mode == "topdown".
"""

from __future__ import annotations

import math
import time
from typing import Optional

import carla

from vse_common.geometry import get_ground_height

def clamp_orbit_pitch(pitch_deg: float, distance: float, pivot_z: float,
                      ground_z: Optional[float], margin: float = 2.0,
                      pitch_min: float = -85.0, pitch_max: float = -10.0) -> float:
    """Clamp an orbit pitch (deg, negative = looking down) into [pitch_min, pitch_max],
    steepening it further if needed so the camera eye (pivot_z + distance*sin(|pitch|))
    stays at least margin above ground_z (None falls back to the pivot plane)."""
    ground = pivot_z if ground_z is None else float(ground_z)
    need = (ground + margin) - pivot_z
    min_abs = 0.0
    if distance > 1e-6 and need > 0.0:
        min_abs = math.degrees(math.asin(min(1.0, need / distance)))
    # Signed clamp first so an over-the-horizon drag (positive pitch) lands on the
    # shallowest allowed tilt, then steepen for terrain clearance.
    pitch = min(max(pitch_deg, pitch_min), pitch_max)
    abs_pitch = min(max(abs(pitch), min_abs), abs(pitch_min))
    return -abs_pitch

def _sample_ground_or_none(world, x: float, y: float) -> Optional[float]:
    """Ground height at (x, y) from a real raycast hit, or None when nothing was found.
    get_ground_height's final fallback echoes the probe Z (0.0 here) — garbage a caller
    must not mistake for terrain (it sank the orbit camera on unstreamed Tartu tiles)."""
    if world is None:
        return None
    try:
        meta = get_ground_height(
            world,
            carla.Location(x, y, 0.0),
            debug=False,
            probe_on_miss=False,  # avoid slow grid probing during camera gestures
            return_metadata=True,
        )
    except Exception:
        return None
    if not isinstance(meta, dict) or meta.get('source') == 'fallback':
        return None
    try:
        return float(meta['height'])
    except Exception:
        return None

class TopDownCamera:
    """
    Controls the top-down camera position, zoom, and navigation for the map viewer.
    Provides transformation for CARLA camera sensor placement.
    """

    # 3D orbit view tuning knobs (editor-only; ignored while view_mode == "topdown").
    ORBIT_PITCH_MIN = -85.0      # deg: steepest tilt (near top-down; never flips over the pivot)
    ORBIT_PITCH_MAX = -10.0      # deg: shallowest tilt (keeps the horizon out of most of the frame)
    ORBIT_MIN_DISTANCE = 5.0     # m: closest wheel zoom to the pivot
    ORBIT_MAX_DISTANCE = 1000.0  # m: mirrors max_height; bounds large-map streaming-anchor jumps
    ORBIT_ZOOM_FACTOR = 1.15     # multiplicative wheel step (delta +1 -> x1.15 out, -1 -> /1.15 in)
    ORBIT_DEG_PER_PIXEL = 0.25   # right-drag orbit sensitivity (deg of yaw/pitch per screen px)
    ORBIT_GROUND_MARGIN = 2.0    # m: camera eye stays at least this far above sampled terrain

    def __init__(self, initial_x=0, initial_y=0, initial_height=200):
        self.center_x = initial_x
        self.center_y = initial_y
        self.height = initial_height

        # Navigation settings
        self.pan_speed = 5.0
        self.zoom_speed = 5.0
        self.min_height = 40.0
        self.max_height = 1000.0
        self.navigation_height_floor = 38.0  # Maintain consistent navigation speed when zoomed in

        # Debug: Track camera movement for raycast after stop
        self.is_moving = False
        self.world = None  # Will be set by CameraImageProcessor

        # Terrain-aware zoom cache
        self._cached_ground_height = None
        self._cached_ground_height_time = 0.0
        self._ground_height_cache_duration = 0.2  # seconds

        # 3D orbit view state (editor-only). The pivot's XY reuses center_x/center_y;
        # height is frozen while orbiting and only rewritten on exit_orbit().
        # "view_mode" is deliberately NOT named like playback_camera_mode -- that is a
        # different axis whose "topdown" string means the playback follow view.
        self.view_mode = "topdown"   # "topdown" | "orbit"
        self.pivot_z = 0.0           # ground Z under the pivot (world m)
        self.orbit_yaw = 270.0       # deg; 270 == top-down screen orientation
        self.orbit_pitch = -45.0     # deg, negative = looking down at the pivot
        self.orbit_distance = 200.0  # m camera eye -> pivot
        # Orbit ground caches, miss-aware and keyed by position + TTL. Kept separate
        # from _cached_ground_height, whose miss value (the probe Z) must keep its
        # legacy meaning for the top-down zoom floor. Entry: (x, y, ground_or_None);
        # attribute None = nothing cached yet.
        self._cached_eye_ground = None
        self._cached_eye_ground_time = 0.0
        self._cached_pivot_ground = None
        self._cached_pivot_ground_time = 0.0

    def get_carla_transform(self):
        """Get CARLA transform: straight down, or the orbit pose in the 3D view."""
        if self.view_mode == "orbit":
            return carla.Transform(
                self._orbit_location(),
                carla.Rotation(pitch=self.orbit_pitch, yaw=self.orbit_yaw, roll=0.0),
            )
        location = carla.Location(self.center_x, self.center_y, self.height)
        rotation = carla.Rotation(pitch=-90.0, yaw=270.0, roll=0.0)
        return carla.Transform(location, rotation)

    def view_state(self) -> tuple:
        """Hashable camera-pose signature; any change must invalidate projection caches."""
        if self.view_mode == "orbit":
            return ("orbit", self.center_x, self.center_y, self.pivot_z,
                    self.orbit_yaw, self.orbit_pitch, self.orbit_distance)
        return ("topdown", self.center_x, self.center_y, self.height)

    def enter_orbit(self) -> None:
        """Tilt into the 3D orbit view around the current top-down view center."""
        ground = self._get_terrain_height_at_pivot()
        self.pivot_z = float(ground) if ground is not None else 0.0
        self.orbit_yaw = 270.0
        self.orbit_pitch = -45.0
        self.orbit_distance = max(self.ORBIT_MIN_DISTANCE,
                                  min(self.ORBIT_MAX_DISTANCE, self.height - self.pivot_z))
        self.view_mode = "orbit"
        self._clamp_orbit_pose()

    def exit_orbit(self) -> None:
        """Return to top-down over the pivot at the equivalent visual height."""
        # Mirror the top-down zoom-in floor (terrain-aware: ground+5, min 5) instead of
        # the blunt min_height, so a close orbit zoom survives the Tab round trip.
        floor = max(self.pivot_z + 5.0, 5.0)
        self.height = max(floor, min(self.max_height, self.pivot_z + self.orbit_distance))
        self.view_mode = "topdown"

    def orbit(self, d_yaw_deg: float, d_pitch_deg: float) -> None:
        """Right-drag: rotate the 3D view around the pivot (pitch clamped vs terrain)."""
        self.orbit_yaw = (self.orbit_yaw + d_yaw_deg) % 360.0
        self.orbit_pitch += d_pitch_deg
        self._clamp_orbit_pose()
        self.start_moving()

    def _orbit_location(self) -> carla.Location:
        """Camera eye position: pivot minus the view-forward vector times distance."""
        pitch = math.radians(self.orbit_pitch)
        yaw = math.radians(self.orbit_yaw)
        fx = math.cos(pitch) * math.cos(yaw)
        fy = math.cos(pitch) * math.sin(yaw)
        fz = math.sin(pitch)  # negative pitch -> forward points down at the pivot
        return carla.Location(
            self.center_x - fx * self.orbit_distance,
            self.center_y - fy * self.orbit_distance,
            self.pivot_z - fz * self.orbit_distance,
        )

    def _clamp_orbit_pose(self) -> None:
        """Bound distance, then keep the camera eye above the terrain.

        Order matters: refresh pivot_z from the ground under the pivot (miss-aware — an
        unknown sample keeps the previous value rather than injecting garbage), bound the
        distance, range-clamp the pitch (with a pivot-plane margin so the eye never dips
        below the pivot), and only THEN sample the terrain under the final eye XY and
        lift pivot_z by any remaining deficit. The lift changes only Z, so the sample
        position is exactly where the eye ends up — clamping first and sampling second
        is what makes "never below ground" hold at the final pose (sampling the
        pre-clamp eye certified clearance for a spot the eye then left). On a raycast
        miss (unstreamed large-map tiles) the pose keeps the last known pivot_z and
        applies no lift — bounded staleness, never garbage. Hillsides *between* pivot
        and eye can still clip through the view: accepted for v1.
        """
        ground_pivot = self._get_terrain_height_at_pivot()
        if ground_pivot is not None:
            self.pivot_z = float(ground_pivot)
        self.orbit_distance = max(self.ORBIT_MIN_DISTANCE,
                                  min(self.ORBIT_MAX_DISTANCE, self.orbit_distance))
        self.orbit_pitch = clamp_orbit_pitch(
            self.orbit_pitch, self.orbit_distance, self.pivot_z, None,
            self.ORBIT_GROUND_MARGIN, self.ORBIT_PITCH_MIN, self.ORBIT_PITCH_MAX)
        ground_eye = self._get_terrain_height_at_eye()
        if ground_eye is not None:
            eye_z = self.pivot_z + self.orbit_distance * math.sin(abs(math.radians(self.orbit_pitch)))
            deficit = (float(ground_eye) + self.ORBIT_GROUND_MARGIN) - eye_z
            if deficit > 0.0:
                self.pivot_z += deficit

    def _get_terrain_height_at_eye(self) -> Optional[float]:
        """Ground height under the orbit eye XY, or None when unknown (raycast miss).

        Cached by position AND time: valid while the eye stays within 2 m of the
        sampled XY and the sample is fresher than the TTL — the eye moves during
        gestures, and a time-only cache would certify clearance for a spot the eye
        already left. Misses are cached too (as None) so a persistent-miss area
        doesn't raycast on every event.
        """
        eye = self._orbit_location()
        now = time.time()
        cached = self._cached_eye_ground
        if (cached is not None and
                (now - self._cached_eye_ground_time) < self._ground_height_cache_duration and
                math.hypot(eye.x - cached[0], eye.y - cached[1]) < 2.0):
            return cached[2]
        ground = _sample_ground_or_none(self.world, eye.x, eye.y)
        self._cached_eye_ground = (eye.x, eye.y, ground)
        self._cached_eye_ground_time = now
        return ground

    def _get_terrain_height_at_pivot(self) -> Optional[float]:
        """Ground height under the orbit pivot, or None when unknown (raycast miss).
        Same position+TTL cache as the eye sampler, in its own slot."""
        now = time.time()
        cached = self._cached_pivot_ground
        if (cached is not None and
                (now - self._cached_pivot_ground_time) < self._ground_height_cache_duration and
                math.hypot(self.center_x - cached[0], self.center_y - cached[1]) < 2.0):
            return cached[2]
        ground = _sample_ground_or_none(self.world, self.center_x, self.center_y)
        self._cached_pivot_ground = (self.center_x, self.center_y, ground)
        self._cached_pivot_ground_time = now
        return ground

    def start_moving(self):
        """Mark that camera is moving"""
        self.is_moving = True

    def stop_moving(self, *, debug_raycast: bool = True):
        """Mark that camera stopped (prototype: no raycast on stop)."""
        if self.is_moving:
            self.is_moving = False

    def get_navigation_height(self):
        """Get the height value used for camera navigation responsiveness."""
        if self.view_mode == "orbit":
            # Speed scales with zoom, same feel as top-down height scaling.
            return max(self.orbit_distance, self.navigation_height_floor)
        return max(self.height, self.navigation_height_floor)

    def pan(self, dx, dy, dt):
        """Pan the camera by world units, scaled by height"""
        if self.view_mode == "orbit":
            # Inputs are screen-relative (right, down); rotate by the orbit yaw so the
            # pivot slides in view space. Identity at yaw==270 (top-down orientation),
            # so WASD / drag feel unchanged right after entering the 3D view.
            height_scale = self.get_navigation_height() / 200.0
            effective_speed = self.pan_speed * height_scale
            yaw = math.radians(self.orbit_yaw)
            self.center_x += (-math.sin(yaw) * dx - math.cos(yaw) * dy) * dt * effective_speed
            self.center_y += (math.cos(yaw) * dx - math.sin(yaw) * dy) * dt * effective_speed
            # No cache invalidation here: _clamp_orbit_pose refreshes pivot_z from the
            # 0.2 s TTL cache, capping ground raycasts at ~5/s during drags.
            self._clamp_orbit_pose()
            self.start_moving()
            return
        height_scale = self.get_navigation_height() / 200.0
        effective_speed = self.pan_speed * height_scale

        self.center_x += dx * dt * effective_speed
        self.center_y += dy * dt * effective_speed

        # Invalidate terrain cache since we moved
        self._cached_ground_height = None

        # Mark that camera is moving
        self.start_moving()

    def zoom(self, zoom_delta):
        """Zoom by changing camera height. Terrain-aware minimum."""
        if self.view_mode == "orbit":
            # 3D view: multiplicative dolly toward/away from the pivot; the pose clamp
            # bounds the distance and keeps the eye above terrain.
            self.orbit_distance *= self.ORBIT_ZOOM_FACTOR ** zoom_delta
            self._clamp_orbit_pose()
            return
        # Calculate effective minimum height based on terrain
        effective_min = self.min_height  # Default fallback: 40.0

        # Only do fresh raycast when zooming IN, but always use cached terrain for minimum
        if zoom_delta < 0:
            # Zooming in - do raycast (will cache result)
            ground_height = self._get_terrain_height_at_center()
            if ground_height is not None:
                terrain_min = ground_height + 5.0
                effective_min = max(terrain_min, 5.0)
        else:
            # Zooming out - use cached terrain height if available (no new raycast)
            if self._cached_ground_height is not None:
                terrain_min = self._cached_ground_height + 5.0
                effective_min = max(terrain_min, 5.0)

        # Apply zoom with terrain-aware minimum
        new_height = self.height + zoom_delta * self.zoom_speed
        self.height = max(effective_min, min(self.max_height, new_height))

    def update_min_height_from_terrain(self, world, height_buffer=5.0):
        """Prototype: terrain raycast disabled for camera navigation."""
        try:
            buffer_z = float(height_buffer)
        except Exception:
            buffer_z = 0.0

        if buffer_z > self.min_height:
            self.min_height = buffer_z
        if self.height < self.min_height:
            self.height = self.min_height

    def _get_terrain_height_at_center(self):
        """
        Get ground height at camera center, using cache if fresh.
        Returns None if raycast fails or world is unavailable.
        """
        import time

        # Check cache freshness
        now = time.time()
        if (self._cached_ground_height is not None and
                (now - self._cached_ground_height_time) < self._ground_height_cache_duration):
            return self._cached_ground_height

        # Need fresh raycast
        if self.world is None:
            return None

        try:
            location = carla.Location(self.center_x, self.center_y, 0.0)
            ground_z = get_ground_height(
                self.world,
                location,
                debug=False,
                probe_on_miss=False  # Avoid slow grid probing during zoom
            )

            # Cache the result
            self._cached_ground_height = ground_z
            self._cached_ground_height_time = now
            return ground_z

        except Exception:
            # On any error, return cached value if available, else None
            return self._cached_ground_height
