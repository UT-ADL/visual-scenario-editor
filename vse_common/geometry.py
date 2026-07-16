"""Shared CARLA geometry helpers for the Visual Scenario Editor.

Raycast ground detection with large-map tile-offset compensation, tile-file
probing and large-map detection. Used by BOTH the editor (vse.py) and
playback (vse_play.py) — extracted verbatim from their previously duplicated
copies (byte-identical block verified before the move).

Note on `_RAYCAST_TILE_OFFSET_GUESS`: a module-level learned cache of the
current map's tile XY offset. Since this module is shared, the editor and
in-process playback now share ONE cache per process (previously two copies) —
same map, same process, so the learned offset is equally valid for both.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Set, Tuple

import carla


# Some large maps (composed of streamed tiles) can produce `world.cast_ray()` hits in a
# tile-local XY frame. This manifests as hit locations that do not lie on the input ray
# segment (often an XY translation of ~hundreds of meters, e.g. +/-250m). VSE compensates
# by detecting the mismatch and re-casting the ray with an inferred XY offset.
_RAYCAST_TILE_OFFSET_GUESS: Tuple[float, float] = (0.0, 0.0)


def _shift_location_xy(location: carla.Location, dx: float, dy: float) -> carla.Location:
    return carla.Location(float(location.x) + float(dx), float(location.y) + float(dy), float(location.z))


def _compute_expected_xy_on_ray_at_z(
    ray_start: carla.Location,
    ray_end: carla.Location,
    hit_z: float,
) -> Optional[Tuple[float, float, float]]:
    dz = float(ray_end.z) - float(ray_start.z)
    if abs(dz) <= 1e-6:
        return None
    t = (float(hit_z) - float(ray_start.z)) / dz
    x = float(ray_start.x) + (float(ray_end.x) - float(ray_start.x)) * t
    y = float(ray_start.y) + (float(ray_end.y) - float(ray_start.y)) * t
    return x, y, t


def cast_ray_with_tile_offset_compensation(
    world: carla.World,
    ray_start: carla.Location,
    ray_end: carla.Location,
    *,
    cached_map: Optional[carla.Map] = None,
    probe_on_miss: bool = True,
    debug: bool = False,
) -> Tuple[List[object], Dict[str, object]]:
    """
    Cast a ray with automatic compensation for CARLA large-map tile XY offsets.

    Returns a tuple ``(hits, meta)`` where ``hits`` is the same list as returned by
    ``world.cast_ray`` and ``meta`` includes information about any correction:
      - offset_guess_xy: (dx, dy) used for the accepted cast
      - corrected: bool
      - attempts: int
      - misalignment_m: float (distance from hit XY to expected XY on the ray)
    """
    global _RAYCAST_TILE_OFFSET_GUESS

    map_name = getattr(cached_map, 'name', None) or '<unknown>'
    base_offset = _RAYCAST_TILE_OFFSET_GUESS

    def _cast(dx: float, dy: float) -> List[object]:
        return world.cast_ray(_shift_location_xy(ray_start, dx, dy), _shift_location_xy(ray_end, dx, dy))

    def _measure(hit_location: carla.Location) -> Optional[Tuple[float, float, float]]:
        expected = _compute_expected_xy_on_ray_at_z(ray_start, ray_end, float(hit_location.z))
        if expected is None:
            return None
        expected_x, expected_y, _t = expected
        dx = expected_x - float(hit_location.x)
        dy = expected_y - float(hit_location.y)
        return dx, dy, math.hypot(dx, dy)

    def _offset_key(dx: float, dy: float) -> Tuple[int, int]:
        return int(round(float(dx))), int(round(float(dy)))

    attempts = 0
    attempted_offsets: Set[Tuple[int, int]] = set()

    accept_threshold_m = 1.0
    recast_threshold_m = 5.0

    def _attempt_offset(guess_dx: float, guess_dy: float) -> Optional[Tuple[List[object], Dict[str, object]]]:
        nonlocal attempts
        global _RAYCAST_TILE_OFFSET_GUESS

        key = _offset_key(guess_dx, guess_dy)
        if key in attempted_offsets:
            return None
        attempted_offsets.add(key)

        attempts += 1
        hits = _cast(guess_dx, guess_dy)
        if not hits:
            return None

        hit = hits[0]
        measurement = _measure(hit.location)
        if measurement is None:
            accepted_offset = (guess_dx, guess_dy)
            corrected = abs(guess_dx) > 1e-6 or abs(guess_dy) > 1e-6
            _RAYCAST_TILE_OFFSET_GUESS = accepted_offset
            return hits, {
                'map_name': map_name,
                'offset_guess_xy': accepted_offset,
                'corrected': corrected,
                'attempts': attempts,
                'misalignment_m': 0.0,
            }

        corr_dx, corr_dy, misalignment = measurement
        misalignment_m = float(misalignment)
        if misalignment_m <= accept_threshold_m:
            accepted_offset = (guess_dx, guess_dy)
            corrected = abs(guess_dx) > 1e-6 or abs(guess_dy) > 1e-6
            _RAYCAST_TILE_OFFSET_GUESS = accepted_offset
            return hits, {
                'map_name': map_name,
                'offset_guess_xy': accepted_offset,
                'corrected': corrected,
                'attempts': attempts,
                'misalignment_m': misalignment_m,
            }

        if misalignment_m < recast_threshold_m:
            return None

        inferred_dx = float(guess_dx) + float(corr_dx)
        inferred_dy = float(guess_dy) + float(corr_dy)

        key2 = _offset_key(inferred_dx, inferred_dy)
        if key2 in attempted_offsets:
            return None
        attempted_offsets.add(key2)

        attempts += 1
        hits2 = _cast(inferred_dx, inferred_dy)
        if not hits2:
            if debug:
                print(
                    f"[Raycast] Tile offset compensation failed (map={map_name}) "
                    f"guess=({guess_dx:.2f},{guess_dy:.2f}) inferred=({inferred_dx:.2f},{inferred_dy:.2f}) "
                    f"misalign={misalignment_m:.2f}m"
                )
            return None

        hit2 = hits2[0]
        measurement2 = _measure(hit2.location)
        if measurement2 is None:
            accepted_offset = (inferred_dx, inferred_dy)
            _RAYCAST_TILE_OFFSET_GUESS = accepted_offset
            if debug:
                print(
                    f"[Raycast] Tile offset compensation accepted (map={map_name}) "
                    f"offset=({accepted_offset[0]:.2f},{accepted_offset[1]:.2f})"
                )
            return hits2, {
                'map_name': map_name,
                'offset_guess_xy': accepted_offset,
                'corrected': True,
                'attempts': attempts,
                'misalignment_m': misalignment_m,
            }

        _, _, misalignment2 = measurement2
        if float(misalignment2) <= accept_threshold_m:
            accepted_offset = (inferred_dx, inferred_dy)
            _RAYCAST_TILE_OFFSET_GUESS = accepted_offset
            if debug:
                print(
                    f"[Raycast] Tile offset compensation accepted (map={map_name}) "
                    f"offset=({accepted_offset[0]:.2f},{accepted_offset[1]:.2f}) "
                    f"misalign={float(misalignment2):.2f}m attempts={attempts}"
                )
            return hits2, {
                'map_name': map_name,
                'offset_guess_xy': accepted_offset,
                'corrected': True,
                'attempts': attempts,
                'misalignment_m': float(misalignment2),
            }

        if debug:
            print(
                f"[Raycast] Tile offset compensation failed (map={map_name}) "
                f"guess=({guess_dx:.2f},{guess_dy:.2f}) inferred=({inferred_dx:.2f},{inferred_dy:.2f}) "
                f"misalign={misalignment_m:.2f}m"
            )

        return None

    offsets_to_try: List[Tuple[float, float]] = []
    if abs(base_offset[0]) > 1e-6 or abs(base_offset[1]) > 1e-6:
        offsets_to_try.append(base_offset)
    offsets_to_try.append((0.0, 0.0))

    for guess_dx, guess_dy in offsets_to_try:
        result = _attempt_offset(guess_dx, guess_dy)
        if result is not None:
            return result

    def _gcd_step_from_offset(offset: Tuple[float, float]) -> int:
        dx = int(round(abs(float(offset[0]))))
        dy = int(round(abs(float(offset[1]))))
        if dx == 0 and dy == 0:
            return 0
        step = math.gcd(dx, dy)
        return step if step > 0 else (dx or dy)

    # If we got no hits at all, we likely need a different tile-origin offset (e.g., Town11/Town12).
    # Probing can be expensive, so callers may disable it (e.g., for camera navigation); the mismatch-based
    # correction above still applies whenever we get a hit.
    if not probe_on_miss:
        return [], {
            'map_name': map_name,
            'offset_guess_xy': (0.0, 0.0),
            'corrected': False,
            'attempts': attempts,
            'misalignment_m': 0.0,
        }

    # Probe a small grid of candidate offsets to find any hit, then the usual mismatch logic will infer the exact offset.
    base_step = _gcd_step_from_offset(base_offset)
    candidate_steps: List[int] = []
    for step in (base_step, 1000, 2000, 250, 500):
        if step and step not in candidate_steps:
            candidate_steps.append(step)

    # Fast-path guess: for streaming large maps, the correct tile-origin shift is often close to a rounded
    # multiple of a common step size (e.g., 1000m/2000m). Try a small set of quantized offsets first to
    # avoid a full grid scan.
    ref_x = float(ray_start.x)
    ref_y = float(ray_start.y)
    for step in candidate_steps:
        step_f = float(step)
        if step_f <= 0:
            continue

        def _quantized_choices(v: float) -> List[float]:
            q = v / step_f
            return sorted({
                -math.floor(q) * step_f,
                -math.ceil(q) * step_f,
                -round(q) * step_f,
            })

        for qdx in _quantized_choices(ref_x):
            for qdy in _quantized_choices(ref_y):
                if abs(qdx) > 20000.0 or abs(qdy) > 20000.0:
                    continue
                result = _attempt_offset(qdx, qdy)
                if result is not None:
                    return result

    candidate_centers: List[Tuple[float, float]] = []
    if abs(base_offset[0]) > 1e-6 or abs(base_offset[1]) > 1e-6:
        candidate_centers.append(base_offset)
    candidate_centers.append((0.0, 0.0))

    radius_steps = 8
    max_abs_offset_m = 20000.0
    index_pairs = [(ix, iy) for ix in range(-radius_steps, radius_steps + 1) for iy in range(-radius_steps, radius_steps + 1)]
    index_pairs.sort(key=lambda pair: (max(abs(pair[0]), abs(pair[1])), abs(pair[0]) + abs(pair[1])))
    for center_dx, center_dy in candidate_centers:
        for step in candidate_steps:
            step_f = float(step)
            for ix, iy in index_pairs:
                probe_dx = float(center_dx) + float(ix) * step_f
                probe_dy = float(center_dy) + float(iy) * step_f
                if abs(probe_dx) > max_abs_offset_m or abs(probe_dy) > max_abs_offset_m:
                    continue
                result = _attempt_offset(probe_dx, probe_dy)
                if result is not None:
                    if debug and (abs(float(probe_dx)) > 1e-6 or abs(float(probe_dy)) > 1e-6):
                        print(
                            f"[Raycast] Tile offset probe accepted (map={map_name}) "
                            f"probe=({probe_dx:.0f},{probe_dy:.0f}) attempts={attempts}"
                        )
                    return result

    return [], {
        'map_name': map_name,
        'offset_guess_xy': (0.0, 0.0),
        'corrected': False,
        'attempts': attempts,
        'misalignment_m': 0.0,
    }


def _has_tile_files_on_disk(short_name: str, map_folder: str = "") -> bool:
    """Check if a map has _Tile_ .umap files on disk (indicates a large/tiled map).

    *map_folder*, when provided, restricts the search to that specific
    subdirectory under ``Maps/`` so that tile files belonging to a
    *different* map with a similar name are not matched.
    """
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root or not short_name:
        return False
    import glob as _glob
    maps_root = os.path.join(carla_root, "CarlaUE4", "Content", "Carla", "Maps")
    if map_folder:
        pattern = os.path.join(maps_root, map_folder, f"{short_name}_Tile_*.umap")
    else:
        pattern = os.path.join(maps_root, "**", f"{short_name}_Tile_*.umap")
    return bool(_glob.glob(pattern, recursive=not map_folder))


# Hits at/below an excluded actor's feet plane must survive the filter: for a
# standing (or floating) walker the true ground is at/below its feet, and a
# zone that extends below the feet swallows it — the sampler then falls back
# to the caller's reference Z (the floating-pedestrian bug). Keep a small
# tolerance above the feet so coincident lower-capsule hits still filter.
_EXCLUSION_FEET_TOLERANCE_M = 0.05
_EXCLUSION_HEAD_MARGIN_M = 0.5


def _actor_exclusion_zones(exclude_actors):
    """Build (x, y, center_z, xy_radius, extent_z) zones for live actors."""
    zones = []
    for actor in exclude_actors:
        try:
            if not actor or not actor.is_alive:
                continue
            actor_location = actor.get_transform().location
            bbox = actor.bounding_box
            zones.append((
                actor_location.x,
                actor_location.y,
                actor_location.z + bbox.location.z,
                math.hypot(bbox.extent.x, bbox.extent.y) + 0.3,
                bbox.extent.z,
            ))
        except Exception:
            continue
    return zones


def _hit_in_exclusion_zone(hit_location, zones):
    """True if a raycast hit lies on an excluded actor's body (feet..head)."""
    return any(
        math.hypot(hit_location.x - zx, hit_location.y - zy) <= zr
        and (zz - ze + _EXCLUSION_FEET_TOLERANCE_M) < hit_location.z
        and hit_location.z <= (zz + ze + _EXCLUSION_HEAD_MARGIN_M)
        for zx, zy, zz, zr, ze in zones
    )


def get_ground_height(world, location, debug=False, cached_map=None, *, return_metadata=False, probe_on_miss=True, ignore_labels=None, exclude_actors=None):
    """
    Get the actual ground height at a location using raycast.
    Tries raycasting first, then falls back to waypoint height, then original Z if all else fails.

    For large maps with tiles, automatically detects and compensates for tile-local coordinate offsets.

    Args:
        world: CARLA world object
        location: carla.Location to check
        debug: bool - Enable debug logging for coordinate testing (default False)
        probe_on_miss: bool - Whether to probe grid of offsets when raycast misses (default True, can be slow on large maps)
        ignore_labels: optional collection of carla.CityObjectLabel values whose
            raycast hits are skipped (e.g. Pedestrians for walker skeleton hits)
        exclude_actors: optional iterable of carla.Actor whose bodies must not
            count as ground — raycast hits inside their bounding cylinders are
            skipped. Used to keep actor placeholders (pedestrian capsules,
            vehicle roofs) from being sampled as terrain; labels can't do this
            because walker capsule hits report NONE, which is also legitimate
            terrain on custom maps.
    Returns:
        float: Detected ground height (Z coordinate)
    """
    try:
        # Cast a ray from high above down to the ground, covering elevated actors
        vertical_buffer_up = 200.0
        vertical_search_down = max(400.0, abs(location.z) + 200.0)
        start_location = carla.Location(
            location.x,
            location.y,
            location.z + vertical_buffer_up,
        )
        end_location = carla.Location(
            location.x,
            location.y,
            location.z - vertical_search_down,
        )

        # Perform raycast to detect ground
        raycast_result, raycast_meta = cast_ray_with_tile_offset_compensation(
            world,
            start_location,
            end_location,
            cached_map=cached_map,
            probe_on_miss=probe_on_miss,
            debug=debug,
        )

        if raycast_result and ignore_labels:
            raycast_result = [
                h for h in raycast_result if getattr(h, "label", None) not in ignore_labels
            ]

        if raycast_result and exclude_actors:
            zones = _actor_exclusion_zones(exclude_actors)
            if zones:
                raycast_result = [
                    h for h in raycast_result
                    if not _hit_in_exclusion_zone(h.location, zones)
                ]

        if raycast_result:
            # Found ground, use that height directly
            hit = raycast_result[0]

            ground_height = hit.location.z

            if debug:
                offset_guess = raycast_meta.get('offset_guess_xy', (0.0, 0.0))
                corrected = bool(raycast_meta.get('corrected', False))
                attempts = int(raycast_meta.get('attempts', 0) or 0)
                misalignment = float(raycast_meta.get('misalignment_m', 0.0) or 0.0)

                print(f"\n{'='*60}")
                print("RAYCAST DEBUG - Ground Height Sample")
                print(f"{'='*60}")
                print(f"Map name: {raycast_meta.get('map_name', '<unknown>')}")
                print(f"Input location (X, Y, Z): ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")
                print(f"Ray start (X, Y, Z):      ({start_location.x:.2f}, {start_location.y:.2f}, {start_location.z:.2f})")
                print(f"Ray end (X, Y, Z):        ({end_location.x:.2f}, {end_location.y:.2f}, {end_location.z:.2f})")
                print(f"Hit location (X, Y, Z):   ({hit.location.x:.2f}, {hit.location.y:.2f}, {hit.location.z:.2f})")
                print(f"Hit label: {hit.label}")
                print(f"Ground height: {ground_height:.2f}")
                print(f"Tile compensation: {'YES' if corrected else 'no'}")
                print(f"Offset guess (dx, dy): ({float(offset_guess[0]):.2f}, {float(offset_guess[1]):.2f})")
                print(f"Misalignment (m): {misalignment:.2f} (attempts={attempts})")
                print(f"{'='*60}\n")

            result = {'height': ground_height, 'source': 'raycast', 'raycast_meta': raycast_meta}
            return result if return_metadata else ground_height
        else:
            # No ground found, try to use waypoint height (project to road) if map data is available
            waypoint = None
            if cached_map is not None:
                try:
                    waypoint = cached_map.get_waypoint(location, project_to_road=True)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Waypoint lookup failed: {e}")
                    waypoint = None
            if waypoint:
                waypoint_height = waypoint.transform.location.z
                if debug:
                    print(f"[DEBUG] Raycast failed, using waypoint height: {waypoint_height:.2f}")
                result = {'height': waypoint_height, 'source': 'waypoint'}
                return result if return_metadata else waypoint_height

            # Fallback to original height if all else fails
            fallback_height = location.z
            if debug:
                print(f"[DEBUG] Raycast and waypoint failed, using original Z: {fallback_height:.2f}")
            result = {'height': fallback_height, 'source': 'fallback'}
            return result if return_metadata else fallback_height

    except Exception as e:
        # On error, fallback to original Z
        if debug:
            print(f"[DEBUG ERROR] Exception in get_ground_height: {e}")
        fallback_height = location.z
        result = {'height': fallback_height, 'source': 'fallback'}
        return result if return_metadata else fallback_height


def is_large_map(map_or_name) -> bool:
    """Heuristic detection for CARLA large maps (streamed tiles).

    Accepts either a ``carla.Map`` (or any object with a ``name`` attribute)
    or the map name string itself. Unifies the previously duplicated
    ``vse.is_large_map_name(name)`` and ``vse_play._is_large_map(map)``.
    """
    if os.environ.get("VSE_FORCE_LARGE_MAP") == "1":
        return True
    if map_or_name is None:
        return False
    if isinstance(map_or_name, str):
        name = map_or_name.lower()
    else:
        try:
            name = str(getattr(map_or_name, "name", "") or "").lower()
        except Exception:
            name = ""
    # Check for tile files on disk (catches maps with sublevels like tallinn_demo)
    parts = name.split('/')
    short = parts[-1]
    # Use the parent folder (e.g. "tartu_large_awmini") so we don't match
    # tile files belonging to a different map with a similar short name.
    folder = parts[-2] if len(parts) >= 2 else ""
    if short and _has_tile_files_on_disk(short, map_folder=folder):
        return True
    return False
