#!/usr/bin/env python

from __future__ import annotations

"""
vse_play.py - Scenario Playback for CARLA ScenarioRunner
=========================================================

This module provides scenario playback functionality for the Visual Scenario Editor (VSE).
It handles vehicle and pedestrian simulation, waypoint following, triggers, and integration
with ScenarioRunner.

TABLE OF CONTENTS
-----------------
Line ~97:   UTILITY FUNCTIONS
            - Map detection, timeouts, actor cleanup, data provider init

Line ~238:  ROUTE PLANNING HELPERS
            - GlobalRoutePlanner wrapper, ROS integration, traffic light matching

Line ~510:  GEOMETRY & MATH UTILITIES
            - Heading calculations, angle normalization, arc/linear sampling

Line ~763:  ROUTE REFINEMENT
            - _refine_vehicle_route() - route interpolation and speed capping

Line ~825:  DATA CLASSES
            - RouteSegment, RoutePoint, VehicleData, TriggerableActor, triggers

Line ~910:  VEHICLE CONTROLLER
            - VehicleController class - threaded vehicle navigation

Line ~1412: PEDESTRIAN BEHAVIORS
            - Walker orientation, smooth turns, walking behaviors

Line ~1676: BEHAVIOR TREE COMPONENTS
            - Sequences, spawn behaviors, route execution, tracking

Line ~1875: CRITERIA & MONITORS
            - Arrival criteria, trigger monitors (traffic light, pedestrian, vehicle)

Line ~2350: MAIN SCENARIO CLASS
            - vse_play(BasicScenario) - main scenario implementation

Line ~3922: CLEANUP & SIGNAL HANDLING
            - Global cleanup, signal handlers, atexit registration

Line ~3964: STANDALONE RUNNER UTILITIES
            - Path resolution, scenario argument parsing

Line ~4101: EXTERNAL EGO HANDLING
            - External ego vehicle detection and adoption

Line ~4159: MINI RUNNER
            - MiniRunner class - standalone scenario execution

Line ~4550: MAIN ENTRY POINT
            - main() function and argument parsing
"""

import argparse
import atexit
import importlib
import json
import math
import os
import faulthandler
import random
import signal
import subprocess
import sys
import threading
import time
import multiprocessing
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import carla
import py_trees
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from py_trees import common as py_trees_common
from py_trees.behaviour import Behaviour
from py_trees.composites import Parallel, Sequence
from py_trees.trees import BehaviourTree

_SCENARIO_RUNNER_ROOT = os.environ.get("SCENARIO_RUNNER_ROOT")
if _SCENARIO_RUNNER_ROOT:
    _scenario_runner_path = Path(_SCENARIO_RUNNER_ROOT).resolve()
    _scenario_runner_str = str(_scenario_runner_path)
    if _scenario_runner_str not in sys.path:
        sys.path.insert(0, _scenario_runner_str)

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    AtomicBehavior,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest,
    Criterion,
    EndofRoadTest,
    InRouteTest,
    KeepLaneTest,
    OffRoadTest,
    OnSidewalkTest,
    OutsideRouteLanesTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
    WrongLaneTest,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.route_manipulation import downsample_route, interpolate_trajectory
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.weather_sim import RouteWeatherBehavior
from srunner.scenariomanager.lights_sim import RouteLightsBehavior

from types import SimpleNamespace

# Module-level logger for vse_play
logger = logging.getLogger(__name__)

_spawned_vehicles: List[carla.Actor] = []
_current_scenario: Optional["vse_play"] = None
_current_scenario_lock = threading.Lock()  # Protect _current_scenario access
euler_from_quaternion = None

_WALKER_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 165, 0),
    (255, 105, 180),
]

_TRAFFIC_LIGHT_FINGERPRINT_SCALE = 4.0  # ~0.25m quantization

EGO_DEFAULT_SPEED_KMH = 40.0

_LARGE_MAP_TOPOLOGY_THRESHOLD = 10000

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


def get_ground_height(world, location, debug=False, cached_map=None, *, return_metadata=False, probe_on_miss=True):
    """
    Get the actual ground height at a location using raycast.
    Tries raycasting first, then falls back to waypoint height, then original Z if all else fails.

    For large maps with tiles, automatically detects and compensates for tile-local coordinate offsets.

    Args:
        world: CARLA world object
        location: carla.Location to check
        debug: bool - Enable debug logging for coordinate testing (default False)
        probe_on_miss: bool - Whether to probe grid of offsets when raycast misses (default True, can be slow on large maps)
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


# =============================================================================
# UTILITY FUNCTIONS
# Map detection, timeouts, actor cleanup, CARLA data provider initialization
# =============================================================================


def _is_large_map(carla_map: Optional[carla.Map]) -> bool:
    """Heuristic: treat streaming/tiled and very large topology maps as 'large'."""
    if os.environ.get("VSE_FORCE_LARGE_MAP") == "1":
        return True
    if carla_map is None:
        return False
    try:
        name = str(getattr(carla_map, "name", "") or "").lower()
    except Exception:
        name = ""
    if "large" in name:
        return True
    try:
        topo = carla_map.get_topology()
        return len(topo) >= _LARGE_MAP_TOPOLOGY_THRESHOLD
    except Exception:
        return False


def _temporary_client_timeout(client: Optional[carla.Client], timeout_s: float):
    """Context manager that temporarily sets the CARLA client timeout."""
    _state = getattr(_temporary_client_timeout, "_state", None)
    if _state is None:
        _state = threading.local()
        _temporary_client_timeout._state = _state  # type: ignore[attr-defined]

    class _TimeoutCtx:
        def __enter__(self_nonlocal):
            if client:
                depth = getattr(_state, "depth", 0)
                if depth <= 0:
                    try:
                        client.set_timeout(float(timeout_s))
                    except Exception:
                        pass
                _state.depth = depth + 1
            return self_nonlocal

        def __exit__(self_nonlocal, exc_type, exc, tb):
            if client:
                depth = getattr(_state, "depth", 1)
                depth = max(0, depth - 1)
                _state.depth = depth
                if depth == 0:
                    try:
                        client.set_timeout(10.0)
                    except Exception:
                        pass
            return False

    return _TimeoutCtx()


def _destroy_actor_ids(
    client: Optional[carla.Client],
    actor_ids: List[int],
    *,
    do_tick: bool,
    log_fn=None,
) -> None:
    """Destroy actors by id using a single batched command."""
    if not client:
        return
    cleaned: List[int] = []
    seen = set()
    for raw in actor_ids:
        try:
            actor_id = int(raw)
        except Exception:
            continue
        if actor_id <= 0 or actor_id in seen:
            continue
        seen.add(actor_id)
        cleaned.append(actor_id)
    if not cleaned:
        return
    try:
        commands = [carla.command.DestroyActor(actor_id) for actor_id in cleaned]
        client.apply_batch_sync(commands, do_tick=bool(do_tick))
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Failed to destroy actors %s: %s", cleaned, exc)
        if log_fn:
            try:
                log_fn(f"Failed to destroy actors {cleaned}: {exc}")
            except Exception:
                pass
        else:
            print(f"[Cleanup] Failed to destroy actors {cleaned}: {exc}")


def _init_carla_data_provider(world: carla.World, client: Optional[carla.Client] = None) -> bool:
    """
    Initialize ScenarioRunner's CarlaDataProvider without blocking on large maps.

    Returns True when the "fast init" path was used (no GlobalRoutePlanner build).
    """
    if not world:
        return False
    carla_map = None
    try:
        carla_map = world.get_map()
    except Exception:
        carla_map = None

    if client:
        try:
            CarlaDataProvider.set_client(client)
        except Exception:
            pass

    # Fast init: mirror CarlaDataProvider.set_world() minus the GRP build.
    #
    # We avoid constructing CARLA's GlobalRoutePlanner here because it can be slow and
    # (depending on CARLA build/driver/threading) has been observed to crash the Python
    # process during Play/Stop races. VSE drives agents via set_global_plan() and uses
    # explicit routes for criteria/weather, so CDP does not require a pre-built GRP.
    CarlaDataProvider._world = world  # type: ignore[attr-defined]
    try:
        CarlaDataProvider._sync_flag = bool(world.get_settings().synchronous_mode)  # type: ignore[attr-defined]
    except Exception:
        CarlaDataProvider._sync_flag = False  # type: ignore[attr-defined]
    CarlaDataProvider._map = carla_map  # type: ignore[attr-defined]
    try:
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()  # type: ignore[attr-defined]
    except Exception:
        CarlaDataProvider._blueprint_library = None  # type: ignore[attr-defined]
    CarlaDataProvider._grp = None  # type: ignore[attr-defined]
    try:
        CarlaDataProvider.generate_spawn_points()
    except Exception:
        pass
    try:
        CarlaDataProvider.prepare_map()
    except Exception:
        pass
    return True


# =============================================================================
# ROUTE PLANNING HELPERS
# GlobalRoutePlanner wrapper, ROS plan publishing, traffic light matching
# =============================================================================


class _NoopGlobalRoutePlanner(GlobalRoutePlanner):
    """
    Cheap stand-in for CARLA's GlobalRoutePlanner.

    BasicAgent only needs a GRP instance when set_destination/trace_route are used.
    VSE drives agents via set_global_plan(), so we can skip GRP precomputation on large maps.
    """

    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def trace_route(self, *args, **kwargs):  # pragma: no cover - defensive fallback
        return []


def _ros_plan_publisher_process(
    raw_waypoints: List[Tuple[float, float, float, float]],
    downsample_interval: float,
    ros_publish_delay: float,
    agent_path: str,
    skip_interpolation: bool,
) -> None:
    """Child process entry: publish a plan via a ScenarioRunner agent with a fresh ROS node."""
    parent_pid = os.getppid()
    agent_file = None
    try:
        agent_file = str(Path(agent_path).expanduser().resolve())
        agent_dir = str(Path(agent_file).parent)
        module_name = Path(agent_file).stem
        if agent_dir and agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        module_agent = importlib.import_module(module_name)
        if hasattr(module_agent, "get_entry_point"):
            agent_class_name = str(module_agent.get_entry_point())
        else:
            agent_class_name = module_agent.__name__.title().replace("_", "")
        agent_class = getattr(module_agent, agent_class_name)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[MiniRunner] Failed to load agent from {agent_file or agent_path}: {exc}")
        return

    if not raw_waypoints or len(raw_waypoints) < 2:
        print("[MiniRunner] No waypoints available for agent; skipping publish")
        return

    try:
        keypoints: List[carla.Location] = []
        for x, y, z, _yaw in raw_waypoints:
            keypoints.append(carla.Location(x=float(x), y=float(y), z=float(z)))

        route: List[Tuple] = []
        gps_route: List[Tuple] = []
        if len(keypoints) >= 2 and not skip_interpolation:
            try:
                gps_route, route = interpolate_trajectory(keypoints, hop_resolution=1.0)
            except Exception as exc:
                print(f"[MiniRunner] Route interpolation failed for minimal agent: {exc}")
                route = []
                gps_route = []

        if not route:
            for x, y, z, yaw in raw_waypoints:
                loc = carla.Location(x=float(x), y=float(y), z=float(z))
                rot = carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
                tf = carla.Transform(loc, rot)
                route.append((tf, RoadOption.LANEFOLLOW))
                gps_route.append((loc, RoadOption.LANEFOLLOW))
        else:
            try:
                sampled_ids = downsample_route(route, downsample_interval)
                route = [route[i] for i in sampled_ids if 0 <= i < len(route)]
                gps_route = [gps_route[i] for i in sampled_ids if 0 <= i < len(gps_route)]
            except Exception as exc:
                print(f"[MiniRunner] Route downsample failed; using full route: {exc}")
            if not route:
                for x, y, z, yaw in raw_waypoints:
                    loc = carla.Location(x=float(x), y=float(y), z=float(z))
                    rot = carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
                    tf = carla.Transform(loc, rot)
                    route.append((tf, RoadOption.LANEFOLLOW))
                    gps_route.append((loc, RoadOption.LANEFOLLOW))

        def _wait_for_ros_time(agent_obj, sample_route, timeout=5.0) -> bool:
            """Wait until ROS time becomes non-zero."""
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    pose = agent_obj.pose_stamped_from_waypoint(sample_route[0][0])
                    stamp = getattr(pose.header, "stamp", None)
                    if stamp and (getattr(stamp, "secs", 0) != 0 or getattr(stamp, "nsecs", 0) != 0):
                        return True
                except Exception:
                    pass
                time.sleep(0.1)
            return False

        agent = agent_class("")

        try:
            # Warm-up publish (two close goals) to wake subscribers.
            base_loc = keypoints[0]
            yaw_deg = float(raw_waypoints[0][3]) if raw_waypoints else 0.0
            yaw_rad = math.radians(yaw_deg)
            offset_loc = carla.Location(
                x=base_loc.x + math.cos(yaw_rad),
                y=base_loc.y + math.sin(yaw_rad),
                z=base_loc.z,
            )
            base_tf = carla.Transform(base_loc, carla.Rotation(yaw=yaw_deg))
            offset_tf = carla.Transform(offset_loc, carla.Rotation(yaw=yaw_deg))
            warmup_route = [(base_tf, RoadOption.LANEFOLLOW), (offset_tf, RoadOption.LANEFOLLOW)]
            warmup_gps = [(base_loc, RoadOption.LANEFOLLOW), (offset_loc, RoadOption.LANEFOLLOW)]
            agent.set_global_plan(warmup_gps, warmup_route)
            _wait_for_ros_time(agent, warmup_route, timeout=3.0)
            agent.publish_plan()
            print("[MiniRunner] Warm-up plan published (2 goals) before main route")
        except Exception as exc:
            print(f"[MiniRunner] Warm-up publish failed; continuing with main route: {exc}")

        agent.set_global_plan(gps_route, route)
        _wait_for_ros_time(agent, route, timeout=3.0)

        try:
            publish_burst = max(1, int(os.environ.get("VSE_PLAN_PUBLISH_BURST", "1") or 2))
        except Exception:
            publish_burst = 1

        try:
            for idx in range(publish_burst):
                agent.publish_plan()
                agent.global_plan_published = True
                if idx + 1 < publish_burst:
                    time.sleep(max(0.1, ros_publish_delay))
        except Exception as exc:
            print(f"[MiniRunner] Failed to publish plan burst: {exc}")

        try:
            # Keep the ROS publishers alive so late subscribers can still receive the
            # latched Path message. This addresses first-run cases where an external
            # stack connects after the publish completes (otherwise only the final goal
            # may be observed).
            while True:
                try:
                    current_ppid = os.getppid()
                    # Exit if parent died (ppid changed or became 1/init)
                    if current_ppid != parent_pid or current_ppid == 1:
                        break
                except (OSError, AttributeError):
                    # Parent process no longer exists
                    break
                # Shorter sleep for faster response to parent termination
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            try:
                agent.destroy()
            except Exception:
                pass
    finally:
        pass


def _normalize_traffic_light_fingerprint(
    fingerprint: Optional[List[Tuple[int, int, int]]],
) -> Optional[Tuple[Tuple[int, int, int], ...]]:
    """Convert a fingerprint payload to a normalized tuple-of-tuples."""
    if not fingerprint:
        return None
    try:
        return tuple((int(pt[0]), int(pt[1]), int(pt[2])) for pt in fingerprint)
    except Exception:
        return None


def _compute_traffic_light_fingerprint(lights: List[carla.TrafficLight]) -> Optional[Tuple[Tuple[int, int, int], ...]]:
    """Return a quantized, order-independent signature for the provided lights."""
    quantized_locations: List[Tuple[int, int, int]] = []
    for light in lights:
        if not light:
            continue
        try:
            location = light.get_transform().location
        except Exception:
            continue
        scale = _TRAFFIC_LIGHT_FINGERPRINT_SCALE
        quantized_locations.append(
            (
                int(round(location.x * scale)),
                int(round(location.y * scale)),
                int(round(location.z * scale)),
            )
        )
    if not quantized_locations:
        return None
    quantized_locations.sort()
    return tuple(quantized_locations)


def _build_traffic_light_fingerprint_index(world: Optional[carla.World]) -> dict:
    """Return mapping of quantized locations -> list of traffic light actors."""
    index: dict = defaultdict(list)
    if not world:
        print("[TRAFFIC_LIGHT][DEBUG] Fingerprint index build: world is None")
        return index
    try:
        actors = world.get_actors().filter("traffic.traffic_light*")
    except Exception:
        actors = []
    try:
        print(f"[TRAFFIC_LIGHT][DEBUG] Building fingerprint index for world id={id(world)} actors={len(actors)}")
    except Exception:
        pass
    total = 0
    success = 0
    not_alive = 0
    tf_fail = 0
    sample_logs = 0
    for actor in actors:
        if not actor:
            continue
        try:
            alive = actor.is_alive
        except Exception:
            alive = True
        try:
            loc = actor.get_transform().location
            total += 1
        except Exception:
            tf_fail += 1
            if not alive:
                not_alive += 1
            continue
        scale = _TRAFFIC_LIGHT_FINGERPRINT_SCALE
        key = (
            int(round(loc.x * scale)),
            int(round(loc.y * scale)),
            int(round(loc.z * scale)),
        )
        index[key].append(actor)
        success += 1
        if not alive:
            not_alive += 1
        if sample_logs < 5:
            try:
                print(f"[TRAFFIC_LIGHT][DEBUG] Sample light id={actor.id} alive={alive} loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) key={key}")
            except Exception:
                pass
            sample_logs += 1
    try:
        print(f"[TRAFFIC_LIGHT][DEBUG] Fingerprint index built with {len(index)} keys (success={success}, not_alive={not_alive}, tf_fail={tf_fail}, total_seen={total})")
    except Exception:
        pass
    return index


def _match_traffic_lights_by_fingerprint(
    fingerprint: Optional[Tuple[Tuple[int, int, int], ...]],
    index: dict,
) -> List[carla.TrafficLight]:
    """Resolve a fingerprint to live traffic-light actors using the provided index."""
    try:
        print(f"[TRAFFIC_LIGHT][DEBUG] Matching fingerprint {fingerprint} against index size {len(index)}")
    except Exception:
        pass
    if not fingerprint or not index:
        return []
    used_ids: set = set()
    matched: List[carla.TrafficLight] = []
    for coord in fingerprint:
        candidates = index.get(coord, [])
        candidate: Optional[carla.TrafficLight] = None
        for cand in candidates:
            try:
                cand_id = cand.id
            except Exception:
                continue
            if cand_id in used_ids:
                continue
            candidate = cand
            break
        if candidate is None:
            return []
        matched.append(candidate)
        try:
            used_ids.add(candidate.id)
        except Exception:
            pass
    return matched


# =============================================================================
# GEOMETRY & MATH UTILITIES
# Heading calculations, angle normalization, distance, arc/linear sampling
# =============================================================================


def _normalize_yaw(yaw: float) -> float:
    return (yaw + 180.0) % 360.0 - 180.0


def _compute_heading(origin: carla.Location, target: carla.Location, fallback: float) -> float:
    dx = target.x - origin.x
    dy = target.y - origin.y
    if math.isclose(dx, 0.0, abs_tol=1e-3) and math.isclose(dy, 0.0, abs_tol=1e-3):
        return fallback
    return _normalize_yaw(math.degrees(math.atan2(dy, dx)))


def _heading_from_orientation(orientation) -> float:
    """Extract yaw (radians) from a ROS Pose orientation; fallback to 0 on failure."""
    if not orientation:
        return 0.0
    if euler_from_quaternion is None:
        return 0.0
    try:
        quat = (
            float(getattr(orientation, "x", 0.0)),
            float(getattr(orientation, "y", 0.0)),
            float(getattr(orientation, "z", 0.0)),
            float(getattr(orientation, "w", 1.0)),
        )
        _, _, yaw = euler_from_quaternion(quat)
        return float(yaw)
    except Exception:
        return 0.0


def _shortest_angle_difference(target: float, start: float) -> float:
    diff = target - start
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0
    return diff


def _distance(a: carla.Location, b: carla.Location) -> float:
    return float(a.distance(b))


def _grounded_location(world: carla.World, location: carla.Location) -> carla.Location:
    """
    DEPRECATED: Use get_ground_height() instead for more accurate ground detection.
    This function is kept for backward compatibility but now uses the improved raycast method.

    Returns a grounded location with a small 0.2m offset above the ground.
    For pedestrians, use get_ground_height() with bounding box offset instead.
    """
    grounded = carla.Location(location.x, location.y, location.z)
    ground_height = get_ground_height(
        world,
        location,
        debug=False,
        cached_map=world.get_map(),
        probe_on_miss=True
    )
    grounded.z = ground_height + 0.2  # Small offset above ground
    return grounded


def _estimate_turn_radius(curr_heading: float, next_heading: float, len_a: float, len_b: float) -> Optional[float]:
    delta = abs(_normalize_yaw(next_heading - curr_heading))
    if delta < 1e-2:
        return None
    angle_rad = math.radians(delta)
    chord = min(len_a, len_b)
    if chord < 1e-2:
        return None
    return chord / (2.0 * math.sin(angle_rad / 2.0))


def _compute_turn_radius_from_locations(
    prev_loc: Optional[carla.Location],
    curr_loc: carla.Location,
    next_loc: Optional[carla.Location],
) -> Optional[float]:
    """Return radius of circumcircle through the 2D projection of three points."""
    if prev_loc is None or next_loc is None:
        return None

    ax, ay = prev_loc.x, prev_loc.y
    bx, by = curr_loc.x, curr_loc.y
    cx, cy = next_loc.x, next_loc.y

    # Degenerate if points are almost colinear or duplicated
    if (
        math.isclose(ax, bx, abs_tol=1e-4) and math.isclose(ay, by, abs_tol=1e-4)
    ) or (
        math.isclose(bx, cx, abs_tol=1e-4) and math.isclose(by, cy, abs_tol=1e-4)
    ):
        return None

    denom = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denom) < 1e-6:
        return None

    a_sq = ax * ax + ay * ay
    b_sq = bx * bx + by * by
    c_sq = cx * cx + cy * cy

    ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / denom
    uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / denom

    radius = math.hypot(ax - ux, ay - uy)
    if radius < 1e-3:
        return None
    return radius


def _angle_at(center_x: float, center_y: float, loc: carla.Location) -> float:
    return math.atan2(loc.y - center_y, loc.x - center_x)


def _generate_arc_samples(
    prev_tf: Optional[carla.Transform],
    curr_tf: carla.Transform,
    next_tf: carla.Transform,
    step: float,
) -> List[carla.Transform]:
    """Create intermediate transforms between curr and next following an arc if possible."""
    prev_loc = prev_tf.location if prev_tf else None
    curr_loc = curr_tf.location
    next_loc = next_tf.location
    if prev_loc is None:
        return _generate_linear_samples(curr_tf, next_tf, step)

    radius = _compute_turn_radius_from_locations(prev_loc, curr_loc, next_loc)
    if radius is None or radius > 1e6:
        return _generate_linear_samples(curr_tf, next_tf, step)

    ax, ay = prev_loc.x, prev_loc.y
    bx, by = curr_loc.x, curr_loc.y
    cx, cy = next_loc.x, next_loc.y

    denom = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denom) < 1e-6:
        return _generate_linear_samples(curr_tf, next_tf, step)

    a_sq = ax * ax + ay * ay
    b_sq = bx * bx + by * by
    c_sq = cx * cx + cy * cy

    center_x = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / denom
    center_y = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / denom

    start_angle = _angle_at(center_x, center_y, curr_loc)
    end_angle = _angle_at(center_x, center_y, next_loc)

    # Determine turn direction using cross product
    v1x, v1y = curr_loc.x - prev_loc.x, curr_loc.y - prev_loc.y
    v2x, v2y = next_loc.x - curr_loc.x, next_loc.y - curr_loc.y
    cross_z = v1x * v2y - v1y * v2x
    is_left_turn = cross_z > 0.0

    if is_left_turn:
        while end_angle <= start_angle:
            end_angle += 2.0 * math.pi
    else:
        while end_angle >= start_angle:
            end_angle -= 2.0 * math.pi

    arc_length = abs(end_angle - start_angle) * radius
    if arc_length <= step:
        return []

    num_samples = max(1, int(arc_length // step))
    delta_angle = (end_angle - start_angle) / (num_samples + 1)

    samples: List[carla.Transform] = []
    for i in range(1, num_samples + 1):
        angle = start_angle + delta_angle * i
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        z = curr_loc.z + (next_loc.z - curr_loc.z) * (i / (num_samples + 1))

        # Tangent direction along the arc
        if is_left_turn:
            tangent_angle = angle + math.pi / 2.0
        else:
            tangent_angle = angle - math.pi / 2.0

        yaw = _normalize_yaw(math.degrees(tangent_angle))
        samples.append(
            carla.Transform(
                carla.Location(x=float(x), y=float(y), z=float(z)),
                carla.Rotation(
                    pitch=curr_tf.rotation.pitch,
                    yaw=yaw,
                    roll=curr_tf.rotation.roll,
                ),
            )
        )

    return samples


def _generate_linear_samples(
    curr_tf: carla.Transform,
    next_tf: carla.Transform,
    step: float,
) -> List[carla.Transform]:
    curr_loc = curr_tf.location
    next_loc = next_tf.location
    dx = next_loc.x - curr_loc.x
    dy = next_loc.y - curr_loc.y
    dz = next_loc.z - curr_loc.z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance <= step:
        return []
    num_samples = max(1, int(distance // step))
    samples: List[carla.Transform] = []
    for i in range(1, num_samples + 1):
        t = i / (num_samples + 1)
        x = curr_loc.x + dx * t
        y = curr_loc.y + dy * t
        z = curr_loc.z + dz * t
        yaw = _normalize_yaw(
            math.degrees(math.atan2(next_loc.y - curr_loc.y, next_loc.x - curr_loc.x))
        )
        samples.append(
            carla.Transform(
                carla.Location(x=float(x), y=float(y), z=float(z)),
                carla.Rotation(
                    pitch=curr_tf.rotation.pitch,
                    yaw=yaw,
                    roll=curr_tf.rotation.roll,
                ),
            )
        )
    return samples


def _cap_speed_for_radius(
    base_speed: float,
    radius: Optional[float],
    max_lat_acc: float,
    min_radius: float = 1.0,
) -> float:
    """Return the capped speed in km/h respecting a lateral acceleration limit."""
    if base_speed <= 0.0 or radius is None:
        return base_speed
    usable_radius = max(radius, min_radius)
    max_speed_kmh = math.sqrt(max_lat_acc * usable_radius) * 3.6
    return min(base_speed, max_speed_kmh)


# =============================================================================
# ROUTE REFINEMENT
# Interpolates waypoints, applies arc sampling, caps speed for turns
# =============================================================================


def _refine_vehicle_route(
    vehicle_data: VehicleData,
    max_step: float = 2.0,
    max_lat_acc: float = 3.0,
) -> None:
    """Insert intermediate samples and clamp waypoint speeds for a vehicle route."""
    points = vehicle_data.route_points
    if len(points) < 2:
        return

    capped_speeds: List[float] = []
    for idx, route_point in enumerate(points):
        prev_loc = points[idx - 1].transform.location if idx > 0 else None
        next_loc = points[idx + 1].transform.location if idx + 1 < len(points) else None
        radius = _compute_turn_radius_from_locations(prev_loc, route_point.transform.location, next_loc)
        capped_speeds.append(_cap_speed_for_radius(route_point.speed_kmh, radius, max_lat_acc))

    new_points: List[RoutePoint] = []
    first_point = points[0]
    new_points.append(
        RoutePoint(
            transform=first_point.transform,
            speed_kmh=capped_speeds[0],
            idle_time_s=first_point.idle_time_s,
            is_destination=first_point.is_destination,
            speed_deviation_kmh=first_point.speed_deviation_kmh,
        )
    )

    for idx in range(len(points) - 1):
        prev_tf = points[idx - 1].transform if idx > 0 else None
        curr_tf = points[idx].transform
        next_tf = points[idx + 1].transform

        segment_speed = min(capped_speeds[idx], capped_speeds[idx + 1])
        segment_deviation = points[idx].speed_deviation_kmh
        for sample_tf in _generate_arc_samples(prev_tf, curr_tf, next_tf, max_step):
            new_points.append(
                RoutePoint(
                    transform=sample_tf,
                    speed_kmh=segment_speed,
                    idle_time_s=0.0,
                    is_destination=False,
                    speed_deviation_kmh=segment_deviation,
                )
            )

        next_point = points[idx + 1]
        new_points.append(
            RoutePoint(
                transform=next_point.transform,
                speed_kmh=capped_speeds[idx + 1],
                idle_time_s=next_point.idle_time_s,
                is_destination=next_point.is_destination,
                speed_deviation_kmh=next_point.speed_deviation_kmh,
            )
        )

    vehicle_data.route_points = new_points


# =============================================================================
# DATA CLASSES
# Route segments, route points, vehicle data, trigger types
# =============================================================================


@dataclass
class RouteSegment:
    index: int
    start: carla.Location
    target: carla.Location
    speed: float
    distance: float
    heading: float
    idle_after: float
    turn_time: float
    is_destination: bool
    next_heading: Optional[float] = None
    turn_radius: Optional[float] = None


@dataclass
class RoutePoint:
    transform: carla.Transform
    speed_kmh: float
    idle_time_s: float
    is_destination: bool
    speed_deviation_kmh: int = 0


@dataclass
class VehicleData:
    blueprint_id: str
    spawn_location: carla.Location
    spawn_rotation: carla.Rotation
    destination: carla.Location
    route_points: List[RoutePoint]
    initial_speed: float
    destination_speed: float
    color: Optional[str] = None
    initial_idle_time: float = 0.0
    ignore_traffic_lights: bool = False
    ignore_stop_signs: bool = False
    ignore_vehicles: bool = False
    trigger_center: Optional[carla.Location] = None
    trigger_radius: Optional[float] = None
    max_lat_acc: float = 3.0


@dataclass
class TriggerableActor:
    """Base class for actors with triggers - foundation for future per-actor system"""
    center: carla.Location
    radius: float
    activated: bool = False

    def check_activation(self, ego_location: carla.Location) -> bool:
        """Check if ego is within radius using 2D distance. Returns True if just activated."""
        if self.activated:
            return False
        dx = ego_location.x - self.center.x
        dy = ego_location.y - self.center.y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance <= self.radius:
            self.activated = True
            return True
        return False


@dataclass
class TrafficLightTrigger(TriggerableActor):
    """Traffic light trigger with color sequence"""
    ids: List[int] = field(default_factory=list)
    sequence: List[dict] = field(default_factory=list)  # [{color, duration_ticks}]
    current_step: int = 0
    step_start_tick: int = 0
    traffic_lights: List[carla.TrafficLight] = field(default_factory=list)
    sequence_completed: bool = False


@dataclass
class PedestrianTrigger(TriggerableActor):
    """Pedestrian trigger for individual activation"""
    pedestrian_index: int = 0  # Index in pedestrians_data list


@dataclass
class VehicleTrigger(TriggerableActor):
    """Vehicle trigger for individual activation"""
    vehicle_index: int = 0  # Index in vehicles_data list


# =============================================================================
# VEHICLE CONTROLLER
# Threaded vehicle navigation with waypoint following and trigger support
# =============================================================================


class VehicleController:
    ARRIVAL_RADIUS = 2.5
    ARRIVAL_SLACK = 1.5
    STOPPED_SPEED = 0.5
    SLEEP_INTERVAL = 0.05
    WAYPOINT_REACHED_THRESHOLD = 3.0  # meters - when to mark waypoint as reached
    DESTINATION_SLOW_RADIUS = 10.0  # meters before destination to apply final speed
    DESTINATION_SPEED_HYSTERESIS = 6.0  # meters buffer to re-arm destination speed logic

    def __init__(self, agent: BasicAgent, vehicle: carla.Actor, destination: carla.Location,
                 scenario: "vse_play", index: int, route_points: List[RoutePoint], initial_idle_time: float = 0.0,
                 destination_speed: Optional[float] = None, cruise_speed: Optional[float] = None,
                 vehicle_trigger: Optional[VehicleTrigger] = None):
        self.agent = agent
        self.vehicle = vehicle
        self.destination = destination
        self.scenario = scenario
        self.index = index
        self.waypoints = route_points
        self.initial_idle_time = initial_idle_time
        self.destination_speed = destination_speed
        self.cruise_speed = cruise_speed
        self._running = True
        self._stop_event = threading.Event()  # Local stop event for this controller
        self._thread: Optional[threading.Thread] = None
        self._done = False
        self._waypoint_speeds_set = [False] * len(route_points)
        self._waypoint_speed_choices: List[Optional[float]] = [None] * len(route_points)
        self._waypoint_reached = [False] * len(route_points)
        self._waypoint_idle_start = [None] * len(route_points)  # Track when idle started at each waypoint
        self._waypoint_idle_complete = [False] * len(route_points)  # Track if idle time completed
        self._step_counter = 0
        self._destination_speed_applied = False
        self._arrival_debug_last_distance: Optional[float] = None
        self._vehicle_trigger: Optional[VehicleTrigger] = vehicle_trigger
        self._destination_waypoint_index: Optional[int] = None
        for idx in range(len(route_points) - 1, -1, -1):
            if route_points[idx].is_destination:
                self._destination_waypoint_index = idx
                break
        if self._destination_waypoint_index is None and route_points:
            self._destination_waypoint_index = len(route_points) - 1

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _should_stop(self) -> bool:
        """Check if this controller should stop running.

        Thread-safe check that considers both local and scenario-level stop signals.
        """
        if self._stop_event.is_set():
            return True
        if not self._running:
            return True
        if self.scenario:
            # Check scenario's stop event first (faster, no lock needed)
            if hasattr(self.scenario, '_stop_event') and self.scenario._stop_event.is_set():
                return True
            # Also check the boolean flag (for backwards compatibility)
            if not getattr(self.scenario, '_keep_running', True):
                return True
        return False

    def _is_vehicle_valid(self) -> bool:
        """Check if the vehicle actor is still valid and alive."""
        return self.vehicle is not None and self.vehicle.is_alive

    def _get_current_speed_kmh(self) -> float:
        """Get current vehicle speed in km/h"""
        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0
        try:
            velocity = self.vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            return speed_ms * 3.6
        except Exception:
            return 0.0

    def _resolve_destination_speed(self) -> float:
        """Return a reasonable speed to resume with when heading to the final destination."""
        min_resume_kmh = (self.STOPPED_SPEED + 0.1) * 3.6  # avoid being treated as stopped
        chosen_destination_speed = None
        dest_idx = self._destination_waypoint_index
        if dest_idx is not None and 0 <= dest_idx < len(self.waypoints):
            try:
                chosen_destination_speed = self._resolve_waypoint_speed(dest_idx)
            except Exception:
                chosen_destination_speed = None
        candidates: List[Optional[float]] = [
            chosen_destination_speed,
            self.destination_speed,
            self.cruise_speed,
        ]
        for candidate in candidates:
            if candidate is not None and candidate > min_resume_kmh:
                return float(candidate)

        # Fall back to the largest non-null candidate even if it is small/non-positive
        numeric_candidates = [c for c in candidates if c is not None]
        if numeric_candidates:
            max_candidate = max(numeric_candidates)
            if max_candidate > 0.0:
                return max(max_candidate, min_resume_kmh)

        return max(10.0, min_resume_kmh)

    def _resolve_waypoint_speed(self, waypoint_index: int) -> float:
        """Return the chosen (randomized) km/h speed for a waypoint index."""
        if waypoint_index < 0 or waypoint_index >= len(self.waypoints):
            return 0.0
        cached = self._waypoint_speed_choices[waypoint_index]
        if cached is not None:
            return float(cached)

        wp = self.waypoints[waypoint_index]
        planned_speed = float(getattr(wp, "speed_kmh", 0.0) or 0.0)
        deviation_val = getattr(wp, "speed_deviation_kmh", 0)
        try:
            deviation_kmh = int(float(deviation_val or 0))
        except Exception:
            deviation_kmh = 0
        if deviation_kmh < 0:
            deviation_kmh = 0

        base_kmh = int(round(planned_speed))
        if deviation_kmh:
            base_kmh += random.randint(-deviation_kmh, deviation_kmh)
        if base_kmh < 0:
            base_kmh = 0
        chosen_speed = float(base_kmh)
        self._waypoint_speed_choices[waypoint_index] = chosen_speed
        return chosen_speed

    def _resolve_cruise_speed(self) -> float:
        """Speed used to cruise after leaving a waypoint when heading to the destination."""
        min_resume_kmh = (self.STOPPED_SPEED + 0.1) * 3.6
        if self.cruise_speed is not None and self.cruise_speed > min_resume_kmh:
            return float(self.cruise_speed)
        # Comfortable fallback cruise speed when none provided
        return max(30.0, min_resume_kmh)

    def _run(self):
        if self.scenario and getattr(self.scenario, "_debug", False):
            print(f"[VEHICLE] Controller {self.index} thread started (trigger_mode={self.scenario._trigger_mode})")

        # Newly spawned actors can report is_alive == False until the simulator ticks once.
        wait_start = time.monotonic()
        while (
            self._running
            and self.vehicle
            and not self.vehicle.is_alive
            and self.scenario
            and self.scenario._keep_running
            and (time.monotonic() - wait_start) < 2.0
        ):
            time.sleep(self.SLEEP_INTERVAL)

        if (
            self._running
            and self.vehicle
            and not self.vehicle.is_alive
        ):
            if self.scenario and getattr(self.scenario, "_debug", False):
                print(f"[VEHICLE] Controller {self.index}: actor never became alive after spawn")
            self._running = False
            self._update_arrival_state()
            return

        # Wait for global trigger activation if required (actors without personal triggers)
        if (
            self.scenario
            and self.scenario._trigger_mode
            and not self._vehicle_trigger
        ):
            while self._running and self.vehicle and self.vehicle.is_alive and self.scenario._keep_running:
                if getattr(self.scenario, "_global_trigger_released", False):
                    break
                try:
                    self.agent.set_target_speed(0)
                except Exception:
                    pass
                time.sleep(self.SLEEP_INTERVAL)
            if not (
                self._running
                and self.vehicle
                and self.vehicle.is_alive
                and self.scenario
                and self.scenario._keep_running
                and getattr(self.scenario, "_global_trigger_released", False)
            ):
                self._running = False
                self._update_arrival_state()
                return

        trigger_armed = False
        if self._vehicle_trigger:
            logged_wait = False
            while (
                self._running
                and self.vehicle
                and self.vehicle.is_alive
                and self.scenario
                and self.scenario._keep_running
            ):
                if self._vehicle_trigger.activated:
                    trigger_armed = True
                    break
                try:
                    self.agent.set_target_speed(0)
                except Exception:
                    pass
                if self.scenario._debug and not logged_wait:
                    print(
                        f"[VEHICLE] Controller {self.index}: waiting for personal trigger "
                        f"(R={self._vehicle_trigger.radius:.1f}m)"
                    )
                    logged_wait = True
                time.sleep(self.SLEEP_INTERVAL)

            if not (
                self._running
                and self.vehicle
                and self.vehicle.is_alive
                and self.scenario
                and self.scenario._keep_running
                and self._vehicle_trigger.activated
            ):
                if self.scenario and getattr(self.scenario, "_debug", False):
                    print(f"[VEHICLE] Controller {self.index}: trigger wait aborted (running={self._running}, activated={self._vehicle_trigger.activated})")
                self._running = False
                self._update_arrival_state()
                return
        else:
            trigger_armed = True

        # Apply initial idle time once triggers have released the actor
        if self.initial_idle_time > 0.0:
            start_delay = GameTime.get_time()
            while (
                self._running
                and self.vehicle
                and self.vehicle.is_alive
                and self.scenario
                and self.scenario._keep_running
            ):
                elapsed = GameTime.get_time() - start_delay
                if elapsed >= self.initial_idle_time:
                    break
                try:
                    self.agent.set_target_speed(0)
                except Exception:
                    pass
                time.sleep(self.SLEEP_INTERVAL)
            if not (
                self._running
                and self.vehicle
                and self.vehicle.is_alive
                and self.scenario
                and self.scenario._keep_running
            ):
                self._running = False
                self._update_arrival_state()
                return

        if trigger_armed and self.vehicle and self.vehicle.is_alive:
            try:
                self.agent.set_target_speed(self._resolve_cruise_speed())
            except Exception:
                pass

        while not self._should_stop() and self._is_vehicle_valid():
            if self._done:
                break
            now = time.monotonic()
            if self._update_arrival_state(now):
                break
            try:
                # Early exit if vehicle was destroyed during termination
                if not self._is_vehicle_valid():
                    break

                loc = self.vehicle.get_location()
                current_speed = self._get_current_speed_kmh()

                # Check each waypoint for speed changes and idle time
                last_wp_index = len(self.waypoints) - 1
                for i, wp in enumerate(self.waypoints):
                    wp_loc = wp.transform.location
                    distance = float(loc.distance(wp_loc))
                    target_speed = self._resolve_waypoint_speed(i)
                    idle_time = wp.idle_time_s

                    # Fixed detection radius for testing
                    detection_radius = 10.0

                    # Check if we're at a waypoint with idle time
                    if distance < detection_radius and not self._waypoint_reached[i]:
                        if idle_time > 0.0:
                            # Waypoint with idle time: stop the vehicle
                            self.agent.set_target_speed(0)
                            self._waypoint_speeds_set[i] = True
                            self._waypoint_reached[i] = True
                            self._waypoint_idle_start[i] = GameTime.get_time()
                        else:
                            # Normal waypoint: just set the speed
                            if not self._waypoint_speeds_set[i]:
                                self.agent.set_target_speed(target_speed)
                                self._waypoint_speeds_set[i] = True
                            if distance < self.WAYPOINT_REACHED_THRESHOLD:
                                self._waypoint_reached[i] = True

                    # Handle idle waiting at waypoints
                    if self._waypoint_reached[i] and not self._waypoint_idle_complete[i] and idle_time > 0.0:
                        if self._waypoint_idle_start[i] is not None:
                            elapsed = GameTime.get_time() - self._waypoint_idle_start[i]
                            # Keep vehicle stopped during idle period
                            self.agent.set_target_speed(0)
                            if elapsed >= idle_time:
                                # Idle time complete, resume with next waypoint's speed
                                self._waypoint_idle_complete[i] = True
                                # Set speed to next waypoint if available, else keep current
                                if i + 1 < len(self.waypoints):
                                    next_speed = self._resolve_waypoint_speed(i + 1)
                                    resume_speed = next_speed
                                    if resume_speed <= (self.STOPPED_SPEED * 3.6):
                                        resume_speed = self._resolve_destination_speed()
                                    self.agent.set_target_speed(resume_speed)
                                else:
                                    self.agent.set_target_speed(self._resolve_cruise_speed())
                                    self._destination_speed_applied = False

                # Apply destination speed as we approach the final goal
                destination_distance = float(loc.distance(self.destination))
                if destination_distance <= self.DESTINATION_SLOW_RADIUS:
                    if not self._destination_speed_applied:
                        self.agent.set_target_speed(self._resolve_destination_speed())
                        self._destination_speed_applied = True
                else:
                    if destination_distance > self.DESTINATION_SLOW_RADIUS + self.DESTINATION_SPEED_HYSTERESIS:
                        self._destination_speed_applied = False

                # Apply control only if vehicle is still valid
                if self._is_vehicle_valid():
                    control = self.agent.run_step()
                    self.vehicle.apply_control(control)

                self._step_counter += 1
            except Exception as e:
                # Log exceptions but continue loop - don't crash the thread
                logger.exception("VehicleController %d exception in main loop: %s",
                                self.index, e)
                if self.scenario and getattr(self.scenario, "_debug", False):
                    print(f"[VEHICLE] Controller {self.index} exception: {e}")
                break

            # Use interruptible sleep - check stop event more frequently
            if self._stop_event.wait(timeout=self.SLEEP_INTERVAL):
                break  # Stop event was set
        self._running = False
        self._update_arrival_state()
        self._apply_hold()

    def stop(self):
        """Signal the controller thread to stop and wait for it to exit."""
        self._running = False
        self._stop_event.set()  # Signal the thread to wake up if sleeping
        if self._thread and self._thread.is_alive():
            # Wait with longer timeout - we need threads to actually stop
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("VehicleController %d thread did not stop within 5s timeout",
                              self.index)
        self._apply_hold()

    def is_finished(self) -> bool:
        # Avoid declaring completion before we have actually attempted to drive.
        if self.scenario and self.scenario._trigger_mode:
            if self._vehicle_trigger:
                if not self._vehicle_trigger.activated:
                    return False
            elif not getattr(self.scenario, "_global_trigger_released", False):
                return False
        if self._step_counter == 0 and self._running:
            return False
        result = self._update_arrival_state()
        if self.scenario and getattr(self.scenario, "_debug", False):
            print(f"[VEHICLE] Controller {self.index} is_finished -> {result} (steps={self._step_counter}, running={self._running})")
        return result

    def _distance_to_destination(self) -> float:
        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0
        try:
            return float(self.vehicle.get_location().distance(self.destination))
        except Exception:
            return float("inf")

    def _speed(self) -> float:
        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0
        try:
            velocity = self.vehicle.get_velocity()
        except Exception:
            return 0.0
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _log_distance_to_destination(self, distance: float) -> None:
        """Log approach to destination once vehicle is within 5 meters."""
        if not (self.scenario and getattr(self.scenario, "_debug", False)):
            self._arrival_debug_last_distance = None
            return
        if distance <= 5.0:
            rounded_distance = round(distance, 2)
            if (
                self._arrival_debug_last_distance is None
                or abs(self._arrival_debug_last_distance - rounded_distance) >= 0.02
            ):
                speed_kmh = self._speed() * 3.6
                print(
                    f"[VEHICLE] Controller {self.index}: distance {distance:.2f}m to destination "
                    f"(speed={speed_kmh:.2f} km/h)"
                )
                self._arrival_debug_last_distance = rounded_distance
        else:
            self._arrival_debug_last_distance = None

    def _is_idling_at_waypoint(self) -> bool:
        """Check if vehicle is currently idling at an intermediate waypoint."""
        for i in range(len(self.waypoints)):
            if self._waypoint_reached[i] and not self._waypoint_idle_complete[i]:
                idle_time = self.waypoints[i].idle_time_s
                if idle_time > 0.0:
                    return True
        return False

    def _update_arrival_state(self, now: Optional[float] = None) -> bool:
        if self._done:
            return True

        if now is None:
            now = time.monotonic()

        vehicle = self.vehicle
        if not vehicle or not vehicle.is_alive:
            if self.scenario and getattr(self.scenario, "_debug", False):
                print(f"[VEHICLE] Controller {self.index}: vehicle missing or dead; vehicle={vehicle}")
            self._done = True
            return True

        try:
            agent_done = self.agent.done()
        except Exception:
            agent_done = False
        if agent_done:
            distance = self._distance_to_destination()
            self._log_distance_to_destination(distance)
            if distance > self.ARRIVAL_RADIUS * 3.0 or self._step_counter < 5:
                agent_done = False
                if self.scenario and getattr(self.scenario, "_debug", False):
                    print(
                        f"[VEHICLE] Controller {self.index}: suppressing premature agent.done()"
                        f" (distance={distance:.2f}m, steps={self._step_counter})"
                    )
        if agent_done:
            self._done = True
            return True

        # If vehicle is idling at an intermediate waypoint, don't mark as finished
        if self._is_idling_at_waypoint():
            return False

        distance = self._distance_to_destination()
        self._log_distance_to_destination(distance)
        threshold = self.ARRIVAL_RADIUS
        if distance <= threshold:
            if self.scenario and getattr(self.scenario, "_debug", False):
                print(f"[VEHICLE] Controller {self.index}: distance {distance:.2f} within threshold {threshold:.2f}")
            self._done = True
            return True

        if distance <= threshold + self.ARRIVAL_SLACK and self._speed() <= self.STOPPED_SPEED:
            self._done = True
            return True

        return False

    def _apply_hold(self) -> None:
        if not self._done:
            return
        vehicle = self.vehicle
        if not vehicle or not vehicle.is_alive:
            return
        try:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass


class PedestrianArrivalCriterion(AtomicBehavior):
    def __init__(self, walker: carla.Actor, destination: carla.Location, tolerance: float = 1.0):
        super().__init__("PedestrianArrivalCriterion")
        self._walker = walker
        self._destination = destination
        self._tolerance = max(0.1, tolerance)

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._walker or not self._walker.is_alive:
            return py_trees_common.Status.FAILURE
        distance = self._walker.get_location().distance(self._destination)
        if distance <= self._tolerance:
            return py_trees_common.Status.SUCCESS
        return py_trees_common.Status.RUNNING


# =============================================================================
# PEDESTRIAN BEHAVIORS
# Walker orientation, smooth turns, walking behaviors, spawn behaviors
# =============================================================================


class SetWalkerOrientation(AtomicBehavior):
    def __init__(self, walker: carla.Actor, yaw: float, name: str):
        super().__init__(name, walker)
        self._yaw = yaw

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        transform = self._actor.get_transform()
        transform.rotation.yaw = _normalize_yaw(self._yaw)
        self._actor.set_transform(transform)
        self._actor.apply_control(carla.WalkerControl())
        return py_trees_common.Status.SUCCESS


class SmoothTurn(AtomicBehavior):
    def __init__(self, walker: carla.Actor, target_yaw: float, duration: float, name: str,
                 debug_enabled: bool = False):
        super().__init__(name, walker)
        self._target_yaw = _normalize_yaw(target_yaw)
        self._duration = max(0.0, duration)
        self._start_yaw: float = 0.0
        self._delta: float = 0.0
        self._start_time: float = 0.0
        self._debug_enabled = debug_enabled

    def initialise(self) -> None:  # type: ignore[override]
        transform = self._actor.get_transform()
        self._start_yaw = _normalize_yaw(transform.rotation.yaw)
        self._delta = _normalize_yaw(self._target_yaw - self._start_yaw)
        self._start_time = GameTime.get_time()

        if abs(self._delta) < 1e-3 or self._duration <= 0.0:
            transform.rotation.yaw = self._target_yaw
            self._actor.set_transform(transform)
            self._actor.apply_control(carla.WalkerControl())
            self._duration = 0.0

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if self._duration <= 0.0 or abs(self._delta) < 1e-3:
            return py_trees_common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time
        t = min(max(elapsed / self._duration, 0.0), 1.0)
        current_yaw = _normalize_yaw(self._start_yaw + self._delta * t)

        transform = self._actor.get_transform()
        transform.rotation.yaw = current_yaw
        self._actor.set_transform(transform)
        self._actor.apply_control(carla.WalkerControl())

        if t >= 1.0:
            if self._debug_enabled:
                print(f"[SmoothTurn] {self.name} completed: final_yaw={current_yaw:.2f}")
            return py_trees_common.Status.SUCCESS
        return py_trees_common.Status.RUNNING


class EnsureWalkerAt(AtomicBehavior):
    def __init__(self, walker: carla.Actor, target: carla.Transform, tolerance: float, name: str):
        super().__init__(name, walker)
        self._target = target
        self._tolerance = max(0.0, tolerance)

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._actor or not self._actor.is_alive:
            return py_trees_common.Status.FAILURE
        current = self._actor.get_transform()
        distance = current.location.distance(self._target.location)
        if distance > self._tolerance:
            # Calculate proper Z based on ground height + bbox extent
            # Don't use target Z from JSON - it has the editor's estimated height which may be wrong
            try:
                bbox_extent_z = float(getattr(self._actor.bounding_box.extent, "z", 1.0))
            except Exception:
                bbox_extent_z = 1.0

            # Temporarily lift walker out of raycast path (like vse.py does for vehicles)
            try:
                lifted_location = carla.Location(
                    current.location.x,
                    current.location.y,
                    current.location.z + 500.0
                )
                self._actor.set_location(lifted_location)
            except Exception:
                pass

            # Get ground height at target XY position
            target_ground_height = get_ground_height(
                self._actor.get_world(),
                self._target.location,
                debug=False,
                cached_map=self._actor.get_world().get_map(),
                probe_on_miss=True
            )

            # Create corrected transform with grounded Z
            corrected_transform = carla.Transform(
                carla.Location(
                    self._target.location.x,
                    self._target.location.y,
                    target_ground_height + bbox_extent_z
                ),
                self._target.rotation
            )
            self._actor.set_transform(corrected_transform)
        else:
            current.rotation = self._target.rotation
            self._actor.set_transform(current)
        self._actor.apply_control(carla.WalkerControl())
        return py_trees_common.Status.SUCCESS


class GroundedIdle(AtomicBehavior):
    """Anchor a walker at a grounded transform for a fixed duration."""

    def __init__(
        self,
        walker: carla.Actor,
        target: carla.Transform,
        duration: float,
        world: carla.World,
        name: str,
    ):
        super().__init__(name, walker)
        self._target = carla.Transform(
            carla.Location(target.location.x, target.location.y, target.location.z),
            carla.Rotation(
                pitch=target.rotation.pitch,
                yaw=target.rotation.yaw,
                roll=target.rotation.roll,
            ),
        )
        self._duration = max(0.0, duration)
        self._world = world
        self._start_time: float = 0.0

    def initialise(self) -> None:  # type: ignore[override]
        self._start_time = GameTime.get_time()

        # Use the same approach as vse.py with bounding box height offset
        try:
            bbox_extent_z = float(getattr(self._actor.bounding_box.extent, "z", 1.0))
        except Exception:
            bbox_extent_z = 1.0

        # Temporarily lift walker out of raycast path (like vse.py does for vehicles)
        if self._actor and self._actor.is_alive:
            try:
                current_location = self._actor.get_location()
                lifted_location = carla.Location(
                    current_location.x,
                    current_location.y,
                    current_location.z + 500.0
                )
                self._actor.set_location(lifted_location)
            except Exception:
                pass

        ground_height = get_ground_height(
            self._world,
            self._target.location,
            debug=False,
            cached_map=self._world.get_map(),
            probe_on_miss=True
        )

        # Use just bbox_extent_z (not +0.2), matching the fix in other locations
        self._target.location.z = ground_height + bbox_extent_z
        self._apply()

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._actor or not self._actor.is_alive:
            return py_trees_common.Status.FAILURE

        self._apply()

        if self._duration <= 0.0:
            return py_trees_common.Status.SUCCESS

        if GameTime.get_time() - self._start_time >= self._duration:
            return py_trees_common.Status.SUCCESS

        return py_trees_common.Status.RUNNING

    def terminate(self, new_status: py_trees_common.Status) -> None:  # type: ignore[override]
        if self._actor and self._actor.is_alive:
            self._actor.apply_control(carla.WalkerControl())

    def _apply(self) -> None:
        transform = carla.Transform(
            carla.Location(
                self._target.location.x,
                self._target.location.y,
                self._target.location.z,
            ),
            carla.Rotation(
                pitch=self._target.rotation.pitch,
                yaw=self._target.rotation.yaw,
                roll=self._target.rotation.roll,
            ),
        )
        self._actor.set_transform(transform)
        self._actor.apply_control(carla.WalkerControl())


class WalkToTarget(AtomicBehavior):
    def __init__(self, walker: carla.Actor, target: carla.Location, speed: float, tolerance: float,
                 stuck_time: float, desired_yaw: Optional[float] = None, name: str = "WalkToTarget",
                 debug_enabled: bool = False, is_destination: bool = False):
        super().__init__(name, walker)
        self._target = carla.Location(target.x, target.y, target.z)
        self._speed = max(0.01, speed)
        self._tolerance = max(0.01, tolerance)
        self._stuck_time = max(0.5, stuck_time)
        self._last_progress_time: float = 0.0
        self._last_distance: float = float('inf')
        self._desired_yaw: Optional[float] = _normalize_yaw(desired_yaw) if desired_yaw is not None else None
        self._start_yaw: Optional[float] = None
        self._debug_last_log_time: float = -1.0
        self._debug_enabled = debug_enabled
        self._is_destination = is_destination

    def initialise(self) -> None:  # type: ignore[override]
        self._last_progress_time = GameTime.get_time()
        self._last_distance = float('inf')
        if self._actor and self._actor.is_alive:
            try:
                transform = self._actor.get_transform()
                self._start_yaw = _normalize_yaw(transform.rotation.yaw)
            except Exception:
                self._start_yaw = None
        else:
            self._start_yaw = None

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        now = GameTime.get_time()
        location = self._actor.get_location()
        distance = location.distance(self._target)

        if distance <= self._tolerance:
            if self._actor and self._actor.is_alive:
                transform = self._actor.get_transform()
                target_yaw = self._desired_yaw if self._desired_yaw is not None else self._start_yaw
                if target_yaw is not None:
                    transform.rotation.yaw = target_yaw
                self._actor.set_transform(transform)
                self._actor.apply_control(carla.WalkerControl())
            return py_trees_common.Status.SUCCESS

        if distance <= self._tolerance + 0.05:
            if self._actor and self._actor.is_alive:
                transform = self._actor.get_transform()
                target_yaw = self._desired_yaw if self._desired_yaw is not None else self._start_yaw

                # Calculate proper Z based on ground height + bbox extent
                # Don't use target Z from JSON - it has the editor's estimated height which may be wrong
                try:
                    bbox_extent_z = float(getattr(self._actor.bounding_box.extent, "z", 1.0))
                except Exception:
                    bbox_extent_z = 1.0

                # Temporarily lift walker out of raycast path (like vse.py does for vehicles)
                try:
                    lifted_location = carla.Location(
                        transform.location.x,
                        transform.location.y,
                        transform.location.z + 500.0
                    )
                    self._actor.set_location(lifted_location)
                except Exception:
                    pass

                # Get ground height at target XY position
                target_ground_height = get_ground_height(
                    self._actor.get_world(),
                    self._target,
                    debug=False,
                    cached_map=self._actor.get_world().get_map(),
                    probe_on_miss=True
                )

                # Set target location with corrected Z
                transform.location.x = self._target.x
                transform.location.y = self._target.y
                transform.location.z = target_ground_height + bbox_extent_z
                if target_yaw is not None:
                    transform.rotation.yaw = target_yaw
                self._actor.set_transform(transform)
                self._actor.apply_control(carla.WalkerControl())
            return py_trees_common.Status.SUCCESS

        if self._actor and self._actor.is_alive:
            target_yaw = self._desired_yaw if self._desired_yaw is not None else self._start_yaw
            if target_yaw is not None:
                transform = self._actor.get_transform()
                transform.rotation.yaw = target_yaw
                self._actor.set_transform(transform)

        direction_vec = carla.Vector3D(
            self._target.x - location.x,
            self._target.y - location.y,
            0.0,
        )
        length = math.sqrt(direction_vec.x ** 2 + direction_vec.y ** 2)
        if length > 0:
            direction_vec.x /= length
            direction_vec.y /= length

        control = carla.WalkerControl()
        if length > 0:
            control.direction = direction_vec

        speed = self._speed
        if distance < 2.0:
            speed = min(speed, max(0.5, distance / 1.0))
        control.speed = speed
        self._actor.apply_control(control)

        if distance < self._last_distance - 0.05:
            self._last_distance = distance
            self._last_progress_time = now
        else:
            time_since_progress = now - self._last_progress_time
            target_yaw = self._desired_yaw if self._desired_yaw is not None else self._start_yaw

            if self._stuck_time <= 0.0 or time_since_progress > self._stuck_time:
                transform = self._actor.get_transform()
                current_yaw = _normalize_yaw(transform.rotation.yaw)

                # Calculate proper Z based on ground height + bbox extent
                # Don't use target Z from JSON - it has the editor's estimated height which may be wrong
                try:
                    bbox_extent_z = float(getattr(self._actor.bounding_box.extent, "z", 1.0))
                except Exception:
                    bbox_extent_z = 1.0

                # Temporarily lift walker out of raycast path (like vse.py does for vehicles)
                try:
                    lifted_location = carla.Location(
                        transform.location.x,
                        transform.location.y,
                        transform.location.z + 500.0
                    )
                    self._actor.set_location(lifted_location)
                except Exception:
                    pass

                # Get ground height at target XY position
                target_ground_height = get_ground_height(
                    self._actor.get_world(),
                    self._target,
                    debug=False,
                    cached_map=self._actor.get_world().get_map(),
                    probe_on_miss=True
                )

                # Set target location with corrected Z
                transform.location.x = self._target.x
                transform.location.y = self._target.y
                transform.location.z = target_ground_height + bbox_extent_z
                if target_yaw is not None:
                    transform.rotation.yaw = target_yaw
                self._actor.set_transform(transform)
                self._actor.apply_control(carla.WalkerControl())
                if self._debug_enabled:
                    new_yaw = _normalize_yaw(transform.rotation.yaw)
                    print(
                        f"[WalkToTarget] {self.name} fallback: distance={distance:.3f}m "
                        f"elapsed={time_since_progress:.2f}s yaw={current_yaw:.2f}->{new_yaw:.2f}"
                    )
                return py_trees_common.Status.SUCCESS

        return py_trees_common.Status.RUNNING


class SingleRunSequence(Sequence):
    def __init__(self, name: str):
        super().__init__(name=name)
        self._completed = False

    def tick(self):  # type: ignore[override]
        if self._completed:
            self.status = py_trees_common.Status.SUCCESS
            self.feedback_message = "completed"
            yield self
            return

        for node in super().tick():
            yield node
        if self.status == py_trees_common.Status.SUCCESS:
            self._completed = True
            self.feedback_message = "completed"
        elif self.status == py_trees_common.Status.FAILURE:
            self._completed = True
            self.status = py_trees_common.Status.SUCCESS
            self.feedback_message = "completed"


class SpawnWalkerBehaviour(Behaviour):
    def __init__(self, scenario: "vse_play", ped_index: int):
        super().__init__(f"SpawnWalker_{ped_index}")
        self._scenario = scenario
        self._ped_index = ped_index
        self._last_attempt_time: float = -1.0

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._scenario._keep_running:
            self.feedback_message = "scenario terminated"
            return py_trees_common.Status.SUCCESS

        if 0 <= self._ped_index < len(self._scenario.other_actors) and \
                self._scenario.other_actors[self._ped_index]:
            return py_trees_common.Status.SUCCESS

        ped_data = self._scenario.pedestrians_data[self._ped_index]
        pedestrian_type = ped_data["pedestrian_type"]
        spawn_transform = ped_data["spawn_transform"]

        walker = CarlaDataProvider.request_new_actor(pedestrian_type, spawn_transform)
        if walker is None:
            now = GameTime.get_time()
            if self._scenario._debug and (now - self._last_attempt_time >= 1.0):
                print(
                    f"Spawn attempt pending for pedestrian {self._ped_index}: "
                    f"{pedestrian_type} at ({spawn_transform.location.x:.1f}, "
                    f"{spawn_transform.location.y:.1f}, {spawn_transform.location.z:.1f})"
                )
                self._last_attempt_time = now
            self.feedback_message = "spawn pending"
            return py_trees_common.Status.RUNNING

        if 0 <= self._ped_index < len(self._scenario.other_actors):
            self._scenario.other_actors[self._ped_index] = walker
        if hasattr(self._scenario, "_spawned_flags"):
            self._scenario._spawned_flags[self._ped_index] = True

        self._scenario._on_walker_spawned(self._ped_index, walker)
        self.feedback_message = "spawned"
        return py_trees_common.Status.SUCCESS


class DelayBeforeSpawn(Behaviour):
    def __init__(self, duration: float, ped_index: int, scenario: "vse_play"):
        super().__init__(f"InitialDelay_{ped_index}")
        self._duration = max(0.0, duration)
        self._start_time: Optional[float] = None
        self._ped_index = ped_index
        self._last_log_time: float = -1.0
        self._scenario = scenario

    def initialise(self) -> None:  # type: ignore[override]
        self._start_time = GameTime.get_time()

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._scenario._keep_running:
            self.feedback_message = "scenario terminated"
            return py_trees_common.Status.SUCCESS

        ped_trigger = None
        if self._ped_index < len(self._scenario._pedestrian_triggers):
            ped_trigger = self._scenario._pedestrian_triggers[self._ped_index]
        requires_global = self._scenario._pedestrian_requires_global_trigger(self._ped_index)

        if ped_trigger and not ped_trigger.activated:
            self._start_time = None
            self.feedback_message = f"waiting for personal trigger (R={ped_trigger.radius:.1f}m)"
            return py_trees_common.Status.RUNNING

        if requires_global and not getattr(self._scenario, "_global_trigger_released", False):
            self._start_time = None
            self.feedback_message = "waiting for global trigger"
            return py_trees_common.Status.RUNNING

        if self._duration <= 0.0:
            self.feedback_message = "no delay"
            return py_trees_common.Status.SUCCESS

        if self._start_time is None:
            self.initialise()

        elapsed = GameTime.get_time() - (self._start_time or 0.0)
        remaining = self._duration - elapsed

        if remaining <= 0.0:
            self.feedback_message = "delay complete"
            if self._scenario._debug:
                print(f"Initial delay complete for pedestrian {self._ped_index}")
            return py_trees_common.Status.SUCCESS

        remaining = max(0.0, remaining)
        self.feedback_message = f"waiting {remaining:.2f}s"
        if self._scenario._debug and (
            self._last_log_time < 0.0 or
            GameTime.get_time() - self._last_log_time >= 1.0
        ):
            print(
                f"Pedestrian {self._ped_index} initial delay running: "
                f"elapsed={elapsed:.2f}s remaining={remaining:.2f}s"
            )
            self._last_log_time = GameTime.get_time()
        return py_trees_common.Status.RUNNING


class ExecuteRouteBehaviour(Behaviour):
    def __init__(self, scenario: "vse_play", ped_index: int):
        super().__init__(f"ExecuteRoute_{ped_index}")
        self._scenario = scenario
        self._ped_index = ped_index
        self._tree: Optional[BehaviourTree] = None

    def initialise(self) -> None:  # type: ignore[override]
        walker = self._scenario.get_walker(self._ped_index)
        if walker is None or not walker.is_alive:
            return

        if self._tree is None:
            route_root = self._scenario._build_pedestrian_route(self._ped_index, walker)
            self._tree = BehaviourTree(route_root)

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._scenario._keep_running:
            return py_trees_common.Status.SUCCESS

        walker = self._scenario.get_walker(self._ped_index)
        if walker is None:
            self.feedback_message = "waiting for spawn"
            return py_trees_common.Status.RUNNING

        if not walker.is_alive:
            self.feedback_message = "walker destroyed"
            return py_trees_common.Status.FAILURE

        requires_global = self._scenario._pedestrian_requires_global_trigger(self._ped_index)
        if self._ped_index < len(self._scenario._pedestrian_triggers):
            ped_trigger = self._scenario._pedestrian_triggers[self._ped_index]
            if ped_trigger and not ped_trigger.activated:
                self.feedback_message = f"waiting for personal trigger (R={ped_trigger.radius:.1f}m)"
                return py_trees_common.Status.RUNNING

        if requires_global and not self._scenario._global_trigger_released:
            self.feedback_message = "waiting for global trigger"
            return py_trees_common.Status.RUNNING

        if self._tree is None:
            self.initialise()
            if self._tree is None:
                self.feedback_message = "failed to build route"
                return py_trees_common.Status.FAILURE

        self._tree.tick()
        status = self._tree.root.status
        if status == py_trees_common.Status.SUCCESS:
            self._scenario.completion_status[self._ped_index] = True
            return py_trees_common.Status.SUCCESS
        if status == py_trees_common.Status.FAILURE:
            return py_trees_common.Status.FAILURE
        return py_trees_common.Status.RUNNING


class TrackingParallel(Parallel):
    def __init__(self, *, name: str, policy: py_trees_common.ParallelPolicy, scenario: "vse_play") -> None:
        super().__init__(name=name, policy=policy)
        self._scenario = scenario

    def tick(self):  # type: ignore[override]
        for node in super().tick():
            yield node
        if self._scenario._debug:
            print(
                f"[TrackingParallel] name={self.name} status={self.status} "
                f"children={[child.status for child in self.children]}"
            )


class AllPedestriansArrivedCriterion(Criterion):
    def __init__(self, scenario: "vse_play"):
        spectator = None
        try:
            if scenario._world:
                spectator = scenario._world.get_spectator()
        except Exception:
            spectator = None
        if spectator is None:
            try:
                world = CarlaDataProvider.get_world()
                spectator = world.get_spectator() if world else None
            except Exception:
                spectator = None

        super().__init__("AllPedestriansArrived", actor=spectator, optional=False)
        self._scenario = scenario

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        completed_flags = list(self._scenario.completion_status)
        self.actual_value = sum(1 for flag in completed_flags if flag)

        # If no pedestrians, criterion succeeds immediately
        if not completed_flags:
            self.test_status = "SUCCESS"
            return py_trees_common.Status.SUCCESS

        for idx in range(len(completed_flags)):
            if completed_flags[idx]:
                continue

            walker = self._scenario.get_walker(idx)
            if walker is None:
                self.test_status = "RUNNING"
                return py_trees_common.Status.RUNNING

            if not walker.is_alive:
                self.test_status = "FAILURE"
                return py_trees_common.Status.FAILURE

        if all(completed_flags):
            self.test_status = "SUCCESS"
            return py_trees_common.Status.SUCCESS

        self.test_status = "RUNNING"
        return py_trees_common.Status.RUNNING


class DestinationCriterion(AtomicBehavior):
    def __init__(self, vehicle: carla.Actor, destination: carla.Location):
        super().__init__("VehicleDestinationCriterion", vehicle)
        self._vehicle = vehicle
        self._destination = destination

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._vehicle or not self._vehicle.is_alive:
            return py_trees_common.Status.FAILURE
        if self._vehicle.get_location().distance(self._destination) <= VehicleController.ARRIVAL_RADIUS:
            return py_trees_common.Status.SUCCESS
        return py_trees_common.Status.RUNNING


class TrafficLightTriggerMonitor(Behaviour):
    """Monitors and activates traffic light triggers based on ego vehicle position"""

    def __init__(self, scenario: "vse_play"):
        super().__init__("TrafficLightTriggerMonitor")
        self.scenario = scenario
        self._world = None
        self._initialized = False
        self._controlled_lights = []  # For cleanup
        self._completed = False

    def initialise(self) -> None:
        """Initialize monitor and retrieve traffic light actors"""
        # Only initialize once
        if self._initialized:
            return

        self._world = CarlaDataProvider.get_world()
        if not self._world:
            print("[TRAFFIC_LIGHT] ERROR: World not available")
            return

        # Retrieve traffic light actors by ID
        for trigger in self.scenario._traffic_light_triggers:
            trigger.traffic_lights.clear()
            for light_id in trigger.ids:
                actor = self._world.get_actor(light_id)
                if actor is None:
                    print(f"[TRAFFIC_LIGHT] WARNING: Traffic light ID {light_id} not found in world")
                elif 'traffic_light' not in actor.type_id:
                    print(f"[TRAFFIC_LIGHT] WARNING: Actor ID {light_id} is not a traffic light")
                else:
                    trigger.traffic_lights.append(actor)

            if not trigger.traffic_lights:
                print(f"[TRAFFIC_LIGHT] WARNING: Trigger has no valid traffic lights")
                trigger.sequence_completed = True

        self._initialized = True
        print(f"[TRAFFIC_LIGHT] Monitor initialized with {len(self.scenario._traffic_light_triggers)} triggers")

    def update(self) -> py_trees_common.Status:
        """Main update - runs every tick"""
        # Increment global tick counter
        self.scenario._global_tick_counter += 1

        # Check for critical errors
        if not self._world:
            print("[TRAFFIC_LIGHT] ERROR: World not available")
            return py_trees_common.Status.FAILURE

        # If no triggers, return SUCCESS (nothing to do)
        if not self.scenario._traffic_light_triggers:
            return py_trees_common.Status.SUCCESS

        if not self.scenario._keep_running:
            if not self._completed:
                self._finalize_triggers()
                self._completed = True
            else:
                self.cleanup()
            return py_trees_common.Status.SUCCESS

        # Check if ego vehicle exists
        ego_vehicle = None
        if self.scenario.ego_vehicles and len(self.scenario.ego_vehicles) > 0:
            ego_vehicle = self.scenario.ego_vehicles[0]
            if ego_vehicle and not ego_vehicle.is_alive:
                ego_vehicle = None

        # If no ego vehicle, skip activation checks but continue running
        if not ego_vehicle:
            # Still update active sequences
            self._update_active_sequences()
            if self._all_triggers_completed():
                if not self._completed:
                    self.cleanup()
                self._completed = True
                return py_trees_common.Status.SUCCESS
            return py_trees_common.Status.RUNNING

        ego_location = ego_vehicle.get_location()

        # Check for trigger activations
        for idx, trigger in enumerate(self.scenario._traffic_light_triggers):
            if trigger.check_activation(ego_location):
                self._activate_trigger(idx, trigger, ego_location)

        # Update active sequences
        self._update_active_sequences()

        if self._all_triggers_completed():
            if not self._completed:
                self.cleanup()
            self._completed = True
            return py_trees_common.Status.SUCCESS

        return py_trees_common.Status.RUNNING

    def _activate_trigger(self, idx: int, trigger: TrafficLightTrigger, ego_location: carla.Location) -> None:
        """Activate a trigger and start its sequence"""
        print(f"[TRAFFIC_LIGHT] Tick {self.scenario._global_tick_counter}: "
              f"Trigger {idx} activated at ({ego_location.x:.2f}, {ego_location.y:.2f}, {ego_location.z:.2f}), "
              f"radius {trigger.radius:.1f}m")

        # Start first step
        trigger.current_step = 0
        trigger.step_start_tick = self.scenario._global_tick_counter
        trigger.sequence_completed = False
        self._execute_sequence_step(idx, trigger)

    def _execute_sequence_step(self, idx: int, trigger: TrafficLightTrigger) -> None:
        """Execute a single step in the sequence"""
        if trigger.current_step >= len(trigger.sequence):
            # Sequence complete
            self._complete_trigger_sequence(idx, trigger)
            return

        step = trigger.sequence[trigger.current_step]
        color_str = step.get("color", "").lower()
        duration_ticks = step.get("duration_ticks", 0)

        # Map color string to CARLA state
        color_map = {
            "red": carla.TrafficLightState.Red,
            "yellow": carla.TrafficLightState.Yellow,
            "green": carla.TrafficLightState.Green,
            "off": carla.TrafficLightState.Off
        }

        if color_str not in color_map:
            print(f"[TRAFFIC_LIGHT] Tick {self.scenario._global_tick_counter}: "
                  f"WARNING: Invalid color '{step.get('color')}' in trigger {idx} step {trigger.current_step}, skipping")
            # Skip to next step
            trigger.current_step += 1
            trigger.step_start_tick = self.scenario._global_tick_counter
            self._execute_sequence_step(idx, trigger)
            return

        state = color_map[color_str]
        duration_s = duration_ticks / 20.0

        # Set traffic lights to this state
        light_ids = []
        for light in trigger.traffic_lights:
            try:
                light.freeze(True)
                light.set_state(state)
                light_ids.append(light.id)
                if light not in self._controlled_lights:
                    self._controlled_lights.append(light)
            except Exception as e:
                print(f"[TRAFFIC_LIGHT] ERROR setting light {light.id}: {e}")

        print(f"[TRAFFIC_LIGHT] Tick {self.scenario._global_tick_counter}: "
              f"Setting lights {light_ids} to {color_str.capitalize()} for {duration_ticks} ticks ({duration_s:.1f}s)")

    def _update_active_sequences(self) -> None:
        """Update all active sequences and check for step completion"""
        current_tick = self.scenario._global_tick_counter

        for idx, trigger in enumerate(self.scenario._traffic_light_triggers):
            if not trigger.activated or trigger.current_step >= len(trigger.sequence):
                continue

            step = trigger.sequence[trigger.current_step]
            duration_ticks = step.get("duration_ticks", 0)
            elapsed_ticks = current_tick - trigger.step_start_tick

            if elapsed_ticks >= duration_ticks:
                # Step complete
                print(f"[TRAFFIC_LIGHT] Tick {current_tick}: "
                      f"Trigger {idx} step {trigger.current_step + 1}/{len(trigger.sequence)} complete")

                # Move to next step
                trigger.current_step += 1
                trigger.step_start_tick = current_tick
                self._execute_sequence_step(idx, trigger)

    def _complete_trigger_sequence(self, idx: int, trigger: TrafficLightTrigger) -> None:
        """Complete a trigger sequence and release traffic lights"""
        if trigger.sequence_completed:
            return

        light_ids = [light.id for light in trigger.traffic_lights]

        # Unfreeze all lights in this trigger
        for light in trigger.traffic_lights:
            try:
                light.freeze(False)
            except Exception as e:
                print(f"[TRAFFIC_LIGHT] ERROR unfreezing light {light.id}: {e}")

        print(f"[TRAFFIC_LIGHT] Tick {self.scenario._global_tick_counter}: "
              f"Trigger {idx} sequence complete, releasing lights {light_ids}")
        trigger.sequence_completed = True

    def _all_triggers_completed(self) -> bool:
        """Return True when every trigger has completed its sequence"""
        if not self.scenario._traffic_light_triggers:
            return True

        for trigger in self.scenario._traffic_light_triggers:
            if trigger.sequence and not trigger.sequence_completed:
                return False
        return True

    def _finalize_triggers(self) -> None:
        """Force completion of all active triggers and release their lights"""
        for idx, trigger in enumerate(self.scenario._traffic_light_triggers):
            if trigger.activated and not trigger.sequence_completed:
                trigger.current_step = len(trigger.sequence)
                self._complete_trigger_sequence(idx, trigger)
        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup - unfreeze all controlled traffic lights"""
        if not self._controlled_lights:
            return

        print(f"[TRAFFIC_LIGHT] Cleanup: Unfreezing {len(self._controlled_lights)} controlled traffic lights")

        for light in self._controlled_lights:
            try:
                light.freeze(False)
            except Exception as e:
                print(f"[TRAFFIC_LIGHT] ERROR during cleanup for light {light.id}: {e}")

        self._controlled_lights.clear()


class PedestrianTriggerMonitor(Behaviour):
    """Monitors and activates pedestrian triggers based on ego vehicle position"""

    def __init__(self, scenario: "vse_play"):
        super().__init__("PedestrianTriggerMonitor")
        self.scenario = scenario

    def update(self) -> py_trees_common.Status:
        """Main update - runs every tick"""
        # Check if scenario is complete
        if not self.scenario._keep_running:
            return py_trees_common.Status.SUCCESS

        # Read tick counter for logging (don't increment - traffic light monitor does that)
        current_tick = self.scenario._global_tick_counter

        # If no pedestrian triggers, return SUCCESS (nothing to do)
        if not self.scenario._pedestrian_triggers or not any(self.scenario._pedestrian_triggers):
            return py_trees_common.Status.SUCCESS

        # Check if ego vehicle exists
        ego_vehicle = None
        if self.scenario.ego_vehicles and len(self.scenario.ego_vehicles) > 0:
            ego_vehicle = self.scenario.ego_vehicles[0]
            if ego_vehicle and not ego_vehicle.is_alive:
                ego_vehicle = None

        # If no ego vehicle, auto-activate all triggers on first update
        if not ego_vehicle:
            if not self.scenario._pedestrian_auto_triggered:
                self.scenario._pedestrian_auto_triggered = True
                activated_count = 0
                for ped_trigger in self.scenario._pedestrian_triggers:
                    if ped_trigger and not ped_trigger.activated:
                        ped_trigger.activated = True
                        activated_count += 1
                if activated_count > 0:
                    print(f"[PEDESTRIAN_TRIGGER] Tick {current_tick}: "
                          f"No ego vehicle - auto-activating {activated_count} pedestrian triggers")
            if all(t is None or t.activated for t in self.scenario._pedestrian_triggers):
                return py_trees_common.Status.SUCCESS
            return py_trees_common.Status.RUNNING

        ego_location = ego_vehicle.get_location()

        # Check for trigger activations
        for idx, ped_trigger in enumerate(self.scenario._pedestrian_triggers):
            if ped_trigger and ped_trigger.check_activation(ego_location):
                print(f"[PEDESTRIAN_TRIGGER] Tick {current_tick}: "
                      f"Pedestrian {idx} trigger activated at "
                      f"({ego_location.x:.2f}, {ego_location.y:.2f}, {ego_location.z:.2f}), "
                      f"radius {ped_trigger.radius:.1f}m")

        if all(t is None or t.activated for t in self.scenario._pedestrian_triggers):
            return py_trees_common.Status.SUCCESS
        return py_trees_common.Status.RUNNING


class VehicleTriggerMonitor(Behaviour):
    """Monitors and activates vehicle triggers based on ego vehicle position"""

    def __init__(self, scenario: "vse_play"):
        super().__init__("VehicleTriggerMonitor")
        self.scenario = scenario

    def update(self) -> py_trees_common.Status:
        if not self.scenario._keep_running:
            return py_trees_common.Status.SUCCESS

        current_tick = self.scenario._global_tick_counter

        if not self.scenario._vehicle_triggers or not any(self.scenario._vehicle_triggers):
            return py_trees_common.Status.SUCCESS

        ego_vehicle = None
        if self.scenario.ego_vehicles and len(self.scenario.ego_vehicles) > 0:
            ego_vehicle = self.scenario.ego_vehicles[0]
            if ego_vehicle and not ego_vehicle.is_alive:
                ego_vehicle = None

        if not ego_vehicle:
            if not self.scenario._vehicle_auto_triggered:
                self.scenario._vehicle_auto_triggered = True
                activated = 0
                for veh_trigger in self.scenario._vehicle_triggers:
                    if veh_trigger and not veh_trigger.activated:
                        veh_trigger.activated = True
                        activated += 1
                if activated > 0:
                    print(f"[VEHICLE_TRIGGER] Tick {current_tick}: No ego vehicle - auto-activating {activated} vehicle triggers")
            if all(t is None or t.activated for t in self.scenario._vehicle_triggers):
                return py_trees_common.Status.SUCCESS
            return py_trees_common.Status.RUNNING

        ego_location = ego_vehicle.get_location()

        for idx, veh_trigger in enumerate(self.scenario._vehicle_triggers):
            if veh_trigger and veh_trigger.check_activation(ego_location):
                print(
                    f"[VEHICLE_TRIGGER] Tick {current_tick}: Vehicle {idx} trigger activated at "
                    f"({ego_location.x:.2f}, {ego_location.y:.2f}, {ego_location.z:.2f}), "
                    f"radius {veh_trigger.radius:.1f}m"
                )

        if all(t is None or t.activated for t in self.scenario._vehicle_triggers):
            return py_trees_common.Status.SUCCESS
        return py_trees_common.Status.RUNNING


class VehicleLightsBehavior(py_trees.behaviour.Behaviour):
    """Turn vehicle lights on/off based on weather, without touching map day/night cycle."""

    SUN_ALTITUDE_THRESHOLD_1 = 15
    SUN_ALTITUDE_THRESHOLD_2 = 165
    CLOUDINESS_THRESHOLD = 80
    FOG_THRESHOLD = 40
    COMBINED_THRESHOLD = 10

    def __init__(self, ego_vehicle, radius=50, radius_increase=15, name="VehicleLightsBehavior"):
        super().__init__(name)
        self._ego_vehicle = ego_vehicle
        self._radius = radius
        self._radius_increase = radius_increase
        self._world = CarlaDataProvider.get_world()
        self._vehicle_lights = (
            carla.VehicleLightState.Position
            | carla.VehicleLightState.LowBeam
            | carla.VehicleLightState.HighBeam
            | carla.VehicleLightState.Fog
        )

    def _get_night_mode(self, weather):
        altitude_dist = weather.sun_altitude_angle - self.SUN_ALTITUDE_THRESHOLD_1
        altitude_dist = min(altitude_dist, self.SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle)
        cloudiness_dist = self.CLOUDINESS_THRESHOLD - weather.cloudiness
        fog_density_dist = self.FOG_THRESHOLD - weather.fog_density

        if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
            return True

        joined_threshold = int(altitude_dist < self.COMBINED_THRESHOLD)
        joined_threshold += int(cloudiness_dist < self.COMBINED_THRESHOLD)
        joined_threshold += int(fog_density_dist < self.COMBINED_THRESHOLD)

        return joined_threshold >= 2

    def update(self):
        new_status = py_trees_common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._ego_vehicle)
        if not location:
            return new_status

        try:
            weather = self._world.get_weather()
        except Exception:
            return new_status

        night_mode = self._get_night_mode(weather)
        ego_speed = CarlaDataProvider.get_velocity(self._ego_vehicle) or 0.0
        radius = max(self._radius, self._radius_increase * ego_speed)

        try:
            all_vehicles = CarlaDataProvider.get_all_actors().filter("*vehicle.*")
        except Exception:
            all_vehicles = []

        for vehicle in all_vehicles:
            try:
                dist_ok = vehicle.get_location().distance(location) <= radius
                lights = vehicle.get_light_state()
                if night_mode and dist_ok:
                    lights |= self._vehicle_lights
                else:
                    lights &= ~self._vehicle_lights
                vehicle.set_light_state(carla.VehicleLightState(lights))
            except Exception:
                continue

        return new_status


# =============================================================================
# MAIN SCENARIO CLASS
# vse_play(BasicScenario) - main scenario implementation for ScenarioRunner
# =============================================================================


class vse_play(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=18000):
        global _current_scenario

        self.timeout = timeout
        # Keep reference to the incoming config for downstream helpers that expect it.
        self.config = config
        self._world = world
        self._map = world.get_map()
        self._large_map_active = _is_large_map(self._map)
        self._debug = debug_mode

        # Thread synchronization primitives
        self._state_lock = threading.Lock()  # Protects _keep_running and other shared state
        self._stop_event = threading.Event()  # Signaled when scenario should stop
        self._keep_running = True
        self._highlight_callback_id = None
        self._cleanup_done = False
        self._scenario_completed = False
        self._completion_reason: Optional[str] = None

        self._scenario_json_path = self._resolve_scenario_json(config.name)

        self._raw_pedestrian_entries: List[dict] = []
        self._raw_vehicle_entries: List[dict] = []

        self._vehicles_data: List[VehicleData] = []
        self._vehicle_controllers: List[VehicleController] = []
        self._vehicle_actors: List[carla.Actor] = []
        self._ego_route_for_criteria: List[Tuple[carla.Transform, RoadOption]] = []
        self._ego_destination: Optional[carla.Location] = None

        # Trigger system
        self._trigger_data: Optional[dict] = None  # Trigger zone data: {x, y, z, radius}
        self._trigger_mode: bool = False  # True if scenario has a trigger
        self._scenario_triggered: bool = False  # True if trigger has been activated
        self._global_trigger_released: bool = False

        self._weather_keyframes: List[Tuple[float, carla.WeatherParameters]] = []

        self._split_actor_entries()
        self._global_trigger_released = not self._trigger_mode

        self._weather_keyframes = self._load_weather_keyframes()

        self.pedestrians_data: List[dict] = []
        self.routes: List[List[dict]] = []
        self.all_segments: List[List[RouteSegment]] = []
        self.final_destinations: List[carla.Location] = []
        self.completion_status: List[bool] = []
        self.expected_durations: List[float] = []
        self.other_actors: List[Optional[carla.Actor]] = []
        self._spawned_flags: List[bool] = []
        self._walker_colors: List[carla.Color] = []
        self._criteria_nodes: List[py_trees.behaviour.Behaviour] = []

        # Pedestrian and vehicle trigger systems - initialize before _prepare_routes()
        self._pedestrian_triggers: List[Optional[PedestrianTrigger]] = []
        self._pedestrian_auto_triggered = False  # Track if auto-triggered when no ego
        self._vehicle_triggers: List[Optional[VehicleTrigger]] = []
        self._vehicle_auto_triggered = False

        # Initialize CarlaDataProvider (avoid GRP precompute on large maps).
        try:
            _init_carla_data_provider(world)
        except Exception:
            pass

        self._load_config_from_json()
        self._prepare_routes()
        self._ego_route_for_criteria = self._load_ego_route_for_criteria()

        if self._weather_keyframes:
            try:
                config.weather = self._weather_keyframes[0][1]
            except Exception:
                pass

        super().__init__("vse_play", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

        # Register this scenario as the current one (thread-safe)
        global _current_scenario
        with _current_scenario_lock:
            _current_scenario = self

        # Global tick counter for deterministic timing (used by traffic lights, future: pedestrians)
        self._global_tick_counter = 0

        # Traffic light trigger system
        self._traffic_light_triggers: List[TrafficLightTrigger] = []
        self._traffic_light_monitor = None  # Reference for cleanup

        # Load traffic light triggers
        self._load_traffic_light_triggers()

    def _resolve_scenario_json(self, scenario_name: Optional[str]) -> str:
        # Prefer an explicit path provided via environment (set by the launcher/UI)
        env_path = os.environ.get("VSE_SCENARIO_JSON_PATH")
        if env_path:
            candidate = os.path.abspath(os.path.expanduser(env_path))
            if os.path.isfile(candidate):
                return candidate

        script_dir = os.path.dirname(os.path.abspath(__file__))

        if not scenario_name:
            raise RuntimeError(
                "No scenario name provided. Please load a scenario file in the VSE editor before running."
            )

        # Try scenario_name.json in script directory
        candidate = os.path.join(script_dir, f"{scenario_name}.json")
        if os.path.isfile(candidate):
            return candidate

        # Scenario file not found
        raise RuntimeError(
            f"Unable to locate scenario data for '{scenario_name}'. Expected file: {candidate}\n"
            f"Please ensure the scenario file exists and has been saved in the VSE editor."
        )

    def _split_actor_entries(self) -> None:
        json_path = self._scenario_json_path

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        actors = data.get("vehicles", [])
        if not actors:
            ego_entry = data.get("ego_vehicle")
            if not ego_entry:
                raise RuntimeError(f"{json_path} does not contain any actors")
            # Allow ego-only scenarios: keep vehicle lists empty.

        for entry in actors:
            actor_type = entry.get("type", "")
            if actor_type.startswith("walker."):
                self._raw_pedestrian_entries.append(entry)
            else:
                self._raw_vehicle_entries.append(entry)

        # Load trigger data
        trigger_data = data.get("trigger")
        if trigger_data:
            loc = trigger_data.get("location", {})
            self._trigger_data = {
                'x': loc.get('x', 0.0),
                'y': loc.get('y', 0.0),
                'z': loc.get('z', 0.0),
                'radius': trigger_data.get('radius', 10.0)
            }
            self._trigger_mode = True
            print(f"[TRIGGER] Loaded trigger at ({self._trigger_data['x']:.2f}, {self._trigger_data['y']:.2f}, {self._trigger_data['z']:.2f}) with radius {self._trigger_data['radius']:.2f}m")
            print(f"[TRIGGER] Scenario will start when ego vehicle enters trigger zone")
        else:
            print("[TRIGGER] No trigger found - scenario will start immediately")

    def _load_config_from_json(self) -> None:
        for idx, actor in enumerate(self._raw_pedestrian_entries):
            pedestrian_type = actor.get("type", "walker.pedestrian.0001")
            loc = actor["location"]
            rot = actor["rotation"]

            spawn_location = carla.Location(loc["x"], loc["y"], loc["z"])
            spawn_rotation = carla.Rotation(rot["pitch"], rot["yaw"], rot["roll"])

            default_speed_kmh = actor.get("speed_km_h", 5.0)
            initial_idle_time = actor.get("idle_time_s", 0.0)
            default_turn_time = actor.get("turn_time_s", 0.0)

            waypoints = actor.get("waypoints", [])
            if not waypoints:
                raise RuntimeError(f"No waypoints defined for pedestrian {idx}")

            route = []
            for wp in waypoints:
                wp_loc = wp["location"]
                deviation_val = wp.get("speed_deviation_km_h", 0)
                try:
                    deviation_val = int(float(deviation_val or 0))
                except Exception:
                    deviation_val = 0
                if deviation_val < 0:
                    deviation_val = 0
                route.append({
                    "location": carla.Location(wp_loc["x"], wp_loc["y"], wp_loc["z"]),
                    "speed_km_h": wp.get("speed_km_h", default_speed_kmh),
                    "speed_deviation_km_h": deviation_val,
                    "idle_time_s": wp.get("idle_time_s", 0.0),
                    "turn_time_s": wp.get("turn_time_s", default_turn_time),
                    "is_destination": str(wp.get("index")) == "destination",
                })

            # Load pedestrian trigger data
            trigger_data = actor.get("trigger")
            pedestrian_trigger = None
            if trigger_data:
                center_data = trigger_data["center"]
                pedestrian_trigger = {
                    "center": carla.Location(
                        x=center_data.get("x", 0.0),
                        y=center_data.get("y", 0.0),
                        z=center_data.get("z", 0.0)
                    ),
                    "radius": trigger_data["radius"]
                }

            ped_data = {
                "pedestrian_type": pedestrian_type,
                "spawn_location": spawn_location,
                "spawn_rotation": spawn_rotation,
                "spawn_transform": carla.Transform(spawn_location, spawn_rotation),
                "route": route,
                "initial_idle_time": initial_idle_time,
                "default_speed_kmh": default_speed_kmh,
                "default_turn_time": default_turn_time,
            }
            if pedestrian_trigger:
                ped_data["trigger"] = pedestrian_trigger
            self.pedestrians_data.append(ped_data)
            self.routes.append(route)
            self.completion_status.append(False)

    def _prepare_routes(self) -> None:
        if self.pedestrians_data:
            for ped_idx, ped_data in enumerate(self.pedestrians_data):
                # Use exact spawn location from JSON (already validated by editor)
                # The editor computes Z using raycast + pedestrian height offset (~1.1m)
                # Re-raycasting here with different parameters causes spawn failures
                spawn_location = carla.Location(
                    ped_data["spawn_location"].x,
                    ped_data["spawn_location"].y,
                    ped_data["spawn_location"].z
                )
                spawn_rotation = ped_data["spawn_rotation"]
                spawn_rotation.yaw = _normalize_yaw(spawn_rotation.yaw)
                route = ped_data["route"]
                initial_idle_time = ped_data["initial_idle_time"]

                segments = []
                current_location = spawn_location
                current_heading = spawn_rotation.yaw

                for index, waypoint in enumerate(route, start=1):
                    # Use exact waypoint location from JSON (already validated by editor)
                    target = carla.Location(
                        waypoint["location"].x,
                        waypoint["location"].y,
                        waypoint["location"].z
                    )
                    planned_speed_kmh = waypoint.get("speed_km_h", 0.0)
                    try:
                        planned_speed_kmh = float(planned_speed_kmh)
                    except Exception:
                        planned_speed_kmh = 0.0
                    deviation_val = waypoint.get("speed_deviation_km_h", 0)
                    try:
                        deviation_kmh = int(float(deviation_val or 0))
                    except Exception:
                        deviation_kmh = 0
                    if deviation_kmh < 0:
                        deviation_kmh = 0

                    speed_kmh = int(round(planned_speed_kmh))
                    if deviation_kmh:
                        speed_kmh += random.randint(-deviation_kmh, deviation_kmh)
                    if speed_kmh < 0:
                        speed_kmh = 0
                    speed_mps = speed_kmh / 3.6
                    distance = _distance(current_location, target)
                    heading = _compute_heading(current_location, target, current_heading)

                    segment = RouteSegment(
                        index=index,
                        start=current_location,
                        target=target,
                        speed=speed_mps,
                        distance=distance,
                        heading=heading,
                        idle_after=max(0.0, waypoint["idle_time_s"]),
                        turn_time=max(0.0, waypoint["turn_time_s"]),
                        is_destination=waypoint["is_destination"],
                    )
                    segments.append(segment)
                    current_location = target
                    current_heading = heading

                for idx, segment in enumerate(segments):
                    if idx + 1 < len(segments):
                        nxt = segments[idx + 1]
                        segment.next_heading = nxt.heading
                        segment.turn_radius = _estimate_turn_radius(
                            segment.heading,
                            nxt.heading,
                            max(segment.distance, 1.0),
                            max(nxt.distance, 1.0),
                        )
                    else:
                        segment.next_heading = segment.heading
                        segment.turn_radius = None

                total_time = initial_idle_time
                for segment in segments:
                    travel_time = segment.distance / segment.speed if segment.speed > 0 else 0.0
                    total_time += travel_time + segment.idle_after
                    if not segment.is_destination:
                        total_time += segment.turn_time

                self.all_segments.append(segments)
                self.expected_durations.append(total_time)
                final_destination = segments[-1].target if segments else spawn_location
                self.final_destinations.append(final_destination)

                ped_data["spawn_location"] = spawn_location
                ped_data["spawn_rotation"] = spawn_rotation
                ped_data["spawn_transform"] = carla.Transform(spawn_location, spawn_rotation)
                ped_data["segments"] = segments
                ped_data["expected_duration"] = total_time
                ped_data["final_destination"] = final_destination
        else:
            self.all_segments = []
            self.expected_durations = []
            self.final_destinations = []

        # Create pedestrian trigger instances
        self._pedestrian_triggers = []
        for ped_idx, ped_data in enumerate(self.pedestrians_data):
            trigger_data = ped_data.get("trigger")
            if trigger_data:
                trigger = PedestrianTrigger(
                    center=trigger_data["center"],
                    radius=trigger_data["radius"],
                    pedestrian_index=ped_idx
                )
                self._pedestrian_triggers.append(trigger)
                print(f"[PEDESTRIAN_TRIGGER] Loaded trigger for pedestrian {ped_idx}: "
                      f"center ({trigger.center.x:.2f}, {trigger.center.y:.2f}, {trigger.center.z:.2f}), "
                      f"radius {trigger.radius:.1f}m")
            else:
                # No trigger for this pedestrian (future-proof, though not expected)
                self._pedestrian_triggers.append(None)

        if self._pedestrian_triggers and any(t is not None for t in self._pedestrian_triggers):
            active_count = sum(1 for t in self._pedestrian_triggers if t is not None)
            print(f"[PEDESTRIAN_TRIGGER] Loaded {active_count} pedestrian triggers")

        self._vehicles_data.clear()
        for entry in self._raw_vehicle_entries:
            loc = entry["location"]
            rot = entry["rotation"]
            spawn_location = carla.Location(loc["x"], loc["y"], loc["z"])
            spawn_rotation = carla.Rotation(rot["pitch"], rot["yaw"], rot["roll"])
            destination_speed = entry.get("destination_speed_km_h", entry.get("speed_km_h", 30.0))
            route_points: List[RoutePoint] = []
            destination: Optional[carla.Location] = None
            trigger_center: Optional[carla.Location] = None
            trigger_radius: Optional[float] = None
            max_lat_acc = float(entry.get("max_lat_acc", 3.0) or 3.0)
            if max_lat_acc <= 0.0:
                max_lat_acc = 3.0

            for wp in entry.get("waypoints", []):
                wp_loc = wp["location"]
                location = carla.Location(wp_loc["x"], wp_loc["y"], wp_loc["z"])
                yaw_value = wp.get("yaw")
                yaw = spawn_rotation.yaw if yaw_value is None else float(yaw_value)
                rotation = carla.Rotation(spawn_rotation.pitch, yaw, spawn_rotation.roll)
                deviation_val = wp.get("speed_deviation_km_h", 0)
                try:
                    deviation_val = int(float(deviation_val or 0))
                except Exception:
                    deviation_val = 0
                if deviation_val < 0:
                    deviation_val = 0
                is_destination = str(wp.get("index")) == "destination"
                if is_destination:
                    destination = carla.Location(location.x, location.y, location.z)
                    destination_speed = wp.get("speed_km_h", destination_speed)
                route_points.append(
                    RoutePoint(
                        transform=carla.Transform(location, rotation),
                        speed_kmh=wp.get("speed_km_h", entry.get("speed_km_h", 30.0)),
                        idle_time_s=wp.get("idle_time_s", 0.0),
                        is_destination=is_destination,
                        speed_deviation_kmh=deviation_val,
                    )
                )

            if destination is None:
                if route_points:
                    destination = route_points[-1].transform.location
                else:
                    destination = spawn_location

            vehicle_trigger_data = entry.get("trigger")
            if vehicle_trigger_data:
                center_data = vehicle_trigger_data.get("center", {})
                trigger_center = carla.Location(
                    x=float(center_data.get("x", 0.0)),
                    y=float(center_data.get("y", 0.0)),
                    z=float(center_data.get("z", 0.0)),
                )
                trigger_radius = max(5.0, float(vehicle_trigger_data.get("radius", 5.0)))

            self._vehicles_data.append(VehicleData(
                blueprint_id=entry.get("type", "vehicle.tesla.model3"),
                spawn_location=spawn_location,
                spawn_rotation=spawn_rotation,
                destination=destination,
                route_points=route_points,
                initial_speed=entry.get("speed_km_h", 30.0),
                destination_speed=destination_speed,
                color=entry.get("color", None),
                initial_idle_time=entry.get("idle_time_s", 0.0),
                ignore_traffic_lights=entry.get("ignore_traffic_lights", False),
                ignore_stop_signs=entry.get("ignore_stop_signs", False),
                ignore_vehicles=entry.get("ignore_vehicles", False),
                trigger_center=trigger_center,
                trigger_radius=trigger_radius,
                max_lat_acc=max_lat_acc,
            ))

        for veh_data in self._vehicles_data:
            _refine_vehicle_route(veh_data, max_lat_acc=veh_data.max_lat_acc)

        self._vehicle_auto_triggered = False
        self._vehicle_triggers = []
        for veh_idx, veh_data in enumerate(self._vehicles_data):
            if veh_data.trigger_center and veh_data.trigger_radius is not None:
                trigger = VehicleTrigger(
                    center=carla.Location(
                        veh_data.trigger_center.x,
                        veh_data.trigger_center.y,
                        veh_data.trigger_center.z,
                    ),
                    radius=veh_data.trigger_radius,
                    vehicle_index=veh_idx,
                )
                self._vehicle_triggers.append(trigger)
                print(f"[VEHICLE_TRIGGER] Loaded trigger for vehicle {veh_idx}: "
                      f"center ({trigger.center.x:.2f}, {trigger.center.y:.2f}, {trigger.center.z:.2f}), "
                      f"radius {trigger.radius:.1f}m")
            else:
                self._vehicle_triggers.append(None)

        if self._vehicle_triggers and any(t is not None for t in self._vehicle_triggers):
            active_vehicle_triggers = sum(1 for t in self._vehicle_triggers if t is not None)
            print(f"[VEHICLE_TRIGGER] Loaded {active_vehicle_triggers} vehicle triggers")

    def _load_ego_route_for_criteria(self) -> List[Tuple[carla.Transform, RoadOption]]:
        """Build an ego route (Transform, RoadOption) list for route-based criteria."""
        route: List[Tuple[carla.Transform, RoadOption]] = []
        try:
            with open(self._scenario_json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"[CRITERIA] Unable to read scenario JSON for ego route: {exc}")
            return route

        ego_entry = data.get("ego_vehicle")
        if not ego_entry:
            for entry in data.get("vehicles", []):
                if str(entry.get("role", "")).lower() == "ego_vehicle":
                    ego_entry = entry
                    break
        if not ego_entry:
            print("[CRITERIA] No ego vehicle entry found; route criteria disabled.")
            return route

        waypoints = ego_entry.get("waypoints") or []
        # Ensure the route starts at the ego spawn location if it's not already included.
        try:
            spawn_loc = ego_entry.get("location", {})
            spawn_rot = ego_entry.get("rotation", {})
            start_x = float(spawn_loc.get("x", 0.0))
            start_y = float(spawn_loc.get("y", 0.0))
            start_z = float(spawn_loc.get("z", 0.0))
            start_yaw = float(spawn_rot.get("yaw", 0.0))
            need_insert = True
            if waypoints:
                first_loc = waypoints[0].get("location", {})
                try:
                    first_x = float(first_loc.get("x", 0.0))
                    first_y = float(first_loc.get("y", 0.0))
                    first_z = float(first_loc.get("z", 0.0))
                    if (
                        abs(first_x - start_x) <= 0.25
                        and abs(first_y - start_y) <= 0.25
                        and abs(first_z - start_z) <= 0.5
                    ):
                        need_insert = False
                except Exception:
                    pass
            if need_insert:
                waypoints.insert(
                    0,
                    {
                        "index": 0,
                        "location": {"x": start_x, "y": start_y, "z": start_z},
                        "yaw": start_yaw,
                        "speed_km_h": ego_entry.get("speed_km_h", 0.0),
                        "idle_time_s": 0.0,
                        "turn_time_s": 0.0,
                        "auto_generated": True,
                        "is_destination": False,
                    },
                )
                print("[CRITERIA] Prepended ego spawn location as start waypoint for criteria route")
        except Exception:
            pass

        if len(waypoints) < 2:
            print("[CRITERIA] Ego route requires at least 2 waypoints; route criteria disabled.")
            return route

        def _fallback_yaw(idx: int) -> float:
            loc = waypoints[idx].get("location", {})
            prev_loc = waypoints[idx - 1].get("location", {}) if idx > 0 else None
            next_loc = waypoints[idx + 1].get("location", {}) if idx + 1 < len(waypoints) else None
            ref_prev = prev_loc if prev_loc and prev_loc.get("x") is not None else None
            ref_next = next_loc if next_loc and next_loc.get("x") is not None else None
            if ref_next:
                dx = float(ref_next.get("x", 0.0)) - float(loc.get("x", 0.0))
                dy = float(ref_next.get("y", 0.0)) - float(loc.get("y", 0.0))
            elif ref_prev:
                dx = float(loc.get("x", 0.0)) - float(ref_prev.get("x", 0.0))
                dy = float(loc.get("y", 0.0)) - float(ref_prev.get("y", 0.0))
            else:
                return 0.0
            return _normalize_yaw(math.degrees(math.atan2(dy, dx)))

        keypoints: List[carla.Location] = []
        transforms: List[carla.Transform] = []
        destination_loc: Optional[carla.Location] = None

        for idx, wp in enumerate(waypoints):
            loc_data = wp.get("location", {})
            try:
                location = carla.Location(
                    x=float(loc_data.get("x", 0.0)),
                    y=float(loc_data.get("y", 0.0)),
                    z=float(loc_data.get("z", 0.0)),
                )
            except Exception:
                continue
            yaw_val = wp.get("yaw")
            try:
                yaw = float(yaw_val) if yaw_val is not None else _fallback_yaw(idx)
            except Exception:
                yaw = _fallback_yaw(idx)

            # Snap to nearest driving lane to align route criteria with the map
            snapped_tf = None
            try:
                world_map = self._map or (self._world.get_map() if self._world else None)
                if world_map:
                    wp_lane = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp_lane:
                        snapped_tf = wp_lane.transform
            except Exception:
                snapped_tf = None

            if snapped_tf:
                transform = snapped_tf
                location = snapped_tf.location
                yaw = snapped_tf.rotation.yaw
            else:
                transform = carla.Transform(location, carla.Rotation(yaw=yaw))

            transforms.append(transform)
            keypoints.append(location)

            is_dest = bool(wp.get("is_destination")) or str(wp.get("index")) == "destination"
            if is_dest and destination_loc is None:
                destination_loc = location

        if destination_loc is None and transforms:
            destination_loc = transforms[-1].location
        if destination_loc:
            self._ego_destination = destination_loc

        should_interpolate = (
            len(keypoints) >= 2
            and not (_is_large_map(self._map) and os.environ.get("VSE_FORCE_ROUTE_INTERPOLATION") != "1")
        )
        if should_interpolate:
            try:
                _, interpolated = interpolate_trajectory(keypoints, hop_resolution=1.0)
                if interpolated:
                    route = list(interpolated)
            except Exception as exc:
                if self._debug:
                    print(f"[CRITERIA] interpolate_trajectory failed for ego route: {exc}")
        elif len(keypoints) >= 2 and self._debug:
            print("[CRITERIA] Large map active; skipping interpolate_trajectory for ego route")

        if not route:
            route = [(tf, RoadOption.LANEFOLLOW) for tf in transforms]

        if route:
            print(f"[CRITERIA] Ego route loaded for criteria with {len(route)} waypoint(s).")
        else:
            print("[CRITERIA] Ego route unavailable; route-based criteria will be skipped.")

        return route

    def _load_weather_keyframes(self) -> List[Tuple[float, carla.WeatherParameters]]:
        """Load weather keyframes from the scenario JSON."""
        parsed: List[Tuple[float, carla.WeatherParameters]] = []
        try:
            with open(self._scenario_json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"[Weather] Unable to read scenario JSON for weather: {exc}")
            data = None

        raw_keyframes = data.get("weather_keyframes") if isinstance(data, dict) else None

        def _dict_to_weather(payload: dict) -> Optional[carla.WeatherParameters]:
            if not isinstance(payload, dict):
                return None
            weather = carla.WeatherParameters()
            for name, val in payload.items():
                if name == "route_percentage":
                    continue
                if hasattr(weather, name):
                    try:
                        setattr(weather, name, float(val))
                    except Exception:
                        pass
            return weather

        if isinstance(raw_keyframes, list):
            cleaned: List[Tuple[float, carla.WeatherParameters]] = []
            for frame in raw_keyframes:
                if not isinstance(frame, dict):
                    continue
                try:
                    pct = float(frame.get("route_percentage", 0.0))
                except Exception:
                    continue
                weather = _dict_to_weather(frame)
                if weather is None:
                    continue
                cleaned.append((max(0.0, min(100.0, pct)), weather))

            if cleaned:
                cleaned.sort(key=lambda item: item[0])
                if cleaned[0][0] > 0.0:
                    cleaned.insert(0, (0.0, cleaned[0][1]))
                if cleaned[-1][0] < 100.0:
                    cleaned.append((100.0, cleaned[-1][1]))
                parsed = cleaned

        if not parsed and getattr(self.config, "weather_keyframes", None):
            for pct, weather in getattr(self.config, "weather_keyframes", []):
                try:
                    parsed.append((float(pct), weather))
                except Exception:
                    continue

        if parsed and len(parsed) == 1:
            parsed.append((100.0, parsed[0][1]))

        return parsed

    def _initialize_environment(self, world):
        """Apply start weather and friction if present."""
        start_weather = None
        if self._weather_keyframes:
            start_weather = self._weather_keyframes[0][1]
        if start_weather is None:
            start_weather = getattr(self.config, "weather", None) or carla.WeatherParameters()

        try:
            world.set_weather(start_weather)
        except Exception as exc:
            print(f"[Weather] Failed to set start weather: {exc}")

        # Preserve BasicScenario friction handling
        if getattr(self.config, "friction", None) is not None:
            friction_bp = world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(self.config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            world.spawn_actor(friction_bp, transform)

    def _create_weather_behavior(self):
        """Animate weather along the ego route when keyframes are defined."""
        if len(self._weather_keyframes) <= 1:
            return None
        if not self.ego_vehicles:
            return None
        if not self._ego_route_for_criteria:
            return None
        try:
            return RouteWeatherBehavior(
                self.ego_vehicles[0],
                self._ego_route_for_criteria,
                list(self._weather_keyframes),
                debug=self._debug,
            )
        except Exception as exc:
            print(f"[Weather] Failed to create route weather behavior: {exc}")
            return None

    def _create_lights_behavior(self):
        """Use ScenarioRunner lights on CARLA towns; vehicle-only lights on tartu_demo."""
        world_name = ""
        try:
            world_name = (self._world.get_map().name or "").lower()
        except Exception:
            world_name = ""

        if "tartu_demo" in world_name:
            if not self.ego_vehicles:
                return None
            try:
                return VehicleLightsBehavior(self.ego_vehicles[0], radius=100)
            except Exception as exc:
                print(f"[Lights] Failed to create vehicle-only lights behavior: {exc}")
                return None

        if not self.ego_vehicles:
            return None
        try:
            combined = py_trees.composites.Parallel(
                name="LightsCombined",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            )
            combined.add_child(RouteLightsBehavior(self.ego_vehicles[0], 100))
            combined.add_child(VehicleLightsBehavior(self.ego_vehicles[0], radius=100))
            return combined
        except Exception as exc:
            print(f"[Lights] Failed to create combined lights behavior: {exc}")
            return None

    def _load_traffic_light_triggers(self) -> None:
        """Load traffic light triggers from JSON"""
        try:
            with open(self._scenario_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            triggers_data = data.get("traffic_light_triggers", [])
            if not triggers_data:
                print("[TRAFFIC_LIGHT] No traffic light triggers defined in scenario")
                return

            print(f"[TRAFFIC_LIGHT][DEBUG] Loading {len(triggers_data)} traffic light triggers from JSON")

            for entry in triggers_data:
                if not isinstance(entry, dict):
                    continue
                if "ids_live" not in entry and "ids" in entry:
                    legacy_ids = entry.pop("ids")
                    entry["ids_live"] = legacy_ids
                    entry.setdefault("ids_reference", list(legacy_ids))

            if not self._world:
                try:
                    self._world = CarlaDataProvider.get_world()
                except Exception:
                    self._world = None
            if not self._world:
                print("[TRAFFIC_LIGHT] ERROR: World not available for traffic light trigger loading")
                return

            try:
                settings = self._world.get_settings()
                print(f"[TRAFFIC_LIGHT][DEBUG] World id={id(self._world)} sync={settings.synchronous_mode} fixed_delta={settings.fixed_delta_seconds}")
            except Exception:
                print(f"[TRAFFIC_LIGHT][DEBUG] World id={id(self._world)} (settings unavailable)")

            # Ensure the actor list is populated before fingerprint matching
            start_wait = time.monotonic()
            while True:
                try:
                    lights = self._world.get_actors().filter("traffic.traffic_light*")
                except Exception:
                    lights = []
                if lights:
                    print(f"[TRAFFIC_LIGHT][DEBUG] Detected {len(lights)} traffic lights before matching")
                    break
                if time.monotonic() - start_wait >= 5.0:
                    print("[TRAFFIC_LIGHT][DEBUG] Timed out waiting for traffic lights to appear")
                    break
                try:
                    self._world.wait_for_tick(0.5)
                except Exception:
                    pass

            fingerprint_index = _build_traffic_light_fingerprint_index(self._world)
            print(f"[TRAFFIC_LIGHT][DEBUG] Fingerprint index size after initial build: {len(fingerprint_index)}")

            def _refresh_fingerprint_index() -> dict:
                """Ensure we have a populated actor list before fingerprint matching."""
                nonlocal fingerprint_index
                # Try for a few seconds to get a populated index (handles ROS-tick startup).
                start_local = time.monotonic()
                while time.monotonic() - start_local < 5.0:
                    try:
                        if self._world:
                            # In ROS-tick mode this waits for an external tick; in own-tick mode it advances one frame.
                            self._world.wait_for_tick(0.5)
                    except Exception:
                        pass
                    fingerprint_index = _build_traffic_light_fingerprint_index(self._world)
                    if fingerprint_index:
                        print(f"[TRAFFIC_LIGHT][DEBUG] Fingerprint index repopulated with {len(fingerprint_index)} entries")
                        break
                return fingerprint_index

            for idx, trigger_data in enumerate(triggers_data):
                # Validate required fields (must have center/radius and at least fingerprint or IDs)
                ids_live = trigger_data.get("ids_live")
                ids_reference = trigger_data.get("ids_reference")
                fingerprint_payload = trigger_data.get("fingerprint")
                normalized_fp = _normalize_traffic_light_fingerprint(fingerprint_payload)
                print(f"[TRAFFIC_LIGHT][DEBUG] Trigger {idx}: ids_live={ids_live} ids_reference={ids_reference} fp={normalized_fp}")
                if ("center" not in trigger_data or "radius" not in trigger_data):
                    print(f"[TRAFFIC_LIGHT] WARNING: Trigger {idx} missing center/radius, skipping")
                    continue
                if normalized_fp is None and ids_live is None and ids_reference is None:
                    print(f"[TRAFFIC_LIGHT] WARNING: Trigger {idx} missing identifiers (fingerprint/ids), skipping")
                    continue

                sequence = trigger_data.get("sequence", [])
                if not sequence:
                    print(f"[TRAFFIC_LIGHT] WARNING: Trigger {idx} has empty sequence, skipping")
                    continue

                # Create trigger
                center_data = trigger_data["center"]
                center = carla.Location(
                    x=center_data.get("x", 0.0),
                    y=center_data.get("y", 0.0),
                    z=center_data.get("z", 0.0)
                )

                resolved_ids: List[int] = []
                resolution_source = None

                # Fingerprint-first to survive ID churn; if we have a fingerprint, do not fall back to mismatched IDs.
                if normalized_fp:
                    matched_lights = _match_traffic_lights_by_fingerprint(normalized_fp, fingerprint_index)
                    if not matched_lights:
                        # Actor list might be empty before the first tick; refresh once and retry.
                        new_index = _refresh_fingerprint_index()
                        matched_lights = _match_traffic_lights_by_fingerprint(normalized_fp, new_index)
                    print(f"[TRAFFIC_LIGHT][DEBUG] Trigger {idx} fingerprint match -> {[l.id for l in matched_lights]}")
                    if matched_lights:
                        resolved_ids = [light.id for light in matched_lights]
                        resolution_source = "fingerprint"
                    else:
                        print(f"[TRAFFIC_LIGHT] WARNING: Trigger {idx} fingerprint did not match any live lights; skipping")
                        continue

                # No fingerprint provided: fall back to IDs.
                if not resolved_ids and normalized_fp is None and ids_live is not None:
                    try:
                        resolved_ids = [int(value) for value in ids_live]
                        resolution_source = "ids_live"
                    except Exception:
                        resolved_ids = []

                if not resolved_ids and normalized_fp is None and ids_reference is not None:
                    try:
                        resolved_ids = [int(value) for value in ids_reference]
                        resolution_source = "ids_reference"
                    except Exception:
                        resolved_ids = []

                if not resolved_ids:
                    print(f"[TRAFFIC_LIGHT] WARNING: Trigger {idx} could not resolve any traffic lights; skipping")
                    continue

                trigger = TrafficLightTrigger(
                    center=center,
                    radius=trigger_data["radius"],
                    ids=resolved_ids,
                    sequence=sequence
                )

                self._traffic_light_triggers.append(trigger)
                source_str = f" via {resolution_source}" if resolution_source else ""
                print(f"[TRAFFIC_LIGHT] Loaded trigger {idx}: {len(trigger.ids)} lights, "
                      f"{len(sequence)} steps, radius {trigger.radius}m{source_str}")

            print(f"[TRAFFIC_LIGHT] Loaded {len(self._traffic_light_triggers)} traffic light triggers")

        except Exception as e:
            print(f"[TRAFFIC_LIGHT] ERROR loading triggers: {e}")

    def get_walker(self, ped_index: int) -> Optional[carla.Actor]:
        if 0 <= ped_index < len(self.other_actors):
            return self.other_actors[ped_index]
        return None

    def _on_walker_spawned(self, ped_index: int, walker: carla.Actor) -> None:
        if 0 <= ped_index < len(self.other_actors):
            self.other_actors[ped_index] = walker
        if self._debug:
            loc = walker.get_location()
            print(
                "Pedestrian "
                f"{ped_index} spawned at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})"
            )

    def _build_pedestrian_route(self, ped_index: int, walker: carla.Actor) -> Behaviour:
        ped_data = self.pedestrians_data[ped_index]
        segments: List[RouteSegment] = ped_data["segments"]
        spawn_rotation: carla.Rotation = ped_data["spawn_rotation"]

        sequence = SingleRunSequence(f"VSEPedestrianRoute_{ped_index}")

        initial_heading = segments[0].heading if segments else spawn_rotation.yaw
        initial_transform = carla.Transform(
            ped_data["spawn_location"],
            carla.Rotation(
                pitch=spawn_rotation.pitch,
                yaw=initial_heading,
                roll=spawn_rotation.roll,
            ),
        )
        sequence.add_child(ActorTransformSetter(walker, initial_transform, name=f"SetInitialPose_{ped_index}"))

        for segment in segments:
            seg_name = f"Ped{ped_index}_Segment{segment.index}"

            if segment.distance > 0.05 and segment.speed > 0:
                # Use larger tolerance for destination segments to prevent overshooting
                tolerance = 0.5 if segment.is_destination else 0.15
                sequence.add_child(WalkToTarget(
                    walker,
                    segment.target,
                    segment.speed,
                    tolerance=tolerance,
                    stuck_time=0.0,
                    desired_yaw=segment.heading,
                    debug_enabled=self._debug,
                    is_destination=segment.is_destination,
                    name=f"{seg_name}_Walk",
                ))
            else:
                sequence.add_child(ActorTransformSetter(
                    walker,
                    carla.Transform(
                        segment.target,
                        carla.Rotation(
                            pitch=spawn_rotation.pitch,
                            yaw=segment.heading,
                            roll=spawn_rotation.roll,
                        ),
                    ),
                    name=f"{seg_name}_Snap",
                ))

            align_transform = carla.Transform(
                segment.target,
                carla.Rotation(
                    pitch=spawn_rotation.pitch,
                    yaw=segment.heading,
                    roll=spawn_rotation.roll,
                ),
            )

            sequence.add_child(EnsureWalkerAt(
                walker,
                align_transform,
                tolerance=0.05,
                name=f"{seg_name}_Align",
            ))

            if segment.idle_after > 0:
                idle_transform = carla.Transform(
                    segment.target,
                    carla.Rotation(
                        pitch=spawn_rotation.pitch,
                        yaw=segment.heading,
                        roll=spawn_rotation.roll,
                    ),
                )
                sequence.add_child(GroundedIdle(
                    walker,
                    idle_transform,
                    duration=segment.idle_after,
                    world=self._world,
                    name=f"{seg_name}_Idle",
                ))

            if segment.next_heading is not None and not segment.is_destination:
                turn_duration = max(0.0, segment.turn_time)
                sequence.add_child(SmoothTurn(
                    walker,
                    segment.next_heading,
                    duration=turn_duration,
                    name=f"{seg_name}_Turn",
                    debug_enabled=self._debug,
                ))

        return sequence

    def _initialize_actors(self, config) -> None:  # pylint: disable=unused-argument
        self.other_actors = [None] * len(self.pedestrians_data)
        self._spawned_flags = [False] * len(self.pedestrians_data)
        self._walker_colors = []
        if self.pedestrians_data:
            for idx in range(len(self.pedestrians_data)):
                r, g, b = _WALKER_COLOR_PALETTE[idx % len(_WALKER_COLOR_PALETTE)]
                self._walker_colors.append(carla.Color(r, g, b))

        if self._highlight_callback_id is None:
            self._highlight_callback_id = self._world.on_tick(self._highlight_walkers)

        self._spawn_all_vehicles()

    def _spawn_all_vehicles(self) -> None:
        if not self._vehicles_data:
            return

        blueprint_lib = self._world.get_blueprint_library()

        for idx, data in enumerate(self._vehicles_data):
            blueprint = blueprint_lib.find(data.blueprint_id)

            # Apply color if available (not for pedestrians)
            if data.color and not data.blueprint_id.startswith('walker.'):
                if blueprint.has_attribute('color'):
                    blueprint.set_attribute('color', data.color)
            if blueprint.has_attribute('role_name'):
                blueprint.set_attribute('role_name', f"scenario_{idx}")

            spawn_location = carla.Location(
                data.spawn_location.x,
                data.spawn_location.y,
                data.spawn_location.z + 1.0,
            )
            spawn_transform = carla.Transform(spawn_location, data.spawn_rotation)
            vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)
            if not vehicle:
                waypoint = self._map.get_waypoint(spawn_location, project_to_road=True)
                if waypoint:
                    fallback = waypoint.transform
                    fallback.location.z += 1.0
                    vehicle = self._world.try_spawn_actor(blueprint, fallback)
            if not vehicle:
                print(f"Failed to spawn vehicle {idx} type {data.blueprint_id}")
                continue

            _spawned_vehicles.append(vehicle)
            CarlaDataProvider.register_actor(vehicle)
            self._vehicle_actors.append(vehicle)

            try:
                agent = BasicAgent(
                    vehicle,
                    target_speed=data.initial_speed,
                    map_inst=self._map,
                    grp_inst=_NoopGlobalRoutePlanner(),
                )
            except Exception:
                # Fall back to constructing the agent without an explicit map instance,
                # but still avoid GRP precomputation (VSE uses set_global_plan()).
                try:
                    agent = BasicAgent(
                        vehicle,
                        target_speed=data.initial_speed,
                        grp_inst=_NoopGlobalRoutePlanner(),
                    )
                except Exception:
                    agent = BasicAgent(vehicle, target_speed=data.initial_speed)
            agent.ignore_traffic_lights(data.ignore_traffic_lights)
            agent.ignore_stop_signs(data.ignore_stop_signs)
            agent.ignore_vehicles(data.ignore_vehicles)
            agent.follow_speed_limits(False)

            self._setup_vehicle_route(agent, vehicle, data)

            controller = VehicleController(
                agent,
                vehicle,
                data.destination,
                self,
                idx,
                data.route_points,
                data.initial_idle_time,
                destination_speed=data.destination_speed,
                cruise_speed=data.initial_speed,
                vehicle_trigger=self._vehicle_triggers[idx] if idx < len(self._vehicle_triggers) else None,
            )
            self._vehicle_controllers.append(controller)
            controller.start()

    def _trace_route_with_retry(
        self,
        agent: BasicAgent,
        start_location: carla.Location,
        target_location: carla.Location,
        description: str,
        max_attempts: int = 6,
        initial_delay: float = 0.2,
    ) -> Optional[List[Tuple]]:
        delay = max(0.0, float(initial_delay))
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            segment: Optional[List[Tuple]] = None
            try:
                segment = agent._global_planner.trace_route(start_location, target_location)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                segment = None

            if segment:
                if self._debug:
                    print(
                        f"[ROUTE] {description}: success on attempt {attempt} "
                        f"with {len(segment)} waypoints"
                    )
                return segment

            if attempt < max_attempts:
                if self._debug:
                    error_msg = f", error={last_error}" if last_error else ""
                    print(
                        f"[ROUTE] {description}: attempt {attempt} returned no waypoints"
                        f"{error_msg}; backing off {delay:.2f}s"
                    )
                if delay > 0.0:
                    time.sleep(delay)
                delay = min(delay * 1.5 if delay > 0.0 else 0.2, 1.5)

        warning = (
            f"[ROUTE] {description}: failed to build route after {max_attempts} attempts"
        )
        if last_error:
            warning = f"{warning}; last error: {last_error}"
        print(warning)
        return None

    def _setup_vehicle_route(self, agent: BasicAgent, vehicle: carla.Actor, data: VehicleData) -> None:
        class WP:
            """Lightweight waypoint wrapper for BasicAgent routes."""
            def __init__(self, transform: carla.Transform):
                self.transform = transform
                self.road_id = 0  # Placeholder for BasicAgent compatibility
                self.lane_id = 0  # Placeholder for BasicAgent compatibility

        try:
            vehicle_loc = vehicle.get_location()
            route: List[Tuple] = []

            if data.route_points:
                for point in data.route_points:
                    transform = carla.Transform(
                        carla.Location(point.transform.location.x, point.transform.location.y, point.transform.location.z),
                        point.transform.rotation,
                    )
                    custom_wp = WP(transform)
                    route.append((custom_wp, carla.LaneChange.NONE))
            elif data.destination:
                if vehicle_loc.distance(data.destination) > 0.1:
                    custom_dest = WP(carla.Transform(data.destination))
                    route.append((custom_dest, carla.LaneChange.NONE))

            if route:
                agent.set_global_plan(route)
                print(
                    f"[ROUTE] vehicle {vehicle.id if vehicle else 'unknown'}: "
                    f"global plan contains {len(route)} entries"
                )
            else:
                print(
                    f"[ROUTE] vehicle {vehicle.id if vehicle else 'unknown'}: "
                    "no route entries accumulated; vehicle will stay put"
                )
        except Exception:
            pass

    def _create_behavior(self) -> Behaviour:
        root = TrackingParallel(
            name="VSEMultiPedestrianParallel",
            policy=py_trees_common.ParallelPolicy.SUCCESS_ON_ALL,
            scenario=self,
        )
        self._root_parallel: Optional[Parallel] = root

        # Spawn all pedestrians immediately to avoid race conditions with trigger
        for ped_idx, ped_data in enumerate(self.pedestrians_data):
            branch = SingleRunSequence(f"Pedestrian_{ped_idx}_Branch")
            # Spawn first, without any initial delay - the delay will be applied to movement instead
            branch.add_child(SpawnWalkerBehaviour(self, ped_idx))
            initial_idle_time = ped_data["initial_idle_time"]
            if initial_idle_time > 0:
                # Apply the delay before route execution, not before spawning
                branch.add_child(DelayBeforeSpawn(initial_idle_time, ped_idx, self))
            branch.add_child(ExecuteRouteBehaviour(self, ped_idx))
            root.add_child(branch)

        class WalkAllPedestrians(py_trees.behaviour.Behaviour):
            def __init__(self, scenario):
                super().__init__("WalkAllPedestrians")
                self.scenario = scenario
                self.running_time = 0
                self.pedestrian_reached = [False] * len(scenario.other_actors)

            def update(self):
                if not self.scenario._keep_running:
                    return py_trees_common.Status.SUCCESS

                # If no pedestrians, immediately return success
                if not self.scenario.pedestrians_data:
                    return py_trees_common.Status.SUCCESS

                all_done = True
                for i in range(len(self.scenario.pedestrians_data)):
                    walker = self.scenario.get_walker(i)

                    if walker is None:
                        all_done = False
                        continue

                    if not walker.is_alive:
                        all_done = all_done and self.pedestrian_reached[i]
                        continue

                    destination = self.scenario.final_destinations[i]
                    distance = walker.get_location().distance(destination)
                    if distance <= 1.0:
                        if not self.pedestrian_reached[i]:
                            if self.scenario._debug:
                                print(
                                    f"Pedestrian {i} reached destination! "
                                    f"(distance: {distance:.1f}m)"
                                )
                            self.pedestrian_reached[i] = True
                            self.scenario.completion_status[i] = True

                            # Ground the pedestrian at destination to prevent falling
                            # Use the same approach as vse.py with bounding box height offset
                            try:
                                bbox_extent_z = float(getattr(walker.bounding_box.extent, "z", 1.0))
                            except Exception:
                                bbox_extent_z = 1.0

                            # Temporarily lift walker out of raycast path (like vse.py does for vehicles)
                            try:
                                current_loc = walker.get_location()
                                lifted_location = carla.Location(
                                    current_loc.x,
                                    current_loc.y,
                                    current_loc.z + 500.0
                                )
                                walker.set_location(lifted_location)
                            except Exception:
                                pass

                            # Get ground height at current XY position
                            ground_height = get_ground_height(
                                self.scenario.world,
                                current_loc,
                                debug=False,
                                cached_map=self.scenario.world.get_map(),
                                probe_on_miss=True
                            )

                            # Set walker with corrected Z (just bbox_extent_z, no extra offset)
                            grounded_transform = walker.get_transform()
                            grounded_transform.location.z = ground_height + bbox_extent_z
                            walker.set_transform(grounded_transform)
                            walker.apply_control(carla.WalkerControl())
                        all_done = all_done and self.pedestrian_reached[i]
                    else:
                        all_done = False

                self.running_time += 1
                if self.running_time > self.scenario.timeout * 100:
                    self.scenario._terminate_scenario("Scenario timeout reached")
                    return py_trees_common.Status.SUCCESS

                # All pedestrians reached destination
                if all_done:
                    return py_trees_common.Status.SUCCESS

                return py_trees_common.Status.RUNNING

        root.add_child(WalkAllPedestrians(self))

        has_vehicles = bool(self._vehicles_data)

        class MonitorVehicles(py_trees.behaviour.Behaviour):
            def __init__(self, scenario_ref: "vse_play"):
                super().__init__("MonitorVehicles")
                self.scenario = scenario_ref

            def update(self):
                if not self.scenario._keep_running:
                    return py_trees_common.Status.SUCCESS
                if not self.scenario._vehicle_controllers:
                    return (
                        py_trees_common.Status.RUNNING
                        if self.scenario._vehicles_data
                        else py_trees_common.Status.SUCCESS
                    )
                all_done = True
                for controller in self.scenario._vehicle_controllers:
                    if controller and not controller.is_finished():
                        all_done = False
                        break
                return py_trees_common.Status.SUCCESS if all_done else py_trees_common.Status.RUNNING

        class CombinedMonitor(py_trees.behaviour.Behaviour):
            def __init__(self, scenario_ref: "vse_play"):
                super().__init__("CombinedMonitor")
                self.scenario = scenario_ref

            def update(self):
                if not self.scenario._keep_running:
                    return py_trees_common.Status.SUCCESS
                has_destination = getattr(self.scenario, "_ego_destination", None) is not None
                if not self.scenario._vehicle_controllers:
                    vehicles_done = not self.scenario._vehicles_data
                else:
                    vehicles_done = all(
                        controller.is_finished() for controller in self.scenario._vehicle_controllers
                    )
                pedestrians_done = (
                    not self.scenario.pedestrians_data or
                    all(self.scenario.completion_status)
                )
                triggers_done = self.scenario._are_required_triggers_satisfied()
                if vehicles_done and pedestrians_done and triggers_done and not has_destination:
                    if not self.scenario._scenario_completed:
                        self.scenario._completion_reason = "All actors reached destination and triggers activated"
                        self.scenario._keep_running = False
                        self.scenario._scenario_completed = True
                    return py_trees_common.Status.SUCCESS
                return py_trees_common.Status.RUNNING

        class TriggerMonitor(py_trees.behaviour.Behaviour):
            def __init__(self, scenario_ref: "vse_play"):
                super().__init__("TriggerMonitor")
                self.scenario = scenario_ref

            def update(self):
                # If scenario is done, return success to allow parallel to complete
                if not self.scenario._keep_running:
                    return py_trees_common.Status.SUCCESS

                # If no trigger mode, always return running
                if not self.scenario._trigger_mode:
                    return py_trees_common.Status.SUCCESS

                # If already triggered, just keep running
                if self.scenario._scenario_triggered:
                    return py_trees_common.Status.RUNNING

                # Check if ego vehicle exists
                if not self.scenario.ego_vehicles or len(self.scenario.ego_vehicles) == 0:
                    return py_trees_common.Status.RUNNING

                ego_vehicle = self.scenario.ego_vehicles[0]
                if not ego_vehicle or not ego_vehicle.is_alive:
                    return py_trees_common.Status.RUNNING

                # Get ego vehicle location
                ego_loc = ego_vehicle.get_location()

                # Get trigger data
                trigger = self.scenario._trigger_data
                if not trigger:
                    return py_trees_common.Status.RUNNING

                # Calculate 2D distance to trigger center
                dx = ego_loc.x - trigger['x']
                dy = ego_loc.y - trigger['y']
                distance = math.sqrt(dx * dx + dy * dy)

                # Check if ego is within trigger radius
                if distance <= trigger['radius']:
                    self.scenario._scenario_triggered = True
                    self.scenario._global_trigger_released = True
                    print(f"[TRIGGER] Activated! Ego vehicle entered trigger zone (distance: {distance:.2f}m)")
                    print(f"[TRIGGER] Starting scenario...")
                    self.scenario._activate_all_personal_triggers()

                return py_trees_common.Status.RUNNING

        combined_root = py_trees.composites.Parallel(
            name="CombinedScenarioParallel",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        combined_root.add_child(root)
        if has_vehicles:
            combined_root.add_child(MonitorVehicles(self))
        combined_root.add_child(CombinedMonitor(self))
        combined_root.add_child(TriggerMonitor(self))

        # Add traffic light trigger monitor
        traffic_light_monitor = TrafficLightTriggerMonitor(self)
        self._traffic_light_monitor = traffic_light_monitor  # Store reference for cleanup
        combined_root.add_child(traffic_light_monitor)

        # Add pedestrian trigger monitor
        pedestrian_trigger_monitor = PedestrianTriggerMonitor(self)
        combined_root.add_child(pedestrian_trigger_monitor)
        vehicle_trigger_monitor = VehicleTriggerMonitor(self)
        combined_root.add_child(vehicle_trigger_monitor)

        return combined_root

    def _highlight_walkers(self, snapshot):  # pylint: disable=unused-argument
        """No-op: pedestrian highlight rings are disabled."""
        return

    def _setup_scenario_trigger(self, config):  # pylint: disable=unused-argument
        """
        Override the default scenario trigger behavior.
        VSE scenarios should start immediately, not wait for ego vehicle movement.
        The scenario has its own trigger system if needed.
        """
        return None

    def _create_test_criteria(self):  # pylint: disable=unused-argument
        criteria = [AllPedestriansArrivedCriterion(self)]

        # Track ego-only criteria separately for reporting
        self._ego_criteria: List[Criterion] = []

        ego_actor = self.ego_vehicles[0] if self.ego_vehicles else None
        if ego_actor:
            ego_criteria: List[Criterion] = [
                CollisionTest(ego_actor),
                KeepLaneTest(ego_actor),
                OffRoadTest(ego_actor),
                EndofRoadTest(ego_actor),
                OnSidewalkTest(ego_actor),
                RunningRedLightTest(ego_actor),
                RunningStopTest(ego_actor),
            ]
            if self._ego_route_for_criteria:
                try:
                    ego_criteria.extend(
                        [
                            OutsideRouteLanesTest(ego_actor, self._ego_route_for_criteria),
                            WrongLaneTest(ego_actor),
                            InRouteTest(ego_actor, self._ego_route_for_criteria),
                            RouteCompletionTest(ego_actor, self._ego_route_for_criteria),
                        ]
                    )
                except Exception as exc:
                    print(f"[CRITERIA] Failed to add ego route criteria: {exc}")
            else:
                print("[CRITERIA] No ego route loaded; skipping ego route-based criteria.")
            self._ego_criteria.extend(ego_criteria)
            criteria.extend(ego_criteria)

        # Add destination criteria for scenario vehicles
        for idx, vehicle in enumerate(self._vehicle_actors):
            if not vehicle:
                continue
            if idx < len(self._vehicles_data):
                criteria.append(DestinationCriterion(vehicle, self._vehicles_data[idx].destination))
        self._criteria_nodes = criteria
        return criteria

    def _activate_all_personal_triggers(self) -> None:
        """Handle global trigger activation by releasing actors without personal triggers."""
        ped_without_personal = sum(1 for trigger in self._pedestrian_triggers if trigger is None)
        veh_without_personal = sum(1 for trigger in self._vehicle_triggers if trigger is None)

        pending_ped_triggers = sum(
            1 for trigger in self._pedestrian_triggers if trigger and not trigger.activated
        )
        pending_vehicle_triggers = sum(
            1 for trigger in self._vehicle_triggers if trigger and not trigger.activated
        )

        if ped_without_personal or veh_without_personal:
            print(
                f"[TRIGGER] Global trigger releasing {ped_without_personal} pedestrian(s) and "
                f"{veh_without_personal} vehicle(s) without personal triggers"
            )
        if pending_ped_triggers or pending_vehicle_triggers:
            print(
                f"[TRIGGER] Global trigger active; still waiting on "
                f"{pending_ped_triggers} pedestrian and {pending_vehicle_triggers} vehicle personal trigger(s)"
            )

        self._global_trigger_released = True
        self._pedestrian_auto_triggered = True
        self._vehicle_auto_triggered = True

    def _are_required_triggers_satisfied(self) -> bool:
        """Return True when every non-traffic-light trigger has been activated."""
        if self._requires_global_trigger() and not self._global_trigger_released:
            return False

        for trigger in self._pedestrian_triggers:
            if trigger and not trigger.activated:
                return False

        for trigger in self._vehicle_triggers:
            if trigger and not trigger.activated:
                return False

        return True

    def _requires_global_trigger(self) -> bool:
        """Return True if any actor depends on the global trigger to start."""
        if not self._trigger_mode:
            return False

        if any(trigger is None for trigger in self._pedestrian_triggers):
            return True
        if any(trigger is None for trigger in self._vehicle_triggers):
            return True

        return False

    def _pedestrian_requires_global_trigger(self, ped_index: int) -> bool:
        """Return True if the specified pedestrian relies on the global trigger."""
        if not self._trigger_mode:
            return False
        if ped_index < 0 or ped_index >= len(self._pedestrian_triggers):
            return False
        return self._pedestrian_triggers[ped_index] is None

    def _terminate_scenario(self, reason=""):
        if self._cleanup_done:
            return

        if (not reason or reason == "Scenario terminated by manager") and self._completion_reason:
            reason = self._completion_reason

        if reason:
            print(f"Terminating scenario: {reason}")
            logger.info("Terminating scenario: %s", reason)

        # Signal all threads to stop using both the flag and event
        with self._state_lock:
            self._keep_running = False
        self._stop_event.set()

        # Debug: log active controllers before stopping
        if getattr(self, "_debug", False):
            active = sum(1 for c in self._vehicle_controllers if c._thread and c._thread.is_alive())
            print(f"[TERMINATE] Stopping {active} active vehicle controllers")
            logger.debug("Stopping %d active vehicle controllers", active)

        # Cleanup traffic light monitor
        if hasattr(self, '_traffic_light_monitor') and self._traffic_light_monitor:
            self._traffic_light_monitor.cleanup()

        # Stop all controllers - this signals each thread to exit
        for controller in self._vehicle_controllers:
            controller.stop()

        # Wait for all threads to truly exit before destroying actors
        # Use longer timeout and multiple passes to ensure threads exit
        max_wait_time = 10.0  # Total seconds to wait for all threads
        wait_start = time.monotonic()
        while time.monotonic() - wait_start < max_wait_time:
            still_alive = [c for c in self._vehicle_controllers
                          if c._thread and c._thread.is_alive()]
            if not still_alive:
                break
            for controller in still_alive:
                controller._thread.join(timeout=0.5)

        # Log and warn about threads that didn't stop cleanly
        still_alive_count = sum(1 for c in self._vehicle_controllers
                                if c._thread and c._thread.is_alive())
        if still_alive_count > 0:
            logger.warning("CRITICAL: %d controller threads still alive after %.1fs - "
                          "destroying actors anyway (may crash)",
                          still_alive_count, max_wait_time)
            if getattr(self, "_debug", False):
                print(f"[TERMINATE] CRITICAL: {still_alive_count} controller threads "
                      f"still alive after {max_wait_time}s")

        self._vehicle_controllers.clear()

        actor_ids: List[int] = []
        for vehicle in self._vehicle_actors[:]:
            if vehicle and vehicle.is_alive:
                try:
                    actor_ids.append(int(vehicle.id))
                except Exception:
                    pass
            if vehicle in _spawned_vehicles:
                _spawned_vehicles.remove(vehicle)
        self._vehicle_actors.clear()

        for walker in self.other_actors[:]:
            if walker and walker.is_alive:
                try:
                    actor_ids.append(int(walker.id))
                except Exception:
                    pass
        self.other_actors.clear()

        if actor_ids:
            # Don't tick world during actor destruction - it can timeout on large maps
            do_tick = False
            try:
                client = CarlaDataProvider.get_client()
            except Exception:
                client = None
            with _temporary_client_timeout(client, timeout_s=60.0):
                _destroy_actor_ids(client, actor_ids, do_tick=do_tick)
        self._walker_colors = []
        self._cleanup_done = True

    def terminate(self):
        self._terminate_scenario("Scenario terminated by manager")
        super().terminate()

    def __del__(self):
        try:
            self._terminate_scenario("Destructor cleanup")
        except Exception:
            # Log but don't raise - destructors shouldn't propagate exceptions
            logger.debug("Exception during destructor cleanup", exc_info=True)


# =============================================================================
# CLEANUP & SIGNAL HANDLING
# Global cleanup, signal handlers, atexit registration
# =============================================================================


def _global_cleanup(*_):
    """Global cleanup handler for atexit and signal handlers.

    Uses lock to safely access the global scenario reference.
    """
    with _current_scenario_lock:
        scenario = _current_scenario
    if scenario:
        try:
            scenario._terminate_scenario("Global cleanup")
        except Exception:
            logger.exception("Error during global cleanup")


def _signal_handler(signum, frame):
    _global_cleanup()
    raise SystemExit(0)


def _report_criteria_results(scenario_instance, criteria_nodes: List[py_trees.behaviour.Behaviour], json_path: str) -> None:
    """Deprecated: scenario_runner output handles reporting."""
    return


_install_handlers_flag = os.environ.get(
    "VSE_PLAY_INSTALL_HANDLERS",
    "0" if __name__ == "__main__" else "1",
)
_install_handlers = str(_install_handlers_flag).strip().lower() not in ("0", "false", "no", "off")
if _install_handlers:
    atexit.register(_global_cleanup)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass


def get_available_scenarios():
    return {"vse_play": vse_play}


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"Required file not found: {path}")


def _resolve_scenario_arg(scenario_arg: str) -> Tuple[str, Path, Path]:
    """
    Accept a scenario name or JSON path and return (scenario_name, json_path, config_path).
    The config XML is expected to live alongside the JSON.
    """
    candidates: List[Path] = []

    # Direct path or path-like string
    expanded = Path(os.path.expanduser(scenario_arg)).resolve()
    candidates.append(expanded)
    # If user passed something without .json, try adding it
    if expanded.suffix != ".json":
        candidates.append(expanded.with_suffix(".json"))

    # Also try relative to CWD if the first resolve failed because the original was relative
    cwd = Path.cwd()
    rel = cwd / scenario_arg
    candidates.append(rel)
    if rel.suffix != ".json":
        candidates.append(rel.with_suffix(".json"))

    # Finally, try next to this script (legacy behavior)
    script_dir = Path(__file__).resolve().parent
    script_rel = script_dir / scenario_arg
    candidates.append(script_rel)
    if script_rel.suffix != ".json":
        candidates.append(script_rel.with_suffix(".json"))

    json_path: Optional[Path] = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            json_path = cand
            break

    if json_path is None:
        raise FileNotFoundError(
            f"Could not find scenario JSON for '{scenario_arg}'. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    scenario_name = json_path.stem
    config_path = json_path.parent / "vse_play_scenario.xml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"vse_play_scenario.xml not found next to {json_path}. "
            f"Expected at: {config_path}"
        )

    return scenario_name, json_path.resolve(), config_path.resolve()


def _resolve_json_arg(scenario_arg: str) -> Path:
    """Resolve a scenario argument to an existing JSON file path."""
    candidates: List[Path] = []

    expanded = Path(os.path.expanduser(scenario_arg)).resolve()
    candidates.append(expanded)
    if expanded.suffix != ".json":
        candidates.append(expanded.with_suffix(".json"))

    cwd = Path.cwd()
    rel = cwd / scenario_arg
    candidates.append(rel)
    if rel.suffix != ".json":
        candidates.append(rel.with_suffix(".json"))

    script_dir = Path(__file__).resolve().parent
    script_rel = script_dir / scenario_arg
    candidates.append(script_rel)
    if script_rel.suffix != ".json":
        candidates.append(script_rel.with_suffix(".json"))

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not find scenario JSON for '{scenario_arg}'. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _scenario_has_ego(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("ego_vehicle"):
        return True
    for entry in data.get("vehicles", []) or []:
        if isinstance(entry, dict) and str(entry.get("role", "")).lower() == "ego_vehicle":
            return True
    return False


def _base_ego_roles() -> set:
    """Return default role names that are treated as ego vehicles."""
    return {"ego_vehicle", "hero", "ego", "player"}


def _expected_ego_roles(data: object) -> set:
    # Mirror VSE editor external-ego role detection, but include a couple of
    # common aliases used by other stacks.
    roles = _base_ego_roles()
    if not isinstance(data, dict):
        return roles
    ego_record = data.get("ego_vehicle")
    if isinstance(ego_record, dict):
        role = ego_record.get("role")
        if role:
            roles.add(str(role))
    elif isinstance(ego_record, list):
        for entry in ego_record:
            if isinstance(entry, dict):
                role = entry.get("role")
                if role:
                    roles.add(str(role))
    return {str(r).lower() for r in roles if r}


def _scenario_ego_blueprint_id(data: object) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    ego_record = data.get("ego_vehicle")
    if isinstance(ego_record, dict):
        ego_type = ego_record.get("type")
        if isinstance(ego_type, str) and ego_type.strip():
            return ego_type.strip()
    for entry in data.get("vehicles", []) or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("role", "")).lower() == "ego_vehicle":
            ego_type = entry.get("type")
            if isinstance(ego_type, str) and ego_type.strip():
                return ego_type.strip()
    return None


def _find_external_ego(
    world: carla.World,
    expected_roles: set,
    *,
    blueprint_id: Optional[str] = None,
) -> Optional[carla.Actor]:
    try:
        actors = world.get_actors().filter("vehicle.*")
    except Exception:
        actors = []
    blueprint_candidates: List[carla.Actor] = []
    for actor in actors:
        if not actor or not getattr(actor, "is_alive", False):
            continue
        if blueprint_id:
            try:
                if actor.type_id == blueprint_id:
                    blueprint_candidates.append(actor)
            except Exception:
                pass
        try:
            role_name = actor.attributes.get("role_name", "").lower()
        except Exception:
            role_name = ""
        if role_name and role_name in expected_roles:
            return actor
    if blueprint_id and len(blueprint_candidates) == 1:
        return blueprint_candidates[0]
    return None


def _build_command(
    scenario_name: str,
    config_file: Path,
    script_path: Path,
    result_file: Optional[Path] = None,
) -> List[str]:
    scenario_runner_root = os.environ.get("SCENARIO_RUNNER_ROOT")
    if not scenario_runner_root:
        raise EnvironmentError("SCENARIO_RUNNER_ROOT is not set.")
    scenario_runner = Path(scenario_runner_root) / "scenario_runner.py"
    if not scenario_runner.exists():
        raise FileNotFoundError(f"scenario_runner.py not found at {scenario_runner}")

    cmd = [
        sys.executable,
        str(scenario_runner),
        "--scenario",
        scenario_name,
        "--configFile",
        str(_resolve_path(config_file)),
        "--additionalScenario",
        str(_resolve_path(script_path)),
        "--waitForEgo",
    ]
    if result_file:
        cmd.extend(["--file", str(result_file)])
    return cmd


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a VSE scenario JSON via the built-in MiniRunner (no ScenarioRunner, no XML).",
    )
    parser.add_argument(
        "scenario",
        help="Scenario JSON path or name (e.g., my_scenario.json or /path/to/my_scenario.json).",
    )
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    parser.add_argument("--client-timeout", type=float, default=float(os.environ.get("CARLA_TIMEOUT", "10.0")))
    parser.add_argument(
        "--tick-mode",
        choices=("auto", "own", "ros"),
        default="auto",
        help="World tick source: auto (match VSE), own (call world.tick), ros (wait_for_tick).",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=None,
        help="Fixed delta seconds when forcing sync mode (default: world setting or 0.05).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.environ.get("VSE_SCENARIO_RUNNER_TIMEOUT", "18000.0") or 18000.0),
        help="Scenario timeout seconds (default: $VSE_SCENARIO_RUNNER_TIMEOUT or 18000).",
    )
    parser.add_argument(
        "--wait-for-ego",
        action="store_true",
        help="Attach to an existing ego vehicle in the world (like VSE external-ego mode).",
    )
    parser.add_argument(
        "--external-ego-actor-id",
        type=int,
        default=None,
        help="Explicit external ego actor id to attach/warp (implies --wait-for-ego).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to a ScenarioRunner-compatible agent Python file (required when using an external ego / tick-mode=ros).",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def _configure_logging(debug_enabled: bool) -> None:
    # Configure logging based on --debug flag
    if debug_enabled:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("Debug logging enabled via --debug flag")
    else:
        # Only show warnings and errors by default
        logging.basicConfig(
            level=logging.WARNING,
            format='%(name)s - %(levelname)s - %(message)s'
        )


def _load_scenario_json(parser: argparse.ArgumentParser, scenario_arg: str) -> Tuple[Path, dict]:
    try:
        json_path = _resolve_json_arg(scenario_arg)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            scenario_data = json.load(fh)
    except Exception as exc:
        parser.error(f"Failed to read scenario JSON: {exc}")
    return json_path, scenario_data


def _connect_carla(
    parser: argparse.ArgumentParser,
    *,
    host: str,
    port: int,
    client_timeout: float,
) -> Tuple[carla.Client, carla.World]:
    try:
        client = carla.Client(host, int(port))
        client.set_timeout(float(client_timeout))
        world = client.get_world()
    except Exception as exc:
        parser.error(f"Failed to connect to CARLA at {host}:{port}: {exc}")
    return client, world


def _resolve_external_ego(
    parser: argparse.ArgumentParser,
    *,
    world: carla.World,
    args: argparse.Namespace,
    scenario_data: dict,
) -> Tuple[bool, Optional[int], Optional[carla.Actor]]:
    scenario_has_ego = _scenario_has_ego(scenario_data)
    expected_roles = _expected_ego_roles(scenario_data)
    ego_blueprint_id = _scenario_ego_blueprint_id(scenario_data)

    external_ego_id = None
    external_ego_actor = None
    if args.external_ego_actor_id is not None:
        try:
            actor = world.get_actor(int(args.external_ego_actor_id))
        except Exception:
            actor = None
        if not actor or not getattr(actor, "is_alive", False):
            parser.error(f"External ego actor id {args.external_ego_actor_id} not found/alive in world.")
        external_ego_id = int(args.external_ego_actor_id)
        external_ego_actor = actor
    elif args.tick_mode in ("auto", "ros") or args.wait_for_ego:
        # Auto-detect an already-spawned ego and use it as the ego vehicle source + tick authority.
        external_actor = _find_external_ego(world, expected_roles, blueprint_id=ego_blueprint_id)

        # Optional grace period to allow external stacks to spawn/register the ego
        # before we decide to run in standalone mode.
        #
        # Default is 1s (helps avoid a race where the ego exists but hasn't shown
        # up in the actor list yet). Override via $VSE_EXTERNAL_EGO_DETECT_TIMEOUT_S.
        try:
            detect_timeout_s = float(os.environ.get("VSE_EXTERNAL_EGO_DETECT_TIMEOUT_S", "1.0") or 1.0)
        except Exception:
            detect_timeout_s = 0.0
        if args.wait_for_ego:
            detect_timeout_s = max(detect_timeout_s, 10.0)
        detect_timeout_s = max(0.0, detect_timeout_s)

        if external_actor is None and detect_timeout_s > 0.0:
            deadline = time.time() + detect_timeout_s
            while external_actor is None and time.time() < deadline:
                try:
                    world.wait_for_tick(0.25)
                except Exception:
                    time.sleep(0.25)
                external_actor = _find_external_ego(world, expected_roles, blueprint_id=ego_blueprint_id)

        if external_actor is not None:
            external_ego_id = int(external_actor.id)
            external_ego_actor = external_actor
        elif args.wait_for_ego:
            parser.error("External ego requested but not found in world actor list.")

    return scenario_has_ego, external_ego_id, external_ego_actor


def _resolve_tick_mode_and_agent(
    parser: argparse.ArgumentParser,
    *,
    args: argparse.Namespace,
    scenario_has_ego: bool,
    external_ego_id: Optional[int],
) -> Tuple[str, Optional[str], bool]:
    tick_mode = args.tick_mode
    if tick_mode == "auto":
        tick_mode = "ros" if external_ego_id is not None else "own"

    if tick_mode == "ros" and external_ego_id is None:
        parser.error("Tick mode 'ros' requires an external ego vehicle, but none was detected in the world.")

    if tick_mode == "ros":
        if not args.agent:
            parser.error("External ego detected; --agent /path/to/agent.py is required to publish the route.")
    else:
        if args.agent:
            parser.error("--agent was provided but no external ego is in use (tick-mode is not 'ros').")

    agent_path: Optional[str] = None
    if args.agent:
        try:
            agent_candidate = Path(str(args.agent)).expanduser().resolve()
        except Exception:
            agent_candidate = None
        if not agent_candidate or not agent_candidate.exists() or not agent_candidate.is_file():
            parser.error(f"Agent file not found: {args.agent}")
        agent_path = str(agent_candidate)

    wait_for_ego = bool(args.wait_for_ego or args.external_ego_actor_id is not None)
    if not wait_for_ego:
        wait_for_ego = bool(tick_mode == "ros" and scenario_has_ego and external_ego_id is not None)

    return tick_mode, agent_path, wait_for_ego


def _resolve_fixed_delta(world: carla.World, fixed_delta_arg: Optional[float]) -> float:
    fixed_delta = fixed_delta_arg
    if fixed_delta is None:
        fixed_delta = 0.05
        try:
            settings = world.get_settings()
            if settings and getattr(settings, "fixed_delta_seconds", None):
                fixed_delta = float(settings.fixed_delta_seconds)
        except Exception:
            fixed_delta = 0.05
    return float(fixed_delta)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(bool(args.debug))

    json_path, scenario_data = _load_scenario_json(parser, args.scenario)
    client, world = _connect_carla(
        parser,
        host=str(args.host),
        port=int(args.port),
        client_timeout=float(args.client_timeout),
    )
    scenario_has_ego, external_ego_id, external_ego_actor = _resolve_external_ego(
        parser,
        world=world,
        args=args,
        scenario_data=scenario_data,
    )
    tick_mode, agent_path, wait_for_ego = _resolve_tick_mode_and_agent(
        parser,
        args=args,
        scenario_has_ego=scenario_has_ego,
        external_ego_id=external_ego_id,
    )

    if external_ego_actor is not None and getattr(external_ego_actor, "is_alive", False):
        try:
            role_name = external_ego_actor.attributes.get("role_name", "")
        except Exception:
            role_name = ""
        try:
            type_id = external_ego_actor.type_id
        except Exception:
            type_id = ""
        print(f"[MiniRunner] Using external ego actor {external_ego_id} ({type_id}) role='{role_name}'")

    fixed_delta = _resolve_fixed_delta(world, args.fixed_delta)

    timeout_s = max(30.0, float(args.timeout_s or 18000.0))

    finish: dict = {"reason": None}

    def _on_finish(reason: str):
        finish["reason"] = reason
        print(f"[MiniRunner] Scenario finished ({reason})")

    runner = MiniRunner(
        client=client,
        world=world,
        json_path=str(json_path),
        tick_mode=str(tick_mode),
        fixed_delta=float(fixed_delta),
        wait_for_ego=bool(wait_for_ego),
        ego_role_name="ego_vehicle",
        timeout_s=float(timeout_s),
        external_ego_actor_id=external_ego_id,
        log_fn=lambda msg: print(f"[MiniRunner] {msg}"),
        debug=bool(args.debug),
        on_finish=_on_finish,
        agent_path=agent_path,
    )

    print(f"[MiniRunner] Running {json_path} (mode={tick_mode}, timeout={timeout_s}s)")
    runner.start()

    thread = getattr(runner, "_thread", None)
    if not thread:
        return 1

    try:
        while thread.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("[MiniRunner] Stop requested by user.")
        try:
            runner.request_stop()
        except Exception:
            pass

        # Give the worker thread a chance to clean up, but do not crash on a
        # second Ctrl-C; instead force cleanup and exit.
        deadline = time.time() + 5.0
        while thread.is_alive() and time.time() < deadline:
            try:
                thread.join(timeout=0.25)
            except KeyboardInterrupt:
                break

        try:
            runner._cleanup("stopped by user")
        except KeyboardInterrupt:
            pass
        except Exception:
            pass

        try:
            thread.join(timeout=1.0)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        return 130

    reason = str(finish.get("reason") or "")
    if reason.lower().startswith("failed") or reason.lower().startswith("timeout"):
        return 1
    return 0


# ------------------------------------------------------------
# MiniRunner: lightweight in-process runner for VSE
# ------------------------------------------------------------


class _ConfigStub:
    """Minimal config object to satisfy vse_play without ScenarioRunner."""

    def __init__(self, name: str, town: str, weather: carla.WeatherParameters, weather_keyframes=None):
        self.name = name
        self.town = town
        self.weather = weather
        self.weather_keyframes = weather_keyframes or []
        self.friction = None
        self.other_actors = []
        self.route = []
        self.route_var_name = None
        self.trigger_points = []


# =============================================================================
# MINI RUNNER
# Standalone scenario execution without full ScenarioRunner infrastructure
# =============================================================================


class MiniRunner:
    """Standalone runner for vse_play that does not require scenario_runner."""

    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        json_path: str,
        tick_mode: str = "own",  # "own" or "ros"
        fixed_delta: float = 0.05,
        wait_for_ego: bool = False,
        ego_role_name: str = "ego_vehicle",
        timeout_s: float = 18000.0,
        external_ego_actor_id: Optional[int] = None,
        log_fn=None,
        debug: bool = False,
        on_finish=None,
        ros_publish_delay: float = 1.5,
        agent_path: Optional[str] = None,
    ):
        self.client = client
        self.world = world
        self.json_path = json_path
        self.tick_mode = tick_mode
        self.fixed_delta = max(0.001, float(fixed_delta or 0.05))
        self.wait_for_ego = wait_for_ego
        self.ego_role_name = ego_role_name
        self.timeout_s = max(1.0, float(timeout_s or 18000.0))
        self.external_ego_actor_id = external_ego_actor_id
        self.log = log_fn or (lambda msg: print(f"[MiniRunner] {msg}"))
        self.debug = debug
        self.on_finish = on_finish
        self.ros_publish_delay = max(0.0, float(ros_publish_delay))
        self.agent_path = agent_path
        try:
            self.ros_goal_interval = max(0.0, float(os.environ.get("VSE_GOAL_INTERVAL", "0.25")))
        except Exception:
            self.ros_goal_interval = 0.25
        try:
            self.ros_downsample_interval = max(1.0, float(os.environ.get("VSE_DOWNSAMPLE_INTERVAL", "42") or 42))
        except Exception:
            self.ros_downsample_interval = 42.0
        # Cache JSON waypoints once per runner instance
        self._route_json_waypoints: List[Tuple[float, float, float, float]] = []
        self._route_json_destination: Optional[carla.Location] = None
        self._full_route_for_criteria: Optional[List[Tuple[carla.Transform, RoadOption]]] = None
        self._ros_agent_proc: Optional[multiprocessing.Process] = None
        self._prepped_ego_actor: Optional[carla.Actor] = None
        self._spawned_internal_ego_id: Optional[int] = None
        self._cleanup_lock = threading.Lock()
        self._cleanup_invoked = False

        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._running = False
        self._scenario: Optional[vse_play] = None
        self._restore_settings: Optional[carla.WorldSettings] = None
        self._restore_tm_sync: Optional[bool] = None
        self._restore_weather: Optional[carla.WeatherParameters] = None
        self._ego_controller: Optional[VehicleController] = None
        self._internal_ego_vehicle_data: Optional[VehicleData] = None
        self._ego_arrival_mark: Optional[float] = None
        self._has_ego_vehicle: Optional[bool] = None
        self._forced_green_lights: List[carla.Actor] = []
        self._large_map_active = False
        self._skip_route_interpolation = False
        try:
            carla_map = world.get_map() if world else None
        except Exception:
            carla_map = None
        self._large_map_active = _is_large_map(carla_map)
        self._skip_route_interpolation = (
            self._large_map_active and os.environ.get("VSE_FORCE_ROUTE_INTERPOLATION") != "1"
        )

    def _diag(self, message: str) -> None:
        if not self.debug:
            return
        try:
            self.log(f"DIAG {message}")
        except Exception:
            pass

    def _enable_faulthandler(self) -> None:
        if not self.debug:
            return
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        thread = getattr(self, "_thread", None)
        if thread and thread.is_alive():
            return True
        return self._running

    def start(self):
        self._enable_faulthandler()
        self._diag("start: enter")

        if self._thread and self._thread.is_alive():
            self._diag("start: thread already alive")
            return

        if self.tick_mode == "ros":
            if not self.agent_path:
                self.log("External ego tick-mode is 'ros' but no --agent was provided; aborting.")
                return
            # Ensure CarlaDataProvider is initialized *before* forking the ROS plan
            # publisher process. The child process may call interpolate_trajectory(),
            # which relies on CarlaDataProvider state; without this, first-run route
            # publishes can degrade to “destination only”.
            try:
                if self.world and not self._skip_route_interpolation:
                    _init_carla_data_provider(self.world, self.client)
                elif self.client:
                    CarlaDataProvider.set_client(self.client)
            except Exception:
                pass

            raw_waypoints, destination = self._load_json_waypoints()
            if raw_waypoints and not self._route_json_waypoints:
                try:
                    self._route_json_waypoints = self._build_route_for_ros(raw_waypoints)
                except Exception:
                    self._route_json_waypoints = []
            if destination is None and self._route_json_waypoints:
                try:
                    last = self._route_json_waypoints[-1]
                    destination = carla.Location(x=float(last[0]), y=float(last[1]), z=float(last[2]))
                except Exception:
                    destination = None
            self._route_json_destination = destination
            if self.wait_for_ego:
                ego_actor = self._prepped_ego_actor if self._prepped_ego_actor else self._prepare_ego_actor()
                if ego_actor is None:
                    self.log("External ego requested but not found; aborting.")
                    return
                self._prepped_ego_actor = ego_actor
            self._start_minimal_agent_publish(raw_waypoints)

        self._diag("start: thread-create")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._diag("start: thread-started")

    def request_stop(self):
        self._stop_requested = True

    def _run_preflight(
        self,
    ) -> Optional[Tuple[str, Optional[VehicleData], Optional[carla.Actor]]]:
        self._preflight_failure_reason = ""
        if not self.world or not self.client:
            self._diag("run: missing world/client")
            self.log("Missing CARLA world/client; aborting.")
            self._preflight_failure_reason = "failed: missing CARLA world/client"
            return None

        if not os.path.isfile(self.json_path):
            self._diag("run: scenario json missing")
            self.log(f"Scenario JSON not found: {self.json_path}")
            self._preflight_failure_reason = f"failed: scenario JSON not found: {self.json_path}"
            return None

        scenario_name = Path(self.json_path).stem
        self.log(f"Starting scenario '{scenario_name}' in mini-runner")

        env = os.environ.copy()
        env["VSE_SCENARIO_JSON_PATH"] = os.path.abspath(self.json_path)
        os.environ.update({"VSE_SCENARIO_JSON_PATH": env["VSE_SCENARIO_JSON_PATH"]})

        raw_waypoints, destination = self._load_json_waypoints()
        self._route_json_destination = destination
        if raw_waypoints and not self._route_json_waypoints:
            self._route_json_waypoints = list(raw_waypoints)

        if self._stop_requested:
            self._preflight_failure_reason = "stopped by user"
            return None

        self._diag("run: init-data-provider")
        try:
            _init_carla_data_provider(self.world, self.client)
        except Exception as exc:
            self.log(f"Failed to initialize CarlaDataProvider: {exc}")
            self._preflight_failure_reason = f"failed: data provider init failed: {exc}"
            return None
        self._diag("run: init-data-provider done")
        GameTime.restart()

        if self._stop_requested:
            self._preflight_failure_reason = "stopped by user"
            return None

        self._diag("run: apply-sync-settings")
        self._apply_sync_settings()
        self._diag("run: apply-sync-settings done")
        if self._stop_requested:
            self._preflight_failure_reason = "stopped by user"
            return None
        internal_ego_vehicle_data: Optional[VehicleData] = None
        if not self.wait_for_ego:
            internal_ego_vehicle_data = self._load_internal_ego_vehicle_data()
            self._internal_ego_vehicle_data = internal_ego_vehicle_data
            if internal_ego_vehicle_data and internal_ego_vehicle_data.destination:
                self._route_json_destination = internal_ego_vehicle_data.destination

        ego_actor: Optional[carla.Actor] = None
        if self._has_ego_vehicle is not False:
            self._diag("run: prepare-ego")
            ego_actor = self._prepped_ego_actor if self._prepped_ego_actor else self._prepare_ego_actor()
            if ego_actor is not None:
                try:
                    role = ego_actor.attributes.get("role_name", "")
                except Exception:
                    role = ""
                self._diag(f"run: prepare-ego done (actor_id={getattr(ego_actor, 'id', '?')}, role='{role}')")
            else:
                self._diag("run: prepare-ego done (actor=None)")
        else:
            self.wait_for_ego = False
        if self.wait_for_ego and ego_actor is None:
            self.log("External ego requested but not found; aborting.")
            self._preflight_failure_reason = "failed: external ego requested but not found"
            return None
        self._prepped_ego_actor = ego_actor

        if self._stop_requested:
            self._preflight_failure_reason = "stopped by user"
            return None

        # Build a stable (no-GRP) route cache for criteria/weather and optional ROS publish.
        if raw_waypoints:
            self._diag("run: build-route-cache")
            try:
                self._route_json_waypoints = self._build_route_for_ros(raw_waypoints)
            except Exception:
                # Keep raw waypoints if route build fails.
                self._route_json_waypoints = list(raw_waypoints)
            self._diag("run: build-route-cache done")
            if self._stop_requested:
                self._preflight_failure_reason = "stopped by user"
                return None

        return scenario_name, internal_ego_vehicle_data, ego_actor

    def _run_build_scenario(
        self,
        *,
        scenario_name: str,
        ego_actor: Optional[carla.Actor],
        internal_ego_vehicle_data: Optional[VehicleData],
    ) -> bool:
        self._diag("run: build-scenario enter")
        if self._restore_weather is None and self.world:
            try:
                self._restore_weather = self.world.get_weather()
            except Exception:
                self._restore_weather = None

        weather_keyframes = self._load_json_weather_keyframes()
        start_weather = weather_keyframes[0][1] if weather_keyframes else None
        if start_weather:
            try:
                self.world.set_weather(start_weather)
                self.log("Applied start weather from scenario JSON")
            except Exception as exc:
                self.log(f"Failed to apply start weather: {exc}")
        elif self.world:
            start_weather = self.world.get_weather()

        config = _ConfigStub(
            name=scenario_name,
            town=self.world.get_map().name if self.world else "Unknown",
            weather=start_weather or (self.world.get_weather() if self.world else carla.WeatherParameters()),
            weather_keyframes=weather_keyframes,
        )

        ego_list = [ego_actor] if ego_actor else []
        self._scenario = vse_play(
            self.world,
            ego_list,
            config,
            randomize=False,
            debug_mode=self.debug,
            criteria_enable=True,
            timeout=self.timeout_s,
        )
        self._diag("run: build-scenario vse_play init done")
        # Set ego destination from JSON waypoint data for arrival detection
        if self._route_json_destination and self._scenario:
            dest_loc = self._route_json_destination
            try:
                self._scenario._ego_destination = dest_loc
                self.log(f"Ego destination set to ({dest_loc.x:.2f}, {dest_loc.y:.2f}, {dest_loc.z:.2f})")
            except Exception:
                pass
        criteria_route = self._criteria_route_from_waypoints()
        if criteria_route and self._scenario:
            try:
                self._scenario._ego_route_for_criteria = criteria_route
                if self._route_json_destination and getattr(self._scenario, "_ego_destination", None) is None:
                    self._scenario._ego_destination = self._route_json_destination
                self.log(f"Ego route for criteria set from agent plan ({len(criteria_route)} points).")
            except Exception as exc:
                self.log(f"Failed to set ego criteria route from agent plan: {exc}")
        if ego_actor and internal_ego_vehicle_data and not self.wait_for_ego:
            self._ego_controller = self._start_internal_ego_controller(ego_actor, internal_ego_vehicle_data)
            if not self._ego_controller:
                self.log("Internal ego controller not started; vehicle will remain stationary/manual.")
        elif not self.wait_for_ego and internal_ego_vehicle_data is None:
            self.log("No ego waypoints available; internal autopilot disabled.")

        # For external ego with ignore_traffic_lights, force all lights green
        if self.wait_for_ego and ego_actor:
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    scenario_data = json.load(f)
                ego_entries = scenario_data.get("ego_vehicle", [])
                if isinstance(ego_entries, dict):
                    ego_entries = [ego_entries]
                for ego_entry in ego_entries:
                    if ego_entry.get("ignore_traffic_lights", False):
                        self._force_all_traffic_lights_green()
                        break
            except Exception as exc:
                self.log(f"Failed to check external ego ignore_traffic_lights: {exc}")

        has_non_ego_actors = False
        if self._scenario:
            try:
                raw_vehicles = getattr(self._scenario, "_raw_vehicle_entries", None) or []
                raw_pedestrians = getattr(self._scenario, "_raw_pedestrian_entries", None) or []
                has_non_ego_actors = bool(raw_vehicles or raw_pedestrians)
            except Exception:
                has_non_ego_actors = False
        return has_non_ego_actors

    def _run(self):
        self._running = True
        self._diag("run: enter")
        reason = ""
        start_system_time = time.time()
        start_game_time = GameTime.get_time()
        end_system_time = start_system_time
        end_game_time = start_game_time
        try:
            preflight = self._run_preflight()
            if preflight is None:
                reason = reason or getattr(self, "_preflight_failure_reason", "") or "failed: preflight"
                return
            scenario_name, internal_ego_vehicle_data, ego_actor = preflight
            has_non_ego_actors = self._run_build_scenario(
                scenario_name=scenario_name,
                ego_actor=ego_actor,
                internal_ego_vehicle_data=internal_ego_vehicle_data,
            )

            start_time = time.monotonic()
            tick_interval = self.fixed_delta if self.tick_mode == "own" else None
            last_tick_time = time.perf_counter()
            last_dist_log = 0.0
            while not self._stop_requested:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.timeout_s:
                    reason = "timeout"
                    self.log(f"Scenario timeout reached after {elapsed:.1f}s")
                    break

                # Pace own-tick mode to the configured fixed delta (e.g., 20 Hz) so
                # manual control speed isn't tied to render FPS.
                if tick_interval:
                    now = time.perf_counter()
                    remaining = tick_interval - (now - last_tick_time)
                    if remaining > 0:
                        time.sleep(remaining)
                        continue
                    last_tick_time = now

                snapshot = self._tick_world()
                if not snapshot:
                    continue

                GameTime.on_carla_tick(snapshot.timestamp)
                CarlaDataProvider.on_carla_tick()

                if self._scenario and getattr(self._scenario, "scenario_tree", None):
                    try:
                        self._scenario.scenario_tree.tick_once()
                    except Exception:
                        pass

                if self._scenario:
                    # Check ego arrival to terminate early when a destination exists
                    arrived, dist = self._check_ego_arrival(self._scenario, return_distance=True)
                    now = time.monotonic()
                    if now - last_dist_log >= 2.0:
                        try:
                            ego = self._scenario.ego_vehicles[0] if self._scenario.ego_vehicles else None
                            dest = getattr(self._scenario, "_ego_destination", None)
                            if ego and dest:
                                self.log(f"Ego distance to destination: {dist:.2f} m")
                        except Exception:
                            pass
                        last_dist_log = now
                    if arrived:
                        if self.wait_for_ego:
                            # External ego: require 1s stable arrival
                            if self._ego_arrival_mark is None:
                                self._ego_arrival_mark = now
                            if now - (self._ego_arrival_mark or now) >= 2.0:
                                reason = "ego arrived"
                                break
                        else:
                            # Local agent: no extra dwell time once within radius
                            reason = "ego arrived"
                            break
                    else:
                        self._ego_arrival_mark = None

                    # If there is no ego destination, end once all actors/triggers are done,
                    # but only when non-ego actors exist (ego-only manual runs stay active).
                    if getattr(self._scenario, "_ego_destination", None) is None:
                        if has_non_ego_actors and self._actors_completed_without_ego(self._scenario):
                            reason = "actors completed (no ego destination)"
                            break

            if self._stop_requested:
                reason = reason or "stopped by user"
        except Exception as exc:  # pylint: disable=broad-except
            reason = reason or f"failed: {exc}"
            self.log(f"Error during mini-runner execution: {exc}")
        finally:
            end_system_time = time.time()
            try:
                end_game_time = GameTime.get_time()
            except Exception:
                end_game_time = start_game_time
            try:
                self._emit_ego_result_table(
                    start_system_time=start_system_time,
                    start_game_time=start_game_time,
                    end_system_time=end_system_time,
                    end_game_time=end_game_time,
                    reason=reason,
                )
            except Exception as exc:
                self.log(f"Failed to emit ego result table: {exc}")
            try:
                self._cleanup(reason)
            finally:
                self._running = False

    def _emit_ego_result_table(
        self,
        *,
        start_system_time: float,
        start_game_time: Optional[float],
        end_system_time: float,
        end_game_time: Optional[float],
        reason: str,
    ) -> None:
        """Print a scenario_runner-style table for ego-only criteria."""
        result_file: Optional[str] = None
        try:
            json_path = Path(self.json_path).resolve()
            result_file = str(json_path.with_suffix(".txt"))
        except Exception:
            result_file = None

        def _write_fallback(text: str) -> None:
            if not result_file:
                return
            try:
                Path(result_file).write_text(text, encoding="utf-8")
            except Exception:
                pass

        if not self._scenario:
            _write_fallback(f"Scenario finished ({reason})\n")
            return
        ego_actor = self._scenario.ego_vehicles[0] if self._scenario.ego_vehicles else None
        if not ego_actor:
            _write_fallback(f"Scenario finished ({reason})\n(No ego vehicle)\n")
            return

        ego_criteria = list(getattr(self._scenario, "_ego_criteria", []))
        if not ego_criteria:
            ego_criteria = [
                c for c in self._scenario.get_criteria() if getattr(c, "actor", None) is ego_actor
            ]
        if not ego_criteria:
            _write_fallback(f"Scenario finished ({reason})\n(No ego criteria)\n")
            return

        result_label = self._resolve_result_label(reason, ego_criteria)
        duration_system = max(0.0, end_system_time - start_system_time)
        try:
            duration_game = max(0.0, (end_game_time or 0.0) - (start_game_time or 0.0))
        except Exception:
            duration_game = 0.0

        proxy = _EgoCriteriaProxy(self._scenario, ego_criteria)
        stub = SimpleNamespace(
            scenario=proxy,
            scenario_tree=self._scenario.scenario_tree,
            ego_vehicles=self._scenario.ego_vehicles,
            other_actors=self._scenario.other_actors,
            start_system_time=start_system_time,
            end_system_time=end_system_time,
            scenario_duration_system=duration_system,
            scenario_duration_game=duration_game,
        )

        ResultOutputProvider(
            stub,
            result_label,
            stdout=True,
            filename=result_file,
            junitfile=None,
            jsonfile=None,
        ).write()

    @staticmethod
    def _normalize_reason(reason: Optional[str]) -> str:
        return reason.lower().strip() if reason else ""

    def _resolve_result_label(self, reason: str, ego_criteria: List[Criterion]) -> str:
        """Determine final result label similar to scenario_runner."""
        label = "SUCCESS"
        normalized = self._normalize_reason(reason)
        if normalized.startswith("timeout"):
            label = "TIMEOUT"
        elif normalized.startswith("stopped by user") or normalized.startswith("failed"):
            label = "FAILURE"
        elif normalized and not normalized.startswith("actors completed"):
            label = "FAILURE"

        if label != "TIMEOUT":
            for criterion in ego_criteria:
                if not getattr(criterion, "optional", False) and criterion.test_status not in (
                    "SUCCESS",
                    "ACCEPTABLE",
                ):
                    label = "FAILURE"
                    break
        return label

    def _load_json_waypoints(self) -> Tuple[List[Tuple[float, float, float, float]], Optional[carla.Location]]:
        """Load ego waypoints (x, y, z, yaw_deg) from the scenario JSON."""
        json_path = Path(self.json_path).resolve()
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.log(f"Failed to read scenario JSON {json_path}: {exc}")
            return [], None

        ego_entry = data.get("ego_vehicle")
        if not ego_entry:
            for entry in data.get("vehicles", []):
                if str(entry.get("role", "")).lower() == "ego_vehicle":
                    ego_entry = entry
                    break
        if not ego_entry:
            self._has_ego_vehicle = False
            self.log("No ego_vehicle entry found in scenario JSON; running without ego waypoints")
            return [], None
        self._has_ego_vehicle = True

        waypoints = ego_entry.get("waypoints", [])
        if not waypoints:
            self.log("No waypoints found for ego_vehicle in scenario JSON")
            return [], None

        # Ensure the route starts at the ego spawn location.
        try:
            spawn_loc = ego_entry.get("location", {})
            spawn_rot = ego_entry.get("rotation", {})
            start_x = float(spawn_loc.get("x", 0.0))
            start_y = float(spawn_loc.get("y", 0.0))
            start_z = float(spawn_loc.get("z", 0.0))
            start_yaw = float(spawn_rot.get("yaw", 0.0))
            need_insert = True
            if waypoints:
                first_loc = waypoints[0].get("location", {})
                try:
                    first_x = float(first_loc.get("x", 0.0))
                    first_y = float(first_loc.get("y", 0.0))
                    first_z = float(first_loc.get("z", 0.0))
                    if (
                        abs(first_x - start_x) <= 0.25
                        and abs(first_y - start_y) <= 0.25
                        and abs(first_z - start_z) <= 0.5
                    ):
                        need_insert = False
                except Exception:
                    pass
            if need_insert:
                waypoints.insert(
                    0,
                    {
                        "index": 0,
                        "location": {"x": start_x, "y": start_y, "z": start_z},
                        "yaw": start_yaw,
                        "speed_km_h": ego_entry.get("speed_km_h", 0.0),
                        "idle_time_s": 0.0,
                        "turn_time_s": 0.0,
                        "auto_generated": True,
                        "is_destination": False,
                    },
                )
                self.log("Prepended ego spawn location as start waypoint for routing")
        except Exception:
            pass

        def _fallback_yaw(idx: int) -> float:
            """Estimate yaw if missing by looking at neighboring waypoints."""
            loc = waypoints[idx].get("location", {})
            prev_loc = waypoints[idx - 1].get("location", {}) if idx > 0 else None
            next_loc = waypoints[idx + 1].get("location", {}) if idx + 1 < len(waypoints) else None
            ref_prev = prev_loc if prev_loc and prev_loc.get("x") is not None else None
            ref_next = next_loc if next_loc and next_loc.get("x") is not None else None
            if ref_next:
                dx = float(ref_next.get("x", 0.0)) - float(loc.get("x", 0.0))
                dy = float(ref_next.get("y", 0.0)) - float(loc.get("y", 0.0))
            elif ref_prev:
                dx = float(loc.get("x", 0.0)) - float(ref_prev.get("x", 0.0))
                dy = float(loc.get("y", 0.0)) - float(ref_prev.get("y", 0.0))
            else:
                return 0.0
            return _normalize_yaw(math.degrees(math.atan2(dy, dx)))

        raw_waypoints: List[Tuple[float, float, float, float]] = []
        destination_loc: Optional[carla.Location] = None
        destination_wp: Optional[dict] = None

        for idx, wp in enumerate(waypoints):
            wp_loc = wp.get("location", {})
            try:
                x = float(wp_loc.get("x", 0.0))
                y = float(wp_loc.get("y", 0.0))
                z = float(wp_loc.get("z", 0.0))
                yaw_val = wp.get("yaw")
                yaw = float(yaw_val) if yaw_val is not None else _fallback_yaw(idx)
            except Exception:
                continue
            raw_waypoints.append((x, y, z, yaw))

            is_dest = bool(wp.get("is_destination")) or str(wp.get("index")) == "destination"
            if is_dest and destination_wp is None:
                destination_wp = wp

        if destination_wp is None and waypoints:
            destination_wp = waypoints[-1]

        if destination_wp:
            loc = destination_wp.get("location", {})
            try:
                destination_loc = carla.Location(
                    x=float(loc.get("x", 0.0)),
                    y=float(loc.get("y", 0.0)),
                    z=float(loc.get("z", 0.0)),
                )
            except Exception:
                destination_loc = None

        self.log(f"Loaded {len(raw_waypoints)} waypoint(s) from {json_path}")

        return raw_waypoints, destination_loc

    def _load_json_weather_keyframes(self) -> List[Tuple[float, carla.WeatherParameters]]:
        """Load weather keyframes from the scenario JSON."""
        json_path = Path(self.json_path).resolve()
        keyframes: List[Tuple[float, carla.WeatherParameters]] = []
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.log(f"Failed to read scenario JSON {json_path} for weather: {exc}")
            return keyframes

        raw_keyframes = data.get("weather_keyframes") if isinstance(data, dict) else None
        if not isinstance(raw_keyframes, list):
            return keyframes

        def _to_weather(payload: dict) -> Optional[carla.WeatherParameters]:
            if not isinstance(payload, dict):
                return None
            weather = carla.WeatherParameters()
            for name, val in payload.items():
                if name == "route_percentage":
                    continue
                if hasattr(weather, name):
                    try:
                        setattr(weather, name, float(val))
                    except Exception:
                        pass
            return weather

        for frame in raw_keyframes:
            if not isinstance(frame, dict):
                continue
            try:
                pct = float(frame.get("route_percentage", 0.0))
            except Exception:
                continue
            weather = _to_weather(frame)
            if weather is None:
                continue
            keyframes.append((max(0.0, min(100.0, pct)), weather))

        if not keyframes:
            return keyframes

        keyframes.sort(key=lambda item: item[0])
        if keyframes[0][0] > 0.0:
            keyframes.insert(0, (0.0, keyframes[0][1]))
        if keyframes[-1][0] < 100.0:
            keyframes.append((100.0, keyframes[-1][1]))
        if len(keyframes) == 1:
            keyframes.append((100.0, keyframes[0][1]))
        return keyframes

    def _start_minimal_agent_publish(self, raw_waypoints: List[Tuple[float, float, float, float]]) -> None:
        """Spawn a one-off agent publisher in a child process."""
        if not raw_waypoints or len(raw_waypoints) < 2:
            self.log("No waypoints available for agent; skipping publish")
            return
        if not self.world:
            self.log("World unavailable; cannot initialize agent publish")
            return
        if not self.agent_path:
            self.log("Agent path unavailable; cannot initialize agent publish")
            return

        # Stop any previous publisher before starting a fresh one.
        self._stop_ros_agent_process()

        try:
            proc = multiprocessing.Process(
                target=_ros_plan_publisher_process,
                args=(
                    raw_waypoints,
                    self.ros_downsample_interval,
                    self.ros_publish_delay,
                    self.agent_path,
                    self._skip_route_interpolation,
                ),
                daemon=True,
            )
            proc.start()
            self._ros_agent_proc = proc
            self.log(f"Agent subprocess started (pid {proc.pid}) to publish route with {len(raw_waypoints)} points")
        except Exception as exc:
            self.log(f"Failed to launch agent subprocess: {exc}")

    def _stop_ros_agent_process(self):
        """Terminate any running agent subprocess."""
        proc = getattr(self, "_ros_agent_proc", None)
        if not proc:
            return
        try:
            if proc.is_alive():
                # Request graceful termination
                proc.terminate()
                # Wait up to 5 seconds for graceful exit
                proc.join(timeout=5.0)

                # If still alive, force kill
                if proc.is_alive():
                    self.log("ROS agent process did not terminate gracefully, force killing...")
                    proc.kill()
                    # Wait briefly for kill to take effect
                    proc.join(timeout=1.0)

                # Final check
                if proc.is_alive():
                    self.log(f"WARNING: ROS agent process (pid {proc.pid}) could not be terminated")
        except Exception as exc:
            try:
                self.log(f"Exception during ROS agent cleanup: {exc}")
            except Exception:
                pass
        finally:
            self._ros_agent_proc = None

    def _load_internal_ego_vehicle_data(self) -> Optional[VehicleData]:
        """Build VehicleData for the ego when running without external control."""
        json_path = Path(self.json_path).resolve()
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.log(f"Failed to read scenario JSON {json_path}: {exc}")
            return None

        ego_entry = data.get("ego_vehicle")
        if not ego_entry:
            for entry in data.get("vehicles", []):
                if str(entry.get("role", "")).lower() == "ego_vehicle":
                    ego_entry = entry
                    break
        if not ego_entry:
            self._has_ego_vehicle = False
            return None
        self._has_ego_vehicle = True

        default_speed = EGO_DEFAULT_SPEED_KMH
        dest_speed_val = ego_entry.get("destination_speed_km_h")
        try:
            if dest_speed_val is not None:
                dest_speed_val = float(dest_speed_val)
        except Exception:
            dest_speed_val = None

        spawn_loc = ego_entry.get("location", {})
        spawn_rot = ego_entry.get("rotation", {})
        spawn_location = carla.Location(
            x=float(spawn_loc.get("x", 0.0)),
            y=float(spawn_loc.get("y", 0.0)),
            z=float(spawn_loc.get("z", 0.0)),
        )
        spawn_rotation = carla.Rotation(
            pitch=float(spawn_rot.get("pitch", 0.0)),
            yaw=float(spawn_rot.get("yaw", 0.0)),
            roll=float(spawn_rot.get("roll", 0.0)),
        )

        waypoints_data = ego_entry.get("waypoints", [])
        if not waypoints_data:
            return None

        def _fallback_yaw(idx: int) -> float:
            """Estimate yaw if missing by looking at neighboring waypoints."""
            loc = waypoints_data[idx].get("location", {})
            prev_loc = waypoints_data[idx - 1].get("location", {}) if idx > 0 else None
            next_loc = waypoints_data[idx + 1].get("location", {}) if idx + 1 < len(waypoints_data) else None
            ref_prev = prev_loc if prev_loc and prev_loc.get("x") is not None else None
            ref_next = next_loc if next_loc and next_loc.get("x") is not None else None
            if ref_next:
                dx = float(ref_next.get("x", 0.0)) - float(loc.get("x", 0.0))
                dy = float(ref_next.get("y", 0.0)) - float(loc.get("y", 0.0))
            elif ref_prev:
                dx = float(loc.get("x", 0.0)) - float(ref_prev.get("x", 0.0))
                dy = float(loc.get("y", 0.0)) - float(ref_prev.get("y", 0.0))
            else:
                return 0.0
            return _normalize_yaw(math.degrees(math.atan2(dy, dx)))

        route_points: List[RoutePoint] = []
        destination_loc: Optional[carla.Location] = None
        for idx, wp in enumerate(waypoints_data):
            wp_loc = wp.get("location", {})
            try:
                x = float(wp_loc.get("x", 0.0))
                y = float(wp_loc.get("y", 0.0))
                z = float(wp_loc.get("z", 0.0))
            except Exception:
                continue
            yaw_val = wp.get("yaw")
            try:
                yaw = float(yaw_val) if yaw_val is not None else _fallback_yaw(idx)
            except Exception:
                yaw = _fallback_yaw(idx)
            speed_val = wp.get("speed_km_h", default_speed)
            try:
                speed_val = float(speed_val)
            except Exception:
                speed_val = default_speed
            if speed_val <= 0.0:
                speed_val = default_speed
            idle_val = wp.get("idle_time_s", 0.0)
            try:
                idle_val = float(idle_val)
            except Exception:
                idle_val = 0.0
            deviation_val = wp.get("speed_deviation_km_h", 0)
            try:
                deviation_val = int(float(deviation_val or 0))
            except Exception:
                deviation_val = 0
            if deviation_val < 0:
                deviation_val = 0
            is_dest = bool(wp.get("is_destination")) or str(wp.get("index")) == "destination"
            transform = carla.Transform(
                carla.Location(x=float(x), y=float(y), z=float(z)),
                carla.Rotation(pitch=spawn_rotation.pitch, yaw=yaw, roll=spawn_rotation.roll),
            )
            route_points.append(
                RoutePoint(
                    transform=transform,
                    speed_kmh=speed_val,
                    idle_time_s=idle_val,
                    is_destination=is_dest,
                    speed_deviation_kmh=deviation_val,
                )
            )
            if is_dest:
                destination_loc = transform.location

        # If only a single destination is provided, prepend the spawn pose as the start.
        if len(route_points) == 1:
            spawn_tf = carla.Transform(spawn_location, spawn_rotation)
            start_speed = route_points[0].speed_kmh if route_points else default_speed
            route_points.insert(
                0,
                RoutePoint(
                    transform=spawn_tf,
                    speed_kmh=start_speed,
                    idle_time_s=float(ego_entry.get("idle_time_s", 0.0) or 0.0),
                    is_destination=False,
                    speed_deviation_kmh=0,
                ),
            )

        # Ensure last waypoint is treated as destination if none flagged
        if route_points and destination_loc is None:
            route_points[-1].is_destination = True
            destination_loc = route_points[-1].transform.location

        # Drop leading duplicates so the agent advances past the spawn pose
        filtered: List[RoutePoint] = []
        for rp in route_points:
            if filtered:
                try:
                    if rp.transform.location.distance(filtered[-1].transform.location) < 0.75:
                        continue
                except Exception:
                    pass
            filtered.append(rp)
        route_points = filtered

        if len(route_points) < 2:
            return None

        if destination_loc is None:
            destination_loc = route_points[-1].transform.location

        try:
            max_lat_acc_val = float(ego_entry.get("max_lat_acc", 3.0) or 3.0)
        except Exception:
            max_lat_acc_val = 3.0
        if max_lat_acc_val <= 0.0:
            max_lat_acc_val = 3.0

        ignore_flags = {
            "traffic_lights": bool(ego_entry.get("ignore_traffic_lights", False)),
            "stop_signs": bool(ego_entry.get("ignore_stop_signs", False)),
            "vehicles": bool(ego_entry.get("ignore_vehicles", False)),
        }

        initial_speed = route_points[0].speed_kmh if route_points else default_speed
        dest_speed_val = dest_speed_val or route_points[-1].speed_kmh
        if dest_speed_val <= 0.0:
            dest_speed_val = default_speed

        vehicle_data = VehicleData(
            blueprint_id=ego_entry.get("type", "vehicle.lexus.utlexus"),
            spawn_location=spawn_location,
            spawn_rotation=spawn_rotation,
            destination=destination_loc,
            route_points=route_points,
            initial_speed=initial_speed,
            destination_speed=dest_speed_val,
            color=ego_entry.get("color"),
            initial_idle_time=float(ego_entry.get("idle_time_s", 0.0) or 0.0),
            ignore_traffic_lights=ignore_flags["traffic_lights"],
            ignore_stop_signs=ignore_flags["stop_signs"],
            ignore_vehicles=ignore_flags["vehicles"],
            trigger_center=None,
            trigger_radius=None,
            max_lat_acc=max_lat_acc_val,
        )
        _refine_vehicle_route(vehicle_data, max_lat_acc=vehicle_data.max_lat_acc)
        return vehicle_data

    def _build_route_for_ros(self, raw_waypoints: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Downsample/interpolate waypoints along roads to mimic carla_minimal_agent planning and cache full route."""
        if not raw_waypoints or len(raw_waypoints) < 2:
            return raw_waypoints

        if self._skip_route_interpolation:
            try:
                self.log("Large map active; skipping road interpolation (using waypoint corridor order).")
            except Exception:
                pass

            route: List[Tuple] = []
            for x, y, z, yaw in raw_waypoints:
                try:
                    tf = carla.Transform(
                        carla.Location(x=float(x), y=float(y), z=float(z)),
                        carla.Rotation(yaw=_normalize_yaw(float(yaw))),
                    )
                    route.append((tf, RoadOption.LANEFOLLOW))
                except Exception:
                    continue
            try:
                self._full_route_for_criteria = list(route) if len(route) >= 2 else None
            except Exception:
                self._full_route_for_criteria = None

            sampled_route = route
            try:
                sampled_ids = downsample_route(route, self.ros_downsample_interval)
                sampled_route = [route[i] for i in sampled_ids if 0 <= i < len(route)]
            except Exception:
                sampled_route = route

            sampled: List[Tuple[float, float, float, float]] = []
            for tf, _opt in sampled_route:
                try:
                    sampled.append(
                        (
                            float(tf.location.x),
                            float(tf.location.y),
                            float(tf.location.z),
                            float(tf.rotation.yaw),
                        )
                    )
                except Exception:
                    continue
            return sampled if sampled else raw_waypoints

        keypoints: List[carla.Location] = []
        for x, y, z, _yaw in raw_waypoints:
            try:
                keypoints.append(carla.Location(x=float(x), y=float(y), z=float(z)))
            except Exception as exc:
                self.log(f"Location build failed for ({x},{y},{z}): {exc}")
        if len(keypoints) >= 2:
            try:
                gps_route, route = interpolate_trajectory(keypoints, hop_resolution=1.0)
                try:
                    self._full_route_for_criteria = list(route)
                except Exception:
                    self._full_route_for_criteria = None
                sampled_ids = downsample_route(route, self.ros_downsample_interval)
                sampled: List[Tuple[float, float, float, float]] = []
                for idx in sampled_ids:
                    if idx < 0 or idx >= len(route):
                        continue
                    tf = route[idx][0]  # Transform
                    sampled.append(
                        (
                            float(tf.location.x),
                            float(tf.location.y),
                            float(tf.location.z),
                            float(tf.rotation.yaw),
                        )
                    )
                if sampled:
                    self.log(
                        f"Interpolated route built with {len(route)} points; "
                        f"downsampled to {len(sampled)} for ROS publish"
                    )
                    return sampled
            except Exception as exc:
                self.log(f"Route interpolation failed; using raw waypoints: {exc}")

        # Fallback: raw waypoints as transforms
        sampled: List[Tuple[float, float, float, float]] = []
        for x, y, z, yaw in raw_waypoints:
            try:
                sampled.append((float(x), float(y), float(z), float(yaw)))
            except Exception:
                continue
        return sampled if sampled else raw_waypoints

    def _criteria_route_from_waypoints(self) -> Optional[List[Tuple[carla.Transform, RoadOption]]]:
        """Convert the route waypoints into a route for criteria."""
        if self._full_route_for_criteria and len(self._full_route_for_criteria) >= 2:
            return self._full_route_for_criteria

        if not self._route_json_waypoints or len(self._route_json_waypoints) < 2:
            return None

        route: List[Tuple[carla.Transform, RoadOption]] = []
        for idx, (x, y, z, yaw) in enumerate(self._route_json_waypoints):
            try:
                tf = carla.Transform(
                    carla.Location(x=float(x), y=float(y), z=float(z)),
                    carla.Rotation(yaw=_normalize_yaw(float(yaw))),
                )
                route.append((tf, RoadOption.LANEFOLLOW))
            except Exception as exc:
                self.log(f"Criteria route build failed at {idx}: {exc}")

        return route if len(route) >= 2 else None

    def _start_internal_ego_controller(self, ego_actor: Optional[carla.Actor], vehicle_data: Optional[VehicleData]) -> Optional[VehicleController]:
        """Launch a VehicleController for the ego in own-tick mode."""
        if not ego_actor or not vehicle_data or not self.world or not self._scenario:
            return None
        try:
            try:
                agent = BasicAgent(
                    ego_actor,
                    target_speed=vehicle_data.initial_speed,
                    map_inst=self.world.get_map(),
                    grp_inst=_NoopGlobalRoutePlanner(),
                )
            except Exception:
                try:
                    agent = BasicAgent(
                        ego_actor,
                        target_speed=vehicle_data.initial_speed,
                        grp_inst=_NoopGlobalRoutePlanner(),
                    )
                except Exception:
                    agent = BasicAgent(ego_actor, target_speed=vehicle_data.initial_speed)
            agent.ignore_traffic_lights(vehicle_data.ignore_traffic_lights)
            agent.ignore_stop_signs(vehicle_data.ignore_stop_signs)
            agent.ignore_vehicles(vehicle_data.ignore_vehicles)
            agent.follow_speed_limits(False)
            try:
                self._scenario._setup_vehicle_route(agent, ego_actor, vehicle_data)
            except Exception:
                pass
            controller = VehicleController(
                agent,
                ego_actor,
                vehicle_data.destination,
                self._scenario,
                index=-1,
                route_points=vehicle_data.route_points,
                initial_idle_time=vehicle_data.initial_idle_time,
                destination_speed=vehicle_data.destination_speed,
                cruise_speed=vehicle_data.initial_speed,
                vehicle_trigger=None,
            )
            controller.start()
            return controller
        except Exception as exc:
            self.log(f"Failed to start internal ego controller: {exc}")
            return None

    def _check_ego_arrival(self, scenario_ref: "vse_play", return_distance: bool = False) -> Tuple[bool, float]:
        """Check if the external ego has reached the destination."""
        ego_list = scenario_ref.ego_vehicles if scenario_ref else []
        if not ego_list:
            return (False, float("inf"))
        ego = ego_list[0]
        if not ego or not ego.is_alive:
            return (False, float("inf"))
        dest = getattr(scenario_ref, "_ego_destination", None)
        if not dest:
            return (False, float("inf"))
        try:
            distance = float(ego.get_location().distance(dest))
        except Exception:
            distance = float("inf")
        arrival_radius = 3.25 if self.wait_for_ego else 2.5
        arrived = distance <= arrival_radius
        return (arrived, distance)

    def _actors_completed_without_ego(self, scenario_ref: "vse_play") -> bool:
        """Return True when all non-ego actors and required triggers have finished."""
        if not scenario_ref:
            return False

        vehicles_done = (
            not scenario_ref._vehicle_controllers or
            all((c is None) or c.is_finished() for c in scenario_ref._vehicle_controllers)
        )
        pedestrians_done = (
            not scenario_ref.pedestrians_data or
            all(scenario_ref.completion_status)
        )
        triggers_done = scenario_ref._are_required_triggers_satisfied()

        # Traffic light triggers are considered done if none exist or all sequences completed
        traffic_done = True
        if scenario_ref._traffic_light_triggers:
            traffic_done = all(
                (not t.sequence) or t.sequence_completed or (not t.activated)
                for t in scenario_ref._traffic_light_triggers
            )

        return vehicles_done and pedestrians_done and triggers_done and traffic_done

    def _apply_sync_settings(self):
        """Configure world/TrafficManager for own-tick mode; remember settings for restore."""
        if self.tick_mode != "own":
            return
        world = self.world
        if not world:
            return
        try:
            self._diag("sync: get_settings")
            current = world.get_settings()

            def clone_settings(src: carla.WorldSettings) -> carla.WorldSettings:
                dst = carla.WorldSettings()
                dst.no_rendering_mode = getattr(src, "no_rendering_mode", False)
                dst.synchronous_mode = getattr(src, "synchronous_mode", False)
                dst.fixed_delta_seconds = getattr(src, "fixed_delta_seconds", 0.0)
                dst.max_substep_delta_time = getattr(src, "max_substep_delta_time", 0.01)
                dst.max_substeps = getattr(src, "max_substeps", 10)
                if hasattr(dst, "deterministic_ragdolls"):
                    dst.deterministic_ragdolls = getattr(src, "deterministic_ragdolls", False)
                if hasattr(dst, "substepping"):
                    dst.substepping = getattr(src, "substepping", False)
                if hasattr(dst, "max_culling_distance"):
                    dst.max_culling_distance = getattr(src, "max_culling_distance", 0.0)
                if hasattr(dst, "tile_stream_distance"):
                    dst.tile_stream_distance = getattr(src, "tile_stream_distance", 3000.0)
                if hasattr(dst, "actor_active_distance"):
                    dst.actor_active_distance = getattr(src, "actor_active_distance", 2000.0)
                if hasattr(dst, "spectator_as_ego"):
                    dst.spectator_as_ego = getattr(src, "spectator_as_ego", True)
                return dst

            self._restore_settings = clone_settings(current)

            already_sync = False
            try:
                already_sync = bool(getattr(current, "synchronous_mode", False))
            except Exception:
                already_sync = False
            try:
                current_fixed = float(getattr(current, "fixed_delta_seconds", 0.0) or 0.0)
            except Exception:
                current_fixed = 0.0

            if already_sync and abs(current_fixed - float(self.fixed_delta)) < 1e-6:
                try:
                    CarlaDataProvider._sync_flag = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                self.log(
                    f"World already in sync mode with fixed_delta={current_fixed:.4f} (no change needed)"
                )
                return

            new_settings = clone_settings(current)
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = self.fixed_delta
            self._diag("sync: apply_settings")
            try:
                world.apply_settings(new_settings, timeout=5.0)  # type: ignore[call-arg]
            except TypeError:
                try:
                    world.apply_settings(new_settings, 5.0)  # type: ignore[misc]
                except TypeError:
                    world.apply_settings(new_settings)
            self._diag("sync: apply_settings done")
            try:
                CarlaDataProvider._sync_flag = True  # type: ignore[attr-defined]
            except Exception:
                pass

            # TrafficManager sync setup is intentionally skipped here:
            # VSE controls actors via BasicAgent/VehicleController, and TM RPC calls
            # have been observed to block in some environments (causing Play/Stop hangs).
            self._restore_tm_sync = None
            self.log(f"World switched to sync mode with fixed_delta={self.fixed_delta:.4f}")
        except Exception as exc:
            self.log(f"Failed to apply sync settings: {exc}")

    def _restore_sync_settings(self):
        world = self.world
        if not world:
            return
        try:
            if self._restore_settings:
                self._diag("sync: restore apply_settings")
                try:
                    world.apply_settings(self._restore_settings, timeout=5.0)  # type: ignore[call-arg]
                except TypeError:
                    try:
                        world.apply_settings(self._restore_settings, 5.0)  # type: ignore[misc]
                    except TypeError:
                        world.apply_settings(self._restore_settings)
                self._diag("sync: restore apply_settings done")
                try:
                    CarlaDataProvider._sync_flag = bool(self._restore_settings.synchronous_mode)  # type: ignore[attr-defined]
                except Exception:
                    pass
            if self._restore_tm_sync is not None:
                try:
                    tm = self.client.get_trafficmanager(8000)
                    tm.set_synchronous_mode(self._restore_tm_sync)
                except Exception:
                    pass
        except Exception:
            pass

    def _prepare_ego_actor(self) -> Optional[carla.Actor]:
        data = {}
        try:
            with open(self.json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.log(f"Failed to read scenario JSON: {exc}")
            return None

        ego_entry = data.get("ego_vehicle", {})
        ego_roles = _base_ego_roles()
        if self.ego_role_name:
            ego_roles.add(self.ego_role_name.lower())

        if not ego_entry:
            self._has_ego_vehicle = False
            return None
        self._has_ego_vehicle = True

        if self.wait_for_ego:
            # Prefer direct actor id if provided by VSE
            if self.external_ego_actor_id:
                try:
                    actor = self.world.get_actor(self.external_ego_actor_id) if self.world else None
                except Exception:
                    actor = None
                if actor and actor.is_alive:
                    spawn_tf = self._ego_transform_from_json(ego_entry, actor.get_transform())
                    if not self._is_spawn_clear(spawn_tf.location, actor.id):
                        self.log("Ego spawn location blocked; aborting.")
                        return None
                    try:
                        actor.set_transform(spawn_tf)
                        actor.set_target_velocity(carla.Vector3D())
                        actor.set_target_angular_velocity(carla.Vector3D())
                        try:
                            CarlaDataProvider.register_actor(actor, spawn_tf)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    return actor

            actor = None
            # Refresh world reference in case VSE swapped it
            try:
                if self.client:
                    self.world = self.client.get_world()
            except Exception:
                pass
            time.sleep(0.1)
            for attempt in range(10):
                try:
                    # Let the world progress so external ego can be registered
                    if self.tick_mode == "ros":
                        self.world.wait_for_tick(0.2)
                    else:
                        self.world.tick()
                except Exception:
                    pass
                actor = self._find_actor_by_role(ego_roles)
                if actor:
                    break
                time.sleep(0.1)
            if actor is None:
                self.log("External ego requested but not found in world actor list.")
                self._debug_list_roles()
                return None
            spawn_tf = self._ego_transform_from_json(ego_entry, actor.get_transform())
            if not self._is_spawn_clear(spawn_tf.location, actor.id):
                self.log("Ego spawn location blocked; aborting.")
                return None
            try:
                actor.set_transform(spawn_tf)
                actor.set_target_velocity(carla.Vector3D())
                actor.set_target_angular_velocity(carla.Vector3D())
                try:
                    CarlaDataProvider.register_actor(actor, spawn_tf)
                except Exception:
                    pass
            except Exception:
                pass
            return actor

        # Internal ego flow: reuse if already present, otherwise spawn a fresh manual-control ego
        existing = self._find_actor_by_role(ego_roles)
        if existing:
            spawn_tf = self._ego_transform_from_json(ego_entry, existing.get_transform())
            if not self._is_spawn_clear(spawn_tf.location, existing.id):
                self.log("Ego spawn location blocked; aborting.")
                return None
            try:
                existing.set_transform(spawn_tf)
                existing.set_target_velocity(carla.Vector3D())
                existing.set_target_angular_velocity(carla.Vector3D())
            except Exception:
                pass
            try:
                CarlaDataProvider.register_actor(existing, spawn_tf)
            except Exception:
                pass
            return existing

        blueprint_id = ego_entry.get("type") or "vehicle.tesla.model3"
        spawn_tf = self._ego_transform_from_json(ego_entry, carla.Transform())
        blueprint_lib = self.world.get_blueprint_library() if self.world else None
        blueprint = None
        if blueprint_lib:
            try:
                blueprint = blueprint_lib.find(blueprint_id)
            except Exception:
                blueprint = None
            if not blueprint:
                try:
                    blueprint = blueprint_lib.find("vehicle.tesla.model3")
                except Exception:
                    blueprint = None
            if blueprint and blueprint.has_attribute("role_name"):
                # LARGE MAP FIX: Use "autopilot" role to prevent VSE cleanup queries from crashing
                if self._large_map_active:
                    blueprint.set_attribute("role_name", "autopilot")
                else:
                    blueprint.set_attribute("role_name", self.ego_role_name or "ego_vehicle")
        if not blueprint or not self.world:
            self.log("Unable to resolve ego blueprint or world; ego will not spawn.")
            return None

        spawn_tf.location.z += 0.5
        if not self._is_spawn_clear(spawn_tf.location, None):
            self.log("Ego spawn location blocked; aborting.")
            return None

        actor = self.world.try_spawn_actor(blueprint, spawn_tf)
        if not actor:
            self.log("Failed to spawn ego vehicle at requested transform.")
            return None

        try:
            CarlaDataProvider.register_actor(actor, spawn_tf)
        except Exception:
            pass
        try:
            self._spawned_internal_ego_id = int(actor.id)
        except Exception:
            self._spawned_internal_ego_id = None
        return actor

    def _find_actor_by_role(self, roles: set) -> Optional[carla.Actor]:
        worlds = []
        if self.world:
            worlds.append(self.world)
        if self.client:
            try:
                cw = self.client.get_world()
                if cw not in worlds:
                    worlds.append(cw)
            except Exception:
                pass
        for w in worlds:
            try:
                actors = list(w.get_actors())
            except RuntimeError:
                continue
            for actor in actors:
                try:
                    role_name = actor.attributes.get("role_name", "").lower()
                    if role_name in roles and actor.is_alive:
                        return actor
                except Exception:
                    continue
        return None

    def _debug_list_roles(self):
        """Log current vehicle actors and their roles (debug helper)."""
        if not self.world:
            return
        try:
            actors = self.world.get_actors()
            roles = []
            for a in actors:
                if not a or not a.is_alive:
                    continue
                try:
                    loc = a.get_location()
                except Exception:
                    loc = None
                roles.append(
                    (a.id, a.attributes.get("role_name", ""), a.type_id, loc)
                )
            self.log(f"Vehicle roles observed: {roles}")
        except Exception:
            pass

    def _ego_transform_from_json(self, entry: dict, fallback: carla.Transform) -> carla.Transform:
        loc = entry.get("location", {}) if isinstance(entry, dict) else {}
        rot = entry.get("rotation", {}) if isinstance(entry, dict) else {}
        return carla.Transform(
            carla.Location(
                x=float(loc.get("x", fallback.location.x)),
                y=float(loc.get("y", fallback.location.y)),
                z=float(loc.get("z", fallback.location.z)),
            ),
            carla.Rotation(
                pitch=float(rot.get("pitch", fallback.rotation.pitch)),
                yaw=float(rot.get("yaw", fallback.rotation.yaw)),
                roll=float(rot.get("roll", fallback.rotation.roll)),
            ),
        )

    def _is_spawn_clear(self, location: carla.Location, ignore_id: Optional[int]) -> bool:
        if not self.world:
            return True
        try:
            actors = self.world.get_actors().filter("vehicle.*")
        except RuntimeError:
            return True
        for actor in actors:
            if not actor or not actor.is_alive:
                continue
            if ignore_id is not None and actor.id == ignore_id:
                continue
            try:
                if actor.get_location().distance(location) < 2.0:
                    return False
            except Exception:
                continue
        return True

    def _tick_world(self) -> Optional[carla.WorldSnapshot]:
        if not self.world:
            return None
        try:
            if self.tick_mode == "own":
                tick = getattr(self.world, "tick", None)
                if callable(tick):
                    try:
                        tick(timeout=0.5)  # type: ignore[call-arg]
                    except TypeError:
                        try:
                            tick(0.5)  # type: ignore[misc]
                        except TypeError:
                            tick()
                else:
                    self.world.tick()
                return self.world.get_snapshot()
            # Use a short timeout so stop requests are responsive even if an
            # external tick source stalls.
            return self.world.wait_for_tick(0.5)
        except Exception:
            return None

    def _cleanup(self, reason: str):
        with self._cleanup_lock:
            if self._cleanup_invoked:
                return
            self._cleanup_invoked = True
        self._stop_requested = True
        # Restore any traffic lights forced green for external ego
        try:
            self._restore_forced_green_lights()
        except Exception:
            pass
        if self._scenario:
            try:
                self._scenario._terminate_scenario(reason or "cleanup")  # type: ignore[attr-defined]
            except Exception:
                pass
            self._scenario = None
        try:
            if self._ego_controller:
                try:
                    self._ego_controller.stop()
                except Exception:
                    pass
                self._ego_controller = None
        except Exception:
            pass
        try:
            self._stop_ros_agent_process()
        except Exception:
            pass
        try:
            ego_id = getattr(self, "_spawned_internal_ego_id", None)
            if ego_id is not None:
                with _temporary_client_timeout(self.client, timeout_s=60.0):
                    _destroy_actor_ids(
                        self.client,
                        [int(ego_id)],
                        do_tick=False,  # Don't tick - can timeout on large maps
                        log_fn=self.log,
                    )
                self._spawned_internal_ego_id = None
        except Exception:
            pass
        # Keep restore operations responsive; large timeouts here can make Stop feel stuck.
        with _temporary_client_timeout(self.client, timeout_s=10.0):
            try:
                if self._restore_weather is not None and self.world:
                    self.world.set_weather(self._restore_weather)
            except Exception:
                pass
            self._restore_weather = None
            self._restore_sync_settings()
        self.log(f"Scenario ended ({reason or 'finished'})")
        if self.on_finish:
            try:
                self.on_finish(reason or "finished")
            except Exception:
                pass

    def _force_all_traffic_lights_green(self) -> int:
        """Force all traffic lights in the world to green (for external ego ignore_traffic_lights).

        Returns the number of lights forced green.
        """
        if not self.world:
            return 0
        try:
            lights = self.world.get_actors().filter("traffic.traffic_light*")
        except Exception:
            return 0
        count = 0
        for light in lights:
            try:
                if not light:
                    continue
                light.freeze(True)
                light.set_state(carla.TrafficLightState.Green)
                self._forced_green_lights.append(light)
                count += 1
            except Exception as e:
                self.log(f"Failed to force light {getattr(light, 'id', '?')} green: {e}")
        if count > 0:
            self.log(f"Forced {count} traffic lights to GREEN (external ego ignore_traffic_lights)")
        else:
            self.log("No traffic lights found to force green")
        return count

    def _restore_forced_green_lights(self) -> None:
        """Unfreeze traffic lights that were forced green for external ego."""
        if not self._forced_green_lights:
            return
        self.log(f"Restoring {len(self._forced_green_lights)} traffic lights to normal operation")
        for light in self._forced_green_lights:
            try:
                if light:
                    light.freeze(False)
            except Exception:
                pass
        self._forced_green_lights.clear()


class _EgoCriteriaProxy:
    """Minimal adapter to expose only ego criteria to ResultOutputProvider."""

    def __init__(self, scenario: vse_play, criteria: List[Criterion]):
        self._scenario = scenario
        self._criteria = criteria

    @property
    def timeout(self) -> float:
        try:
            return float(getattr(self._scenario, "timeout", 0))
        except Exception:
            return 0.0

    def get_criteria(self) -> List[Criterion]:
        return self._criteria


if __name__ == "__main__":
    sys.exit(main())
