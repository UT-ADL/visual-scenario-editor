"""The vse_play scenario class (moved verbatim from vse_play.py).

The class keeps its historical lowercase name: result tables print
"Results of Scenario: vse_play" and the golden corpus depends on it.

Also owns the process-global current-scenario reference used by the
atexit/signal cleanup handlers; install_handlers_from_env() is the ONLY
installer (the vse_play.py shim calls it on plain import, preserving the
old module's import-time gate semantics; VSE_PLAY_INSTALL_HANDLERS=0
disables, as the editor does around its lazy MiniRunner import).
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import signal
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import carla
import py_trees
from py_trees import common as py_trees_common
from py_trees.behaviour import Behaviour
from py_trees.composites import Parallel, Sequence

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    AtomicBehavior,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest,
    Criterion,
    InRouteTest,
    KeepLaneTest,
    OffRoadTest,
    OnSidewalkTest,
    OutsideRouteLanesTest,
    RouteCompletionTest,
    RunningStopTest,
    WrongLaneTest,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.weather_sim import RouteWeatherBehavior
from srunner.scenariomanager.lights_sim import RouteLightsBehavior
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.route_manipulation import interpolate_trajectory

from vse_common.actor_cache import clear_bounding_box_cache
from vse_common.env import env_float
from vse_common.geometry import get_ground_height, is_large_map as _is_large_map
from vse_common.traffic_lights import (
    TRAFFIC_LIGHT_FINGERPRINT_SCALE as _TRAFFIC_LIGHT_FINGERPRINT_SCALE,
    build_traffic_light_fingerprint_index as _build_traffic_light_fingerprint_index,
    compute_traffic_light_fingerprint as _compute_traffic_light_fingerprint,
    match_traffic_lights_by_fingerprint as _match_traffic_lights_by_fingerprint,
    normalize_traffic_light_fingerprint as _normalize_traffic_light_fingerprint,
)
from vse_playback.behaviors import (
    AllPedestriansArrivedCriterion,
    DelayBeforeSpawn,
    DestinationCriterion,
    EnsureWalkerAt,
    ExecuteRouteBehaviour,
    GroundedIdle,
    GeometricCollisionTest,
    PedestrianTriggerMonitor,
    SingleRunSequence,
    SpawnWalkerBehaviour,
    TrackingParallel,
    TrafficLightTriggerMonitor,
    VehicleLightsBehavior,
    VehicleTriggerMonitor,
    WalkToTarget,
)
from vse_playback.models import (
    PedestrianTrigger,
    RoutePoint,
    RouteSegment,
    TrafficLightTrigger,
    VehicleData,
    VehicleTrigger,
)
from vse_playback.npc_agent import _NoopGlobalRoutePlanner, _NpcBasicAgent, _apply_npc_detection_tuning
from vse_playback.route import (
    _compute_heading,
    _distance,
    _normalize_yaw,
    _refine_vehicle_route,
    _resolve_walk_speed_mps,
)
from vse_playback.srunner_compat import GuardedRunningRedLightTest
from vse_playback.vehicle_control import VehicleController, _AsyncSafeVehicleControl, _SafeVehicleControl
from vse_playback.world_utils import (
    _WALKER_GROUND_IGNORE_LABELS,
    _destroy_actor_ids,
    _init_carla_data_provider,
    _temporary_client_timeout,
    _walker_ground_height,
)

logger = logging.getLogger(__name__)

_spawned_vehicles: List[carla.Actor] = []
_current_scenario: Optional["vse_play"] = None
_current_scenario_lock = threading.Lock()  # Protect _current_scenario access

_WALKER_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 165, 0),
    (255, 105, 180),
]


class vse_play(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=18000, vehicle_control_mode=None,
                 ego_physics_off=False):
        global _current_scenario

        # True when the ego runs with CARLA physics disabled (external/VIL ego). Set before
        # super().__init__() because that triggers _create_test_criteria, which uses it to choose
        # the geometric collision criterion over the (then-dead) sensor-based CollisionTest.
        self._ego_physics_off = bool(ego_physics_off)
        self._vehicle_control_mode_override = vehicle_control_mode
        self.timeout = timeout
        # Keep reference to the incoming config for downstream helpers that expect it.
        self.config = config
        self._world = world
        self._map = world.get_map()
        self._large_map_active = _is_large_map(self._map)
        self._debug = debug_mode
        # srunner may have load_world'ed for this scenario (actor ids restart per episode);
        # drop any bounding boxes cached by a previous run or by the in-process editor.
        clear_bounding_box_cache()

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
            print("[TRIGGER] No trigger found in scenario data")

        # Enable trigger mode when ego vehicle is present, so NPCs without
        # personal triggers must wait for a global trigger (and never activate
        # if no global trigger exists in the scene).
        ego_entry = data.get("ego_vehicle")
        if not ego_entry:
            for entry in data.get("vehicles", []):
                if str(entry.get("role", "")).lower() == "ego_vehicle":
                    ego_entry = entry
                    break
        if ego_entry and not self._trigger_mode:
            self._trigger_mode = True
            print("[TRIGGER] Ego vehicle present - NPCs require triggers to activate")

        if not self._trigger_mode:
            print("[TRIGGER] No ego vehicle - scenario will start immediately")

        if self._vehicle_control_mode_override is not None:
            self.vehicle_control_mode = self._vehicle_control_mode_override
        else:
            self.vehicle_control_mode = data.get('vehicle_control_mode', 'basic_agent')
        if self.vehicle_control_mode != 'basic_agent':
            print(f"[NPC Control] Mode: {self.vehicle_control_mode}")

    def _load_config_from_json(self) -> None:
        for idx, actor in enumerate(self._raw_pedestrian_entries):
            pedestrian_type = actor.get("type", "walker.pedestrian.0001")
            loc = actor["location"]
            rot = actor["rotation"]

            spawn_location = carla.Location(loc["x"], loc["y"], loc["z"])
            spawn_rotation = carla.Rotation(rot["pitch"], rot["yaw"], rot["roll"])

            default_speed_kmh = actor.get("speed_km_h", 5.0)
            initial_idle_time = actor.get("idle_time_s", 0.0)

            # A pedestrian with no waypoints simply stands still at its spawn pose for the
            # whole scenario (mirrors a vehicle with no waypoints). The empty route flows
            # through _prepare_routes / _build_pedestrian_route, which both handle the
            # empty-segments case, and the pedestrian is immediately marked "arrived".
            waypoints = actor.get("waypoints", [])

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

                # Departure-speed semantics: the speed/deviation assigned to a
                # point governs the leg that LEAVES it. The first leg leaves the
                # spawn at the pedestrian's initial speed (no deviation); each
                # later leg carries the speed of the waypoint just departed. The
                # destination waypoint's speed is therefore never used to travel.
                departure_speed_kmh = ped_data.get("default_speed_kmh", 0.0)
                departure_deviation_kmh = 0

                for index, waypoint in enumerate(route, start=1):
                    # A leg's speed is the *departure* speed of the point it leaves.
                    # A planned departure speed of 0 means the pedestrian cannot
                    # traverse this leg: it stands still at the current point (spawn,
                    # or the waypoint it just reached) for the rest of the scenario
                    # rather than teleporting to the leg's target. Truncate the route
                    # here — the last reachable point becomes the effective
                    # destination (forced to stop, so the walker halts instead of
                    # walking straight through and drifting), and the shortened (or
                    # empty) segment list flows through the existing handling.
                    try:
                        planned_departure_kmh = float(departure_speed_kmh)
                    except (TypeError, ValueError):
                        planned_departure_kmh = 0.0
                    # Match how _resolve_walk_speed_mps rounds: any speed that rounds
                    # to 0 km/h (i.e. < 0.5) cannot be walked. Deviation is ignored on
                    # purpose — authoring 0 means "stop", not "maybe move a little".
                    if int(round(planned_departure_kmh)) <= 0:
                        if segments:
                            segments[-1].is_destination = True
                        break

                    # Use exact waypoint location from JSON (already validated by editor)
                    target = carla.Location(
                        waypoint["location"].x,
                        waypoint["location"].y,
                        waypoint["location"].z
                    )
                    speed_mps = _resolve_walk_speed_mps(departure_speed_kmh, departure_deviation_kmh)
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
                        is_destination=waypoint["is_destination"],
                    )
                    segments.append(segment)
                    # This waypoint's own speed/deviation becomes the departure
                    # speed for the next leg (carried until the following waypoint).
                    departure_speed_kmh = waypoint.get("speed_km_h", 0.0)
                    departure_deviation_kmh = waypoint.get("speed_deviation_km_h", 0)
                    current_location = target
                    current_heading = heading

                total_time = initial_idle_time
                for segment in segments:
                    travel_time = segment.distance / segment.speed if segment.speed > 0 else 0.0
                    total_time += travel_time + segment.idle_after

                self.all_segments.append(segments)
                self.expected_durations.append(total_time)
                final_destination = segments[-1].target if segments else spawn_location
                self.final_destinations.append(final_destination)

                # Spawn the walker already facing its first waypoint, before any trigger fires.
                # segments[0].heading is the bearing to the first waypoint (same atan2 the editor
                # uses) and matches what ActorTransformSetter applies at trigger time, so there is
                # no visible snap. Deriving from segments (not the stored yaw) also corrects older
                # scenarios saved with rotation 0,0,0. Routeless pedestrians keep their spawn yaw.
                if segments:
                    spawn_rotation.yaw = segments[0].heading
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

            # Insert spawn as first route point for proper speed profiling
            initial_speed = entry.get("speed_km_h", 30.0)
            route_points.append(
                RoutePoint(
                    transform=carla.Transform(spawn_location, spawn_rotation),
                    speed_kmh=initial_speed,
                    idle_time_s=0.0,
                    is_destination=False,
                    speed_deviation_kmh=0,
                )
            )

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
            # Interpolate at a coarse hop so InRouteTest's fixed 5-point look-ahead window spans more
            # distance (5 points x hop): at 3 m that's ~15 m, enough that the criterion's route index
            # can't stall when the externally-driven ego advances several metres between ticks
            # (teleports / tick gaps), which otherwise inflated the measured off-route distance and
            # failed InRouteTest spuriously. Tunable via $VSE_CRITERIA_ROUTE_HOP_M (default 3.0).
            hop_m = env_float("VSE_CRITERIA_ROUTE_HOP_M", 3.0)
            hop_m = min(25.0, max(0.5, hop_m))
            try:
                _, interpolated = interpolate_trajectory(keypoints, hop_resolution=hop_m)
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

            walkable = segment.distance > 0.05 and segment.speed > 0
            # Waypoints without idle are walked straight through: no align/stop,
            # the next leg's WalkToTarget turns smoothly into the new direction.
            pass_through = walkable and segment.idle_after <= 0 and not segment.is_destination

            if walkable:
                # Larger tolerance where exact arrival doesn't matter (pass-through
                # corners and the final destination) to prevent overshooting. The
                # pass-through tolerance must also exceed the walker's minimum
                # turning radius (speed / turn rate) or a sharp corner could orbit
                # the waypoint instead of arriving.
                if pass_through:
                    min_turn_radius = segment.speed / math.radians(WalkToTarget.TURN_RATE_DEG_S)
                    tolerance = max(0.5, 1.2 * min_turn_radius)
                else:
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
                    pass_through=pass_through,
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

            if not pass_through:
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

            # NPCs that ignore vehicles keep the stock BasicAgent (deterministic, no
            # obstacle detection). NPCs that DO consider vehicles use _NpcBasicAgent,
            # which adds graduated braking; their detection range/brake strength is then
            # widened by _apply_npc_detection_tuning() below.
            agent_cls = BasicAgent if data.ignore_vehicles else _NpcBasicAgent
            try:
                agent = agent_cls(
                    vehicle,
                    target_speed=data.initial_speed,
                    map_inst=self._map,
                    grp_inst=_NoopGlobalRoutePlanner(),
                )
            except Exception:
                # Fall back to constructing the agent without an explicit map instance,
                # but still avoid GRP precomputation (VSE uses set_global_plan()).
                try:
                    agent = agent_cls(
                        vehicle,
                        target_speed=data.initial_speed,
                        grp_inst=_NoopGlobalRoutePlanner(),
                    )
                except Exception:
                    agent = agent_cls(vehicle, target_speed=data.initial_speed)
            agent.ignore_traffic_lights(data.ignore_traffic_lights)
            agent.ignore_stop_signs(data.ignore_stop_signs)
            agent.ignore_vehicles(data.ignore_vehicles)
            agent.follow_speed_limits(False)
            if not data.ignore_vehicles:
                _apply_npc_detection_tuning(agent)

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
                control_mode=self.vehicle_control_mode,
            )
            controller._ignore_traffic_lights = data.ignore_traffic_lights
            self._vehicle_controllers.append(controller)
            controller.start()


    def _setup_vehicle_route(self, agent: BasicAgent, vehicle: carla.Actor, data: VehicleData) -> None:
        class WP:
            """Lightweight waypoint wrapper for BasicAgent/BehaviorAgent routes."""
            class _LaneMarking:
                lane_change = carla.LaneChange.NONE
            def __init__(self, transform: carla.Transform):
                self.transform = transform
                self.road_id = 0
                self.lane_id = 0
                self.is_junction = False
                self.lane_type = carla.LaneType.Driving
                self.left_lane_marking = WP._LaneMarking()
                self.right_lane_marking = WP._LaneMarking()
            def get_left_lane(self):
                return None
            def get_right_lane(self):
                return None

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

                            # Get ground height at current XY position
                            current_loc = walker.get_location()
                            ground_height = _walker_ground_height(
                                self.scenario.world,
                                walker,
                                current_loc,
                                cached_map=self.scenario.world.get_map(),
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
            # An external/VIL ego runs with CARLA physics disabled (see _run_preflight), so the
            # sensor-based CollisionTest can never fire. Swap in the geometric (bounding-box)
            # collision test for that case; keep the sensor-based one when physics is on.
            ego_physics_off = bool(getattr(self, "_ego_physics_off", False))
            collision_criterion = (
                GeometricCollisionTest(self, ego_actor) if ego_physics_off
                else CollisionTest(ego_actor)
            )
            ego_criteria: List[Criterion] = [
                collision_criterion,
                KeepLaneTest(ego_actor),
                OffRoadTest(ego_actor),
                OnSidewalkTest(ego_actor),
                GuardedRunningRedLightTest(ego_actor),
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

def install_handlers_from_env(default: str = "1") -> None:
    """Install atexit/SIGTERM/SIGINT cleanup handlers unless disabled.

    Reproduces the historical import-time gate of the monolithic
    vse_play.py: VSE_PLAY_INSTALL_HANDLERS overrides *default*; values
    0/false/no/off disable installation.
    """
    flag = os.environ.get("VSE_PLAY_INSTALL_HANDLERS", default)
    if str(flag).strip().lower() in ("0", "false", "no", "off"):
        return
    atexit.register(_global_cleanup)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass
