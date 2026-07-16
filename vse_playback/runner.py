"""MiniRunner — the in-process scenario runner (moved verbatim from
vse_play.py): world/tick ownership (own vs ros), external-ego adoption,
scenario-tree execution with the detached, leaf-ticked criteria tree,
result-table emission and idempotent cleanup. Plus the scenario-data ego
role helpers shared with the CLI.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import math
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import carla
import py_trees
from py_trees import common as py_trees_common
from py_trees.trees import BehaviourTree

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion, InRouteTest
from srunner.scenariomanager.timer import GameTime
from srunner.tools.route_manipulation import downsample_route, interpolate_trajectory

from vse_common.env import env_float
from vse_common.geometry import get_ground_height, is_large_map as _is_large_map
from types import SimpleNamespace

from srunner.scenariomanager.scenarioatomics.atomic_criteria import RouteCompletionTest

from vse_playback.models import RoutePoint, VehicleData
from vse_playback.npc_agent import _NoopGlobalRoutePlanner, _NpcBasicAgent, _apply_npc_detection_tuning
from vse_playback.ros_publisher import _ros_plan_publisher_process
from vse_playback.route import _compute_heading, _distance, _normalize_yaw, _refine_vehicle_route
from vse_playback.scenario import vse_play
from vse_playback.vehicle_control import VehicleController
from vse_playback.world_utils import (
    _cdp_location_or_live,
    _cdp_purge_actor_id,
    _destroy_actor_ids,
    _init_carla_data_provider,
    _temporary_client_timeout,
)

logger = logging.getLogger(__name__)

EGO_DEFAULT_SPEED_KMH = 40.0


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
        agent_mode: str = "autopilot",
        agent_behavior: str = "normal",
        vehicle_control_mode: str = "basic_agent",
        disable_ego_collision: bool = False,
        disable_ego_physics: bool = False,
        ignore_actor_ids: Optional[Set[int]] = None,
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
        self.agent_mode = agent_mode if agent_mode in ("autopilot", "human", "custom") else "autopilot"
        self.agent_behavior = agent_behavior if agent_behavior in ("cautious", "normal", "aggressive") else "normal"
        self.vehicle_control_mode = vehicle_control_mode if vehicle_control_mode in ("basic_agent", "velocity") else "basic_agent"
        self._disable_ego_collision = bool(disable_ego_collision)
        self._disable_ego_physics = bool(disable_ego_physics)
        self.ros_goal_interval = max(0.0, env_float("VSE_GOAL_INTERVAL", 0.25))
        self.ros_downsample_interval = max(1.0, env_float("VSE_DOWNSAMPLE_INTERVAL", 42.0))
        # Cache JSON waypoints once per runner instance
        self._route_json_waypoints: List[Tuple[float, float, float, float]] = []
        self._route_json_destination: Optional[carla.Location] = None
        self._full_route_for_criteria: Optional[List[Tuple[carla.Transform, RoadOption]]] = None
        self._ros_agent_proc: Optional[multiprocessing.Process] = None
        # Graceful cancel handshake with the agent subprocess: _stop_ros_agent_process sets
        # cancel_now and waits for cancel_done so the route is cancelled while the agent is
        # still healthy (clean /planning/cancel_route reply) instead of during SIGTERM teardown.
        self._ros_agent_cancel_now = None
        self._ros_agent_cancel_done = None
        self._prepped_ego_actor: Optional[carla.Actor] = None
        # Preview actors the editor destroyed right before Play: DestroyActor only
        # materializes on the next tick, so in a sync world with no tick source between
        # runs they still appear alive (e.g. the placeholder ego standing on the spawn
        # point). _is_spawn_clear must skip them or the run aborts as "blocked".
        self._spawn_clear_ignore_ids: Set[int] = set(ignore_actor_ids or ())
        self._spawned_internal_ego_id: Optional[int] = None
        self._cleanup_lock = threading.Lock()
        self._cleanup_invoked = False

        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._server_lost = False  # set when the run aborts on a dead server (fix-08)
        self._running = False
        self._scenario: Optional[vse_play] = None
        self._restore_settings: Optional[carla.WorldSettings] = None
        self._restore_tm_sync: Optional[bool] = None
        self._restore_weather: Optional[carla.WeatherParameters] = None
        self._ego_controller: Optional[VehicleController] = None
        self._internal_ego_vehicle_data: Optional[VehicleData] = None
        self._ego_arrival_mark: Optional[float] = None
        self._last_world_frame: Optional[int] = None
        self._criteria_tree = None  # detached from scenario_tree; ticked separately in _run
        self._criteria_nodes: List = []  # individual criterion leaves ticked each frame
        self._criteria_armed = False  # criteria start evaluating once ego localized at route start
        self._route_start: Optional[carla.Location] = None
        self._route_start_enter_t: Optional[float] = None  # when ego entered start radius (arm dwell)
        # External/VIL ego: seconds the ego must remain within the arrival radius before the run
        # ends ("stable arrival" dwell). Default 2.0 (settle so the table doesn't pop the instant
        # the ego clips the radius); set $VSE_EXTERNAL_EGO_ARRIVAL_DWELL_S to tune (0 = end
        # immediately on arrival, like the local agent).
        self._external_ego_arrival_dwell_s = max(
            0.0, env_float("VSE_EXTERNAL_EGO_ARRIVAL_DWELL_S", 2.0)
        )
        self._has_ego_vehicle: Optional[bool] = None
        # Why the last _prepare_ego_actor() call returned None (empty = no reason
        # logged). Lets callers skip the generic "external ego not found" line when a
        # more specific abort reason (e.g. spawn blocked) was already printed.
        self._ego_prepare_failure: str = ""
        self._forced_green_lights: List[carla.Actor] = []
        self._large_map_active = False
        self._skip_route_interpolation = False
        try:
            carla_map = world.get_map() if world else None
        except Exception:
            carla_map = None
        self._large_map_active = _is_large_map(carla_map)
        # External-ego routes are published to the ROS agent (carla_minimal_agent) and consumed by
        # autoware_mini's lanelet2 global planner, which re-routes on its own Lanelet2 map from the
        # goal/via points and never reads the OpenDRIVE-interpolated corridor. The GRP/OpenDRIVE
        # interpolation is therefore wasted work for that path, so skip it for any map size (not just
        # large maps). Set VSE_FORCE_ROUTE_INTERPOLATION=1 to restore GRP interpolation — needed only
        # if the route is consumed directly (e.g. autoware_mini global_planner:=carla).
        self._skip_route_interpolation = (
            os.environ.get("VSE_FORCE_ROUTE_INTERPOLATION") != "1"
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

        if self.tick_mode == "ros" and self.wait_for_ego:
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
                    # _prepare_ego_actor already logged the specific reason (e.g. spawn
                    # blocked); only fall back to the generic line when it stayed silent.
                    if not self._ego_prepare_failure:
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
            # _prepare_ego_actor already logged the specific reason (e.g. spawn blocked);
            # only fall back to the generic line when it stayed silent.
            if not self._ego_prepare_failure:
                self.log("External ego requested but not found; aborting.")
            self._preflight_failure_reason = (
                f"failed: {self._ego_prepare_failure or 'external ego requested but not found'}"
            )
            return None
        self._prepped_ego_actor = ego_actor

        # Optional "ghost ego": disable collision response on the ego only (physics stays on so
        # it remains drivable). Covers both internal and external/VIL ego since both converge here.
        if ego_actor is not None and self._disable_ego_collision:
            try:
                ego_actor.set_collisions(False)
                self.log("Ego collision DISABLED (ghost ego).")
            except Exception as exc:
                self.log(f"Failed to disable ego collision: {exc}")

        # Optional: disable CARLA physics ONLY on an external/VIL ego, whose motion is driven by
        # an outside stack via teleport. The internal/local ego is driven by physics (apply_control)
        # and held on the road by physics, so we must NOT disable it (it would freeze / fall through
        # the ground). For the local ego, "Ego Physics off" reduces to "collision off" (handled above).
        if ego_actor is not None and self._disable_ego_physics and self.wait_for_ego:
            try:
                ego_actor.set_simulate_physics(False)
                self.log("Ego physics DISABLED (external/VIL ego).")
            except Exception as exc:
                self.log(f"Failed to disable ego physics: {exc}")

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
            vehicle_control_mode=self.vehicle_control_mode,
            ego_physics_off=bool(self._disable_ego_physics and self.wait_for_ego),
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

        # Detach the criteria from the scenario tree and tick each one ourselves (in _run). Two
        # reasons:
        #  - scenario_tree is a SUCCESS_ON_ONE Parallel of [behavior, criteria_tree]; when the
        #    behavior subtree (pedestrians AND vehicles) finishes, py_trees stops the still-running
        #    criteria_tree (terminate -> INIT->FAILURE), which froze RouteCompletionTest at ~2.38%
        #    when the behavior finished before the ego arrived.
        #  - criteria_tree itself is a Parallel that stops all children if any one reaches FAILURE.
        # Ticking each criterion leaf individually isolates them: no actor finishing early and no
        # other criterion can terminate RouteCompletionTest, so it evaluates for the whole drive.
        self._criteria_tree = None
        self._criteria_nodes = []
        try:
            st = getattr(self._scenario, "scenario_tree", None)
            ct = getattr(self._scenario, "criteria_tree", None)
            if st is not None and ct is not None and ct in getattr(st, "children", []):
                st.remove_child(ct)
                self._criteria_tree = ct
                self._criteria_nodes = list(getattr(ct, "children", []))
                self._diag(f"run: detached criteria_tree ({len(self._criteria_nodes)} criteria) from scenario_tree")
        except Exception as exc:
            self.log(f"Could not detach criteria_tree (criteria may terminate early): {exc}")

        # Route start for the criteria arm-gate (see _ego_at_route_start). First point of the criteria
        # route, falling back to the JSON ego destination's counterpart (the loaded route start).
        self._route_start = None
        try:
            route = getattr(self._scenario, "_ego_route_for_criteria", None)
            if route:
                self._route_start = route[0][0].location
        except Exception:
            self._route_start = None

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
        self._last_world_frame = None
        self._criteria_armed = False
        self._route_start_enter_t = None
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
            # Abort the run if the server stops answering (crash/segfault): a
            # continuous streak of failed ticks (None) longer than this means
            # the server is gone. Without it a mid-run crash spins until
            # timeout_s (default 5 h). _tick_world returns None ONLY on an RPC
            # exception — tick-stream starvation falls back to get_snapshot()
            # (non-None) — so this never trips on a healthy-but-slow server.
            server_lost_after_s = env_float("VSE_PLAY_SERVER_LOST_S", 5.0)
            first_dead_tick_t = None
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
                    now = time.monotonic()
                    if first_dead_tick_t is None:
                        first_dead_tick_t = now
                    elif now - first_dead_tick_t > server_lost_after_s:
                        reason = "CARLA server lost"
                        self._server_lost = True
                        self.log("CARLA server unreachable — "
                                 f"aborting run after {now - first_dead_tick_t:.1f}s")
                        break
                    continue
                first_dead_tick_t = None

                # Only advance time/criteria once per real world frame. In ros tick-mode _tick_world
                # falls back to world.get_snapshot() when wait_for_tick starves (e.g. the in-GUI
                # MiniRunner thread losing the tick stream); that can return the same frame
                # repeatedly, and processing it again would double-feed GameTime and busy-spin.
                frame = getattr(snapshot, "frame", None)
                if frame is not None and frame == self._last_world_frame:
                    time.sleep(0.005)
                    continue
                self._last_world_frame = frame

                GameTime.on_carla_tick(snapshot.timestamp)
                CarlaDataProvider.on_carla_tick()

                if self._scenario and getattr(self._scenario, "scenario_tree", None):
                    try:
                        self._scenario.scenario_tree.tick_once()
                    except Exception:
                        pass
                # Tick each criterion individually (detached from scenario_tree) so neither the
                # behavior subtree finishing nor any sibling criterion can terminate the criteria
                # mid-run (which froze RouteCompletionTest at ~2.38%). See _run_build_scenario / docs.
                # Don't evaluate criteria until the externally-driven ego is localized at the route
                # start. At run start awmini's pose relay can briefly hold the ego at its previous
                # pose (e.g. the prior run's destination, far off-route) until /initialpose
                # re-localizes it. InRouteTest is terminal — a single off-route sample freezes it at
                # FAILURE forever (RouteCompletion survives because it's non-terminal) — so we gate
                # all criteria ticking until the ego is within arm radius of the route start. Once
                # armed it stays armed for the rest of the run.
                if self._criteria_nodes:
                    if not self._criteria_armed:
                        self._criteria_armed = self._ego_at_route_start()
                        if self._criteria_armed:
                            # Confirm the ego's data-provider speed at arm time: a stale ~0 here
                            # (while the ego is moving) is the CollisionTest "< EPSILON" skip cause.
                            try:
                                _ego = self._scenario.ego_vehicles[0] if (
                                    self._scenario and self._scenario.ego_vehicles) else None
                                _spd = CarlaDataProvider.get_velocity(_ego) if _ego else float("nan")
                                self._diag(f"criteria armed; ego CDP speed={_spd:.2f} m/s")
                            except Exception:
                                pass
                    if self._criteria_armed:
                        for _criterion in self._criteria_nodes:
                            try:
                                _criterion.tick_once()
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
                        if self.wait_for_ego and self._external_ego_arrival_dwell_s > 0.0:
                            # External ego: require a brief stable-arrival dwell (tunable; default 0)
                            if self._ego_arrival_mark is None:
                                self._ego_arrival_mark = now
                            if now - (self._ego_arrival_mark or now) >= self._external_ego_arrival_dwell_s:
                                reason = "ego arrived"
                                break
                        else:
                            # Local agent (or dwell disabled): end as soon as within radius
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

        # Drive the criteria to a terminal status before reading them. The runner loop breaks on
        # ego arrival without ticking the scenario tree to completion, so py_trees never calls
        # terminate() on the criteria and they stay at their birth value "INIT".
        self._finalize_ego_criteria(ego_criteria, reason)

        result_label = self._resolve_result_label(reason, ego_criteria)
        duration_system = max(0.0, end_system_time - start_system_time)
        try:
            duration_game = max(0.0, (end_game_time or 0.0) - (start_game_time or 0.0))
        except Exception:
            duration_game = 0.0

        proxy = _EgoCriteriaProxy(self._scenario, ego_criteria)
        # scenario.other_actors holds only walkers (vehicles live in _vehicle_actors); the writer
        # lists them as separate "NPC vehicles" / "Pedestrians" groups, so other_actors stays
        # empty here. Walker slots can be None after a respawn — filter them.
        stub = SimpleNamespace(
            scenario=proxy,
            scenario_tree=self._scenario.scenario_tree,
            ego_vehicles=self._scenario.ego_vehicles,
            other_actors=[],
            start_system_time=start_system_time,
            end_system_time=end_system_time,
            scenario_duration_system=duration_system,
            scenario_duration_game=duration_game,
        )

        _GroupedResultOutput(
            stub,
            result_label,
            stdout=True,
            filename=result_file,
            vehicles=[a for a in getattr(self._scenario, "_vehicle_actors", []) if a],
            walkers=[a for a in self._scenario.other_actors if a],
        ).write()

    @staticmethod
    def _normalize_reason(reason: Optional[str]) -> str:
        return reason.lower().strip() if reason else ""

    def _finalize_ego_criteria(self, ego_criteria: List[Criterion], reason: str) -> None:
        """Bring ego criteria to a terminal status before the result table is read.

        The runner loop breaks as soon as the ego arrives (or on timeout/stop) without ticking the
        scenario tree to a terminal state, so py_trees never calls terminate() on the criteria and
        they remain at their birth value "INIT". We finalize them here, mirroring what
        scenario_tree.stop() would do at clean teardown:

        - On ego arrival, credit RouteCompletionTest as complete (100% / SUCCESS). Autoware Mini
          stops a few metres short of the final waypoint, so the >99% gate never trips even though
          the ego reached the destination (within the arrival radius).
        - Then terminate() every ego criterion: the base Criterion.terminate flips remaining
          INIT/RUNNING criteria (the no-violation detectors) to SUCCESS; criteria that already
          failed stay FAILURE, and RouteCompletionTest (if not credited above, i.e. the ego did not
          arrive) correctly becomes FAILURE.
        """
        arrived = self._normalize_reason(reason) == "ego arrived"
        for criterion in ego_criteria:
            if (
                arrived
                and isinstance(criterion, RouteCompletionTest)
                and criterion.test_status in ("INIT", "RUNNING")
            ):
                criterion.actual_value = 100
                criterion.test_status = "SUCCESS"
        for criterion in ego_criteria:
            try:
                criterion.terminate(py_trees_common.Status.INVALID)
            except Exception:
                pass

    def _resolve_result_label(self, reason: str, ego_criteria: List[Criterion]) -> str:
        """Determine final result label similar to scenario_runner."""
        label = "SUCCESS"
        normalized = self._normalize_reason(reason)
        if normalized.startswith("timeout"):
            label = "TIMEOUT"
        elif normalized.startswith("stopped by user") or normalized.startswith("failed"):
            label = "FAILURE"
        elif normalized and not (
            normalized.startswith("actors completed") or normalized.startswith("ego arrived")
        ):
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
            cancel_now = multiprocessing.Event()
            cancel_done = multiprocessing.Event()
            proc = multiprocessing.Process(
                target=_ros_plan_publisher_process,
                args=(
                    raw_waypoints,
                    self.ros_downsample_interval,
                    self.ros_publish_delay,
                    self.agent_path,
                    self._skip_route_interpolation,
                    cancel_now,
                    cancel_done,
                ),
                daemon=True,
            )
            proc.start()
            self._ros_agent_proc = proc
            self._ros_agent_cancel_now = cancel_now
            self._ros_agent_cancel_done = cancel_done
            self.log(f"Agent subprocess started (pid {proc.pid}) to publish route with {len(raw_waypoints)} points")
        except Exception as exc:
            self.log(f"Failed to launch agent subprocess: {exc}")

    def _stop_ros_agent_process(self):
        """Terminate any running agent subprocess."""
        proc = getattr(self, "_ros_agent_proc", None)
        if not proc:
            return
        cancel_now = getattr(self, "_ros_agent_cancel_now", None)
        cancel_done = getattr(self, "_ros_agent_cancel_done", None)
        try:
            if proc.is_alive():
                # Graceful cancel first: ask the agent to cancel the route while it is still
                # healthy (clean /planning/cancel_route reply, no "returned no response"),
                # then terminate. Wait briefly for the cancel to complete.
                if cancel_now is not None:
                    try:
                        cancel_now.set()
                        if cancel_done is not None:
                            cancel_done.wait(timeout=1.0)
                    except Exception:
                        pass
                # Request graceful termination
                proc.terminate()
                # Wait briefly for graceful exit
                proc.join(timeout=1.5)

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
            self._ros_agent_cancel_now = None
            self._ros_agent_cancel_done = None

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
        # Insert spawn as first route point for proper speed profiling
        route_points.append(
            RoutePoint(
                transform=carla.Transform(spawn_location, spawn_rotation),
                speed_kmh=default_speed,
                idle_time_s=float(ego_entry.get("idle_time_s", 0.0) or 0.0),
                is_destination=False,
                speed_deviation_kmh=0,
            )
        )
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

        # Human mode: skip autopilot entirely; manual keyboard control only
        if self.agent_mode == "human":
            self.log("Ego agent mode: human — autopilot disabled, keyboard control only.")
            return None

        try:
            agent = self._create_ego_agent(ego_actor, vehicle_data)
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

    def _create_ego_agent(self, ego_actor: carla.Actor, vehicle_data: VehicleData):
        """Create the appropriate agent for the ego vehicle based on agent_mode/agent_behavior."""
        # Try BehaviorAgent first (when in autopilot mode)
        if self.agent_mode == "autopilot":
            try:
                from agents.navigation.behavior_agent import BehaviorAgent
                behavior = self.agent_behavior  # "cautious", "normal", or "aggressive"
                try:
                    agent = BehaviorAgent(
                        ego_actor,
                        behavior=behavior,
                        map_inst=self.world.get_map(),
                        grp_inst=_NoopGlobalRoutePlanner(),
                    )
                except Exception:
                    try:
                        agent = BehaviorAgent(
                            ego_actor,
                            behavior=behavior,
                            grp_inst=_NoopGlobalRoutePlanner(),
                        )
                    except Exception:
                        agent = BehaviorAgent(ego_actor, behavior=behavior)
                self.log(f"Ego agent: BehaviorAgent (behavior={behavior})")
                return agent
            except ImportError:
                self.log("BehaviorAgent not available; falling back to BasicAgent.")

        # Fallback to BasicAgent
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
        self.log("Ego agent: BasicAgent (fallback)")
        return agent

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
            distance = float(_cdp_location_or_live(ego).distance(dest))
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
        self._ego_prepare_failure = ""
        data = {}
        try:
            with open(self.json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.log(f"Failed to read scenario JSON: {exc}")
            self._ego_prepare_failure = "failed to read scenario JSON"
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
                        self._ego_prepare_failure = "ego spawn location blocked"
                        return None
                    try:
                        actor.set_transform(spawn_tf)
                        actor.set_target_velocity(carla.Vector3D())
                        actor.set_target_angular_velocity(carla.Vector3D())
                        try:
                            _cdp_purge_actor_id(actor.id)
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
                self._ego_prepare_failure = "external ego not found in world actor list"
                self._debug_list_roles()
                return None
            spawn_tf = self._ego_transform_from_json(ego_entry, actor.get_transform())
            if not self._is_spawn_clear(spawn_tf.location, actor.id):
                self.log("Ego spawn location blocked; aborting.")
                self._ego_prepare_failure = "ego spawn location blocked"
                return None
            try:
                actor.set_transform(spawn_tf)
                actor.set_target_velocity(carla.Vector3D())
                actor.set_target_angular_velocity(carla.Vector3D())
                try:
                    _cdp_purge_actor_id(actor.id)
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
                self._ego_prepare_failure = "ego spawn location blocked"
                return None
            try:
                existing.set_transform(spawn_tf)
                existing.set_target_velocity(carla.Vector3D())
                existing.set_target_angular_velocity(carla.Vector3D())
            except Exception:
                pass
            try:
                _cdp_purge_actor_id(existing.id)
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
            self._ego_prepare_failure = "unable to resolve ego blueprint or world"
            return None

        spawn_tf.location.z += 0.5
        if not self._is_spawn_clear(spawn_tf.location, None):
            self.log("Ego spawn location blocked; aborting.")
            self._ego_prepare_failure = "ego spawn location blocked"
            return None

        actor = self.world.try_spawn_actor(blueprint, spawn_tf)
        if not actor:
            self.log("Failed to spawn ego vehicle at requested transform.")
            self._ego_prepare_failure = "failed to spawn ego vehicle"
            return None

        try:
            _cdp_purge_actor_id(actor.id)
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
            if actor.id in self._spawn_clear_ignore_ids:
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
            try:
                snap = self.world.wait_for_tick(0.5)
                if snap is not None:
                    return snap
            except Exception:
                pass
            # wait_for_tick starved (it can lose the tick stream when MiniRunner runs as a thread
            # inside the GUI process, which consumes the ticks). The world is still advancing, so
            # read the current snapshot directly — the run loop de-dups by frame number so this
            # never double-feeds GameTime.
            return self.world.get_snapshot()
        except Exception:
            return None

    def _ego_at_route_start(self) -> bool:
        """True once the ego has been *stably* localized at the criteria route start.

        Gates criteria evaluation (see _run) so the localization transient isn't seen by the terminal
        InRouteTest: at run start awmini's relay can briefly flicker the ego back to its previous pose
        (far off-route) even after VSE places it at the start — a single such sample freezes
        InRouteTest at FAILURE forever. We therefore require the ego to stay within the arm radius
        (8 m, below InRouteTest's 15 m offroad_min) continuously for an arm dwell before arming; any
        flicker out of the radius resets the dwell, so arming happens only after the flicker settles.
        Dwell tunable via $VSE_CRITERIA_ARM_DWELL_S (default 1.0 s).
        """
        if self._route_start is None:
            # No known start (e.g. route unavailable): don't block criteria.
            return True
        sc = self._scenario
        ego = sc.ego_vehicles[0] if (sc and sc.ego_vehicles) else None
        if ego is None:
            return False
        try:
            dist = ego.get_location().distance(self._route_start)
        except Exception:
            return False
        now = time.monotonic()
        if dist <= 8.0:
            if self._route_start_enter_t is None:
                self._route_start_enter_t = now
            dwell = env_float("VSE_CRITERIA_ARM_DWELL_S", 1.0)
            return (now - self._route_start_enter_t) >= max(0.0, dwell)
        # Flicked out of the start radius — reset the stability timer.
        self._route_start_enter_t = None
        return False

    def _cleanup(self, reason: str):
        with self._cleanup_lock:
            if self._cleanup_invoked:
                return
            self._cleanup_invoked = True
        self._stop_requested = True
        # When the server is gone (fix-08) every teardown RPC would otherwise
        # stall on the client timeout (the 60 s ego-destroy in particular),
        # just moving the 5 h hang into cleanup. Tighten the client timeout so
        # each RPC fails fast; the ordering (destroy -> restore -> on_finish ->
        # ROS teardown) is untouched — only the per-call timeout shrinks. The
        # actors/settings die with the server anyway.
        lost = getattr(self, "_server_lost", False)
        if lost and self.client:
            try:
                self.client.set_timeout(1.5)
            except Exception:
                pass
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
            ego_id = getattr(self, "_spawned_internal_ego_id", None)
            if ego_id is not None:
                with _temporary_client_timeout(self.client, timeout_s=(1.5 if lost else 60.0)):
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
        with _temporary_client_timeout(self.client, timeout_s=(1.5 if lost else 10.0)):
            try:
                if self._restore_weather is not None and self.world:
                    self.world.set_weather(self._restore_weather)
            except Exception:
                pass
            self._restore_weather = None
            self._restore_sync_settings()
        self.log(f"Scenario ended ({reason or 'finished'})")
        # Show the results (on_finish opens the result dialog) BEFORE tearing down the ROS agent
        # subprocess, whose graceful route-cancel + terminate can take a few seconds. This removes
        # that delay from the user-visible "results appear" time for the external/VIL ego. The local
        # agent has no such subprocess, so its ordering is unaffected.
        if self.on_finish:
            try:
                self.on_finish(reason or "finished")
            except Exception:
                pass
        try:
            self._stop_ros_agent_process()
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


class _GroupedResultOutput(ResultOutputProvider):
    """ResultOutputProvider that lists NPC vehicles and pedestrians as separate groups.

    The stock writer prints every non-ego actor under a single "> Other actors:" header. The
    stub is fed other_actors=[], and the empty block that produces (exactly
    " > Other actors:\\n\\n\\n") is swapped for a "> NPC vehicles:" and a "> Pedestrians:" group.
    create_output_text() feeds both stdout and the .txt file, so the two stay identical.
    """

    def __init__(self, data, result, stdout: bool, filename: Optional[str],
                 vehicles: List[carla.Actor], walkers: List[carla.Actor]):
        super().__init__(data, result, stdout=stdout, filename=filename)
        self._vehicles = list(vehicles)
        self._walkers = list(walkers)

    def create_output_text(self) -> str:
        groups = " > NPC vehicles:\n"
        for actor in self._vehicles:
            groups += "{}; ".format(actor)
        groups += "\n\n > Pedestrians:\n"
        for actor in self._walkers:
            groups += "{}; ".format(actor)
        groups += "\n\n"
        return super().create_output_text().replace(" > Other actors:\n\n\n", groups, 1)


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
