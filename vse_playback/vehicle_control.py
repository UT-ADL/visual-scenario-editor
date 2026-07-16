"""Threaded per-vehicle navigation (moved verbatim from vse_play.py):
_SafeVehicleControl / _AsyncSafeVehicleControl (divide-by-zero and async
set_transform velocity-rollback guards) and VehicleController (waypoint
following, stopped-gated idle timing, pre-release handbrake hold).
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import List, Optional, TYPE_CHECKING

import carla

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import get_speed
from srunner.scenariomanager.actorcontrols.simple_vehicle_control import SimpleVehicleControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from vse_playback.models import RoutePoint, VehicleData, VehicleTrigger
from vse_playback.npc_agent import _NpcBasicAgent, _apply_npc_detection_tuning
from vse_playback.route import _compute_heading, _distance, _normalize_yaw

if TYPE_CHECKING:  # only for type annotations; avoids a runtime circular import
    from vse_playback.scenario import vse_play

logger = logging.getLogger(__name__)

class _SafeVehicleControl(SimpleVehicleControl):
    """SimpleVehicleControl guarded against the divide-by-zero stock SVC raises when the
    current target waypoint coincides with the actor's (cached) location.

    Stock ``_set_new_velocity`` computes ``velocity.x = direction.x / direction_norm * speed``
    with ``direction = next_waypoint - CarlaDataProvider.get_location(actor)``. The waypoint-drop
    loop in ``run_step`` measures distance with the *live* ``actor.get_location()``, while this
    divide uses the *cached* CDP location; when the cached location equals the target (e.g. the
    first step before the cache advances, common on large maps) ``direction_norm`` is 0 and stock
    SVC raises ``ZeroDivisionError``. That exception propagates to VehicleController, which logs it
    and stops the controller thread — so the NPC silently never moves (only happens in velocity /
    "Scripted" mode; ``basic_agent`` never calls SVC). This subclass detects the degenerate case and
    skips driving for that step, returning a ~0 norm so ``run_step`` advances to the next waypoint.
    Used for velocity-mode NPCs in every tick mode.
    """

    # Below this planar distance (m) the target is treated as "already reached": skip the
    # velocity write so the stock divide-by-zero can't fire. Far below the 0.5 m waypoint-drop
    # threshold, so it never interferes with normal driving.
    _MIN_DIRECTION_NORM = 1e-3

    def _set_new_velocity(self, next_location):
        location = CarlaDataProvider.get_location(self._actor) if self._actor else None
        if location is not None:
            if math.hypot(next_location.x - location.x, next_location.y - location.y) < self._MIN_DIRECTION_NORM:
                try:
                    self._actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
                return 0.0
        return super()._set_new_velocity(next_location)


class _AsyncSafeVehicleControl(_SafeVehicleControl):
    """SimpleVehicleControl variant for asynchronous / vehicle-in-the-loop playback.

    Stock SVC steers by ``set_target_angular_velocity`` (simple_vehicle_control.py
    ``_set_new_velocity``), a per-physics-tick control law: it imparts a yaw rate and
    assumes one ``run_step`` per fixed-dt world tick. In async/VIL the ROS bridge
    free-runs physics (``synchronous_mode=False``, ``fixed_delta_seconds=0``), so the
    engine integrates that yaw rate for a variable interval between this controller's
    ~20 Hz ``run_step`` calls -> over-rotation -> opposite correction -> the car spins
    on the spot while the world-frame linear velocity still drags it along the route.

    This subclass keeps SVC's linear / obstacle / traffic-light / brake logic
    (``super()._set_new_velocity``), then cancels the angular velocity and writes the
    body yaw kinematically toward the travel bearing, rate-limited and gated exactly
    like ``WalkToTarget`` (the pedestrian async fix). The linear velocity is world-frame,
    so yaw is cosmetic for translation: this corrects how the car faces, never its path.
    Extends _SafeVehicleControl (divide-by-zero guard); selected only in async, while
    synchronous runs use _SafeVehicleControl directly (no kinematic yaw write).
    """

    # Max body-yaw rate (degrees per simulation second). Yaw does not affect the path
    # (linear velocity is world-frame), so this only bounds how smoothly the heading
    # tracks the travel direction; it also tames the near-waypoint blow-up of SVC's
    # angular law as direction_norm -> 0. Lower than the walker's 360 (cars yaw slower).
    TURN_RATE_DEG_S = 180.0
    # Only write the facing via set_transform when the yaw actually changed by at least
    # this much. In async/VIL each write applies a stale position and snaps the actor
    # back ~1 server frame of motion, so per-tick no-op writes on straight legs would
    # cut effective speed. Same mechanism/value as WalkToTarget.YAW_WRITE_EPSILON_DEG.
    YAW_WRITE_EPSILON_DEG = 0.5

    def __init__(self, actor, args=None):
        super().__init__(actor, args)
        self._last_update_time: Optional[float] = None
        try:
            self._current_yaw: Optional[float] = _normalize_yaw(
                CarlaDataProvider.get_transform(actor).rotation.yaw
            )
        except Exception:
            self._current_yaw = None
        self._last_written_yaw: Optional[float] = self._current_yaw

    def _set_new_velocity(self, next_location):
        # Reuse SVC's linear velocity + obstacle/traffic-light/brake logic and its
        # waypoint-advancement return value.
        direction_norm = super()._set_new_velocity(next_location)

        actor = self._actor
        if actor is None or not actor.is_alive:
            return direction_norm

        # (a) Cancel the persistent angular velocity SVC just set — the spin cause.
        try:
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass

        now = GameTime.get_time()
        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = min(max(now - self._last_update_time, 0.0), 0.25)
        self._last_update_time = now

        location = CarlaDataProvider.get_location(actor)
        if location is None:
            return direction_norm

        # (b) Rate-limit the body yaw toward the travel bearing, then gate the write.
        fallback_yaw = self._current_yaw if self._current_yaw is not None else 0.0
        bearing = _compute_heading(location, next_location, fallback_yaw)
        if self._current_yaw is None:
            self._current_yaw = bearing

        yaw_error = _normalize_yaw(bearing - self._current_yaw)
        max_step = self.TURN_RATE_DEG_S * dt
        if abs(yaw_error) <= max_step:
            self._current_yaw = bearing
        else:
            self._current_yaw = _normalize_yaw(
                self._current_yaw + math.copysign(max_step, yaw_error)
            )

        yaw_changed = (
            self._last_written_yaw is None
            or abs(_normalize_yaw(self._current_yaw - self._last_written_yaw))
            >= self.YAW_WRITE_EPSILON_DEG
        )
        if yaw_changed:
            try:
                transform = actor.get_transform()
                transform.rotation.yaw = self._current_yaw
                actor.set_transform(transform)
                self._last_written_yaw = self._current_yaw
                # set_transform can zero the rigid-body linear velocity on some CARLA
                # builds; re-assert the world-frame velocity SVC computed so a yaw write
                # mid-turn does not stall forward motion until the next ~20 Hz step.
                target_speed = self._target_speed
                if target_speed > 0.0 and direction_norm > 1e-6:
                    direction = next_location - location
                    velocity = carla.Vector3D(
                        direction.x / direction_norm * target_speed,
                        direction.y / direction_norm * target_speed,
                        0.0,
                    )
                    actor.set_target_velocity(velocity)
            except Exception:
                pass

        return direction_norm


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
    IDLE_APPROACH_DECEL = 2.5  # m/s^2 - comfortable deceleration used to brake toward idle waypoints and the destination

    def __init__(self, agent: BasicAgent, vehicle: carla.Actor, destination: carla.Location,
                 scenario: "vse_play", index: int, route_points: List[RoutePoint], initial_idle_time: float = 0.0,
                 destination_speed: Optional[float] = None, cruise_speed: Optional[float] = None,
                 vehicle_trigger: Optional[VehicleTrigger] = None,
                 control_mode: str = "basic_agent"):
        self.agent = agent
        self.vehicle = vehicle
        self.control_mode = control_mode
        self._ignore_traffic_lights = False
        self._svc = None  # SimpleVehicleControl instance (velocity mode only)
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

    def _braking_limited_speed_kmh(self, distance_m: float, arrival_kmh: float) -> float:
        """Max speed (km/h) at distance_m that still reaches arrival_kmh under comfortable braking.

        Uses constant deceleration IDLE_APPROACH_DECEL, so v = sqrt(arrival^2 + 2*a*d). At d=0 this
        returns arrival_kmh; clamping the normal target to it makes braking auto-engage at the right
        distance for the current speed (faster car -> earlier slow-down). Used both to stop at idle
        waypoints (arrival=0) and to reach the destination at its configured speed.
        """
        arrival_ms = max(0.0, arrival_kmh) / 3.6
        d = max(0.0, distance_m)
        v_ms = math.sqrt(arrival_ms ** 2 + 2.0 * self.IDLE_APPROACH_DECEL * d)
        return v_ms * 3.6

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
        # Note: Ego vehicle (index=-1) is exempt from this wait to avoid deadlock
        if (
            self.scenario
            and self.scenario._trigger_mode
            and not self._vehicle_trigger
            and self.index >= 0  # Ego has index=-1, exempt it
        ):
            while self._running and self.vehicle and self.vehicle.is_alive and self.scenario._keep_running:
                if getattr(self.scenario, "_global_trigger_released", False):
                    break
                try:
                    self.agent.set_target_speed(0)
                except Exception:
                    pass
                self._apply_wait_hold()
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
                self._apply_wait_hold()
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
                self._apply_wait_hold()
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

        # Wait for CarlaDataProvider to have a valid location (needs at least one world tick
        # in sync mode). Without this, SVC.run_step() crashes with Location - NoneType when
        # there is no trigger/idle to delay the controller thread.
        if self.control_mode == "velocity":
            loc_wait_start = time.monotonic()
            while self._running and self._is_vehicle_valid() and self.scenario and self.scenario._keep_running:
                if CarlaDataProvider.get_location(self.vehicle) is not None:
                    break
                if time.monotonic() - loc_wait_start > 5.0:
                    logger.warning("VehicleController %d: timed out waiting for CDP location", self.index)
                    break
                time.sleep(self.SLEEP_INTERVAL)

        # Initialize SimpleVehicleControl for deterministic mode
        if self.control_mode == "velocity" and self.vehicle and self.vehicle.is_alive:
            args = {
                'consider_trafficlights': 'false' if self._ignore_traffic_lights else 'true',
            }
            # Both velocity-mode controllers guard SVC's divide-by-zero (target == cached
            # location). In async/VIL playback the world is not in sync mode, where stock SVC's
            # persistent angular-velocity steering also makes NPCs spin, so use the async-safe
            # variant that additionally writes the heading kinematically. Sync runs use the
            # crash-guarded variant only (otherwise identical to stock SVC).
            try:
                _async_world = not bool(CarlaDataProvider._sync_flag)
            except Exception:
                _async_world = False
            svc_cls = _AsyncSafeVehicleControl if _async_world else _SafeVehicleControl
            self._svc = svc_cls(self.vehicle, args=args)
            # Build waypoint list as carla.Transform (what SimpleVehicleControl expects)
            wp_transforms = []
            for rp in self.waypoints:
                wp_transforms.append(rp.transform)
            # Add destination as final waypoint
            wp_transforms.append(carla.Transform(self.destination))
            self._svc.update_waypoints(wp_transforms)
            cruise = self._resolve_cruise_speed()
            self._svc.update_target_speed(cruise / 3.6)  # SVC expects m/s

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

                # Skip ticks where the freshly spawned vehicle still reports the world origin
                # (0,0,0) before its transform has been applied. Measuring waypoint distances from
                # the origin makes the overshoot scan below mark leading waypoints — including idle
                # ones — as already reached, which consumes their idle time at the spawn point.
                if abs(loc.x) < 0.01 and abs(loc.y) < 0.01 and abs(loc.z) < 0.01:
                    self._step_counter += 1
                    if self._stop_event.wait(timeout=self.SLEEP_INTERVAL):
                        break
                    continue

                current_speed = self._get_current_speed_kmh()

                # Advance waypoint progression: from the current active waypoint,
                # check if a NEARBY subsequent waypoint is now closer (car passed the active one)
                # First find current active (first unreached)
                active_wp_idx = None
                for i in range(len(self.waypoints)):
                    if not self._waypoint_reached[i]:
                        active_wp_idx = i
                        break

                if active_wp_idx is not None:
                    active_dist = float(loc.distance(self.waypoints[active_wp_idx].transform.location))
                    # Check if we're close enough to mark reached
                    if active_dist < self.WAYPOINT_REACHED_THRESHOLD:
                        self._waypoint_reached[active_wp_idx] = True
                    else:
                        # Check if a subsequent waypoint is closer (car overshot/diverged)
                        # Only look a limited window ahead to avoid jumping to distant waypoints
                        scan_end = min(active_wp_idx + 15, len(self.waypoints))
                        for j in range(active_wp_idx + 1, scan_end):
                            j_dist = float(loc.distance(self.waypoints[j].transform.location))
                            if j_dist < active_dist:
                                # Car is closer to waypoint j than to active — mark all before j as reached
                                for k in range(active_wp_idx, j):
                                    self._waypoint_reached[k] = True
                                active_dist = j_dist
                                active_wp_idx = j
                            else:
                                break  # distances increasing, stop scanning

                    # Re-find active after possible advancement
                    active_wp_idx = None
                    for i in range(len(self.waypoints)):
                        if not self._waypoint_reached[i]:
                            active_wp_idx = i
                            break

                # Part 1 — idle latch: a waypoint we've reached that still owes idle time.
                # Same condition as _is_idling_at_waypoint. Hold a full stop until the timer
                # elapses, regardless of the waypoint's configured speed or any small overshoot.
                idling_idx = None
                for i in range(len(self.waypoints)):
                    if (self._waypoint_reached[i]
                            and not self._waypoint_idle_complete[i]
                            and self.waypoints[i].idle_time_s > 0.0):
                        idling_idx = i
                        break

                if idling_idx is not None:
                    # Always command a full stop while latched.
                    self.agent.set_target_speed(0)
                    # Count idle time only while the car is actually stopped. The clock is
                    # GameTime (simulation seconds), but it must not advance until the vehicle has
                    # physically halted — otherwise, if the controller thread is starved while the
                    # simulation races ahead (in-process editor runs), the elapsed GameTime could
                    # satisfy the idle before the car ever stopped, and it would sail through.
                    is_stopped = current_speed <= (self.STOPPED_SPEED * 3.6)
                    if is_stopped:
                        if self._waypoint_idle_start[idling_idx] is None:
                            self._waypoint_idle_start[idling_idx] = GameTime.get_time()
                        elapsed = GameTime.get_time() - self._waypoint_idle_start[idling_idx]
                        if elapsed >= self.waypoints[idling_idx].idle_time_s:
                            self._waypoint_idle_complete[idling_idx] = True
                            # Resume with the next waypoint's speed (or destination/cruise fallback).
                            if idling_idx + 1 < len(self.waypoints):
                                resume_speed = self._resolve_waypoint_speed(idling_idx + 1)
                                if resume_speed <= (self.STOPPED_SPEED * 3.6):
                                    resume_speed = self._resolve_destination_speed()
                                self.agent.set_target_speed(resume_speed)
                            else:
                                self.agent.set_target_speed(self._resolve_cruise_speed())
                    else:
                        # Not stopped yet — keep braking and (re)start the clock once halted.
                        self._waypoint_idle_start[idling_idx] = None
                else:
                    # Part 2/3 — normal driving, clamped by speed-based braking toward the next
                    # pending idle waypoint (stop at it) and toward the destination (reach it at
                    # its configured speed). The strictest (nearest/slowest) constraint wins.
                    if active_wp_idx is not None:
                        target_speed = self._resolve_waypoint_speed(active_wp_idx)
                    else:
                        target_speed = self._resolve_cruise_speed()

                    next_idle_idx = None
                    for i in range(len(self.waypoints)):
                        if (not self._waypoint_idle_complete[i]
                                and self.waypoints[i].idle_time_s > 0.0):
                            next_idle_idx = i
                            break
                    if next_idle_idx is not None:
                        idle_dist = float(loc.distance(self.waypoints[next_idle_idx].transform.location))
                        target_speed = min(target_speed,
                                           self._braking_limited_speed_kmh(idle_dist, 0.0))

                    destination_distance = float(loc.distance(self.destination))
                    target_speed = min(
                        target_speed,
                        self._braking_limited_speed_kmh(destination_distance,
                                                        self._resolve_destination_speed()),
                    )

                    self.agent.set_target_speed(target_speed)

                # Apply control only if vehicle is still valid
                if self._is_vehicle_valid():
                    if self.control_mode == "velocity" and self._svc:
                        # Sync speed from agent (updated by waypoint logic above) to SVC
                        speed_ms = getattr(self.agent, '_target_speed', 50.0) / 3.6
                        self._svc.update_target_speed(speed_ms)
                        self._svc.run_step()
                    else:
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
        if self._svc:
            try:
                self._svc.reset()
            except Exception:
                pass
            self._svc = None
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

    def _apply_wait_hold(self) -> None:
        # Held before the actor is released (trigger/idle wait). set_target_speed(0)
        # only stores a number on the agent; nothing applies control until the main
        # drive loop, so on a slope the vehicle would roll. Apply an explicit handbrake
        # hold to keep it parked in place.
        vehicle = self.vehicle
        if not vehicle or not vehicle.is_alive:
            return
        try:
            vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
            )
        except Exception:
            pass
