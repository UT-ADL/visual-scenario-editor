"""py_trees behaviours and criteria for playback (moved verbatim from
vse_play.py): walker orientation/grounding/walking, spawn/delay/route
execution, trigger monitors (traffic-light / pedestrian / vehicle),
geometric collision test (physics-off ego) and vehicle lights.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import carla
import py_trees
from py_trees import common as py_trees_common
from py_trees.behaviour import Behaviour
from py_trees.composites import Parallel, Sequence
from py_trees.trees import BehaviourTree

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.timer import GameTime

from vse_common.actor_cache import cached_bounding_box
from vse_playback.models import PedestrianTrigger, TrafficLightTrigger, VehicleTrigger
from vse_playback.vehicle_control import VehicleController
from vse_playback.route import (
    _compute_heading,
    _normalize_yaw,
    _resolve_walk_speed_mps,
    _shortest_angle_difference,
)
from vse_playback.world_utils import _cdp_location_or_live, _walker_ground_height

if TYPE_CHECKING:  # only for type annotations; avoids a runtime circular import
    from vse_playback.scenario import vse_play

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
                bbox_extent_z = float(getattr(cached_bounding_box(self._actor).extent, "z", 1.0))
            except Exception:
                bbox_extent_z = 1.0

            # Get ground height at target XY position
            target_ground_height = _walker_ground_height(
                self._actor.get_world(),
                self._actor,
                self._target.location,
                cached_map=self._actor.get_world().get_map(),
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

        ground_height = _walker_ground_height(
            self._world,
            self._actor,
            self._target.location,
            cached_map=self._world.get_map(),
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
    # Max body-yaw rate while walking (degrees per simulation second). The walker
    # always moves along its current facing, so this also bounds how sharply the
    # walked path curves through a corner.
    TURN_RATE_DEG_S = 360.0
    # Only write the facing via set_transform when the yaw actually changed by at
    # least this much. In async/VIL mode each write applies a stale position and
    # snaps the walker backward by ~1 server frame of motion, so per-tick no-op
    # writes on straight legs cut the effective walking speed nearly in half.
    YAW_WRITE_EPSILON_DEG = 0.5

    def __init__(self, walker: carla.Actor, target: carla.Location, speed: float, tolerance: float,
                 stuck_time: float, desired_yaw: Optional[float] = None, name: str = "WalkToTarget",
                 debug_enabled: bool = False, is_destination: bool = False, pass_through: bool = False):
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
        self._pass_through = pass_through
        self._current_yaw: Optional[float] = None
        self._last_written_yaw: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self._min_distance: float = float('inf')

    def initialise(self) -> None:  # type: ignore[override]
        self._last_progress_time = GameTime.get_time()
        self._last_distance = float('inf')
        self._min_distance = float('inf')
        self._last_update_time = None
        if self._actor and self._actor.is_alive:
            try:
                transform = self._actor.get_transform()
                self._start_yaw = _normalize_yaw(transform.rotation.yaw)
            except Exception:
                self._start_yaw = None
        else:
            self._start_yaw = None
        self._current_yaw = self._start_yaw
        self._last_written_yaw = self._start_yaw

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        now = GameTime.get_time()
        location = self._actor.get_location()
        # XY distance only: the walker's actor location is its body center (~1 m
        # above ground) while waypoint Z is ground level, so a 3D distance can
        # never reach the arrival tolerance.
        distance = math.hypot(self._target.x - location.x, self._target.y - location.y)

        if self._pass_through:
            # Walk straight through the waypoint: no stop control, no snap, no yaw
            # set — the control stays live so the next leg continues seamlessly.
            if distance <= self._tolerance:
                return py_trees_common.Status.SUCCESS
            # Corner arcs can skim past the tolerance circle on short legs: once the
            # walker moves away again after a close approach, count the waypoint as passed.
            if distance < self._min_distance:
                self._min_distance = distance
            elif self._min_distance < 1.0 and distance > self._min_distance + 0.3:
                return py_trees_common.Status.SUCCESS
        elif distance <= self._tolerance:
            if self._actor and self._actor.is_alive:
                transform = self._actor.get_transform()
                target_yaw = self._desired_yaw if self._desired_yaw is not None else self._start_yaw
                if target_yaw is not None:
                    transform.rotation.yaw = target_yaw
                self._actor.set_transform(transform)
                self._actor.apply_control(carla.WalkerControl())
            return py_trees_common.Status.SUCCESS
        elif distance <= self._tolerance + 0.05:
            if self._actor and self._actor.is_alive:
                transform = self._actor.get_transform()
                target_yaw = self._desired_yaw if self._desired_yaw is not None else self._start_yaw

                # Calculate proper Z based on ground height + bbox extent
                # Don't use target Z from JSON - it has the editor's estimated height which may be wrong
                try:
                    bbox_extent_z = float(getattr(cached_bounding_box(self._actor).extent, "z", 1.0))
                except Exception:
                    bbox_extent_z = 1.0

                # Get ground height at target XY position
                target_ground_height = _walker_ground_height(
                    self._actor.get_world(),
                    self._actor,
                    self._target,
                    cached_map=self._actor.get_world().get_map(),
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

        # Rate-limited facing: swing the body toward the live bearing to the target
        # and walk along the current facing, so motion and body orientation stay
        # aligned (turn-while-walking arc instead of an instant pivot).
        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = min(max(now - self._last_update_time, 0.0), 0.25)
        self._last_update_time = now

        fallback_yaw = self._current_yaw if self._current_yaw is not None else 0.0
        bearing = _compute_heading(location, self._target, fallback_yaw)
        if self._current_yaw is None:
            self._current_yaw = bearing
        yaw_error = _normalize_yaw(bearing - self._current_yaw)
        max_step = self.TURN_RATE_DEG_S * dt
        if abs(yaw_error) <= max_step:
            self._current_yaw = bearing
        else:
            self._current_yaw = _normalize_yaw(self._current_yaw + math.copysign(max_step, yaw_error))

        yaw_changed = (
            self._last_written_yaw is None
            or abs(_normalize_yaw(self._current_yaw - self._last_written_yaw)) >= self.YAW_WRITE_EPSILON_DEG
        )
        if yaw_changed and self._actor and self._actor.is_alive:
            transform = self._actor.get_transform()
            transform.rotation.yaw = self._current_yaw
            self._actor.set_transform(transform)
            self._last_written_yaw = self._current_yaw

        yaw_rad = math.radians(self._current_yaw)
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)

        speed = self._speed
        if not self._pass_through and distance < 2.0:
            speed = min(speed, max(0.5, distance / 1.0))
        control.speed = speed
        self._actor.apply_control(control)

        if distance < self._last_distance - 0.05:
            self._last_distance = distance
            self._last_progress_time = now
        elif abs(_normalize_yaw(bearing - self._current_yaw)) > 15.0:
            # Still swinging toward the new bearing (mid-corner) — distance to the
            # target may legitimately grow; don't count this as stuck.
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
                    bbox_extent_z = float(getattr(cached_bounding_box(self._actor).extent, "z", 1.0))
                except Exception:
                    bbox_extent_z = 1.0

                # Get ground height at target XY position
                target_ground_height = _walker_ground_height(
                    self._actor.get_world(),
                    self._actor,
                    self._target,
                    cached_map=self._actor.get_world().get_map(),
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


def _actor_obb_2d(actor, transform, margin: float = 0.0):
    """Build a 2D oriented bounding box (footprint) + vertical extent for an actor.

    Returns a dict with the four world-space footprint corners, the two box orientation
    axes (for the Separating Axis Theorem), the center, a bounding radius (for broad-phase),
    and the world Z range. Returns None if the geometry can't be read.

    Pure math from the actor's static ``bounding_box`` and a (cached) world transform, so it
    needs no physics — this is what lets us detect collisions on a teleported, physics-disabled
    ego where CARLA's collision sensor never fires.
    """
    if actor is None or transform is None:
        return None
    try:
        bb = cached_bounding_box(actor)  # per-tick path; direct read = blocking RPC on >= 0.9.16
        ext = bb.extent
    except Exception:
        return None

    yaw = math.radians(transform.rotation.yaw)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    # Box-orientation axes in the XY plane.
    ux, uy = cos_y, sin_y           # forward (local +x)
    vx, vy = -sin_y, cos_y          # left    (local +y)

    # World center of the box: actor origin + the box's local offset rotated into world.
    bx, by, bz = bb.location.x, bb.location.y, bb.location.z
    cx = transform.location.x + (bx * cos_y - by * sin_y)
    cy = transform.location.y + (bx * sin_y + by * cos_y)
    cz = transform.location.z + bz

    ex = ext.x + margin
    ey = ext.y + margin
    ez = ext.z + margin

    # Four footprint corners (center ± ex*u ± ey*v).
    corners = [
        (cx + ux * ex + vx * ey, cy + uy * ex + vy * ey),
        (cx + ux * ex - vx * ey, cy + uy * ex - vy * ey),
        (cx - ux * ex - vx * ey, cy - uy * ex - vy * ey),
        (cx - ux * ex + vx * ey, cy - uy * ex + vy * ey),
    ]
    return {
        "corners": corners,
        "axes": [(ux, uy), (vx, vy)],
        "center": (cx, cy),
        "radius": math.hypot(ex, ey),
        "z_min": cz - ez,
        "z_max": cz + ez,
    }


def _obb_overlap_2d(a, b) -> bool:
    """Separating Axis Theorem overlap test for two oriented quads (footprints).

    Two convex boxes overlap iff their projections overlap on every face-normal axis of
    both boxes (4 axes in 2D). A single gap on any axis proves they're separate.
    """
    for axis in (a["axes"][0], a["axes"][1], b["axes"][0], b["axes"][1]):
        ax, ay = axis
        a_dots = [cx * ax + cy * ay for cx, cy in a["corners"]]
        b_dots = [cx * ax + cy * ay for cx, cy in b["corners"]]
        if max(a_dots) < min(b_dots) or max(b_dots) < min(a_dots):
            return False
    return True


class GeometricCollisionTest(Criterion):
    """Physics-free collision criterion for the VIL / external ego.

    A teleport-driven ego has CARLA physics disabled, so the ``sensor.other.collision`` that
    scenario_runner's ``CollisionTest`` relies on never fires. This criterion instead checks,
    each tick, whether the ego's oriented bounding box overlaps any scenario vehicle or
    pedestrian (2D footprint via SAT, gated by a vertical-extent check so stacked actors on a
    bridge/overpass don't false-trigger). Each colliding actor is counted at most once.

    Reports under the name "CollisionTest" with ``actor=ego`` so the results table renders
    identically to the sensor-based criterion it replaces.
    """

    # Extra inflation (metres) applied to every box extent before the overlap test. 0.0 = exact
    # touch. Tuning knob; bump slightly if near-misses should count as contact.
    COLLISION_MARGIN_M = 0.0

    def __init__(self, scenario: "vse_play", ego_actor: carla.Actor):
        super().__init__("CollisionTest", actor=ego_actor, optional=False)
        self._scenario = scenario
        self._ego = ego_actor
        self.actual_value = 0
        self.success_value = 0
        self.units = "times"
        self._collided_ids: set = set()

    def _iter_targets(self):
        """Yield live scenario vehicles and pedestrians, excluding the ego itself."""
        ego_id = self._ego.id if self._ego else None
        for actor in list(self._scenario._vehicle_actors) + list(self._scenario.other_actors):
            if actor is None or actor.id == ego_id:
                continue
            try:
                if not actor.is_alive:
                    continue
            except Exception:
                continue
            yield actor

    def update(self) -> py_trees_common.Status:  # type: ignore[override]
        if not self._ego:
            return py_trees_common.Status.RUNNING
        try:
            if not self._ego.is_alive:
                return py_trees_common.Status.RUNNING
        except Exception:
            return py_trees_common.Status.RUNNING

        ego_tf = CarlaDataProvider.get_transform(self._ego)
        ego_obb = _actor_obb_2d(self._ego, ego_tf, self.COLLISION_MARGIN_M)
        if ego_obb is None:
            return py_trees_common.Status.RUNNING

        for actor in self._iter_targets():
            if actor.id in self._collided_ids:
                continue
            tf = CarlaDataProvider.get_transform(actor)
            obb = _actor_obb_2d(actor, tf, self.COLLISION_MARGIN_M)
            if obb is None:
                continue
            # Broad-phase: skip the SAT math for actors clearly out of reach.
            ex, ey = ego_obb["center"]
            ox, oy = obb["center"]
            if math.hypot(ex - ox, ey - oy) > ego_obb["radius"] + obb["radius"]:
                continue
            # Vertical gate: footprints overlapping but at different heights (e.g. an overpass)
            # is not a collision.
            if ego_obb["z_min"] > obb["z_max"] or obb["z_min"] > ego_obb["z_max"]:
                continue
            if not _obb_overlap_2d(ego_obb, obb):
                continue

            # Collision.
            self._collided_ids.add(actor.id)
            self.actual_value += 1
            self.test_status = "FAILURE"
            self._scenario.log(
                f"[COLLISION] Geometric collision: ego vs {actor.type_id} (id={actor.id}); "
                f"total={self.actual_value}"
            )
            try:
                from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
                is_ped = "walker" in actor.type_id
                ev_type = (
                    TrafficEventType.COLLISION_PEDESTRIAN if is_ped
                    else TrafficEventType.COLLISION_VEHICLE
                )
                self.events.append(TrafficEvent(event_type=ev_type))
            except Exception:
                pass

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

        ego_location = _cdp_location_or_live(ego_vehicle)

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
        trigger.step_start_time = GameTime.get_time()
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
            trigger.step_start_time = GameTime.get_time()
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
            # Phase completion is timed in GameTime seconds, not tree ticks, so a phase holds
            # for its authored duration regardless of tick rate. In sync@20Hz GameTime advances
            # 0.05s/tick, so this flips on the same tick as the old tick count; in async (VIL),
            # where the tree ticks far slower than 20Hz, it no longer over-runs.
            duration_s = duration_ticks / 20.0
            elapsed_s = GameTime.get_time() - trigger.step_start_time

            if elapsed_s >= duration_s:
                # Step complete
                print(f"[TRAFFIC_LIGHT] Tick {current_tick}: "
                      f"Trigger {idx} step {trigger.current_step + 1}/{len(trigger.sequence)} complete")

                # Move to next step
                trigger.current_step += 1
                trigger.step_start_time = GameTime.get_time()
                self._execute_sequence_step(idx, trigger)

    def _complete_trigger_sequence(self, idx: int, trigger: TrafficLightTrigger) -> None:
        """Complete a trigger sequence and release traffic lights"""
        if trigger.sequence_completed:
            return

        light_ids = [light.id for light in trigger.traffic_lights]
        trigger.sequence_completed = True

        # CARLA 0.9.15 gotcha: TrafficLight.freeze() is GLOBAL — the client maps
        # it to FreezeAllTrafficLights, so unfreezing this trigger's lights would
        # also unfreeze every other frozen light in the scene and knock any
        # still-running trigger sequence back to the default controller mid-step.
        # Defer the release until the last active sequence completes; until then
        # this trigger's lights hold their final commanded state.
        others_running = any(
            t is not trigger and t.activated and t.sequence and not t.sequence_completed
            for t in self.scenario._traffic_light_triggers
        )
        if others_running:
            print(f"[TRAFFIC_LIGHT] Tick {self.scenario._global_tick_counter}: "
                  f"Trigger {idx} sequence complete; release of lights {light_ids} deferred "
                  f"(another trigger sequence is still running; freeze is global in CARLA)")
            return

        # Unfreeze this trigger's lights. freeze(False) is global (see above),
        # which at this point — no other sequence running — is exactly the
        # intended release of everything frozen so far.
        for light in trigger.traffic_lights:
            try:
                light.freeze(False)
            except Exception as e:
                print(f"[TRAFFIC_LIGHT] ERROR unfreezing light {light.id}: {e}")

        print(f"[TRAFFIC_LIGHT] Tick {self.scenario._global_tick_counter}: "
              f"Trigger {idx} sequence complete, releasing lights {light_ids}")

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

        ego_location = _cdp_location_or_live(ego_vehicle)

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

        ego_location = _cdp_location_or_live(ego_vehicle)

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

    # Weather is re-fetched at most once per this many SIM seconds (opt-08):
    # night mode follows the GameTime-driven weather keyframes, so the
    # throttle must be sim-time based (wall time would lag fast sync runs).
    WEATHER_REFRESH_SIM_S = 1.0

    def __init__(self, ego_vehicle, radius=50, radius_increase=15, name="VehicleLightsBehavior"):
        super().__init__(name)
        self._ego_vehicle = ego_vehicle
        self._radius = radius
        self._radius_increase = radius_increase
        self._world = CarlaDataProvider.get_world()
        self._cached_weather = None
        self._weather_fetched_at = None
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

        # opt-08: get_weather() was an RPC EVERY tick; weather only moves on
        # the keyframe timescale, so refresh at most once per sim second.
        sim_now = GameTime.get_time()
        if (self._weather_fetched_at is None
                or sim_now - self._weather_fetched_at >= self.WEATHER_REFRESH_SIM_S):
            try:
                self._cached_weather = self._world.get_weather()
                self._weather_fetched_at = sim_now
            except Exception:
                pass  # keep the previous snapshot; first-tick failure handled below
        weather = self._cached_weather
        if weather is None:
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
                # opt-08: locations come from the CDP per-tick cache (all VSE
                # NPCs are CDP-registered); miss -> skip, like a dead actor.
                vehicle_location = CarlaDataProvider.get_location(vehicle)
                if vehicle_location is None:
                    continue
                dist_ok = vehicle_location.distance(location) <= radius
                lights = vehicle.get_light_state()
                if night_mode and dist_ok:
                    lights |= self._vehicle_lights
                else:
                    lights &= ~self._vehicle_lights
                vehicle.set_light_state(carla.VehicleLightState(lights))
            except Exception:
                continue

        return new_status
