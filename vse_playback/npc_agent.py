"""Scripted-NPC agent tuning (moved verbatim from vse_play.py):
GlobalRoutePlanner stub, BasicAgent subclass with detection tuning and the
VSE_NPC_* environment-variable knobs.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import carla

from vse_common.env import env_float

from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import get_speed, get_trafficlight_trigger_location
from agents.navigation.global_route_planner import GlobalRoutePlanner

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


class _NpcBasicAgent(BasicAgent):
    """BasicAgent variant for NPC vehicles that are NOT ignoring vehicles.

    Stock BasicAgent reacts to any detected obstacle with a binary emergency stop
    (full brake on, otherwise nothing). For NPCs that should yield to the ego that
    is both too late-feeling and abrupt. This subclass keeps BasicAgent's detection
    and steering untouched but replaces the *vehicle* response with graduated braking
    proportional to the gap: it eases off when the ego is far inside the detection
    range and brakes harder as it closes, reaching full brake within the stopping
    distance. Red traffic lights still trigger the stock hard stop.

    Detection range and brake strength are tuned separately (see
    ``_apply_npc_detection_tuning``); this class only changes the *response*. Used
    only for ``ignore_vehicles=False`` NPCs, so the deterministic ignore path is
    never routed through here.
    """

    # Comfortable full-stop deceleration assumed when sizing the brake ramp (m/s^2).
    # Matches the value the route backward-pass uses for approach braking.
    _YIELD_DECEL = 6.0
    # Extra standoff distance kept in front of the obstacle / stop line (m).
    _YIELD_MARGIN = 2.0
    # Vehicle-only detection/brake tuning (instance values are set by
    # _apply_npc_detection_tuning). Independent of the red-light tuning below.
    _npc_vehicle_speed_ratio = 1.5
    _npc_vehicle_max_brake = 1.0
    # Red-light "glide to the line" tuning. Detection is widened to at least
    # stopping_distance * _npc_tlight_slack (never below the CARLA-stock 5 + v) so there
    # is always room to ease to a stop; the brake then ramps up toward _npc_tlight_max_brake
    # as the stop line approaches.
    _npc_tlight_slack = 1.6
    _npc_tlight_max_brake = 1.0

    def run_step(self):
        """One navigation step with graduated (rather than binary) braking.

        Vehicle obstacles use graduated braking proportional to the gap. Red lights use the
        same graduated model toward the stop line, but with the detection distance widened
        (>= stopping distance) so the NPC eases to a stop AT the line instead of either
        stopping short or overshooting when it arrives fast.
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        speed_ms = get_speed(self._vehicle) / 3.6
        stopping_distance = (speed_ms * speed_ms) / (2.0 * self._YIELD_DECEL) + self._YIELD_MARGIN

        # Vehicle obstacle detection — widened, vehicle-only lookahead.
        max_vehicle_distance = self._base_vehicle_threshold + self._npc_vehicle_speed_ratio * speed_ms
        affected_by_vehicle, _, distance = self._vehicle_obstacle_detected(
            vehicle_list, max_vehicle_distance)

        # Traffic-light detection — widened to at least stopping_distance * slack (but never
        # below the CARLA-stock 5 + v) so there is room to glide to a stop. The stock
        # distance alone is shorter than the stopping distance at speed -> overshoot.
        max_tlight_distance = max(
            self._base_tlight_threshold + self._speed_ratio * speed_ms,
            stopping_distance * self._npc_tlight_slack,
        )
        affected_by_tlight, tlight = self._affected_by_traffic_light(
            self._lights_list, max_tlight_distance)

        control = self._local_planner.run_step()

        if affected_by_tlight:
            # Glide to the stop line: brake proportional to stopping_distance / gap, so a
            # fast NPC eases down over the (widened) distance and stops at the line, while a
            # slow one only brakes lightly near it — neither stopping short nor overshooting.
            tl_distance = max_tlight_distance
            if tlight is not None:
                try:
                    trigger = get_trafficlight_trigger_location(tlight)
                    tl_distance = self._vehicle.get_location().distance(trigger)
                except Exception:
                    tl_distance = max_tlight_distance
            ratio = max(0.0, min(1.0, stopping_distance / max(tl_distance, 0.1)))
            control.throttle = 0.0
            control.brake = max(control.brake, self._npc_tlight_max_brake * ratio)
        elif affected_by_vehicle:
            # Graduated brake: gentle when the obstacle is far inside the lookahead,
            # full brake once we are within the stopping distance (+ margin).
            ratio = max(0.0, min(1.0, stopping_distance / max(distance, 0.1)))
            control.throttle = 0.0
            control.brake = max(control.brake, self._npc_vehicle_max_brake * ratio)

        return control


def _apply_npc_detection_tuning(agent):
    """Widen obstacle detection and harden obstacle braking for an NPC ``_NpcBasicAgent``.

    Stock BasicAgent defaults look only ~5 m + speed ahead, brake at half strength,
    and (with use_bbs off) only treat a vehicle in the *same lane* as a hazard — so
    NPCs brake too late and miss crossing/adjacent egos. These overrides fix that.

    IMPORTANT: this deliberately does NOT set ``agent._speed_ratio`` or ``agent._max_brake``.
    ``_speed_ratio`` is only the CARLA-default floor of the red-light detection distance, and the
    red light no longer uses ``add_emergency_stop`` (so ``_max_brake`` is unused). Vehicle
    detection/braking lives in the ``_npc_vehicle_*`` attributes; the red-light "glide to the
    line" behaviour lives in the ``_npc_tlight_*`` attributes (see ``_NpcBasicAgent.run_step``).
    ``_base_vehicle_threshold`` and ``_use_bbs_detection`` are vehicle-only, set here directly.

    Only ever called for ``ignore_vehicles=False`` NPCs. Values overridable via ``VSE_NPC_*``.
    """
    agent._base_vehicle_threshold = env_float("VSE_NPC_VEHICLE_THRESHOLD_M", 8.0)
    agent._use_bbs_detection = os.environ.get("VSE_NPC_USE_BBS", "1") == "1"
    agent._npc_vehicle_speed_ratio = env_float("VSE_NPC_DETECTION_SPEED_RATIO", 1.5)
    agent._npc_vehicle_max_brake = env_float("VSE_NPC_MAX_BRAKE", 1.0)
    # Red-light glide-to-line: widen detection to stopping_distance * slack (room to ease to
    # a stop) and ramp the brake up to this ceiling as the line approaches.
    agent._npc_tlight_slack = max(1.0, env_float("VSE_NPC_TLIGHT_SLACK", 1.6))
    agent._npc_tlight_max_brake = env_float("VSE_NPC_TLIGHT_MAX_BRAKE", 1.0)
