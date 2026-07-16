"""Editor scene data types (moved verbatim from vse.py): waypoint TypedDicts,
ignore-flag dict and their clone/normalize helpers used by commands, the
scene state and scenario (de)serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TypedDict, Union, cast

import carla

WaypointIndex = Union[int, str]


class WaypointDataRequired(TypedDict):
    x: float
    y: float
    z: float
    index: WaypointIndex
    speed_km_h: float
    idle_time_s: float
    is_destination: bool


class WaypointData(WaypointDataRequired, total=False):
    yaw: Optional[float]
    auto_generated: bool
    speed_deviation_km_h: float


class VehicleIgnoreFlags(TypedDict):
    traffic_lights: bool
    stop_signs: bool
    vehicles: bool


def clone_waypoint_data(waypoint: WaypointData) -> WaypointData:
    """Return a shallow copy of a waypoint payload with consistent typing."""
    return cast(WaypointData, dict(waypoint))


def clone_waypoint_sequence(waypoints: List[WaypointData]) -> List[WaypointData]:
    """Return a cloned waypoint sequence to avoid mutating original lists."""
    return [clone_waypoint_data(waypoint) for waypoint in waypoints]


def ensure_ignore_flags(flags: Optional[Dict[str, bool]]) -> VehicleIgnoreFlags:
    """Normalize ignore flag dictionaries with default True for vehicles."""
    base: VehicleIgnoreFlags = {
        'traffic_lights': False,
        'stop_signs': False,
        'vehicles': True,
    }
    if not flags:
        return base
    base.update({
        'traffic_lights': bool(flags.get('traffic_lights', False)),
        'stop_signs': bool(flags.get('stop_signs', False)),
        'vehicles': bool(flags.get('vehicles', True)),
    })
    return base


@dataclass
class TrafficLightGroupData:
    """Bundle multiple overlapping traffic lights into a single selectable unit."""

    lights: List[carla.TrafficLight]
    ids: Set[int]
    reference_ids: Optional[Set[int]] = None
    location_fingerprint: Optional[Tuple[Tuple[int, int, int], ...]] = None
    center_location: Optional[Tuple[float, float, float]] = None
    screen_polygon: Optional[List[Tuple[float, float]]] = None
    screen_center: Optional[Tuple[float, float]] = None
    trigger_center: Optional[Dict[str, float]] = None
    trigger_radius: Optional[float] = None
    sequence: List[Dict[str, Union[str, float, int]]] = field(default_factory=list)
    cached_size: int = 0

    @property
    def reference_light(self) -> Optional[carla.TrafficLight]:
        return self.lights[0] if self.lights else None

    def has_trigger(self) -> bool:
        return self.trigger_center is not None and self.trigger_radius is not None

TRAFFIC_LIGHT_CENTROID_MATCH_THRESHOLD = 2.0  # Max distance (m) to treat centroids as identical
TRAFFIC_LIGHT_GROUP_YAW_TOLERANCE_DEG = 30.0  # Max heading difference (deg) for lights to share a group
TRAFFIC_LIGHT_GROUP_ALONG_TRAVEL_CLAMP = 0.5  # Half-extent cap (m) along travel for grouping bboxes
TRAFFIC_LIGHT_STOP_LINE_MARKER_DEPTH = 1.0  # Drawn stop-line strip depth (m) from the box front edge
