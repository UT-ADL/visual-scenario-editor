"""Playback data models (moved verbatim from vse_play.py)."""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import carla

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
    is_destination: bool


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
    step_start_time: float = 0.0  # GameTime seconds at step start; drives phase completion (tick-rate independent)
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
