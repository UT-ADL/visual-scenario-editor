"""Route geometry & refinement (moved verbatim from vse_play.py):
heading/yaw math, arc/linear sampling, walk-speed resolution and
_refine_vehicle_route (waypoint interpolation + lateral-acc speed capping).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

import carla

from vse_common.geometry import get_ground_height
from vse_playback.models import RoutePoint, RouteSegment, VehicleData


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




def _shortest_angle_difference(target: float, start: float) -> float:
    diff = target - start
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0
    return diff


def _distance(a: carla.Location, b: carla.Location) -> float:
    return float(a.distance(b))


def _resolve_walk_speed_mps(planned_speed_kmh, deviation_kmh) -> float:
    """Resolve a concrete walking speed (m/s) from a planned km/h speed and an
    optional +/- deviation. The deviation is applied once via random.randint and
    the result is clamped to >= 0. Used for pedestrian leg speeds."""
    try:
        planned_speed_kmh = float(planned_speed_kmh)
    except Exception:
        planned_speed_kmh = 0.0
    try:
        deviation_kmh = int(float(deviation_kmh or 0))
    except Exception:
        deviation_kmh = 0
    if deviation_kmh < 0:
        deviation_kmh = 0

    speed_kmh = int(round(planned_speed_kmh))
    if deviation_kmh:
        speed_kmh += random.randint(-deviation_kmh, deviation_kmh)
    if speed_kmh < 0:
        speed_kmh = 0
    return speed_kmh / 3.6




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

    # Backward pass: ensure the car can physically brake to each upcoming speed
    max_decel = 6.0  # m/s^2
    for i in range(len(points) - 2, -1, -1):
        dist = points[i].transform.location.distance(points[i + 1].transform.location)
        v_next_ms = capped_speeds[i + 1] / 3.6
        v_max_kmh = math.sqrt(v_next_ms * v_next_ms + 2.0 * max_decel * dist) * 3.6
        if capped_speeds[i] > v_max_kmh:
            capped_speeds[i] = v_max_kmh

    # Forward pass: limit speed recovery to physically achievable acceleration
    max_accel = 4.0  # m/s^2
    for i in range(1, len(points)):
        dist = points[i - 1].transform.location.distance(points[i].transform.location)
        v_prev_ms = capped_speeds[i - 1] / 3.6
        v_max_kmh = math.sqrt(v_prev_ms * v_prev_ms + 2.0 * max_accel * dist) * 3.6
        if capped_speeds[i] > v_max_kmh:
            capped_speeds[i] = v_max_kmh

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

        samples = _generate_arc_samples(prev_tf, curr_tf, next_tf, max_step)
        segment_deviation = points[idx].speed_deviation_kmh
        num_samples = len(samples)
        for s_idx, sample_tf in enumerate(samples):
            frac = (s_idx + 1) / (num_samples + 1)
            sample_speed = capped_speeds[idx] + frac * (capped_speeds[idx + 1] - capped_speeds[idx])
            new_points.append(
                RoutePoint(
                    transform=sample_tf,
                    speed_kmh=sample_speed,
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
