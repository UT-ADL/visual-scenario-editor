"""Shared traffic-light fingerprint identity for editor and playback.

A fingerprint is a sorted tuple of quantized (x, y, z) light locations —
a stable, actor-id-independent identity for a traffic-light group. The
editor writes fingerprints into scenario JSON; playback matches them back
to live actors. Extracted verbatim from the previously duplicated copies in
vse.py / vse_play.py (bodies verified identical before the move).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import carla

TRAFFIC_LIGHT_FINGERPRINT_SCALE = 4.0  # Quantize to ~0.25m resolution for matching stability


def normalize_traffic_light_fingerprint(
    fingerprint: Optional[List[Tuple[int, int, int]]],
) -> Optional[Tuple[Tuple[int, int, int], ...]]:
    """Convert a fingerprint payload to a normalized tuple-of-tuples."""
    if not fingerprint:
        return None
    try:
        return tuple((int(pt[0]), int(pt[1]), int(pt[2])) for pt in fingerprint)
    except Exception:
        return None


def compute_traffic_light_fingerprint(lights: List[carla.TrafficLight]) -> Optional[Tuple[Tuple[int, int, int], ...]]:
    """Return a quantized, order-independent signature for the provided lights."""
    quantized_locations: List[Tuple[int, int, int]] = []
    for light in lights:
        if not light:
            continue
        try:
            location = light.get_transform().location
        except Exception:
            continue
        scale = TRAFFIC_LIGHT_FINGERPRINT_SCALE
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


def build_traffic_light_fingerprint_index(world: Optional[carla.World]) -> dict:
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
        scale = TRAFFIC_LIGHT_FINGERPRINT_SCALE
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


def match_traffic_lights_by_fingerprint(
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
