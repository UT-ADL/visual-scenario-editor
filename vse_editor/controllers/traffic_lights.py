"""Traffic-light engine (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-28).

Discovery/grouping of stop-line groups, the trigger-data store keyed by
location fingerprint, group selection, and the click/menu/scaling handlers.
All state lives on the processor/scene; the processor keeps one-line delegates
(plus staticmethod aliases for the five self-less geometry helpers), so
scenario_io's processor.* calls, undo/redo command reach-ins and
hasattr-string sites are unchanged. The _compute_traffic_light_fingerprint
alias and _normalize_traffic_light_fingerprint stay on the processor
(tests/unit/test_fingerprint.py asserts identity/unbound arity).

Contracts preserved verbatim: processor.traffic_lights is mutated in place
(clear()+extend(), never rebound — WorldCoordinateDetector aliases the list);
_traffic_light_trigger_key backfills group.location_fingerprint as a side
effect its callers depend on; _refresh_traffic_lights re-adopts a replaced
world (writes processor.world + detector.world, world.wait_for_tick(0.5)).
"""

import copy
import math
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, cast

import carla

from vse_editor.commands import SetPersonalTriggerCommand
from vse_editor.constants import (
    DEFAULT_PERSONAL_TRIGGER_RADIUS,
    MIN_PERSONAL_TRIGGER_RADIUS,
)
from vse_editor.rendering.overlays import OverlayMenuRenderer
from vse_editor.scene_types import (
    TRAFFIC_LIGHT_CENTROID_MATCH_THRESHOLD,
    TRAFFIC_LIGHT_GROUP_ALONG_TRAVEL_CLAMP,
    TRAFFIC_LIGHT_GROUP_YAW_TOLERANCE_DEG,
    TrafficLightGroupData,
)
from vse_common.geometry import get_ground_height


def _refresh_traffic_lights(processor) -> None:
    """Update cached traffic light actor list."""
    if not processor.world:
        if processor.traffic_lights:
            processor.traffic_lights.clear()
        if processor.traffic_light_groups:
            processor.traffic_light_groups.clear()
        processor._traffic_light_group_lookup.clear()
        processor._traffic_light_rectangles.clear()
        processor.clear_traffic_light_selection()
        return

    try:
        actor_list = processor.world.get_actors().filter('traffic.traffic_light*')
        try:
            is_empty = len(actor_list) == 0
        except Exception:
            is_empty = False
        if is_empty:
            fallback_list = processor.world.get_actors().filter('traffic_light*')
            try:
                fallback_count = len(fallback_list)
            except Exception:
                fallback_count = -1
            if processor._traffic_light_debug_matching:
                processor._traffic_light_debug(
                    "Primary actor filter returned 0 results; "
                    f"fallback 'traffic_light*' count={fallback_count}"
                )
            actor_list = fallback_list
    except Exception as exc:
        if not processor._traffic_light_refresh_error_logged:
            print(f"[TrafficLights] Failed to query actors: {exc}")
            processor._traffic_light_refresh_error_logged = True
        return

    processor._traffic_light_refresh_error_logged = False

    def _collect_alive(actors: Iterable[carla.Actor]) -> List[carla.TrafficLight]:
        collected: List[carla.TrafficLight] = []
        for actor in actors:
            if not actor:
                continue
            try:
                if not actor.is_alive:
                    continue
            except Exception:
                continue
            collected.append(cast(carla.TrafficLight, actor))
        return collected

    alive_lights = _collect_alive(actor_list)

    if not alive_lights:
        editor_world = None
        editor = getattr(processor, "editor", None)
        if editor and hasattr(editor, "_get_current_world"):
            try:
                editor_world = editor._get_current_world()
            except Exception:
                editor_world = None

        if editor_world and editor_world is not processor.world:
            if processor._traffic_light_debug_matching:
                processor._traffic_light_debug("Detected world replacement; refreshing traffic-light query.")
            processor.world = editor_world
            if processor.coordinate_detector:
                processor.coordinate_detector.world = editor_world

        active_world = processor.world or editor_world
        if active_world:
            try:
                active_world.wait_for_tick(0.5)
            except Exception:
                pass
            try:
                actor_list = active_world.get_actors().filter('traffic.traffic_light*')
                try:
                    is_empty = len(actor_list) == 0
                except Exception:
                    is_empty = False
                if is_empty:
                    actor_list = active_world.get_actors().filter('traffic_light*')
            except Exception:
                actor_list = []
            alive_lights = _collect_alive(actor_list)

        if not alive_lights:
            if processor._traffic_light_debug_matching:
                processor._traffic_light_debug("Actor query returned 0 lights; preserving existing cache.")
            return

    filtered_lights = [light for light in alive_lights if not processor._is_traffic_light_underground(light)]
    if not filtered_lights:
        if processor._traffic_light_debug_matching:
            processor._traffic_light_debug("Traffic lights filtered out as underground; skipping refresh.")
        return

    # Replace list in-place so coordinate detector keeps reference
    processor.traffic_lights.clear()
    processor.traffic_lights.extend(filtered_lights)

    current_ids: Set[int] = {light.id for light in filtered_lights}

    # Rebuild cached trigger rectangles and overlapping groups
    processor._traffic_light_rectangles.clear()
    grouping_meta = []

    for light in processor.traffic_lights:
        # Warm the rectangle cache for drawing/hit-testing (full-size box)
        rectangle_data = processor._get_traffic_light_rectangle_points(light)
        if not rectangle_data:
            continue
        grouping_geometry = processor._get_traffic_light_grouping_geometry(light)
        if not grouping_geometry:
            continue
        bbox, yaw = grouping_geometry
        grouping_meta.append({
            "light": light,
            "bbox": bbox,
            "yaw": yaw,
        })

    processor._build_traffic_light_groups(grouping_meta)

    # Drop selection if no lights from the previously selected group remain
    processor._restore_selected_traffic_light_group(current_ids)

def _bboxes_overlap(
    bbox1: Tuple[float, float, float, float, float, float],
    bbox2: Tuple[float, float, float, float, float, float],
    tolerance_xy: float = 0.25,
    tolerance_z: float = 1.5,
) -> bool:
    """Return True when two bounding boxes overlap within the tolerances."""
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = bbox1
    min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = bbox2
    return (
        max_x1 + tolerance_xy >= min_x2
        and max_x2 + tolerance_xy >= min_x1
        and max_y1 + tolerance_xy >= min_y2
        and max_y2 + tolerance_xy >= min_y1
        and max_z1 + tolerance_z >= min_z2
        and max_z2 + tolerance_z >= min_z1
    )

def _yaws_aligned(
    yaw1: Optional[float],
    yaw2: Optional[float],
    tolerance_deg: float = TRAFFIC_LIGHT_GROUP_YAW_TOLERANCE_DEG,
) -> bool:
    """Return True when two world yaws differ by no more than tolerance_deg.

    Unknown yaws (None) pass the gate so lights are never dropped from grouping.
    """
    if yaw1 is None or yaw2 is None:
        return True
    diff = abs(yaw1 - yaw2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff <= tolerance_deg

def _heading_diff_180(yaw1: float, yaw2: float) -> float:
    """Smallest angle (deg) between two headings, ignoring direction sign."""
    diff = abs(yaw1 - yaw2) % 180.0
    return 180.0 - diff if diff > 90.0 else diff

def _heading_diff_360(yaw1: float, yaw2: float) -> float:
    """Smallest angle (deg) between two directions, sign-sensitive (0-180)."""
    diff = abs(yaw1 - yaw2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff

def _get_travel_yaw_at(processor, location: carla.Location) -> Optional[float]:
    """Return the driving-lane heading (deg) at a world location, or None."""
    cached_map = processor._get_cached_map(refresh=False)
    if cached_map is None:
        return None
    try:
        waypoint = cached_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint:
            return float(waypoint.transform.rotation.yaw)
    except Exception:
        pass
    return None

def _get_traffic_light_grouping_geometry(
    processor, traffic_light: carla.TrafficLight
) -> Optional[Tuple[Tuple[float, float, float, float, float, float], Optional[float]]]:
    """Return (world AABB, world yaw) of the trigger box as used for grouping.

    The half-extent along the travel direction (detected from the road waypoint heading at
    the trigger center) is clamped to TRAFFIC_LIGHT_GROUP_ALONG_TRAVEL_CLAMP so that long
    trigger boxes don't bridge consecutive stop lines along the same road. Drawing and
    hit-testing use the stop-line strip instead (_get_traffic_light_rectangle_points). When
    the travel axis can't be determined, extents stay unclamped (previous behavior).
    """
    trigger_volume = getattr(traffic_light, "trigger_volume", None)
    if trigger_volume is None:
        return None
    try:
        transform = traffic_light.get_transform()
        extent = trigger_volume.extent
    except Exception:
        return None
    if extent is None:
        return None

    trigger_transform = carla.Transform(trigger_volume.location, trigger_volume.rotation)

    yaw: Optional[float] = None
    try:
        yaw = float(transform.rotation.yaw) + float(trigger_volume.rotation.yaw)
    except Exception:
        yaw = None

    half_x = float(extent.x)
    half_y = float(extent.y)
    travel_yaw: Optional[float] = None
    if yaw is not None:
        try:
            center = transform.transform(trigger_transform.transform(carla.Location()))
            travel_yaw = processor._get_travel_yaw_at(center)
        except Exception:
            travel_yaw = None
    if yaw is not None and travel_yaw is not None:
        if processor._heading_diff_180(yaw, travel_yaw) <= processor._heading_diff_180(yaw + 90.0, travel_yaw):
            half_x = min(half_x, TRAFFIC_LIGHT_GROUP_ALONG_TRAVEL_CLAMP)
        else:
            half_y = min(half_y, TRAFFIC_LIGHT_GROUP_ALONG_TRAVEL_CLAMP)

    local_corners = [
        carla.Location(+half_x, +half_y, 0.0),
        carla.Location(-half_x, +half_y, 0.0),
        carla.Location(-half_x, -half_y, 0.0),
        carla.Location(+half_x, -half_y, 0.0),
    ]
    world_corners: List[carla.Location] = []
    for corner in local_corners:
        try:
            world_corners.append(transform.transform(trigger_transform.transform(corner)))
        except Exception:
            return None
    bbox = (
        min(pt.x for pt in world_corners),
        max(pt.x for pt in world_corners),
        min(pt.y for pt in world_corners),
        max(pt.y for pt in world_corners),
        min(pt.z for pt in world_corners),
        max(pt.z for pt in world_corners),
    )
    return bbox, yaw

def _traffic_light_debug(processor, message: str) -> None:
    """Print traffic-light debug information when enabled."""
    if processor._traffic_light_debug_matching:
        print(f"[TrafficLights][Debug] {message}")

def _compute_traffic_light_group_centroid(
    lights: List[carla.TrafficLight],
) -> Optional[Tuple[float, float, float]]:
    """Return centroid of traffic light transforms, or None if unavailable."""
    if not lights:
        return None
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    count = 0
    for light in lights:
        if not light:
            continue
        try:
            location = light.get_transform().location
        except Exception:
            continue
        sum_x += location.x
        sum_y += location.y
        sum_z += location.z
        count += 1
    if count == 0:
        return None
    return (sum_x / count, sum_y / count, sum_z / count)

def _build_traffic_light_groups(processor, grouping_meta: List[Dict[str, object]]) -> None:
    """Cluster overlapping traffic light trigger volumes into selection groups."""
    groups: List[TrafficLightGroupData] = []
    previous_groups: List[TrafficLightGroupData] = list(getattr(processor, "traffic_light_groups", []))
    previous_records: List[Dict[str, object]] = []
    records_by_ids: Dict[frozenset, Dict[str, object]] = {}
    records_by_fp: Dict[Tuple[Tuple[int, int, int], ...], List[Dict[str, object]]] = {}

    def _add_record(
        *,
        live_ids: Set[int],
        reference_ids: Optional[Set[int]],
        fingerprint: Optional[Tuple[Tuple[int, int, int], ...]],
        centroid: Optional[Tuple[float, float, float]],
        trigger_payload: Optional[Tuple[Dict[str, float], float]],
        sequence_payload: Optional[List[Dict[str, Union[str, float, int]]]],
        size: int,
        group_obj: Optional[TrafficLightGroupData],
        source: str,
    ) -> None:
        ids_key = frozenset(live_ids) if live_ids else None
        record = {
            "live_ids": set(live_ids),
            "ids_key": ids_key,
            "reference_ids": set(reference_ids) if reference_ids else None,
            "fingerprint": fingerprint,
            "centroid": centroid,
            "trigger": trigger_payload,
            "sequence": sequence_payload,
            "size": size,
            "group_obj": group_obj,
            "used": False,
            "source": source,
        }
        previous_records.append(record)
        if ids_key:
            # Prefer live group data over snapshots
            if ids_key not in records_by_ids or group_obj is not None:
                records_by_ids[ids_key] = record
        if fingerprint:
            bucket = records_by_fp.setdefault(fingerprint, [])
            if group_obj is not None:
                bucket.insert(0, record)
            else:
                bucket.append(record)

    for existing_group in previous_groups:
        ids_set = set(existing_group.ids)
        if existing_group.reference_ids is None:
            existing_group.reference_ids = set(ids_set)
        fingerprint = existing_group.location_fingerprint
        if fingerprint is None:
            fingerprint = processor._compute_traffic_light_fingerprint(existing_group.lights)
            existing_group.location_fingerprint = fingerprint
        centroid = existing_group.center_location
        if centroid is None:
            centroid = processor._compute_traffic_light_group_centroid(existing_group.lights)
            existing_group.center_location = centroid

        trigger_payload = None
        if existing_group.has_trigger():
            trigger_payload = (
                dict(existing_group.trigger_center or {}),
                float(existing_group.trigger_radius or 0.0),
            )

        sequence_payload: Optional[List[Dict[str, Union[str, float, int]]]] = None
        ids_key = frozenset(ids_set) if ids_set else None
        if existing_group.sequence:
            sequence_payload = copy.deepcopy(existing_group.sequence)
        elif ids_key:
            seq_from_cache = processor.traffic_light_sequences.get(ids_key)
            if seq_from_cache:
                sequence_payload = copy.deepcopy(seq_from_cache)

        size_hint = existing_group.cached_size or len(existing_group.lights) or len(ids_set)
        _add_record(
            live_ids=ids_set,
            reference_ids=set(existing_group.reference_ids) if existing_group.reference_ids else set(ids_set),
            fingerprint=fingerprint,
            centroid=centroid,
            trigger_payload=trigger_payload,
            sequence_payload=sequence_payload,
            size=size_hint,
            group_obj=existing_group,
            source="live",
        )

    if not previous_records and processor._traffic_light_group_snapshots:
        for snapshot in processor._traffic_light_group_snapshots:
            legacy_live_ids = snapshot.get("ids_live")
            legacy_ids = snapshot.get("ids")
            snapshot_ids = set(cast(Set[int], legacy_live_ids if legacy_live_ids is not None else legacy_ids or set()))
            snapshot_reference_ids = set(cast(Set[int], snapshot.get("reference_ids", snapshot_ids)))
            fingerprint = cast(Optional[Tuple[Tuple[int, int, int], ...]], snapshot.get("fingerprint"))
            centroid = cast(Optional[Tuple[float, float, float]], snapshot.get("centroid"))
            trigger_payload = snapshot.get("trigger")
            sequence_payload = snapshot.get("sequence")
            size_hint = int(snapshot.get("size", len(snapshot_ids)))
            _add_record(
                live_ids=snapshot_ids,
                reference_ids=snapshot_reference_ids,
                fingerprint=fingerprint,
                centroid=centroid,
                trigger_payload=cast(Optional[Tuple[Dict[str, float], float]], trigger_payload),
                sequence_payload=cast(Optional[List[Dict[str, Union[str, float, int]]]], copy.deepcopy(sequence_payload)),
                size=size_hint,
                group_obj=None,
                source="snapshot",
            )

    processor._traffic_light_debug(
        "Rebuilding traffic light groups: "
        f"prev_groups={len(previous_groups)}, new_candidates={len(grouping_meta)}"
    )
    if processor._traffic_light_debug_matching:
        preview_limit = 12
        for index, record in enumerate(previous_records):
            if index >= preview_limit:
                remaining = len(previous_records) - preview_limit
                if remaining > 0:
                    processor._traffic_light_debug(
                        f"  ... truncated {remaining} additional cached group record(s)"
                    )
                break
            ids_list = sorted(cast(Set[int], record["live_ids"]))
            processor._traffic_light_debug(
                f"  Prev group ids={ids_list} "
                f"size={record['size']} "
                f"fp={record['fingerprint']} "
                f"centroid={record['centroid']} "
                f"source={record['source']}"
            )

    assigned: Set[int] = set()
    tolerance_xy = 0.25
    tolerance_z = 1.5
    new_sequence_cache: Dict[frozenset, List[Dict[str, Union[str, float, int]]]] = {}

    for entry in grouping_meta:
        light = cast(carla.TrafficLight, entry["light"])
        light_id = light.id
        if light_id in assigned:
            continue

        stack = [entry]
        group_entries: List[Dict[str, object]] = []

        while stack:
            current = stack.pop()
            current_light = cast(carla.TrafficLight, current["light"])
            current_id = current_light.id
            if current_id in assigned:
                continue
            assigned.add(current_id)
            group_entries.append(current)

            current_bbox = cast(Tuple[float, float, float, float, float, float], current["bbox"])
            for other in grouping_meta:
                other_light = cast(carla.TrafficLight, other["light"])
                other_id = other_light.id
                if other_id in assigned:
                    continue
                other_bbox = cast(Tuple[float, float, float, float, float, float], other["bbox"])
                if processor._bboxes_overlap(
                    current_bbox,
                    other_bbox,
                    tolerance_xy=tolerance_xy,
                    tolerance_z=tolerance_z,
                ):
                    if not processor._yaws_aligned(
                        cast(Optional[float], current.get("yaw")),
                        cast(Optional[float], other.get("yaw")),
                    ):
                        if processor._traffic_light_debug_matching:
                            processor._traffic_light_debug(
                                f"  Yaw gate rejected merge: light {current_id} "
                                f"(yaw={current.get('yaw')}) vs light {other_id} "
                                f"(yaw={other.get('yaw')})"
                            )
                        continue
                    stack.append(other)

        if not group_entries:
            continue

        group_entries.sort(key=lambda item: cast(carla.TrafficLight, item["light"]).id)
        group_lights = [cast(carla.TrafficLight, item["light"]) for item in group_entries]
        group_ids = {light.id for light in group_lights}
        group_data = TrafficLightGroupData(
            lights=group_lights,
            ids=group_ids,
            reference_ids=None,
        )
        fingerprint = processor._compute_traffic_light_fingerprint(group_lights) or None
        group_data.location_fingerprint = fingerprint
        group_centroid = processor._compute_traffic_light_group_centroid(group_lights)
        group_data.center_location = group_centroid
        group_data.cached_size = len(group_lights)
        cache_key = frozenset(group_ids)
        matched_record: Optional[Dict[str, object]] = None
        match_reason: Optional[str] = None

        # Prefer fingerprint to survive ID churn
        if matched_record is None and fingerprint:
            for candidate in records_by_fp.get(fingerprint, []):
                if not candidate.get("used"):
                    matched_record = candidate
                    match_reason = "fingerprint"
                    break

        # Next, exact live ids
        if matched_record is None and cache_key in records_by_ids:
            candidate = records_by_ids[cache_key]
            if not candidate.get("used"):
                matched_record = candidate
                match_reason = "id"

        # Next, reference ids if present
        if matched_record is None:
            for candidate in previous_records:
                if candidate.get("used"):
                    continue
                ref_ids = candidate.get("reference_ids")
                if not ref_ids:
                    continue
                try:
                    if frozenset(cast(Set[int], ref_ids)) == cache_key:
                        matched_record = candidate
                        match_reason = "reference_id"
                        break
                except Exception:
                    continue

        # Finally, centroid-based fallback
        if matched_record is None and group_centroid is not None:
            best_record = None
            best_distance = float("inf")
            for candidate in previous_records:
                if candidate.get("used"):
                    continue
                if candidate.get("size") != len(group_lights):
                    continue
                prev_centroid = candidate.get("centroid")
                if prev_centroid is None:
                    continue
                distance = math.dist(group_centroid, prev_centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_record = candidate
            if best_record and best_distance <= TRAFFIC_LIGHT_CENTROID_MATCH_THRESHOLD:
                matched_record = best_record
                match_reason = f"centroid(d={best_distance:.2f})"

        if matched_record:
            matched_record["used"] = True
            match_fp = matched_record.get("fingerprint")
            if match_fp and group_data.location_fingerprint is None:
                group_data.location_fingerprint = cast(Tuple[Tuple[int, int, int], ...], match_fp)
            if group_data.center_location is None:
                prev_centroid = cast(Optional[Tuple[float, float, float]], matched_record.get("centroid"))
                if prev_centroid:
                    group_data.center_location = prev_centroid
            if not group_data.cached_size:
                group_data.cached_size = int(matched_record.get("size", len(group_lights)))

            trigger_payload = matched_record.get("trigger")
            if trigger_payload:
                center_payload, radius_value = cast(Tuple[Dict[str, float], float], trigger_payload)
                if center_payload:
                    group_data.trigger_center = dict(center_payload)
                    group_data.trigger_radius = float(max(MIN_PERSONAL_TRIGGER_RADIUS, radius_value))
                    processor._set_traffic_light_trigger_data(
                        group_data.trigger_center,
                        group_data.trigger_radius,
                        group=group_data,
                        mark_visible=False,
                    )
            else:
                stored_center, stored_radius, _ = processor._get_traffic_light_trigger_data(group=group_data)
                if stored_center and stored_radius is not None:
                    group_data.trigger_center = dict(stored_center)
                    group_data.trigger_radius = float(stored_radius)

            sequence_payload = matched_record.get("sequence")
            if sequence_payload:
                preserved_sequence = copy.deepcopy(
                    cast(List[Dict[str, Union[str, float, int]]], sequence_payload)
                )
                group_data.sequence = processor._normalize_traffic_light_sequence(preserved_sequence)
                new_sequence_cache[cache_key] = copy.deepcopy(group_data.sequence)
            reference_ids_record = matched_record.get("reference_ids")
            if reference_ids_record:
                group_data.reference_ids = set(cast(Set[int], reference_ids_record))
            else:
                group_data.reference_ids = set(group_ids)
            prev_ids = sorted(cast(Set[int], matched_record.get("live_ids", set())))
            processor._traffic_light_debug(
                f"Matched new group ids={sorted(group_ids)} size={len(group_lights)} "
                f"to prev ids={prev_ids} via {match_reason}"
            )
        else:
            group_data.reference_ids = set(group_ids)
            cached_sequence = processor.traffic_light_sequences.get(cache_key)
            if cached_sequence:
                group_data.sequence = processor._normalize_traffic_light_sequence(cached_sequence)
                new_sequence_cache[cache_key] = copy.deepcopy(group_data.sequence)
            if not group_data.has_trigger():
                stored_center, stored_radius, _ = processor._get_traffic_light_trigger_data(group=group_data)
                if stored_center and stored_radius is not None:
                    group_data.trigger_center = dict(stored_center)
                    group_data.trigger_radius = float(stored_radius)
            processor._traffic_light_debug(
                f"No match for new group ids={sorted(group_ids)} size={len(group_lights)} "
                f"fp={fingerprint} centroid={group_centroid}"
            )

        groups.append(group_data)

    groups.sort(key=lambda data: min(data.ids) if data.ids else 0)
    processor.traffic_light_groups = groups

    processor._rebuild_traffic_light_group_lookup()
    if groups:
        processor.traffic_light_sequences = new_sequence_cache

    if groups:
        processor._snapshot_traffic_light_groups(groups)
    elif processor._traffic_light_debug_matching:
        processor._traffic_light_debug(
            "Traffic light groups empty after rebuild; retaining previous snapshot for future matching."
        )

    if (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'traffic_light'):
        key = processor.selected_personal_trigger.get('key')
        if key:
            refreshed_group = processor._find_traffic_light_group_by_key(key)
            if refreshed_group:
                processor.selected_personal_trigger['group'] = refreshed_group

def _snapshot_traffic_light_groups(processor, groups: List[TrafficLightGroupData]) -> None:
    """Cache a snapshot of current traffic light groups for future matching."""
    snapshots: List[Dict[str, object]] = []
    for group in groups:
        if group.has_trigger():
            processor._set_traffic_light_trigger_data(
                group.trigger_center or {},
                group.trigger_radius or 0.0,
                group=group,
                mark_visible=False,
            )
        trigger_payload = None
        if group.has_trigger():
            trigger_payload = (
                dict(group.trigger_center or {}),
                float(group.trigger_radius or 0.0),
            )
        snapshots.append(
            {
                "ids_live": set(group.ids),
                "reference_ids": set(group.reference_ids or set(group.ids)),
                "fingerprint": group.location_fingerprint,
                "centroid": group.center_location,
                "trigger": trigger_payload,
                "sequence": copy.deepcopy(group.sequence) if group.sequence else None,
                "size": group.cached_size or len(group.lights) or len(group.ids),
            }
        )
    processor._traffic_light_group_snapshots = snapshots

def _rebuild_traffic_light_group_lookup(processor) -> None:
    """Refresh lookup from light id to its owning group."""
    processor._traffic_light_group_lookup.clear()
    for group in processor.traffic_light_groups:
        for light in group.lights:
            processor._traffic_light_group_lookup[light.id] = group

def _restore_selected_traffic_light_group(processor, alive_ids: Set[int]) -> None:
    """Restore previously selected group if all of its members are still alive."""
    if not processor._selected_traffic_light_group_ids and not processor._selected_traffic_light_group_fingerprint:
        processor.selected_traffic_light_group = None
        processor._traffic_light_debug("Restore selection: no prior group ids/fingerprint recorded")
        return

    def _apply_restoration(target_group: TrafficLightGroupData) -> None:
        processor.selected_traffic_light_group = target_group
        processor._selected_traffic_light_group_ids = set(target_group.ids)
        processor._selected_traffic_light_group_fingerprint = target_group.location_fingerprint
        processor._update_traffic_light_menu_anchor(target_group)

        if processor.scaling_traffic_light_trigger and processor._traffic_light_scaling_group:
            if processor._traffic_light_scaling_group.ids == target_group.ids:
                processor._traffic_light_scaling_group = target_group

        panel = getattr(getattr(processor, "editor", None), "info_panel", None)
        if panel and panel.object_type == 'traffic_light' and panel.selected_object:
            if panel.selected_object.ids == target_group.ids:
                panel.selected_object = target_group
        processor._traffic_light_debug(
            f"Restored traffic light selection to ids={sorted(target_group.ids)} "
            f"fp={target_group.location_fingerprint}"
        )

    fingerprint = processor._selected_traffic_light_group_fingerprint
    ids_valid = (
        processor._selected_traffic_light_group_ids
        and processor._selected_traffic_light_group_ids.issubset(alive_ids)
    )

    if not alive_ids:
        processor.selected_traffic_light_group = None
        processor._selected_traffic_light_group_ids.clear()
        processor._traffic_light_debug(
            "No alive traffic lights detected; preserving selection fingerprint for future restoration."
        )
        return

    # Prefer fingerprint-based restore to survive ID changes
    if fingerprint:
        for group in processor.traffic_light_groups:
            if group.location_fingerprint == fingerprint:
                _apply_restoration(group)
                return

    if ids_valid:
        for group in processor.traffic_light_groups:
            if group.ids == processor._selected_traffic_light_group_ids:
                _apply_restoration(group)
                return

    processor._traffic_light_debug(
        "Failed to restore traffic light selection. "
        f"stored_ids={sorted(processor._selected_traffic_light_group_ids)} "
        f"stored_fp={processor._selected_traffic_light_group_fingerprint} "
        f"alive_ids_sample={sorted(list(alive_ids))[:6]}"
    )
    processor.clear_traffic_light_selection()

def _is_traffic_light_underground(processor, light: carla.TrafficLight) -> bool:
    """
    Check if traffic light is underground (more than 1m below ground level).
    Uses road waypoint height instead of raycast to avoid hitting structures above
    or the traffic light pole itself.

    Uses caching to avoid expensive waypoint lookups on every refresh.

    Args:
        light: Traffic light to check

    Returns:
        True if light is underground, False otherwise
    """
    if not processor.world:
        return False

    light_id = light.id
    loc = light.get_location()
    current_time = time.time()

    # Check cache (expire after 60 seconds)
    if light_id in processor._traffic_light_ground_cache:
        ground_z, timestamp = processor._traffic_light_ground_cache[light_id]
        if current_time - timestamp < 60.0:  # Cache valid for 60 seconds
            return loc.z < (ground_z - 1.0)

    # Cache miss or expired - get ground height using waypoint (road level)
    # This avoids raycasting issues where it hits structures above the light
    # or the traffic light pole itself
    cached_map = processor._get_cached_map(refresh=False)
    ground_z = loc.z  # Fallback to light's own Z if waypoint fails

    if cached_map is not None:
        try:
            waypoint = cached_map.get_waypoint(loc, project_to_road=True)
            if waypoint:
                ground_z = waypoint.transform.location.z
        except Exception:
            # If waypoint lookup fails, fall back to raycast
            ground_z = get_ground_height(processor.world, loc, debug=False, cached_map=cached_map)

    # Update cache
    processor._traffic_light_ground_cache[light_id] = (ground_z, current_time)

    # Check if underground (more than 1m below ground)
    return loc.z < (ground_z - 1.0)

def handle_traffic_light_click(processor, screen_x: int, screen_y: int) -> bool:
    """Check if a traffic light marker was clicked and select it."""
    if not processor.traffic_lights_visible:
        return False  # Overlay hidden: let clicks pass through to actors underneath
    if not processor.traffic_light_groups:
        return False

    click_point = (screen_x, screen_y)

    for group in processor.traffic_light_groups:
        if not group.lights:
            continue
        polygon_data = processor._get_traffic_light_group_screen_polygon(group)
        if not polygon_data:
            continue

        screen_points, _ = polygon_data
        hit = False
        if len(screen_points) >= 3:
            hit = processor._is_point_inside_polygon(click_point, screen_points)
        elif len(screen_points) == 2:
            hit = processor._is_point_near_segment(click_point, screen_points[0], screen_points[1])
        elif len(screen_points) == 1:
            dx = screen_points[0][0] - screen_x
            dy = screen_points[0][1] - screen_y
            hit = (dx * dx + dy * dy) <= 36.0  # radius ~6px

        if hit:
            if processor.selected_personal_trigger:
                processor.clear_personal_trigger_selection()
            processor.select_traffic_light_group(group)
            return True

    return False

def handle_traffic_light_action_click(processor, screen_x: int, screen_y: int) -> bool:
    """Handle clicks on the traffic light action menu icons."""
    if not processor.traffic_lights_visible:
        return False  # Menu is not rendered while the overlay is hidden
    group = processor.selected_traffic_light_group
    if not group:
        return False

    icons = processor._get_traffic_light_menu_icons(group)
    if not icons:
        return False

    processor._update_traffic_light_menu_anchor(group)
    if not processor.traffic_light_menu_position:
        return False

    action = OverlayMenuRenderer.hit_test(
        processor.traffic_light_menu_position,
        icons,
        (processor.screen_width, processor.screen_height),
        (screen_x, screen_y),
    )
    if not action:
        return False

    if action == 'add_trigger':
        processor.start_personal_trigger_placement('traffic_light', group=group)
        return True
    if action == 'remove_trigger':
        processor.delete_traffic_light_trigger(group)
        return True

    return False

def _get_traffic_light_menu_icons(processor, group: Optional[TrafficLightGroupData]) -> List[str]:
    """Return the ordered icon list for the selected traffic light group."""
    if not group:
        return []
    icons: List[str] = ['add_trigger']
    if group.has_trigger():
        icons.append('remove_trigger')
    if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
        # 3D view is view + select: drop icons that start a placement gesture
        # (render and hit-test share this list).
        icons = [icon for icon in icons if icon != 'add_trigger']
    return icons

def _get_traffic_light_group_menu_anchor(
    processor, group: Optional[TrafficLightGroupData]
) -> Optional[Tuple[float, float]]:
    """Compute the current menu anchor for a traffic light group in screen space."""
    if not group:
        return None

    polygon_data = processor._get_traffic_light_group_screen_polygon(group)
    if polygon_data:
        _, screen_center = polygon_data
        if screen_center:
            return screen_center

    if group.screen_center:
        return group.screen_center
    return None

def _update_traffic_light_menu_anchor(processor, group: Optional[TrafficLightGroupData]) -> None:
    """Update cached menu anchor for the provided traffic light group."""
    anchor = processor._get_traffic_light_group_menu_anchor(group)
    if anchor:
        processor.traffic_light_menu_position = (int(anchor[0]), int(anchor[1]))
    else:
        processor.traffic_light_menu_position = None

def _compute_traffic_light_group_trigger_center(
    processor, group: TrafficLightGroupData
) -> Optional[carla.Location]:
    """Return the world-space center of the traffic light trigger box overlay."""
    centers: List[carla.Location] = []
    fallback_points: List[carla.Location] = []

    for light in group.lights:
        rectangle_data = processor._get_traffic_light_rectangle_points(light)
        if not rectangle_data:
            continue
        corners, center = rectangle_data
        if center is not None:
            centers.append(center)
        elif corners:
            fallback_points.extend(corners)

    if centers:
        avg_x = sum(loc.x for loc in centers) / len(centers)
        avg_y = sum(loc.y for loc in centers) / len(centers)
        avg_z = sum(loc.z for loc in centers) / len(centers)
        return carla.Location(avg_x, avg_y, avg_z)

    if fallback_points:
        avg_x = sum(loc.x for loc in fallback_points) / len(fallback_points)
        avg_y = sum(loc.y for loc in fallback_points) / len(fallback_points)
        avg_z = sum(loc.z for loc in fallback_points) / len(fallback_points)
        return carla.Location(avg_x, avg_y, avg_z)

    return None

def start_traffic_light_trigger_scaling(processor, mouse_pos: Tuple[int, int]) -> None:
    """Begin scaling operation for the selected traffic light group's trigger."""
    group = processor.selected_traffic_light_group
    if not group or not group.has_trigger():
        return

    processor.scaling_traffic_light_trigger = True
    processor._traffic_light_scaling_group = group
    processor.traffic_light_scale_start_pos = mouse_pos
    processor.traffic_light_scale_start_radius = float(group.trigger_radius or DEFAULT_PERSONAL_TRIGGER_RADIUS)
    print(f"Started scaling traffic light trigger for IDs {sorted(group.ids)}")

def update_traffic_light_trigger_scaling(processor, mouse_pos: Tuple[int, int]) -> None:
    """Update trigger radius while scaling a traffic light group."""
    if not processor.scaling_traffic_light_trigger or not processor._traffic_light_scaling_group:
        return

    group = processor._traffic_light_scaling_group
    if not group.has_trigger():
        processor.stop_traffic_light_trigger_scaling()
        return

    delta_y = processor.traffic_light_scale_start_pos[1] - mouse_pos[1]
    new_radius = processor.traffic_light_scale_start_radius + (delta_y * 0.1)
    new_radius = max(MIN_PERSONAL_TRIGGER_RADIUS, min(new_radius, 100.0))
    group.trigger_radius = new_radius
    processor._cache_traffic_light_trigger_payload(group)

def cancel_traffic_light_trigger_scaling(processor) -> bool:
    """Escape-cancel an in-progress traffic-light-trigger scale (restore the
    start radius, no undo command). Returns True when cancelled."""
    if not processor.scaling_traffic_light_trigger:
        return False
    group = processor._traffic_light_scaling_group
    start_radius = getattr(processor, "traffic_light_scale_start_radius", None)
    if group is not None and start_radius is not None and group.has_trigger():
        group.trigger_radius = start_radius
        processor._cache_traffic_light_trigger_payload(group)
    processor.scaling_traffic_light_trigger = False
    processor._traffic_light_scaling_group = None
    return True

def stop_traffic_light_trigger_scaling(processor) -> None:
    """End any active traffic light trigger scaling."""
    if not processor.scaling_traffic_light_trigger:
        return

    if processor._traffic_light_scaling_group and processor._traffic_light_scaling_group.has_trigger():
        print(
            f"Stopped scaling traffic light trigger for IDs "
            f"{sorted(processor._traffic_light_scaling_group.ids)} "
            f"(radius {processor._traffic_light_scaling_group.trigger_radius:.2f} m)"
        )
        group = processor._traffic_light_scaling_group
        start_radius = getattr(processor, "traffic_light_scale_start_radius", None)
        center_snapshot = group.trigger_center or {}
        cached_center, _, resolved_key = processor._get_traffic_light_trigger_data(group=group)
        if not center_snapshot and cached_center:
            center_snapshot = dict(cached_center)
        key = processor._traffic_light_trigger_key(group=group) or resolved_key
        if center_snapshot and start_radius is not None and abs(group.trigger_radius - start_radius) > 1e-4:
            selection = {
                'kind': 'traffic_light',
                'group': group,
                'key': key,
            }
            command = SetPersonalTriggerCommand(
                processor,
                selection,
                dict(center_snapshot),
                group.trigger_radius,
                old_center=copy.deepcopy(center_snapshot),
                old_radius=start_radius,
            )
            editor = getattr(processor, "editor", None)
            if editor:
                editor.execute_command(command)
            else:
                command.execute()
        processor._cache_traffic_light_trigger_payload(group)
        processor._mark_last_visible_traffic_light_trigger(group)
    processor.scaling_traffic_light_trigger = False
    processor._traffic_light_scaling_group = None

def delete_traffic_light_trigger(
    processor, group: Optional[TrafficLightGroupData] = None
) -> bool:
    """Remove the trigger associated with the given traffic light group."""
    target_group = group or processor.selected_traffic_light_group
    if not target_group or not target_group.has_trigger():
        print("No traffic light trigger available to delete.")
        return False

    if processor.scaling_traffic_light_trigger and processor._traffic_light_scaling_group is target_group:
        processor.stop_traffic_light_trigger_scaling()

    processor._delete_traffic_light_trigger_data(group=target_group)
    processor.traffic_light_sequences.pop(frozenset(target_group.ids), None)
    target_group.sequence = []

    # Update snapshot to prevent restoration of deleted trigger
    for snapshot in processor._traffic_light_group_snapshots:
        # Check if this snapshot matches the target group
        ids_match = (snapshot.get("ids_live") == target_group.ids or
                     snapshot.get("reference_ids") == target_group.reference_ids)
        fp_match = (snapshot.get("fingerprint") == target_group.location_fingerprint
                    if target_group.location_fingerprint else False)

        if ids_match or fp_match:
            # Clear trigger and sequence from snapshot
            snapshot["trigger"] = None
            snapshot["sequence"] = None
            break

    print(f"Deleted traffic light trigger for IDs {sorted(target_group.ids)}")
    if (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'traffic_light'
            and processor.selected_personal_trigger.get('group') is target_group):
        processor.clear_personal_trigger_selection()
    return True

def _cache_traffic_light_sequence(processor, group: Optional[TrafficLightGroupData]) -> None:
    if not group or not group.ids:
        return
    key = frozenset(group.ids)
    if group.sequence:
        processor.traffic_light_sequences[key] = copy.deepcopy(group.sequence)
    else:
        processor.traffic_light_sequences.pop(key, None)

def _normalize_traffic_light_sequence(
    processor,
    sequence: Optional[Iterable[Dict[str, Union[str, float, int]]]],
    *,
    coerce_color: bool = False,
) -> List[Dict[str, Union[str, float, int]]]:
    """Normalize a traffic light sequence payload to the expected dict format."""
    if not sequence:
        return []
    return [
        {
            'color': str(entry.get('color', 'Red')) if coerce_color else entry.get('color', 'Red'),
            'duration_s': float(entry.get('duration_s', 0.0)),
            'duration_ticks': int(entry.get('duration_ticks', 0)),
        }
        for entry in sequence
    ]

def _traffic_light_trigger_key(
    processor,
    group: Optional[TrafficLightGroupData] = None,
    *,
    fingerprint: Optional[Iterable[Tuple[int, int, int]]] = None,
    ids: Optional[Iterable[int]] = None,
) -> Optional[Tuple[str, Tuple]]:
    """Return a stable key for storing traffic light trigger data."""
    normalized_fp = processor._normalize_traffic_light_fingerprint(fingerprint)
    if normalized_fp is None and group is not None:
        if group.location_fingerprint is None:
            group.location_fingerprint = processor._compute_traffic_light_fingerprint(group.lights)
        normalized_fp = processor._normalize_traffic_light_fingerprint(group.location_fingerprint)
    if normalized_fp:
        return ('fp', normalized_fp)

    ids_source: Optional[Iterable[int]] = ids
    if ids_source is None and group is not None:
        if group.reference_ids:
            ids_source = group.reference_ids
        else:
            ids_source = group.ids
    if ids_source:
        try:
            ids_tuple = tuple(sorted(int(value) for value in ids_source))
        except Exception:
            ids_tuple = None
        if ids_tuple:
            return ('ids', ids_tuple)
    return None

def _get_traffic_light_trigger_data(
    processor,
    *,
    key: Optional[Tuple[str, Tuple]] = None,
    group: Optional[TrafficLightGroupData] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[Tuple[str, Tuple]]]:
    resolved_key = key or processor._traffic_light_trigger_key(group=group)
    if not resolved_key:
        return None, None, None
    center = processor.traffic_light_trigger_centers.get(resolved_key)
    radius = processor.traffic_light_trigger_radii.get(resolved_key)
    if center and radius is not None and group:
        group.trigger_center = dict(center)
        group.trigger_radius = float(radius)
    return center, radius, resolved_key

def _set_traffic_light_trigger_data(
    processor,
    center: Dict[str, float],
    radius: float,
    *,
    key: Optional[Tuple[str, Tuple]] = None,
    group: Optional[TrafficLightGroupData] = None,
    mark_visible: bool = True,
) -> Optional[Tuple[str, Tuple]]:
    resolved_key = key or processor._traffic_light_trigger_key(group=group)
    if not resolved_key:
        return None
    processor.traffic_light_trigger_centers[resolved_key] = dict(center)
    processor.traffic_light_trigger_radii[resolved_key] = float(radius)
    if group:
        group.trigger_center = dict(center)
        group.trigger_radius = float(radius)
    if mark_visible:
        processor._last_visible_traffic_light_trigger_key = resolved_key
    return resolved_key

def _delete_traffic_light_trigger_data(
    processor,
    *,
    key: Optional[Tuple[str, Tuple]] = None,
    group: Optional[TrafficLightGroupData] = None,
) -> bool:
    resolved_key = key or processor._traffic_light_trigger_key(group=group)
    if not resolved_key:
        return False
    removed = False
    if resolved_key in processor.traffic_light_trigger_centers or resolved_key in processor.traffic_light_trigger_radii:
        processor.traffic_light_trigger_centers.pop(resolved_key, None)
        processor.traffic_light_trigger_radii.pop(resolved_key, None)
        removed = True
    if processor._last_visible_traffic_light_trigger_key == resolved_key:
        processor._last_visible_traffic_light_trigger_key = None
    if group:
        group.trigger_center = None
        group.trigger_radius = None
    return removed

def _find_traffic_light_group_by_key(
    processor,
    key: Optional[Tuple[str, Tuple]],
) -> Optional[TrafficLightGroupData]:
    if not key:
        return None
    for group in processor.traffic_light_groups:
        if processor._traffic_light_trigger_key(group=group) == key:
            return group
    return None

def _cache_traffic_light_trigger_payload(processor, group: Optional[TrafficLightGroupData]) -> None:
    """Legacy helper retained for compatibility with existing call sites."""
    if not group or not group.has_trigger():
        return
    center = group.trigger_center or {}
    radius = group.trigger_radius
    if center and radius is not None:
        processor._set_traffic_light_trigger_data(center, radius, group=group, mark_visible=False)

def _mark_last_visible_traffic_light_trigger(processor, group: Optional[TrafficLightGroupData], *, key=None) -> None:
    """Legacy helper retained for compatibility with existing call sites."""
    resolved_key = key or processor._traffic_light_trigger_key(group=group)
    if resolved_key:
        processor._last_visible_traffic_light_trigger_key = resolved_key

def select_traffic_light_group(processor, group: TrafficLightGroupData) -> bool:
    """Select a traffic light group and log its members."""
    if not group or not group.lights:
        return False

    alive_lights = []
    for light in group.lights:
        try:
            if light and light.is_alive:
                alive_lights.append(light)
        except Exception:
            continue

    if not alive_lights:
        return False

    # Reuse existing group if it still matches after filtering dead lights
    if len(alive_lights) != len(group.lights):
        group.lights = alive_lights
        group.ids = {light.id for light in alive_lights}
        group.location_fingerprint = processor._compute_traffic_light_fingerprint(group.lights)
        group.center_location = processor._compute_traffic_light_group_centroid(group.lights)
        group.cached_size = len(group.lights)
        processor._rebuild_traffic_light_group_lookup()

    if processor.selected_traffic_light_group and group.ids == processor.selected_traffic_light_group.ids:
        processor._update_traffic_light_menu_anchor(group)
        return True

    processor.clear_vehicle_selection(clear_traffic_light=False)
    # Selection kinds stay mutually exclusive: drop any selected global trigger
    # and any group selection
    processor.selected_trigger_index = None
    processor.trigger_action_menu_position = None
    processor.selected_actor_ids = set()
    processor.selected_waypoint_group = None
    if processor.scaling_traffic_light_trigger and processor._traffic_light_scaling_group is not group:
        processor.stop_traffic_light_trigger_scaling()
    if (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'traffic_light'
            and processor.selected_personal_trigger.get('group') is not group):
        processor.clear_personal_trigger_selection()
    processor.selected_traffic_light_group = group
    processor._selected_traffic_light_group_ids = set(group.ids)
    if group.location_fingerprint is None:
        group.location_fingerprint = processor._compute_traffic_light_fingerprint(group.lights)
    if group.center_location is None:
        group.center_location = processor._compute_traffic_light_group_centroid(group.lights)
    group.cached_size = len(group.lights)
    processor._selected_traffic_light_group_fingerprint = group.location_fingerprint
    processor._update_traffic_light_menu_anchor(group)
    processor.traffic_light_menu_hidden_for_camera_pan = False
    processor.traffic_light_overlays_hidden_for_camera_pan = False

    center_snapshot, radius_snapshot, key = processor._get_traffic_light_trigger_data(group=group)
    if center_snapshot and radius_snapshot is not None:
        group.trigger_center = dict(center_snapshot)
        group.trigger_radius = float(radius_snapshot)
        if key:
            processor._last_visible_traffic_light_trigger_key = key
    else:
        # Selecting a group without a trigger should clear any previously visible trigger marker.
        group.trigger_center = None
        group.trigger_radius = None
        processor._last_visible_traffic_light_trigger_key = None

    # Show info panel for traffic light
    panel = getattr(getattr(processor, "editor", None), "info_panel", None)
    if panel:
        screen_width = getattr(processor, "screen_width", 1280)
        screen_height = getattr(processor, "screen_height", 720)
        panel.show(group, 'traffic_light', screen_width, screen_height)

    try:
        reference_light = group.reference_light
        if reference_light:
            location = reference_light.get_transform().location
            ids_str = ", ".join(str(light.id) for light in group.lights)
            print(f"Selected {len(group.lights)} traffic lights [{ids_str}] near ({location.x:.2f}, {location.y:.2f})")
            return True
    except Exception:
        pass

    ids_str = ", ".join(str(light.id) for light in group.lights)
    print(f"Selected {len(group.lights)} traffic lights [{ids_str}]")
    return True

def clear_traffic_light_selection(processor) -> None:
    """Clear any active traffic light selection without stopping active operations."""
    # NOTE: Don't stop scaling here - let it complete like ego vehicle triggers
    # The scaling operation will be stopped explicitly on mouse button release
    processor.selected_traffic_light_group = None
    processor._selected_traffic_light_group_ids.clear()
    processor._selected_traffic_light_group_fingerprint = None
    processor._traffic_light_connector_debug_signature = None
    processor.traffic_light_menu_position = None
    processor.traffic_light_menu_hidden_for_camera_pan = False
    processor.traffic_light_overlays_hidden_for_camera_pan = False

    # Hide info panel if it's showing a traffic light
    panel = getattr(getattr(processor, "editor", None), "info_panel", None)
    if panel and panel.visible and panel.object_type == 'traffic_light':
        panel.hide()
    if (processor.selected_personal_trigger
            and processor.selected_personal_trigger.get('kind') == 'traffic_light'):
        processor.clear_personal_trigger_selection()
    processor._last_visible_traffic_light_trigger_key = None
