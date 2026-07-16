"""Scenario JSON save/load (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-22).

Duck-typed like the Phase-3 overlay renderers: every function takes the
processor as first argument and reaches scene state, live CARLA handles and
the remaining CIP helpers (spawning, traffic-light stores, camera focus,
_reset_scenario_state) through it. The processor keeps thin delegates for
the editor-facing seam (save/load/_collect_scenario_snapshot/
_serialize_waypoints_for_vehicle).

Contract notes preserved from the monolith: the global trigger serializes as
`location`, personal triggers as `center` (asymmetry is part of the format);
`ids`->`ids_live` migration happens in _normalize_traffic_light_trigger_entries;
weather keyframes are sanitized at save time from the SceneModel storage.
"""

import json
import os
import time
import traceback
from typing import Dict, Iterable, List, Optional, Tuple

import carla

from vse_editor.carla_io.spawning import PEDESTRIAN_PLACEMENT_CLEARANCE_M
from vse_editor.constants import (
    DEFAULT_PERSONAL_TRIGGER_RADIUS,
    MIN_PERSONAL_TRIGGER_RADIUS,
    WEATHER_PARAMETER_SPECS,
)
from vse_editor.scene_types import TrafficLightGroupData, WaypointData


def _weather_dict_from_params(processor, weather: "carla.WeatherParameters") -> Dict[str, float]:
    """Convert WeatherParameters to a plain dict for saving."""
    payload: Dict[str, float] = {}
    for spec in WEATHER_PARAMETER_SPECS:
        try:
            payload[spec.name] = float(getattr(weather, spec.name))
        except Exception:
            payload[spec.name] = spec.min_value
    return payload

def _sanitize_weather_keyframes(processor, keyframes: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    """Normalize weather keyframes for saving."""
    cleaned: List[Dict[str, float]] = []

    def _append(payload: Dict[str, float]) -> None:
        try:
            pct = float(payload.get("route_percentage", payload.get("route_percent", payload.get("pct", 0.0))))
        except Exception:
            return
        normalized: Dict[str, float] = {"route_percentage": max(0.0, min(100.0, pct))}
        for spec in WEATHER_PARAMETER_SPECS:
            val = payload.get(spec.name)
            if val is None and spec.name.lower() in payload:
                val = payload.get(spec.name.lower())
            if val is None:
                continue
            try:
                normalized[spec.name] = float(val)
            except Exception:
                normalized[spec.name] = spec.min_value
        cleaned.append(normalized)

    for frame in keyframes or []:
        if isinstance(frame, dict):
            _append(frame)

    if not cleaned:
        base = dict(processor._weather_state)
        if not base:
            try:
                base = _weather_dict_from_params(processor, processor.world.get_weather())
            except Exception:
                base = {}
        default_frame = {"route_percentage": 0.0, **base}
        cleaned = [
            default_frame,
            {**default_frame, "route_percentage": 100.0},
        ]

    # Sort and deduplicate by route percentage (last wins)
    cleaned = sorted(cleaned, key=lambda kf: float(kf.get("route_percentage", 0.0)))
    deduped: List[Dict[str, float]] = []
    for frame in cleaned:
        pct = float(frame.get("route_percentage", 0.0))
        if deduped and abs(deduped[-1].get("route_percentage", 0.0) - pct) < 1e-3:
            deduped[-1].update({k: v for k, v in frame.items() if k != "route_percentage"})
        else:
            deduped.append(frame)

    if deduped[0].get("route_percentage", 0.0) > 0.0:
        deduped.insert(0, {"route_percentage": 0.0, **deduped[0]})
    if deduped[-1].get("route_percentage", 0.0) < 100.0:
        deduped.append({"route_percentage": 100.0, **deduped[-1]})

    # Ensure each frame has all parameters so playback can interpolate reliably.
    for frame in deduped:
        for spec in WEATHER_PARAMETER_SPECS:
            if spec.name not in frame:
                frame[spec.name] = float(processor._weather_state.get(spec.name, spec.min_value))
    return deduped

def _weather_keyframes_for_save(processor) -> List[Dict[str, float]]:
    """Build weather keyframes payload for scenario save.

    Weather storage is unified on SceneModel, so the historical
    "prefer editor copy, fall back to processor copy" dance is gone:
    both names were forwarders to the same list.
    """
    return _sanitize_weather_keyframes(processor, processor._weather_keyframes)

def _serialize_waypoints_for_vehicle(processor, vehicle_id, *, start_location: Optional[carla.Transform] = None):
    """Return serialized waypoint payload and destination speed for a vehicle.

    start_location is ignored for serialization; waypoint list begins with the first user-placed waypoint.
    """
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints:
        return [], processor.get_vehicle_destination_speed(vehicle_id)

    serialized = []
    destination_speed = processor.get_vehicle_destination_speed(vehicle_id)
    is_ego = processor.is_ego_vehicle(vehicle_id)
    default_speed = 40 if is_ego else 50
    total_waypoints = len(waypoints)

    for index, waypoint in enumerate(waypoints):
        is_last = index == total_waypoints - 1
        waypoint_data = {
            'index': "destination" if is_last else index + 1,
            'location': {
                'x': waypoint['x'],
                'y': waypoint['y'],
                'z': waypoint['z'],
            },
            'yaw': waypoint.get('yaw', None),
            'speed_km_h': waypoint.get('speed_km_h', default_speed),
            'speed_deviation_km_h': waypoint.get('speed_deviation_km_h', 0),
            'idle_time_s': waypoint.get('idle_time_s', 0.0),
            'auto_generated': waypoint.get('auto_generated', False),
            'is_destination': is_last,
        }
        try:
            waypoint_data['speed_km_h'] = float(waypoint_data['speed_km_h'])
        except Exception:
            waypoint_data['speed_km_h'] = default_speed
        if waypoint_data['speed_km_h'] <= 0.0:
            waypoint_data['speed_km_h'] = default_speed
        try:
            deviation_value = int(float(waypoint_data.get('speed_deviation_km_h', 0) or 0))
        except Exception:
            deviation_value = 0
        if deviation_value < 0:
            deviation_value = 0
        waypoint_data['speed_deviation_km_h'] = deviation_value
        serialized.append(waypoint_data)
        if is_last:
            destination_speed = waypoint_data['speed_km_h']

    if destination_speed is None:
        destination_speed = default_speed
    return serialized, destination_speed

def _build_vehicle_snapshot(processor, vehicle):
    """Serialize a single actor for scenario saving."""
    if not vehicle or not vehicle.is_alive:
        return None, None

    stored = processor.vehicle_transforms.get(vehicle.id)
    transform = stored if stored is not None else vehicle.get_transform()

    if processor.is_ego_vehicle(vehicle.id):
        payload = {
            'type': vehicle.type_id,
            'location': {
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z,
            },
            'rotation': {
                'pitch': transform.rotation.pitch,
                'yaw': transform.rotation.yaw,
                'roll': transform.rotation.roll,
            },
            'role': 'ego_vehicle',
        }
        color = processor.get_vehicle_color(vehicle.id)
        if color is not None:
            payload['color'] = color
        ignore_flags = processor.get_vehicle_ignore_flags(vehicle.id)
        payload['ignore_traffic_lights'] = ignore_flags['traffic_lights']
        payload['ignore_stop_signs'] = ignore_flags['stop_signs']
        payload['ignore_vehicles'] = ignore_flags['vehicles']
        payload['max_lat_acc'] = float(processor.get_vehicle_max_lat_acc(vehicle.id, 3.0))
        serialized_waypoints, destination_speed = processor._serialize_waypoints_for_vehicle(
            vehicle.id,
            start_location=transform,
        )
        if serialized_waypoints:
            payload['waypoints'] = serialized_waypoints
        if destination_speed is not None:
            payload['destination_speed_km_h'] = destination_speed
        return None, payload

    vehicle_data = {
        'id': vehicle.id,
        'type': vehicle.type_id,
        'location': {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
        },
        'rotation': {
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'roll': transform.rotation.roll,
        },
        'speed_km_h': processor.get_vehicle_speed(vehicle.id, 50),
        'idle_time_s': processor.get_actor_idle_time(vehicle.id, 0.0),
        'waypoints': [],
    }

    color = processor.get_vehicle_color(vehicle.id)
    if color is not None:
        vehicle_data['color'] = color

    ignore_flags = processor.get_vehicle_ignore_flags(vehicle.id)
    vehicle_data['ignore_traffic_lights'] = ignore_flags['traffic_lights']
    vehicle_data['ignore_stop_signs'] = ignore_flags['stop_signs']
    vehicle_data['ignore_vehicles'] = ignore_flags['vehicles']
    if not vehicle.type_id.startswith('walker.'):
        vehicle_data['max_lat_acc'] = float(processor.get_vehicle_max_lat_acc(vehicle.id, 3.0))

    serialized_waypoints, destination_speed = processor._serialize_waypoints_for_vehicle(
        vehicle.id,
        start_location=transform if processor.is_ego_vehicle(vehicle.id) else None,
    )
    if serialized_waypoints:
        vehicle_data['waypoints'] = serialized_waypoints
    if destination_speed is not None:
        vehicle_data['destination_speed_km_h'] = destination_speed

    trigger_center = None
    trigger_radius = None
    if vehicle.type_id.startswith('walker.'):
        trigger_center = processor.pedestrian_trigger_centers.get(vehicle.id)
        trigger_radius = processor.pedestrian_trigger_radii.get(vehicle.id)
    else:
        trigger_center = processor.vehicle_trigger_centers.get(vehicle.id)
        trigger_radius = processor.vehicle_trigger_radii.get(vehicle.id)

    if trigger_center and trigger_radius is not None:
        vehicle_data['trigger'] = {
            'center': {
                'x': float(trigger_center['x']),
                'y': float(trigger_center['y']),
                'z': float(trigger_center['z']),
            },
            'radius': float(trigger_radius),
        }

    return vehicle_data, None

def _serialize_trigger(processor):
    """Return serialized trigger payload if present."""
    if not processor.triggers:
        return None
    trigger = processor.triggers[0]
    return {
        'location': {
            'x': trigger['x'],
            'y': trigger['y'],
            'z': trigger['z'],
        },
        'radius': trigger['radius'],
    }

def _serialize_traffic_light_triggers(processor) -> Optional[List[Dict[str, object]]]:
    """Return serialized traffic light trigger payloads based on the stable trigger store."""
    entries: List[Dict[str, object]] = []
    for key, center in processor.traffic_light_trigger_centers.items():
        radius = processor.traffic_light_trigger_radii.get(key)
        if not center or radius is None:
            continue

        payload: Dict[str, object] = {
            'center': {
                'x': float(center.get('x', 0.0)),
                'y': float(center.get('y', 0.0)),
                'z': float(center.get('z', 0.0)),
            },
            'radius': float(radius),
        }

        group = processor._find_traffic_light_group_by_key(key)
        if group:
            payload['ids_live'] = sorted(group.ids)
            payload['ids_reference'] = sorted(group.reference_ids or set(group.ids))
            if group.location_fingerprint is None:
                group.location_fingerprint = processor._compute_traffic_light_fingerprint(group.lights)
            if group.location_fingerprint:
                payload['fingerprint'] = [
                    [int(pt[0]), int(pt[1]), int(pt[2])] for pt in group.location_fingerprint
                ]
            seq = group.sequence
            if not seq:
                seq = processor.traffic_light_sequences.get(frozenset(group.ids), [])
            if seq:
                payload['sequence'] = processor._normalize_traffic_light_sequence(seq)
        else:
            # Persist fallback identifiers when no live group is available
            if key[0] == 'fp':
                payload['fingerprint'] = [
                    [int(pt[0]), int(pt[1]), int(pt[2])] for pt in key[1]
                ]
            elif key[0] == 'ids':
                payload['ids_reference'] = list(key[1])

        entries.append(payload)

    return entries or None

def _collect_scenario_snapshot(processor, map_name):
    """Create the scenario data payload for saving."""
    save_data = {
        'vehicles': [],
        'timestamp': time.time(),
        'map_name': map_name,
    }
    ego_entry = None

    for vehicle in processor.spawned_vehicles:
        vehicle_data, maybe_ego = _build_vehicle_snapshot(processor, vehicle)
        if maybe_ego is not None:
            ego_entry = maybe_ego
            continue
        if vehicle_data is not None:
            save_data['vehicles'].append(vehicle_data)

    if ego_entry is None:
        ego_entry = processor.get_ego_vehicle_data()
    if ego_entry:
        save_data['ego_vehicle'] = ego_entry

    for group in processor.traffic_light_groups:
        processor._cache_traffic_light_sequence(group)
    trigger_payload = _serialize_trigger(processor)
    if trigger_payload:
        save_data['trigger'] = trigger_payload

    traffic_light_payloads = _serialize_traffic_light_triggers(processor)
    if traffic_light_payloads:
        save_data['traffic_light_triggers'] = traffic_light_payloads

    if processor.vehicle_control_mode != "basic_agent":
        save_data['vehicle_control_mode'] = processor.vehicle_control_mode

    return save_data, ego_entry

def save_waypoint_data_to_file(processor, filename):
    """Save all waypoint data to a JSON file"""
    try:
        # Check if filename is valid
        if not filename or not filename.strip():
            print("Error: Empty filename provided for save operation")
            print("Call stack:")
            traceback.print_stack()
            return False

        # Get map name (strip path, keep only the map name)
        map_obj = processor._get_cached_map(refresh=False)
        if map_obj:
            full_map_name = map_obj.name
            map_name = full_map_name.split('/')[-1] if '/' in full_map_name else full_map_name
        else:
            full_map_name = "unknown"
            map_name = "unknown"

        weather_keyframes = _weather_keyframes_for_save(processor)
        save_data, ego_entry = processor._collect_scenario_snapshot(map_name)
        save_data['weather_keyframes'] = weather_keyframes

        # Ensure directory exists (only if there's a directory path)
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # Write JSON to file
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Saved waypoint data for {len(save_data['vehicles'])} vehicles to {filename}")
        if ego_entry:
            print("Included ego vehicle in scenario data")
        if 'trigger' in save_data:
            print(f"Saved trigger to scenario")
        if save_data.get('traffic_light_triggers'):
            print(
                f"Saved {len(save_data['traffic_light_triggers'])} "
                f"traffic light trigger(s) to scenario"
            )

        return True
        
    except Exception as e:
        print(f"Error saving waypoint data: {e}")
        return False

def _normalize_traffic_light_trigger_entries(processor, trigger_entries):
    """Normalize legacy traffic light trigger payloads in-place."""
    if not isinstance(trigger_entries, list):
        return trigger_entries
    for entry in trigger_entries:
        if not isinstance(entry, dict):
            continue
        if 'ids_live' not in entry and 'ids' in entry:
            legacy_ids = entry.pop('ids')
            entry['ids_live'] = legacy_ids
            entry.setdefault('ids_reference', list(legacy_ids))
    return trigger_entries

def load_waypoint_data_from_file(
    processor,
    filename,
    *,
    preserve_camera: bool = False,
    skip_ego_spawn: bool = False,
    apply_to_actor_only: bool = False,
    external_ego_actor: Optional[carla.Actor] = None,
    preserved_actor_id: Optional[int] = None,
):
    """Load waypoint data from a JSON file"""
    try:
        editor = getattr(processor, "editor", None)
        if editor and hasattr(editor, "_debug_camera_pose"):
            try:
                editor._debug_camera_pose(
                    f"load-waypoints-start preserve_camera={preserve_camera} skip_ego_spawn={skip_ego_spawn} apply_to_actor_only={apply_to_actor_only}"
                )
            except Exception:
                pass

        with open(filename, 'r') as f:
            load_data = json.load(f)

        # Store loaded data for later reference (needed for pedestrian restoration)
        processor.loaded_scenario_data = load_data
        ego_data = load_data.get('ego_vehicle')

        print(f"Loading scenario from {filename}...")
        print(f"Map: {load_data.get('map_name', 'Unknown')}")

        # If we are skipping ego spawn because an external ego is active, reload others while preserving the external.
        if skip_ego_spawn and not apply_to_actor_only:
            external_actor = processor.session.external_ego_actor
            if external_actor and external_actor.is_alive and ego_data:
                try:
                    _apply_ego_data_to_actor(processor, ego_data, external_actor)
                    processor.waypoint_display_vehicle_id = external_actor.id
                    processor.selected_vehicle = external_actor
                    processor.selected_vehicle_is_pedestrian = False
                    processor.refresh_waypoints_carla_debug()
                except Exception as exc:
                    print(f"Failed to apply ego data to external actor during skip_ego_spawn load: {exc}")
                preserved_actor_id = external_actor.id
            elif preserved_actor_id is not None:
                preserved_actor_id = preserved_actor_id
            else:
                # Skip full reload/reset when skipping ego spawn but no external actor to target
                return True

            preserved_waypoints = processor.get_vehicle_waypoints(preserved_actor_id) if preserved_actor_id else []
            processor.cleanup_all_vehicles(preserve_ids={preserved_actor_id} if preserved_actor_id else set())
            # Clear non-ego waypoint/metadata entries
            for vid in list(processor.waypoint_list.keys()):
                if vid != preserved_actor_id:
                    processor.clear_vehicle_waypoints(vid)
            # Clear triggers/traffic-light state
            processor.triggers.clear()
            processor.traffic_light_trigger_centers.clear()
            processor.traffic_light_trigger_radii.clear()
            processor.traffic_light_sequences.clear()
            processor._traffic_light_group_snapshots.clear()
            processor._scenario_active_traffic_light_trigger = None
            processor._last_visible_traffic_light_trigger_key = None
            processor.scaling_traffic_light_trigger = False
            processor._traffic_light_scaling_group = None
            processor.traffic_light_menu_position = None
            for group in processor.traffic_light_groups:
                group.trigger_center = None
                group.trigger_radius = None
            processor.clear_traffic_light_selection()

            context = {
                'first_vehicle_json_location': None,
                'first_vehicle_spawn_location': None,
                'first_ped_json_location': None,
                'first_ped_spawn_location': None,
            }
            blueprint_library = processor.world.get_blueprint_library()
            for vehicle_data in load_data.get('vehicles', []):
                _load_vehicle_entry(processor, vehicle_data, blueprint_library, ego_data, context)

            _apply_trigger_data(processor, load_data.get('trigger'))

            processor.vehicle_control_mode = load_data.get('vehicle_control_mode', 'basic_agent')

            triggers_payload = _normalize_traffic_light_trigger_entries(processor, 
                load_data.get('traffic_light_triggers')
            )
            _apply_traffic_light_trigger_data(processor, triggers_payload)

            if preserved_actor_id is not None:
                if preserved_waypoints:
                    try:
                        processor.set_vehicle_waypoints(preserved_actor_id, preserved_waypoints)
                    except Exception:
                        pass
                processor.waypoint_display_vehicle_id = preserved_actor_id
                processor.selected_vehicle = processor.get_spawned_vehicle(preserved_actor_id) or external_actor
                processor.selected_vehicle_is_pedestrian = False
                try:
                    processor.refresh_waypoints_carla_debug()
                except Exception:
                    pass
            return True

        # If we only need to apply ego data to an existing actor, do it and return
        if apply_to_actor_only:
            if ego_data and external_ego_actor:
                try:
                    _apply_ego_data_to_actor(processor, ego_data, external_ego_actor)
                    processor.waypoint_display_vehicle_id = external_ego_actor.id
                    processor.selected_vehicle = external_ego_actor
                    processor.selected_vehicle_is_pedestrian = False
                    processor.refresh_waypoints_carla_debug()
                except Exception as exc:
                    print(f"Failed to apply ego data to external actor: {exc}")
            return True

        context = {
            'first_vehicle_json_location': None,
            'first_vehicle_spawn_location': None,
            'first_ped_json_location': None,
            'first_ped_spawn_location': None,
        }
        processor._reset_scenario_state()

        blueprint_library = processor.world.get_blueprint_library()
        
        for vehicle_data in load_data.get('vehicles', []):
            _load_vehicle_entry(processor, vehicle_data, blueprint_library, ego_data, context)
        
        print(f"Successfully loaded {len(processor.spawned_vehicles)} vehicles")

        ego_spawn_location = None
        if ego_data and not skip_ego_spawn:
            ego_spawn_location = _spawn_ego_from_data(processor, ego_data, blueprint_library)

        _apply_trigger_data(processor, load_data.get('trigger'))

        processor.vehicle_control_mode = load_data.get('vehicle_control_mode', 'basic_agent')

        triggers_payload = _normalize_traffic_light_trigger_entries(processor, 
            load_data.get('traffic_light_triggers')
        )
        _apply_traffic_light_trigger_data(processor, triggers_payload)

        target_location = _resolve_focus_location(processor, context, ego_spawn_location)
        has_actor_focus = bool(
            context['first_vehicle_spawn_location']
            or context['first_vehicle_json_location']
            or context['first_ped_spawn_location']
            or context['first_ped_json_location']
        )

        if target_location and (not preserve_camera or not has_actor_focus):
            auto_camera_ok = not (processor.session.camera_stream_enabled is False)
            if auto_camera_ok:
                processor.focus_camera_on_location(target_location)
                if editor and hasattr(editor, "_debug_camera_pose"):
                    try:
                        editor._debug_camera_pose("load-waypoints-focus")
                    except Exception:
                        pass

        return True

    except Exception as e:
        print(f"Error loading waypoint data: {e}")
        return False

def _load_vehicle_entry(processor, vehicle_data, blueprint_library, ego_data, context):
    """Spawn and configure a single vehicle entry from loaded scenario data."""
    try:
        if vehicle_data.get('role') == 'ego':
            return

        if ego_data and vehicle_data.get('role') is None and vehicle_data.get('type') == ego_data.get('type'):
            loc = vehicle_data.get('location', {})
            ego_loc = ego_data.get('location', {})
            if loc.get('x') == ego_loc.get('x') and loc.get('y') == ego_loc.get('y'):
                return

        vehicle_bp = blueprint_library.find(vehicle_data['type'])
        if 'color' in vehicle_data and not vehicle_data['type'].startswith('walker.'):
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', vehicle_data['color'])

        location = carla.Location(
            vehicle_data['location']['x'],
            vehicle_data['location']['y'],
            vehicle_data['location']['z'],
        )
        rotation = carla.Rotation(
            vehicle_data['rotation']['pitch'],
            vehicle_data['rotation']['yaw'],
            vehicle_data['rotation']['roll'],
        )
        transform = carla.Transform(location, rotation)

        if vehicle_data['type'].startswith('walker.'):
            if context['first_ped_json_location'] is None:
                context['first_ped_json_location'] = carla.Location(location.x, location.y, location.z)
        else:
            if context['first_vehicle_json_location'] is None:
                context['first_vehicle_json_location'] = carla.Location(location.x, location.y, location.z)

        vehicle = processor.world.try_spawn_actor(vehicle_bp, transform)
        if not vehicle:
            print(f"Failed to spawn vehicle {vehicle_data['type']}")
            return

        vehicle.set_simulate_physics(False)
        processor.spawned_vehicles.append(vehicle)

        default_speed = 5 if vehicle_data.get('type', '').startswith('walker.') else 50
        speed_value = vehicle_data.get('speed_km_h', default_speed)
        idle_value = vehicle_data.get('idle_time_s', 0.0)
        color = vehicle_data.get('color')
        ignore_flags = {
            'traffic_lights': vehicle_data.get('ignore_traffic_lights', False),
            'stop_signs': vehicle_data.get('ignore_stop_signs', False),
            'vehicles': vehicle_data.get('ignore_vehicles', False),
        }
        max_lat_acc_value = float(vehicle_data.get('max_lat_acc', 3.0))
        if max_lat_acc_value <= 0.0:
            max_lat_acc_value = 3.0

        print(f"Vehicle loaded at JSON Z: {location.z:.2f}")

        if vehicle_data['type'].startswith('walker.'):
            # Loading repairs floating walkers: old files may carry an inflated
            # Z from the pre-Phase-8 drag/spawn bug. Only trust a genuine
            # raycast hit, and only correct beyond the tolerance (healthy
            # walkers differ from a fresh sample by <2 cm; bug deltas are
            # >=0.85 m) so grounded scenarios are never touched. On a miss the
            # saved Z is kept verbatim (never worse than before). The corrected
            # Z flows into vehicle_transforms below, so re-saving repairs the
            # file (playback keeps trusting the saved Z by design).
            try:
                ground_sample = processor.get_ground_height_at_location(
                    location.x, location.y,
                    reference_z=location.z,
                    return_metadata=True,
                )
                if (ground_sample.get('source') == 'raycast'
                        and ground_sample.get('height') is not None):
                    bbox_z = float(getattr(vehicle.bounding_box.extent, 'z', 1.0))
                    corrected_z = (ground_sample['height'] + bbox_z
                                   + PEDESTRIAN_PLACEMENT_CLEARANCE_M)
                    tolerance = getattr(processor, '_ped_load_reground_tolerance_m', 0.2)
                    # Downward-only: the drag/spawn bug only ever RAISED
                    # walkers, so genuine repairs always lower Z. An upward
                    # "correction" means the first surface the ray hit is
                    # above the walker (tree canopy / overhang on city maps,
                    # proven on tartu_demo) — never trust it.
                    if location.z - corrected_z > tolerance:
                        print(f"[Ped Ground] corrected walker {vehicle.id} "
                              f"z {location.z:.2f} -> {corrected_z:.2f}")
                        location = carla.Location(location.x, location.y, corrected_z)
                        transform = carla.Transform(location, rotation)
                        vehicle.set_transform(transform)
            except Exception as exc:
                print(f"[Ped Ground] load re-ground skipped: {exc}")

        actual_location = vehicle.get_location()
        if vehicle_data['type'].startswith('walker.'):
            if context['first_ped_spawn_location'] is None:
                context['first_ped_spawn_location'] = carla.Location(
                    actual_location.x, actual_location.y, actual_location.z
                )
        else:
            if context['first_vehicle_spawn_location'] is None:
                context['first_vehicle_spawn_location'] = carla.Location(
                    actual_location.x, actual_location.y, actual_location.z
                )

        waypoints = []
        for wp_data in vehicle_data.get('waypoints', []):
            deviation_val = wp_data.get('speed_deviation_km_h', 0)
            try:
                deviation_val = int(float(deviation_val or 0))
            except Exception:
                deviation_val = 0
            if deviation_val < 0:
                deviation_val = 0
            waypoints.append({
                'x': wp_data['location']['x'],
                'y': wp_data['location']['y'],
                'z': wp_data['location']['z'],
                'index': wp_data.get('index'),
                'yaw': wp_data.get('yaw', None),
                'speed_km_h': wp_data.get('speed_km_h', 50),
                'speed_deviation_km_h': deviation_val,
                'idle_time_s': wp_data.get('idle_time_s', 0.0),
                'is_destination': wp_data.get('is_destination', False),
            })

        destination_speed_value = vehicle_data.get('destination_speed_km_h')
        processor.initialize_vehicle_metadata(
            vehicle.id,
            speed=speed_value,
            destination_speed=destination_speed_value,
            idle_time=idle_value,
            color=color,
            ignore_flags=ignore_flags,
            max_lat_acc=max_lat_acc_value,
        )
        processor.vehicle_transforms[vehicle.id] = carla.Transform(location, rotation)

        if waypoints:
            for idx, waypoint in enumerate(waypoints):
                is_last = idx == len(waypoints) - 1
                waypoint['index'] = "destination" if is_last else idx + 1
                waypoint['is_destination'] = is_last
                if is_last and destination_speed_value is not None:
                    waypoint['speed_km_h'] = destination_speed_value

            processor.set_vehicle_waypoints(vehicle.id, waypoints)
            processor._cache_destination_speed(vehicle.id)
            print(f"Loaded vehicle {vehicle_data['type']} with {len(waypoints)} waypoints")
        else:
            processor.clear_vehicle_waypoints(vehicle.id)
            processor.set_vehicle_destination_speed(vehicle.id, destination_speed_value)
            print(f"Loaded vehicle {vehicle_data['type']} with no waypoints")

        if 'trigger' in vehicle_data:
            trigger_data = vehicle_data['trigger']
            if 'center' in trigger_data and 'radius' in trigger_data:
                center = trigger_data['center']
                radius = float(trigger_data['radius'])
                if vehicle_data['type'].startswith('walker.'):
                    processor.pedestrian_trigger_centers[vehicle.id] = {
                        'x': float(center['x']),
                        'y': float(center['y']),
                        'z': float(center['z']),
                    }
                    processor.pedestrian_trigger_radii[vehicle.id] = max(MIN_PERSONAL_TRIGGER_RADIUS, radius)
                    print(f"Loaded pedestrian trigger: radius {radius:.2f}m")
                else:
                    if not processor.is_ego_vehicle(vehicle.id):
                        processor.vehicle_trigger_centers[vehicle.id] = {
                            'x': float(center['x']),
                            'y': float(center['y']),
                            'z': float(center['z']),
                        }
                        processor.vehicle_trigger_radii[vehicle.id] = max(MIN_PERSONAL_TRIGGER_RADIUS, radius)
                        print(f"Loaded vehicle trigger: radius {radius:.2f}m")

    except Exception as exc:
        print(f"Error loading vehicle {vehicle_data.get('type', 'unknown')}: {exc}")

def _apply_ego_data_to_actor(processor, ego_data: dict, actor: carla.Actor) -> None:
    """Apply ego metadata and waypoints to an existing actor."""
    if not ego_data or not actor or not actor.is_alive:
        return

    ego_default_speed = 40.0
    try:
        ego_default_speed = float(ego_data.get('speed_km_h', ego_default_speed) or ego_default_speed)
    except Exception:
        ego_default_speed = 40.0
    try:
        destination_speed_value = float(ego_data.get('destination_speed_km_h')) if ego_data.get('destination_speed_km_h') is not None else None
    except Exception:
        destination_speed_value = None
    ignore_flags = {
        'traffic_lights': bool(ego_data.get('ignore_traffic_lights', False)),
        'stop_signs': bool(ego_data.get('ignore_stop_signs', False)),
        'vehicles': bool(ego_data.get('ignore_vehicles', False)),
    }
    try:
        max_lat_acc_value = float(ego_data.get('max_lat_acc', 3.0) or 3.0)
    except Exception:
        max_lat_acc_value = 3.0
    if max_lat_acc_value <= 0.0:
        max_lat_acc_value = 3.0
    ego_color = ego_data.get('color')

    processor.initialize_vehicle_metadata(
        actor.id,
        speed=ego_default_speed,
        destination_speed=destination_speed_value,
        idle_time=0.0,            color=ego_color,
        ignore_flags=ignore_flags,
        max_lat_acc=max_lat_acc_value,
    )

    raw_waypoints = ego_data.get('waypoints', [])
    waypoints: List[WaypointData] = []
    if isinstance(raw_waypoints, list):
        total = len(raw_waypoints)
        for idx, wp_data in enumerate(raw_waypoints):
            waypoint_entry = {
                'x': wp_data.get('location', {}).get('x', 0.0),
                'y': wp_data.get('location', {}).get('y', 0.0),
                'z': wp_data.get('location', {}).get('z', 0.0),
                'index': wp_data.get('index'),
                'yaw': wp_data.get('yaw', None),
                'speed_km_h': wp_data.get('speed_km_h', ego_default_speed),
                'idle_time_s': wp_data.get('idle_time_s', 0.0),
                'is_destination': wp_data.get('is_destination', idx == total - 1),
            }
            deviation_val = wp_data.get('speed_deviation_km_h', 0)
            try:
                deviation_val = int(float(deviation_val or 0))
            except Exception:
                deviation_val = 0
            if deviation_val < 0:
                deviation_val = 0
            waypoint_entry['speed_deviation_km_h'] = deviation_val
            try:
                speed_val = float(waypoint_entry['speed_km_h'])
            except Exception:
                speed_val = ego_default_speed
            if speed_val <= 0.0:
                speed_val = ego_default_speed
            waypoint_entry['speed_km_h'] = speed_val
            if waypoint_entry['is_destination'] and destination_speed_value is not None:
                waypoint_entry['speed_km_h'] = destination_speed_value
            waypoints.append(waypoint_entry)

    if waypoints:
        processor.set_vehicle_waypoints(actor.id, waypoints)
        processor._cache_destination_speed(actor.id)
    else:
        processor.clear_vehicle_waypoints(actor.id)
        processor.set_vehicle_destination_speed(actor.id, destination_speed_value)

def _spawn_ego_from_data(processor, ego_data, blueprint_library):
    """Spawn ego vehicle from loaded scenario data."""
    ego_spawn_location = None
    if not ego_data:
        return None

    try:
        destination_speed_value = float(ego_data.get('destination_speed_km_h')) if ego_data.get('destination_speed_km_h') is not None else None
    except Exception:
        destination_speed_value = None
    ego_default_speed = 40.0
    try:
        ego_default_speed = float(ego_data.get('speed_km_h', ego_default_speed) or ego_default_speed)
    except Exception:
        ego_default_speed = 40.0
    ignore_flags = {
        'traffic_lights': bool(ego_data.get('ignore_traffic_lights', False)),
        'stop_signs': bool(ego_data.get('ignore_stop_signs', False)),
        'vehicles': bool(ego_data.get('ignore_vehicles', False)),
    }
    try:
        max_lat_acc_value = float(ego_data.get('max_lat_acc', 3.0) or 3.0)
    except Exception:
        max_lat_acc_value = 3.0
    if max_lat_acc_value <= 0.0:
        max_lat_acc_value = 3.0
    ego_color = ego_data.get('color')

    def _apply_loaded_ego_waypoints(target_vehicle_id: Optional[int]) -> None:
        if target_vehicle_id is None:
            return
        raw_waypoints = ego_data.get('waypoints', [])
        waypoints: List[WaypointData] = []
        if isinstance(raw_waypoints, list):
            total = len(raw_waypoints)
            for idx, wp_data in enumerate(raw_waypoints):
                waypoint_entry = {
                    'x': wp_data.get('location', {}).get('x', 0.0),
                    'y': wp_data.get('location', {}).get('y', 0.0),
                    'z': wp_data.get('location', {}).get('z', 0.0),
                    'index': wp_data.get('index'),
                    'yaw': wp_data.get('yaw', None),
                    'speed_km_h': wp_data.get('speed_km_h', ego_default_speed),
                    'idle_time_s': wp_data.get('idle_time_s', 0.0),
                    'is_destination': wp_data.get('is_destination', idx == total - 1),
                }
                deviation_val = wp_data.get('speed_deviation_km_h', 0)
                try:
                    deviation_val = int(float(deviation_val or 0))
                except Exception:
                    deviation_val = 0
                if deviation_val < 0:
                    deviation_val = 0
                waypoint_entry['speed_deviation_km_h'] = deviation_val
                try:
                    speed_val = float(waypoint_entry['speed_km_h'])
                except Exception:
                    speed_val = ego_default_speed
                if speed_val <= 0.0:
                    speed_val = ego_default_speed
                waypoint_entry['speed_km_h'] = speed_val
                waypoints.append(waypoint_entry)

        if waypoints:
            for idx, waypoint in enumerate(waypoints):
                is_last = idx == len(waypoints) - 1
                waypoint['index'] = "destination" if is_last else idx + 1
                waypoint['is_destination'] = is_last
                if is_last and destination_speed_value is not None:
                    waypoint['speed_km_h'] = destination_speed_value
            processor.set_vehicle_waypoints(target_vehicle_id, waypoints)
            processor._cache_destination_speed(target_vehicle_id)
            print(f"Loaded ego vehicle with {len(waypoints)} waypoints")
        elif destination_speed_value is not None:
            processor.set_vehicle_destination_speed(target_vehicle_id, destination_speed_value)

    try:
        editor = getattr(processor, 'editor', None)
        if editor:
            try:
                editor._detect_external_ego_vehicle()
                editor._refresh_external_ego_actor_reference()
            except Exception as exc:
                print(f"[Scenario] Warning: failed to refresh external ego before spawn ({exc})")
        processor._cleanup_leftover_ego_actor()

        # Always spawn a fresh placeholder ego — the external ego (if present)
        # is kept separate and only used during scenario playback.
        vehicle_bp = blueprint_library.find(ego_data.get('type', 'vehicle.lexus.utlexus'))
        loc_data = ego_data.get('location', {})
        rot_data = ego_data.get('rotation', {})
        base_location = carla.Location(
            loc_data.get('x', 0.0),
            loc_data.get('y', 0.0),
            loc_data.get('z', 0.0),
        )
        rotation = carla.Rotation(
            rot_data.get('pitch', 0.0),
            rot_data.get('yaw', 0.0),
            rot_data.get('roll', 0.0),
        )
        spawn_location = carla.Location(base_location.x, base_location.y, base_location.z + 0.1)
        transform = carla.Transform(spawn_location, rotation)

        ego_actor = processor.world.try_spawn_actor(vehicle_bp, transform)
        if ego_actor:
            ego_actor.set_simulate_physics(False)
            processor.spawned_vehicles.append(ego_actor)
            processor.initialize_vehicle_metadata(
                ego_actor.id,
                speed=ego_default_speed,
                destination_speed=destination_speed_value,
                idle_time=float(ego_data.get('idle_time_s', 0.0) or 0.0),                    color=ego_color,
                ignore_flags=ignore_flags,
                max_lat_acc=max_lat_acc_value,
            )
            processor.register_ego_vehicle(ego_actor, transform)
            _apply_loaded_ego_waypoints(ego_actor.id)
            ego_spawn_location = ego_actor.get_location()
            print(f"Loaded ego vehicle at ({spawn_location.x:.2f}, {spawn_location.y:.2f}, {spawn_location.z:.2f})")
        else:
            fallback_actor = processor.get_ego_vehicle_actor()
            if fallback_actor and fallback_actor.is_alive:
                print(f"Reusing existing ego actor {fallback_actor.id} for scenario spawn")
                external_id = None
                try:
                    external_id = (processor.session.external_ego_actor_id
                                   or processor.session._external_swap_current_id)
                except Exception:
                    external_id = None
                if external_id is not None and fallback_actor.id == external_id:
                    try:
                        fallback_actor.set_simulate_physics(True)
                    except Exception:
                        pass
                else:
                    fallback_actor.set_simulate_physics(False)
                fallback_actor.set_transform(transform)
                processor.spawned_vehicles.append(fallback_actor)
                processor.initialize_vehicle_metadata(
                    fallback_actor.id,
                    speed=ego_default_speed,
                    destination_speed=destination_speed_value,
                    idle_time=float(ego_data.get('idle_time_s', 0.0) or 0.0),                        color=ego_color,
                    ignore_flags=ignore_flags,
                    max_lat_acc=max_lat_acc_value,
                )
                processor.register_ego_vehicle(fallback_actor, transform)
                _apply_loaded_ego_waypoints(fallback_actor.id)
                ego_spawn_location = fallback_actor.get_location()
            else:
                processor.ego_vehicle_transform = transform
                processor.ego_vehicle_blueprint = ego_data.get('type', 'vehicle.lexus.utlexus')
                processor.ego_vehicle_id = None
                ego_spawn_location = carla.Location(base_location.x, base_location.y, base_location.z)
                print(f"Warning: Failed to spawn ego vehicle at ({base_location.x:.2f}, {base_location.y:.2f}, {base_location.z:.2f}). Location might be blocked.")
    except Exception as exc:
        processor.ego_vehicle_transform = carla.Transform(spawn_location, rotation)
        processor.ego_vehicle_blueprint = ego_data.get('type', 'vehicle.lexus.utlexus')
        processor.ego_vehicle_id = None
        print(f"Error loading ego vehicle: {exc}")

    return ego_spawn_location

def _apply_trigger_data(processor, trigger_data):
    """Restore trigger from scenario data."""
    if not trigger_data:
        print("No trigger found in scenario")
        return
    try:
        loc = trigger_data.get('location', {})
        trigger = {
            'x': loc.get('x', 0.0),
            'y': loc.get('y', 0.0),
            'z': loc.get('z', 0.0),
            'radius': trigger_data.get('radius', 10.0),
        }
        processor.triggers.append(trigger)
        print(f"Successfully loaded trigger at ({trigger['x']:.2f}, {trigger['y']:.2f}, {trigger['z']:.2f})")
    except Exception as trigger_exc:
        print(f"Error loading trigger: {trigger_exc}")

def _apply_traffic_light_trigger_data(processor, trigger_entries) -> None:
    """Restore traffic light group triggers from saved scenario data."""
    if not trigger_entries:
        print("No traffic light triggers in scenario")
        return

    if not isinstance(trigger_entries, list):
        print("Warning: Traffic light trigger payload is malformed; expected a list.")
        return

    try:
        processor._refresh_traffic_lights()
    except Exception as exc:
        print(f"Warning: Unable to refresh traffic lights before applying triggers ({exc})")

    if not processor.traffic_light_groups:
        print("Warning: No traffic light groups available to apply trigger data.")
        return

    live_lookup = {frozenset(group.ids): group for group in processor.traffic_light_groups if group.ids}
    reference_lookup: Dict[frozenset, TrafficLightGroupData] = {}
    fingerprint_lookup: Dict[Tuple[Tuple[int, int, int], ...], TrafficLightGroupData] = {}
    for group in processor.traffic_light_groups:
        if group.reference_ids:
            reference_lookup[frozenset(group.reference_ids)] = group
        if group.location_fingerprint:
            fingerprint_lookup[group.location_fingerprint] = group
    restored_count = 0

    for entry in trigger_entries:
        if not isinstance(entry, dict):
            continue
        ids_live = entry.get('ids_live')
        ids_reference = entry.get('ids_reference')
        fingerprint_payload = entry.get('fingerprint')

        group: Optional[TrafficLightGroupData] = None

        def _make_set(values: Optional[Iterable[int]]) -> Optional[frozenset]:
            if not values:
                return None
            try:
                converted = [int(v) for v in values]
            except Exception:
                return None
            return frozenset(converted)

        # Prefer fingerprint matching to survive ID churn
        if group is None and fingerprint_payload:
            try:
                fingerprint_tuple = tuple(
                    (int(pt[0]), int(pt[1]), int(pt[2])) for pt in fingerprint_payload
                )
            except Exception:
                fingerprint_tuple = None
            if fingerprint_tuple:
                group = fingerprint_lookup.get(fingerprint_tuple)

        live_key = _make_set(ids_live)
        if group is None and live_key:
            group = live_lookup.get(live_key)

        if group is None:
            reference_key = _make_set(ids_reference or ids_live)
            if reference_key:
                group = reference_lookup.get(reference_key) or live_lookup.get(reference_key)

        if not group:
            ids_for_warning = ids_live or ids_reference or fingerprint_payload
            print(
                f"Warning: Traffic light trigger references IDs/fingerprint {ids_for_warning}, "
                "but no matching group was found."
            )
            group = None

        normalized_fp = processor._normalize_traffic_light_fingerprint(fingerprint_payload)
        center_payload = entry.get('center', {})
        radius_value = max(MIN_PERSONAL_TRIGGER_RADIUS, float(entry.get('radius', DEFAULT_PERSONAL_TRIGGER_RADIUS)))
        stored_center = {
            'x': float(center_payload.get('x', 0.0)),
            'y': float(center_payload.get('y', 0.0)),
            'z': float(center_payload.get('z', 0.0)),
        }
        key = processor._traffic_light_trigger_key(
            group=group,
            fingerprint=normalized_fp,
            ids=ids_reference or ids_live,
        )
        processor._set_traffic_light_trigger_data(
            stored_center,
            radius_value,
            key=key,
            group=group,
            mark_visible=False,
        )
        processor._cache_traffic_light_trigger_payload(group)
        reference_key = _make_set(ids_reference or ids_live)
        if group:
            if reference_key:
                group.reference_ids = set(reference_key)
            elif group.reference_ids is None:
                group.reference_ids = set(group.ids)
            if fingerprint_payload and not group.location_fingerprint:
                try:
                    group.location_fingerprint = tuple(
                        (int(pt[0]), int(pt[1]), int(pt[2])) for pt in fingerprint_payload
                    )
                except Exception:
                    group.location_fingerprint = group.location_fingerprint
        sequence_payload = entry.get('sequence', [])
        if group:
            if isinstance(sequence_payload, list):
                group.sequence = processor._normalize_traffic_light_sequence(
                    sequence_payload,
                    coerce_color=True,
                )
            else:
                group.sequence = []
        restored_count += 1

        if (
            processor.selected_traffic_light_group is not None
            and group.ids == processor.selected_traffic_light_group.ids
        ):
            processor._update_traffic_light_menu_anchor(group)

        processor._cache_traffic_light_sequence(group)

    if restored_count:
        print(f"Restored {restored_count} traffic light trigger(s) from scenario data")
    else:
        print("No traffic light triggers restored (no matching traffic light groups found)")

def _resolve_focus_location(processor, context, ego_spawn_location):
    """Determine a good camera target after loading a scenario."""
    return (
        ego_spawn_location
        or context['first_vehicle_spawn_location']
        or context['first_vehicle_json_location']
        or context['first_ped_spawn_location']
        or context['first_ped_json_location']
    )
