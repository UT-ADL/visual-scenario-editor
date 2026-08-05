"""Placement controller, part 1 (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-29).

Actor movement/rotation, waypoint creation/editing/click routing, destination
placement + auto-routing, vehicle selection/spawn/delete. Part 2 (step-30)
adds the trigger placement/scaling family to this module. All state lives on
the processor/scene; the processor keeps one-line delegates so editor input
dispatch, commands.py reach-ins (e.g. _adjust_pedestrian_spawn_orientation,
_cache_destination_speed) and lazy attrs (movement_snap_to_lane, set here via
setattr-on-processor semantics) behave unchanged.
"""

import copy
import math
import time
from typing import Optional, Tuple

import carla
import pygame

from vse_common.actor_cache import cached_bounding_box
from vse_common.geometry import get_ground_height
from vse_editor.carla_io.spawning import (
    PEDESTRIAN_PLACEMENT_CLEARANCE_M,
    VEHICLE_PLACEMENT_CLEARANCE_M,
    lowest_spawnable_z,
)
from vse_editor.constants import GROUP_MENU_ICONS, WAYPOINT_MARKER_Z_OFFSET
from vse_editor.commands import (
    CompositeCommand,
    DeleteVehicleCommand,
    DeleteWaypointCommand,
    MoveVehicleCommand,
    MoveWaypointCommand,
    PlaceWaypointCommand,
    SpawnVehicleCommand,
    SplitWaypointCommand,
    UpdateWaypointPropertyCommand,
)
from vse_editor.input_helpers import is_shift_pressed
from vse_editor.rendering.overlays import OverlayMenuRenderer


def start_auto_route_to_destination(processor, vehicle):
    """Enable destination placement mode and guide the user to select a target."""
    if not vehicle or not vehicle.is_alive:
        print("No valid vehicle selected for auto-routing.")
        return

    processor.creating_destination = True
    processor.destination_marker_coordinates = None
    processor.vehicle_menu_position = None  # Hide menu while placing destination

    print("Destination placement mode: Move the mouse and left-click to choose a destination for auto-routing. ESC or right-click to cancel.")

def vehicle_menu_icon_order(processor):
    """Icon row for the selected actor's floating menu.

    Single source of truth for BOTH the hit-test (check_menu_icon_click) and
    the render side (scene_render.render_vehicle_action_menu): the menu lays
    icons out purely by index, so the two lists must never diverge.
    """
    is_ego = processor.is_ego_vehicle(processor.selected_vehicle.id)
    icon_order = ['delete']
    if not processor.selected_vehicle_is_pedestrian:
        icon_order.append('rotate')
    if not is_ego:
        icon_order.append('waypoint')
    if not processor.selected_vehicle_is_pedestrian and not is_ego:
        icon_order.append('autoroute')
    if is_ego and not processor.selected_vehicle_is_pedestrian:
        icon_order.append('ego_destination')
    trigger_supported = False
    if processor.selected_vehicle_is_pedestrian:
        trigger_supported = True
    elif not is_ego:
        trigger_supported = True

    has_personal_trigger = False
    actor_id = processor.selected_vehicle.id
    if processor.selected_vehicle_is_pedestrian:
        has_personal_trigger = actor_id in processor.pedestrian_trigger_radii
    elif not is_ego:
        has_personal_trigger = actor_id in processor.vehicle_trigger_radii

    if trigger_supported:
        icon_order.append('add_trigger')
        if has_personal_trigger:
            icon_order.append('remove_trigger')
    if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
        # 3D view is view + select: keep only click-type icons that don't start a
        # placement or press-drag gesture (render and hit-test share this list).
        icon_order = [icon for icon in icon_order if icon in ('delete', 'remove_trigger')]
    return icon_order

def check_menu_icon_click(processor, mouse_x, mouse_y):
    """Check if a click hits any of the vehicle menu icons"""
    if not (processor.selected_vehicle and processor.vehicle_menu_position):
        return None

    return OverlayMenuRenderer.hit_test(
        processor.vehicle_menu_position,
        vehicle_menu_icon_order(processor),
        (processor.screen_width, processor.screen_height),
        (mouse_x, mouse_y),
    )

def check_vehicle_collision(processor, vehicle, new_transform):
    """Check if a vehicle would collide with other vehicles at the new transform"""
    if not vehicle or not vehicle.is_alive:
        return True  # Collision if vehicle is invalid
    
    # Get the bounding box of the vehicle we're moving/rotating (cached: this runs per
    # mouse-motion event, and a direct read is a blocking RPC on CARLA >= 0.9.16)
    vehicle_bbox = cached_bounding_box(vehicle)
    
    # Create a temporary transform to check collision
    test_location = new_transform.location
    test_rotation = new_transform.rotation
    
    # Check collision with all other vehicles
    for other_vehicle in processor.spawned_vehicles:
        if (other_vehicle and other_vehicle.is_alive and 
            other_vehicle.id != vehicle.id):
            
            other_location = other_vehicle.get_location()
            other_bbox = cached_bounding_box(other_vehicle)
            
            # Calculate 2D distance between vehicle centers
            dx = test_location.x - other_location.x
            dy = test_location.y - other_location.y
            distance_2d = math.sqrt(dx * dx + dy * dy)
            
            # Calculate combined bounding box size (approximate collision check)
            combined_extent = (vehicle_bbox.extent.x + other_bbox.extent.x +
                             vehicle_bbox.extent.y + other_bbox.extent.y) / 2.0
            
            # Add some safety margin
            collision_threshold = combined_extent * 0.8
            
            if distance_2d < collision_threshold:
                return True  # Collision detected
    
    return False  # No collision

def _is_pedestrian_actor(processor, actor_id):
    """Utility to determine if the given actor id belongs to a pedestrian."""
    for actor in processor.spawned_vehicles:
        if actor and actor.is_alive and actor.id == actor_id:
            return actor.type_id.startswith('walker.')
    return False

def _adjust_pedestrian_spawn_orientation(processor, actor_id: int) -> None:
    """Rotate a pedestrian to face its first waypoint."""
    if not processor._is_pedestrian_actor(actor_id):
        return

    waypoints = processor.get_vehicle_waypoints(actor_id)
    if not waypoints:
        return

    first_wp = waypoints[0]
    vehicle = None
    for actor in processor.spawned_vehicles:
        if actor and actor.is_alive and actor.id == actor_id:
            vehicle = actor
            break
    if vehicle is None:
        return

    # Base the write on the editor's own last-commanded transform (the same
    # cache the save path trusts), NOT a live get_transform() read-back: in
    # async mode the live read returns the last server snapshot, which can
    # still show a mid-gesture position — e.g. the +500 m raycast lift at
    # drag-end — and writing that back parked the walker in the sky and
    # clobbered every undo. Live read is only a fallback for actors never
    # placed through a command (load/spawn/move all populate the cache).
    base_transform = processor.vehicle_transforms.get(actor_id)
    if base_transform is None:
        base_transform = vehicle.get_transform()

    dx = first_wp['x'] - base_transform.location.x
    dy = first_wp['y'] - base_transform.location.y
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return

    yaw = math.degrees(math.atan2(dy, dx))
    transform = carla.Transform(
        base_transform.location,
        carla.Rotation(
            pitch=base_transform.rotation.pitch,
            yaw=yaw,
            roll=base_transform.rotation.roll,
        ),
    )
    vehicle.set_transform(transform)
    # Persist the computed heading so it survives saving: the save path reads
    # processor.vehicle_transforms first (see _build_vehicle_snapshot), so without this
    # the pedestrian's rotation would still serialize as its stale spawn yaw.
    processor.vehicle_transforms[actor_id] = carla.Transform(transform.location, transform.rotation)

    if (processor.editor and hasattr(processor.editor, 'info_panel')):
        panel = processor.editor.info_panel
        if (panel.visible and panel.object_type in ('vehicle', 'pedestrian') and
                panel.selected_object == vehicle):
            panel.fields['yaw'] = f"{yaw:.2f}"

def start_vehicle_movement(processor, mouse_pos, snap_to_lane=False):
    """Start moving the selected vehicle"""
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        processor.moving_vehicle = True
        processor.movement_start_mouse_pos = mouse_pos
        processor.movement_start_location = processor.selected_vehicle.get_location()
        processor.movement_snap_to_lane = snap_to_lane  # Store lane snapping preference
        
        if snap_to_lane:
            print(f"Started moving vehicle from ({processor.movement_start_location.x:.2f}, {processor.movement_start_location.y:.2f}) with lane snapping")
        else:
            print(f"Started moving vehicle from ({processor.movement_start_location.x:.2f}, {processor.movement_start_location.y:.2f})")

def update_vehicle_movement(processor, mouse_pos):
    """Update vehicle position based on mouse movement"""
    # Grab-to-move promotion: an armed click becomes a move once the mouse
    # travels past the threshold (a plain click below it only selects).
    if processor.vehicle_drag_armed and not processor.moving_vehicle:
        anchor = processor.movement_start_mouse_pos
        drag_distance = math.hypot(mouse_pos[0] - anchor[0], mouse_pos[1] - anchor[1])
        if drag_distance <= processor.object_drag_threshold:
            return
        # Snap preference is decided at promotion (was: when clicking the
        # Move icon): lane-snap by default, Shift for free placement.
        keys = pygame.key.get_pressed()
        snap_to_lane = not is_shift_pressed(keys)
        if processor.selected_vehicle_is_pedestrian:
            snap_to_lane = False
        processor.vehicle_drag_armed = False
        processor.start_vehicle_movement(anchor, snap_to_lane=snap_to_lane)

    if not (processor.moving_vehicle and processor.selected_vehicle and processor.selected_vehicle.is_alive):
        return
    
    # Calculate mouse delta
    mouse_dx = mouse_pos[0] - processor.movement_start_mouse_pos[0]
    mouse_dy = mouse_pos[1] - processor.movement_start_mouse_pos[1]
    
    # Convert screen movement to world movement
    camera_height = processor.camera_controller.height
    movement_scale = camera_height / 400.0  # Scale based on camera height
    
    world_dx = mouse_dx * movement_scale * 0.1
    world_dy = mouse_dy * movement_scale * 0.1
    
    # Calculate new world position
    new_location = carla.Location(
        processor.movement_start_location.x + world_dx,
        processor.movement_start_location.y + world_dy,
        processor.movement_start_location.z
    )
    
    # Check if lane snapping is enabled
    if hasattr(processor, 'movement_snap_to_lane') and processor.movement_snap_to_lane:
        # Snap to closest lane using fast method (without raycast during movement)
        lane_result = processor.coordinate_detector.find_closest_lane_point_fast(
            new_location.x, new_location.y, new_location.z
        )
        
        if lane_result['success']:
            # Use lane position and orientation (with OpenDRIVE height for now)
            snapped_location = carla.Location(
                lane_result['x'], 
                lane_result['y'], 
                lane_result['z'] + 0.1  # Use OpenDRIVE height + offset during movement
            )
            lane_rotation = carla.Rotation(pitch=0.0, yaw=lane_result['yaw'], roll=0.0)
            new_transform = carla.Transform(snapped_location, lane_rotation)
        else:
            # Fallback to original position if no lane found
            current_transform = processor.selected_vehicle.get_transform()
            new_transform = carla.Transform(new_location, current_transform.rotation)
    else:
        # Normal movement without lane snapping
        current_transform = processor.selected_vehicle.get_transform()
        new_transform = carla.Transform(new_location, current_transform.rotation)
    
    # Check for collision
    if not processor.check_vehicle_collision(processor.selected_vehicle, new_transform):
        # No collision, apply the movement
        processor.selected_vehicle.set_transform(new_transform)

def _group_waypoint_screen_positions(processor):
    """Projected marker positions of the marquee waypoint selection
    (list of (index, x, y)); empty when there is none."""
    group = processor.selected_waypoint_group
    if not group:
        return []
    waypoints = processor.get_vehicle_waypoints(group.get('vehicle_id'))
    positions = []
    for index in group.get('indices', ()):
        if not (0 <= index < len(waypoints)):
            continue
        waypoint = waypoints[index]
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            waypoint['x'], waypoint['y'], waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
        )
        if screen_pos.get('success'):
            positions.append((index, screen_pos['x'], screen_pos['y']))
    return positions

def group_selection_menu_anchor(processor):
    """Screen anchor (top-center of the combined projected extent) for the
    group selection's one-icon menu; None when no live group selection."""
    min_x = max_x = min_y = None

    def feed(x1, y1, x2):
        nonlocal min_x, max_x, min_y
        min_x = x1 if min_x is None else min(min_x, x1)
        max_x = x2 if max_x is None else max(max_x, x2)
        min_y = y1 if min_y is None else min(min_y, y1)

    if processor.selected_actor_ids:
        ids = processor.selected_actor_ids
        for actor in processor.spawned_vehicles:
            if not (actor and actor.is_alive and actor.id in ids):
                continue
            rect = processor.coordinate_detector._projected_actor_rect(actor)
            if rect is not None:
                feed(rect[0], rect[1], rect[2])
    else:
        for _, x, y in _group_waypoint_screen_positions(processor):
            feed(x, y, x)

    if min_x is None:
        return None
    return (int((min_x + max_x) / 2), int(min_y))

def waypoints_in_screen_rect(processor, rect):
    """Marquee capture for waypoints (used when the box caught no actors):
    return (vehicle_id, indices) of the displayed route's waypoints whose
    marker center falls inside the screen rect (x, y, w, h)."""
    vehicle_id = processor.waypoint_display_vehicle_id
    if not vehicle_id:
        return None, []
    rect_min_x, rect_min_y = rect[0], rect[1]
    rect_max_x, rect_max_y = rect[0] + rect[2], rect[1] + rect[3]
    indices = []
    for index, waypoint in enumerate(processor.get_vehicle_waypoints(vehicle_id)):
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            waypoint['x'], waypoint['y'], waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
        )
        if not screen_pos.get('success'):
            continue
        if (rect_min_x <= screen_pos['x'] <= rect_max_x and
                rect_min_y <= screen_pos['y'] <= rect_max_y):
            indices.append(index)
    return vehicle_id, indices

def select_single_waypoint(processor, vehicle_id, waypoint_index):
    """Select one waypoint (marquee caught exactly one): same selection and
    info panel as clicking its marker, without arming a drag."""
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints or not (0 <= waypoint_index < len(waypoints)):
        return False
    processor.selected_waypoint_group = None
    processor.selected_waypoint_vehicle_id = vehicle_id
    processor.selected_waypoint_index = waypoint_index
    processor.waypoint_drag_armed = False
    processor.pending_waypoint_click_action = None
    print(f"Selected waypoint {waypoint_index + 1}")
    if processor.editor and hasattr(processor.editor, 'info_panel'):
        processor.editor.info_panel.show(
            waypoints[waypoint_index],
            'waypoint',
            processor.editor.screen_width,
            processor.editor.screen_height,
            vehicle_id,
            waypoint_index,
        )
    return True

def delete_selected_waypoint_group(processor):
    """Delete every waypoint in the marquee waypoint selection as ONE
    undoable step."""
    group = processor.selected_waypoint_group
    processor.selected_waypoint_group = None
    if not group:
        return False
    vehicle_id = group.get('vehicle_id')
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    indices = sorted(
        {i for i in group.get('indices', ()) if 0 <= i < len(waypoints)},
        reverse=True,
    )
    if not indices:
        return False

    # Drop any single-waypoint selection pointing into the shrinking route
    processor.selected_waypoint_index = None
    processor.selected_waypoint_vehicle_id = None

    # Descending order keeps the remaining indices valid while executing;
    # CompositeCommand.undo runs in reverse (ascending), restoring the route.
    commands = [
        DeleteWaypointCommand(processor, vehicle_id, index, copy.deepcopy(waypoints[index]))
        for index in indices
    ]
    if len(commands) == 1:
        command = commands[0]
    else:
        command = CompositeCommand(commands, description=f"Delete {len(commands)} waypoints")
    processor.editor.execute_command(command)
    panel = getattr(processor.editor, 'info_panel', None)
    if panel and panel.visible and panel.object_type == 'waypoint_group':
        panel.hide()
    print(f"Deleted {len(commands)} selected waypoint(s)")
    return True

def check_group_menu_icon_click(processor, mouse_x, mouse_y):
    """Check if a click hits the group-selection menu icon."""
    anchor = group_selection_menu_anchor(processor)
    if not anchor:
        return None
    return OverlayMenuRenderer.hit_test(
        anchor,
        list(GROUP_MENU_ICONS),
        (processor.screen_width, processor.screen_height),
        (mouse_x, mouse_y),
    )

def delete_selected_actors(processor):
    """Delete every actor in the group selection as ONE undoable step."""
    ids = set(processor.selected_actor_ids or ())
    processor.selected_actor_ids = set()
    if not ids:
        return False

    commands = []
    for actor in list(processor.spawned_vehicles):
        if not (actor and actor.is_alive and actor.id in ids):
            continue
        transform = actor.get_transform()
        waypoints = copy.deepcopy(processor.get_vehicle_waypoints(actor.id))
        speed = processor.get_vehicle_speed(actor.id, 50)
        commands.append(
            DeleteVehicleCommand(processor, actor, actor.type_id, transform, waypoints, speed)
        )
    if not commands:
        return False

    if len(commands) == 1:
        command = commands[0]
    else:
        command = CompositeCommand(commands, description=f"Delete {len(commands)} actors")
    processor.editor.execute_command(command)
    print(f"Deleted {len(commands)} selected actor(s)")
    return True

def actor_under_click(processor, screen_x, screen_y):
    """Screen-space actor hit-test, memoized for the current left press.

    One click can consult it up to three times (personal-zone handler, global
    trigger handler, then handle_vehicle_click); the cache is invalidated at
    the top of _handle_left_click so it never outlives a press.
    """
    cached = getattr(processor, '_click_actor_hit_cache', None)
    if cached is not None and cached[0] == (screen_x, screen_y):
        return cached[1]
    actor = processor.coordinate_detector.check_vehicle_click(screen_x, screen_y)
    processor._click_actor_hit_cache = ((screen_x, screen_y), actor)
    return actor

def cancel_vehicle_rotation(processor):
    """Escape-cancel an in-progress rotation: restore the start yaw, keep the
    selection, push no undo command. Returns True when cancelled."""
    if not processor.rotating_vehicle:
        return False
    actor = processor.selected_vehicle if processor.selected_vehicle and processor.selected_vehicle.is_alive else None
    if actor is not None:
        try:
            transform = actor.get_transform()
            transform.rotation.yaw = processor.rotation_start_yaw
            actor.set_transform(transform)
        except Exception:
            pass
    processor.rotating_vehicle = False
    return True

def cancel_vehicle_drag(processor):
    """Escape-cancel an in-progress vehicle move: restore the start transform,
    keep the selection, push no undo command. Returns True when cancelled."""
    if not processor.moving_vehicle:
        return False
    actor = processor.selected_vehicle if processor.selected_vehicle and processor.selected_vehicle.is_alive else None
    if actor and processor.movement_start_location:
        try:
            rotation = actor.get_transform().rotation
            actor.set_transform(carla.Transform(processor.movement_start_location, rotation))
        except Exception:
            pass
    processor.moving_vehicle = False
    return True

def cancel_waypoint_drag(processor):
    """Escape-cancel an in-progress waypoint move: restore the start coords,
    keep the selection, push no undo command. Returns True when cancelled."""
    if not processor.moving_waypoint:
        return False
    start = processor.waypoint_movement_start_coords
    index = processor.selected_waypoint_index
    waypoints = None
    if processor.selected_waypoint_vehicle_id is not None:
        waypoints = processor.get_vehicle_waypoints(processor.selected_waypoint_vehicle_id)
    if waypoints and start and index is not None and 0 <= index < len(waypoints):
        waypoints[index].update(copy.deepcopy(start))
    processor.moving_waypoint = False
    processor.waypoint_drag_armed = False
    processor.waypoint_original_height = None
    processor.waypoint_movement_start_coords = None
    processor.pending_waypoint_click_action = None
    return True

def get_ground_height_at_location(processor, x, y, reference_z=None, return_metadata=False, probe_on_miss=True):
    """Return ground height metadata at (x, y) using raycast-based sampler.

    Args:
        probe_on_miss: If False, skip expensive grid search when raycast misses (faster for waypoints with known Z)
    """
    location = carla.Location(x, y, reference_z if reference_z is not None else 0.0)
    debug_mode = getattr(processor, 'debug_raycast', False)
    cached_map = processor._get_cached_map(refresh=False)
    result = get_ground_height(
        processor.world,
        location,
        debug=debug_mode,
        cached_map=cached_map,
        return_metadata=True,
        probe_on_miss=probe_on_miss,
        exclude_actors=[v for v in getattr(processor, 'spawned_vehicles', []) if v is not None],
    )
    if return_metadata:
        return result
    return result['height']

def stop_vehicle_movement(processor):
    """Stop moving the vehicle and adjust to final position with proper height"""
    if processor.moving_vehicle:
        actor = processor.selected_vehicle if processor.selected_vehicle and processor.selected_vehicle.is_alive else None
        old_transform = None
        is_pedestrian = bool(actor and processor._is_pedestrian_actor(actor.id))
        ped_height_offset = None
        if hasattr(processor, 'movement_start_location') and processor.movement_start_location:
            # Create original transform for undo
            reference_transform = actor.get_transform() if actor else None
            if reference_transform:
                old_transform = carla.Transform(processor.movement_start_location, reference_transform.rotation)
        
        # For pedestrians, use their bounding box height to determine the placement offset
        if actor and is_pedestrian:
            try:
                bbox_extent_z = float(getattr(actor.bounding_box.extent, "z", 1.0))
            except Exception:
                bbox_extent_z = 1.0
            # Walker transform is body-center: feet sit PEDESTRIAN clearance above ground.
            ped_height_offset = bbox_extent_z + PEDESTRIAN_PLACEMENT_CLEARANCE_M
        
        processor.moving_vehicle = False
        if actor and actor.is_alive:
            final_location = actor.get_location()

            # Determine target XY/rotation based on snapping
            corrected_x = final_location.x
            corrected_y = final_location.y
            corrected_rotation = actor.get_transform().rotation
            if hasattr(processor, 'movement_snap_to_lane') and processor.movement_snap_to_lane:
                lane_result = processor.coordinate_detector.find_closest_lane_point(
                    final_location.x, final_location.y, final_location.z
                )
                if lane_result and lane_result.get('success'):
                    corrected_x = lane_result['x']
                    corrected_y = lane_result['y']
                    corrected_rotation = carla.Rotation(pitch=0.0, yaw=lane_result['yaw'], roll=0.0)

            # Temporarily lift the actor out of the raycast path. The lift is
            # load-bearing for walkers too, not just the vehicle try_spawn
            # probe: an in-place walker yields label-NONE ray hits ~0.7 m
            # ABOVE its bbox head (outside the exclusion band), which then
            # win as "ground" (+2 m placements, measured live 2026-08-05 at
            # the Town02 probe spot). The lift landing in a server snapshot
            # is harmless since fix-27: the post-move heading adjust writes
            # from the vehicle_transforms cache, never a live get_transform()
            # read-back, so nothing re-applies the lift.
            original_transform = actor.get_transform()
            lifted_actor = False
            if original_transform:
                try:
                    lifted_location = carla.Location(
                        original_transform.location.x,
                        original_transform.location.y,
                        original_transform.location.z + 500.0,
                    )
                    actor.set_location(lifted_location)
                    lifted_actor = True
                except Exception:
                    lifted_actor = False

            corrected_transform = None
            try:
                ground_sample = processor.get_ground_height_at_location(
                    corrected_x,
                    corrected_y,
                    reference_z=final_location.z,
                    return_metadata=True,
                )

                ground_height = ground_sample.get('height', final_location.z)
                base_ground = ground_height if ground_height is not None else final_location.z
                if is_pedestrian and ped_height_offset is not None:
                    if ground_sample.get('source') == 'raycast':
                        spawn_height = base_ground + ped_height_offset
                    else:
                        # Raycast miss (unloaded tile / self-exclusion race):
                        # the fallback height is the walker's own body-center
                        # Z, so stacking the offset on it raises the walker
                        # ~1 m per drag. Keep the pre-drag Z instead — XY
                        # still moves, Z can never accumulate.
                        spawn_height = (
                            old_transform.location.z
                            if old_transform is not None
                            else final_location.z
                        )
                else:
                    # The actor is lifted aside, so probe try_spawn at this XY to
                    # find the lowest collision-free height — drag uses set_transform
                    # (no collision gate), so without this a vehicle dragged onto a
                    # curb/sidewalk saves a Z the post-Play restore can't respawn.
                    try:
                        veh_bp = processor.world.get_blueprint_library().find(actor.type_id)
                        spawn_height = lowest_spawnable_z(
                            processor.world, veh_bp, corrected_x, corrected_y,
                            corrected_rotation, base_ground,
                        )
                    except Exception:
                        spawn_height = base_ground + VEHICLE_PLACEMENT_CLEARANCE_M

                corrected_location = carla.Location(
                    corrected_x,
                    corrected_y,
                    spawn_height,
                )
                corrected_transform = carla.Transform(corrected_location, corrected_rotation)
                
                # Update vehicle position with corrected transform
                actor.set_transform(corrected_transform)
            finally:
                if corrected_transform is None and lifted_actor and original_transform:
                    # Restore actor if placement failed
                    try:
                        actor.set_transform(original_transform)
                    except Exception:
                        pass
            
            # Create movement command if vehicle was actually moved
            if (old_transform and 
                (abs(old_transform.location.x - corrected_transform.location.x) > 0.1 or
                 abs(old_transform.location.y - corrected_transform.location.y) > 0.1 or
                 abs(old_transform.location.z - corrected_transform.location.z) > 0.1)):
                
                command = MoveVehicleCommand(
                    processor,
                    actor,
                    old_transform,
                    corrected_transform
                )
                processor.editor.execute_command(command)
            
            print(f"Finished moving vehicle to ({final_location.x:.2f}, {final_location.y:.2f}) - Height adjusted to {corrected_transform.location.z:.2f}")
        # Reset movement tracking
        processor.movement_start_location = None

def start_vehicle_rotation(processor, mouse_y):
    """Start rotating the selected vehicle"""
    if processor.selected_vehicle_is_pedestrian:
        print("Pedestrian orientation follows its first waypoint and cannot be rotated manually.")
        return
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        processor.rotating_vehicle = True
        processor.rotation_start_mouse_y = mouse_y
        current_transform = processor.selected_vehicle.get_transform()
        processor.rotation_start_yaw = current_transform.rotation.yaw
        print(f"Started rotating vehicle from yaw: {processor.rotation_start_yaw:.1f}°")

def update_vehicle_rotation(processor, mouse_y):
    """Update vehicle rotation based on mouse movement"""
    if not (processor.rotating_vehicle and processor.selected_vehicle and processor.selected_vehicle.is_alive):
        return
    
    # Calculate rotation based on mouse Y movement
    mouse_delta = mouse_y - processor.rotation_start_mouse_y
    rotation_sensitivity = 0.5  # Degrees per pixel
    rotation_delta = mouse_delta * rotation_sensitivity
    
    # Calculate new yaw
    new_yaw = processor.rotation_start_yaw + rotation_delta
    
    # Normalize yaw to 0-360 range
    while new_yaw < 0:
        new_yaw += 360
    while new_yaw >= 360:
        new_yaw -= 360
    
    # Create new transform for collision checking
    current_transform = processor.selected_vehicle.get_transform()
    new_transform = carla.Transform(
        current_transform.location,
        carla.Rotation(pitch=current_transform.rotation.pitch, 
                      yaw=new_yaw, 
                      roll=current_transform.rotation.roll)
    )
    
    # Check for collision before applying rotation
    if not processor.check_vehicle_collision(processor.selected_vehicle, new_transform):
        # No collision, apply the rotation
        processor.selected_vehicle.set_transform(new_transform)

def stop_vehicle_rotation(processor):
    """Stop rotating the vehicle"""
    if processor.rotating_vehicle:
        old_transform = None
        if hasattr(processor, 'rotation_start_yaw') and processor.rotation_start_yaw is not None:
            # Create original transform for undo
            current_transform = processor.selected_vehicle.get_transform()
            old_rotation = carla.Rotation(
                current_transform.rotation.pitch,
                processor.rotation_start_yaw,
                current_transform.rotation.roll
            )
            old_transform = carla.Transform(current_transform.location, old_rotation)
        
        processor.rotating_vehicle = False
        if processor.selected_vehicle and processor.selected_vehicle.is_alive:
            final_transform = processor.selected_vehicle.get_transform()
            
            # Create rotation command if vehicle was actually rotated
            if (old_transform and 
                abs(old_transform.rotation.yaw - final_transform.rotation.yaw) > 1.0):  # Only if rotated more than 1 degree
                
                command = MoveVehicleCommand(
                    processor,
                    processor.selected_vehicle,
                    old_transform,
                    final_transform
                )
                processor.editor.execute_command(command)
            
            print(f"Finished rotating vehicle to yaw: {final_transform.rotation.yaw:.1f}°")
        
        # Reset rotation tracking
        processor.rotation_start_yaw = None

def start_waypoint_creation(processor, reset_existing=True):
    """Start waypoint creation mode for the selected vehicle."""
    # Default behavior hides info panel when creation stops unless we're extending an existing path
    processor.preserve_info_panel_on_waypoint_cancel = False
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        vehicle_id = processor.selected_vehicle.id
        existing_waypoints = list(processor.get_vehicle_waypoints(vehicle_id))

        if reset_existing and existing_waypoints:
            print(f"Clearing {len(existing_waypoints)} existing waypoints for actor {vehicle_id}")

        if reset_existing:
            # Reset waypoint data so the next placement starts a fresh path
            processor.set_vehicle_waypoints(vehicle_id, [])
            processor.clear_vehicle_destination_speed(vehicle_id)
            processor.selected_waypoint_vehicle_id = None
            processor.selected_waypoint_index = None
        else:
            # We're extending an existing path; preserve info panel after cancel
            if existing_waypoints:
                processor.preserve_info_panel_on_waypoint_cancel = True
            # Keep current waypoints and prepare to extend from the last one
            if existing_waypoints:
                processor.selected_waypoint_vehicle_id = vehicle_id
                processor.selected_waypoint_index = len(existing_waypoints) - 1
            else:
                processor.selected_waypoint_vehicle_id = None
                processor.selected_waypoint_index = None

        processor.moving_waypoint = False
        processor.waypoint_original_height = None
        processor.waypoint_drag_armed = False
        processor.pending_waypoint_click_action = None
        processor.creating_destination = False

        # Hide info panel if it was showing details for the previous path
        if processor.editor and hasattr(processor.editor, 'info_panel'):
            processor.editor.info_panel.hide()

        processor.creating_waypoint = True
        processor.waypoint_vehicle = processor.selected_vehicle
        
        # Set waypoint display to show this vehicle's waypoints during creation
        processor.waypoint_display_vehicle_id = vehicle_id
        
        # Clear menu position to hide the menu, but keep selected_vehicle for reference
        processor.vehicle_menu_position = None

        # Ensure overlays stop rendering the old path immediately
        processor.refresh_waypoints_carla_debug()

        location = processor.waypoint_vehicle.get_location()
        if reset_existing:
            print(f"Started waypoint creation for vehicle at ({location.x:.2f}, {location.y:.2f})")
        else:
            print(f"Waypoint extension mode for vehicle at ({location.x:.2f}, {location.y:.2f})")
        print("Move mouse to desired waypoint location and click to place waypoint. ESC or right-click to cancel.")
    else:
        print("ERROR: No valid vehicle selected for waypoint creation")

def stop_waypoint_creation(processor, *, clear_waypoints=False):
    """Stop waypoint creation mode."""
    if not processor.creating_waypoint:
        return

    vehicle_id = getattr(processor.waypoint_vehicle, "id", None)
    waypoint_count = len(processor.get_vehicle_waypoints(vehicle_id)) if vehicle_id else 0

    if clear_waypoints and vehicle_id is not None:
        if waypoint_count > 0:
            print(f"Clearing {waypoint_count} waypoint(s) for actor {vehicle_id}")
        processor.clear_vehicle_waypoints(vehicle_id)
        processor.clear_vehicle_destination_speed(vehicle_id)
        processor.selected_waypoint_vehicle_id = None
        processor.selected_waypoint_index = None
        waypoint_count = 0
        processor.refresh_waypoints_carla_debug()

    processor.creating_waypoint = False

    if clear_waypoints:
        print("Waypoint creation cancelled; new waypoints discarded.")
    else:
        print(f"Stopped waypoint creation. Vehicle has {waypoint_count} waypoints.")
        if waypoint_count > 0 and vehicle_id is not None:
            waypoints = processor.get_vehicle_waypoints(vehicle_id)
            print("Waypoint path:")
            for i, wp in enumerate(waypoints):
                print(f"  {i+1}: ({wp['x']:.2f}, {wp['y']:.2f}, {wp['z']:.2f})")

    # Clear all selection states to return to normal mode, but keep waypoint display
    waypoint_vehicle_id = vehicle_id  # Store before clearing
    preserve_panel = processor.preserve_info_panel_on_waypoint_cancel
    processor.waypoint_vehicle = None
    processor.clear_vehicle_selection(
        keep_waypoints=True,
        hide_info_panel=not preserve_panel
    )
    processor.waypoint_drag_armed = False
    # Reset preservation flag after applying it once
    processor.preserve_info_panel_on_waypoint_cancel = False

    # Keep waypoint display active for the vehicle that just had waypoints created
    processor.waypoint_display_vehicle_id = waypoint_vehicle_id

    print("Returned to normal editor mode")

def stop_destination_creation(processor):
    """Cancel destination placement mode and return to normal editor mode."""
    if not processor.creating_destination:
        return

    processor.creating_destination = False
    processor.destination_marker_coordinates = None
    # Deselect the vehicle to return to normal editor mode, mirroring waypoint cancel.
    processor.clear_vehicle_selection()
    print("Destination placement cancelled. Returned to normal editor mode.")

def place_waypoint_at_click(processor, screen_x, screen_y):
    """Place a waypoint at the clicked location"""
    if not (processor.creating_waypoint and processor.waypoint_vehicle and processor.waypoint_vehicle.is_alive):
        return
    
    # Get world coordinates at click position with lane snapping support
    keys = pygame.key.get_pressed()
    snap_to_lane = not is_shift_pressed(keys)  # Inverted: snap by default, shift for free movement

    if processor.waypoint_vehicle and processor.waypoint_vehicle.type_id.startswith('walker.'):
        snap_to_lane = False

    _t0 = time.perf_counter()
    world_coords = processor.coordinate_detector.screen_to_world_coordinates(
        screen_x, screen_y, processor.screen_width, processor.screen_height
    )
    _t1 = time.perf_counter()
    print(f"[TIMING] screen_to_world_coordinates: {(_t1-_t0)*1000:.1f}ms")

    if not world_coords['success']:
        print("Failed to determine world coordinates for waypoint")
        return
    
    # Apply lane snapping if requested
    if snap_to_lane:
        _t2 = time.perf_counter()
        lane_result = processor.coordinate_detector.find_closest_lane_point(
            world_coords['x'], world_coords['y'], world_coords['z']
        )
        _t3 = time.perf_counter()
        print(f"[TIMING] find_closest_lane_point: {(_t3-_t2)*1000:.1f}ms")
        if lane_result['success']:
            world_coords = lane_result
            world_coords['snapped'] = True
            print(f"Snapped waypoint to lane - Road ID: {lane_result['road_id']}, Lane ID: {lane_result['lane_id']}")
        else:
            print("Could not snap waypoint to lane, using original position")
            world_coords['snapped'] = False
    else:
        world_coords['snapped'] = False
    
    # Create waypoint data - ensure ground level placement
    ground_z = world_coords['z']

    # If this is lane snapping, the Z might be elevated, so get actual ground height
    if world_coords.get('snapped', False):
        # For lane-snapped waypoints, ensure we're at ground level
        # Use probe_on_miss=False since we have good reference Z from OpenDRIVE - avoids expensive grid search
        _t4 = time.perf_counter()
        ground_result = processor.get_ground_height_at_location(
            world_coords['x'],
            world_coords['y'],
            reference_z=world_coords.get('z'),
            return_metadata=True,
            probe_on_miss=False,
        )
        _t5 = time.perf_counter()
        print(f"[TIMING] get_ground_height_at_location: {(_t5-_t4)*1000:.1f}ms")
        # If raycast hit, use that height; otherwise fall back to OpenDRIVE Z
        if ground_result and ground_result.get('source') == 'raycast':
            ground_z = ground_result['height']
        # else: keep ground_z from OpenDRIVE (world_coords['z'])
    
    idle_time_default = 0.0
    speed_default = 50
    if processor.waypoint_vehicle:
        if processor.waypoint_vehicle.type_id.startswith('walker.'):
            speed_default = 5
        speed_default = processor.get_vehicle_speed(processor.waypoint_vehicle.id, speed_default)

    waypoint = {
        'x': world_coords['x'],
        'y': world_coords['y'],
        'z': ground_z,  # Use actual ground level
        'index': len(processor.get_vehicle_waypoints(processor.waypoint_vehicle.id)) + 1,
        'yaw': world_coords.get('yaw', None),
        'speed_km_h': speed_default,
        'idle_time_s': idle_time_default
    }
    
    # Create and execute command to place waypoint
    command = PlaceWaypointCommand(
        processor, 
        processor.waypoint_vehicle.id, 
        waypoint, 
        waypoint['index']
    )
    processor.editor.execute_command(command)

    if processor._is_pedestrian_actor(processor.waypoint_vehicle.id):
        waypoints_for_actor = processor.get_vehicle_waypoints(processor.waypoint_vehicle.id)
        if len(waypoints_for_actor) == 1:
            processor._adjust_pedestrian_spawn_orientation(processor.waypoint_vehicle.id)

    waypoint_num = len(processor.get_vehicle_waypoints(processor.waypoint_vehicle.id))
    print(f"Placed waypoint #{waypoint_num} at ({waypoint['x']:.2f}, {waypoint['y']:.2f}, {waypoint['z']:.2f})")

def place_destination_at_click(processor, screen_x, screen_y):
    """Place a destination at the clicked location and auto-route to it"""
    if not (processor.creating_destination and processor.selected_vehicle and processor.selected_vehicle.is_alive):
        return

    # Get world coordinates at click position with lane snapping support
    keys = pygame.key.get_pressed()
    snap_to_lane = not is_shift_pressed(keys)  # Inverted: snap by default, shift for free movement

    world_coords = processor.coordinate_detector.screen_to_world_coordinates(
        screen_x, screen_y, processor.screen_width, processor.screen_height
    )

    if not world_coords['success']:
        print("Failed to determine world coordinates for destination")
        return

    # Apply lane snapping if requested
    if snap_to_lane:
        lane_result = processor.coordinate_detector.find_closest_lane_point(
            world_coords['x'], world_coords['y'], world_coords['z']
        )
        if lane_result['success']:
            world_coords = lane_result
            world_coords['snapped'] = True
            print(f"Snapped destination to lane - Road ID: {lane_result['road_id']}, Lane ID: {lane_result['lane_id']}")
        else:
            print("Could not snap destination to lane, using original position")
            world_coords['snapped'] = False
    else:
        world_coords['snapped'] = False

    # Get ground level for destination
    ground_z = world_coords['z']
    if world_coords.get('snapped', False):
        ground_height = processor.get_ground_height_at_location(
            world_coords['x'],
            world_coords['y'],
            reference_z=world_coords.get('z'),
        )
        if ground_height is not None:
            ground_z = ground_height

    # Create destination data
    destination = {
        'x': world_coords['x'],
        'y': world_coords['y'],
        'z': ground_z,
        'yaw': world_coords.get('yaw', None)
    }

    # Clear existing waypoints for this vehicle
    processor.set_vehicle_waypoints(processor.selected_vehicle.id, [])
    processor.clear_vehicle_destination_speed(processor.selected_vehicle.id)

    # Auto-route to destination
    processor.auto_route_to_destination(processor.selected_vehicle, destination)

    # Exit destination mode
    processor.creating_destination = False
    print(f"Created auto-route to destination at ({destination['x']:.2f}, {destination['y']:.2f}, {destination['z']:.2f})")

def _create_auto_waypoint_data(processor, location: "carla.Location", waypoint_index: int,
                                yaw: float, speed_km_h: float) -> dict:
    """Create a waypoint data dictionary for auto-generated waypoints.

    Args:
        location: The CARLA location for the waypoint
        waypoint_index: The index to assign to this waypoint
        yaw: The yaw rotation at this waypoint
        speed_km_h: The target speed at this waypoint

    Returns:
        A dictionary containing the waypoint data
    """
    return {
        'x': location.x,
        'y': location.y,
        'z': location.z,
        'index': waypoint_index,
        'yaw': yaw,
        'speed_km_h': speed_km_h,
        'auto_generated': True
    }

def auto_route_to_destination(processor, vehicle, destination):
    """Automatically create a route from vehicle's current position to destination"""
    if not vehicle or not vehicle.is_alive:
        return

    # Get vehicle's current location
    vehicle_location = vehicle.get_location()
    start_point = {
        'x': vehicle_location.x,
        'y': vehicle_location.y,
        'z': vehicle_location.z
    }

    # Check if start point is on the road
    start_lane = processor.coordinate_detector.find_closest_lane_point(
        start_point['x'], start_point['y'], start_point['z']
    )

    # Check if destination is on the road
    dest_lane = processor.coordinate_detector.find_closest_lane_point(
        destination['x'], destination['y'], destination['z']
    )

    waypoints_to_add = []
    default_speed = processor.get_vehicle_speed(vehicle.id, 50)

    # Determine routing strategy based on destination type
    if dest_lane['success']:
        # Case 1: Destination is on-road - route directly to it
        if processor.world:
            try:
                map_obj = processor._get_cached_map(refresh=not processor.session._map_refresh_disabled)
                if not map_obj:
                    print("Warning: Could not get map object, falling back to direct path")
                    raise Exception("No map object available")

                # Get waypoints for start and destination
                start_wp = map_obj.get_waypoint(vehicle_location, project_to_road=True)
                dest_wp = map_obj.get_waypoint(
                    carla.Location(x=dest_lane['x'], y=dest_lane['y'], z=dest_lane['z'])
                )

                if start_wp and dest_wp:
                    # Generate route on roads
                    try:
                        current_wp = start_wp
                        target_location = dest_wp.transform.location

                        # Simple waypoint following
                        distance_threshold = 5.0  # meters between waypoints
                        min_distance_from_vehicle = 5.0  # minimum distance from vehicle to prevent extra markers
                        last_added_location = None
                        max_iterations = 200  # Prevent infinite loops

                        for iteration in range(max_iterations):
                            current_location = current_wp.transform.location
                            distance_to_target = current_location.distance(target_location)

                            # Check distance from vehicle starting position
                            vehicle_distance = current_location.distance(vehicle_location)

                            # Add waypoint if we've moved enough distance AND it's far enough from vehicle
                            if (last_added_location is None or current_location.distance(last_added_location) > distance_threshold) and vehicle_distance >= min_distance_from_vehicle:
                                waypoint_data = processor._create_auto_waypoint_data(
                                    current_location,
                                    len(waypoints_to_add) + 1,
                                    current_wp.transform.rotation.yaw,
                                    default_speed
                                )
                                waypoints_to_add.append(waypoint_data)
                                last_added_location = current_location

                            # Check if we've reached the destination
                            if distance_to_target < 10.0:
                                break

                            # Get next waypoint
                            next_waypoints = current_wp.next(5.0)  # Get next waypoint 5 meters ahead
                            if next_waypoints:
                                # Choose the waypoint that gets us closer to target
                                best_wp = min(next_waypoints,
                                            key=lambda w: w.transform.location.distance(target_location))
                                current_wp = best_wp
                            else:
                                break

                        print(f"Generated {len(waypoints_to_add)} waypoints using road following to on-road destination")

                    except Exception as e:
                        print(f"Error computing route: {e}")
                        waypoints_to_add = []

            except Exception as e:
                print(f"Error in on-road routing: {e}")

    else:
        # Case 2: Destination is off-road - route to nearest road point then connect to actual destination
        if processor.world:
            try:
                map_obj = processor._get_cached_map(refresh=not processor.session._map_refresh_disabled)
                if not map_obj:
                    print("Warning: Could not get map object, falling back to direct path")
                    raise Exception("No map object available")

                # Get closest road point to destination
                closest_road_wp = map_obj.get_waypoint(
                    carla.Location(x=destination['x'], y=destination['y'], z=destination['z']),
                    project_to_road=True
                )

                if closest_road_wp:
                    # Route from vehicle to closest road point
                    start_wp = map_obj.get_waypoint(vehicle_location, project_to_road=True)

                    if start_wp:
                        # Generate route to road point near destination
                        try:
                            current_wp = start_wp
                            target_location = closest_road_wp.transform.location

                            # Simple waypoint following
                            distance_threshold = 5.0  # meters between waypoints
                            min_distance_from_vehicle = 5.0  # minimum distance from vehicle to prevent extra markers
                            last_added_location = None
                            max_iterations = 200

                            for iteration in range(max_iterations):
                                current_location = current_wp.transform.location
                                distance_to_target = current_location.distance(target_location)

                                # Check distance from vehicle starting position
                                vehicle_distance = current_location.distance(vehicle_location)

                                # Add waypoint if we've moved enough distance AND it's far enough from vehicle
                                if (last_added_location is None or current_location.distance(last_added_location) > distance_threshold) and vehicle_distance >= min_distance_from_vehicle:
                                    waypoint_data = processor._create_auto_waypoint_data(
                                        current_location,
                                        len(waypoints_to_add) + 1,
                                        current_wp.transform.rotation.yaw,
                                        default_speed
                                    )
                                    waypoints_to_add.append(waypoint_data)
                                    last_added_location = current_location

                                # Check if we've reached the closest road point
                                if distance_to_target < 10.0:
                                    break

                                # Get next waypoint
                                next_waypoints = current_wp.next(5.0)
                                if next_waypoints:
                                    best_wp = min(next_waypoints,
                                                key=lambda w: w.transform.location.distance(target_location))
                                    current_wp = best_wp
                                else:
                                    break

                            # Add connection waypoint at the road exit point if not already there
                            if waypoints_to_add:
                                last_wp = waypoints_to_add[-1]
                                road_exit_dist = ((last_wp['x'] - closest_road_wp.transform.location.x)**2 +
                                                 (last_wp['y'] - closest_road_wp.transform.location.y)**2)**0.5

                                if road_exit_dist > 5.0:  # Only add if not too close to last waypoint
                                    road_exit_point = processor._create_auto_waypoint_data(
                                        closest_road_wp.transform.location,
                                        len(waypoints_to_add) + 1,
                                        closest_road_wp.transform.rotation.yaw,
                                        default_speed
                                    )
                                    waypoints_to_add.append(road_exit_point)

                            print(f"Generated {len(waypoints_to_add)} waypoints to road exit point")

                        except Exception as e:
                            print(f"Error computing route to road exit: {e}")

            except Exception as e:
                print(f"Error in off-road routing: {e}")

    # If we couldn't generate a road route, create a direct path
    if not waypoints_to_add:
        # Don't add vehicle position as waypoint - vehicle itself is the start marker
        # Add intermediate points if needed (for long distances)
        distance = ((destination['x'] - start_point['x'])**2 +
                   (destination['y'] - start_point['y'])**2)**0.5

        if distance > 50:  # If distance is more than 50 meters, add intermediate points
            num_intermediate = int(distance / 10)  # One point every 10 meters
            for i in range(1, num_intermediate):
                t = i / (num_intermediate + 1)
                intermediate = {
                    'x': start_point['x'] + t * (destination['x'] - start_point['x']),
                    'y': start_point['y'] + t * (destination['y'] - start_point['y']),
                    'z': start_point['z'] + t * (destination['z'] - start_point['z']),
                    'index': len(waypoints_to_add) + 1,
                    'speed_km_h': default_speed,
                    'auto_generated': True
                }
                waypoints_to_add.append(intermediate)

    # Ensure we have at least one approach waypoint with movement before the final stop
    if not waypoints_to_add:
        vector_x = destination['x'] - start_point['x']
        vector_y = destination['y'] - start_point['y']
        vector_z = destination['z'] - start_point['z']
        direct_distance = math.sqrt(vector_x ** 2 + vector_y ** 2 + vector_z ** 2)

        if direct_distance > 1.0:  # Only add approach if destination isn't essentially the spawn point
            dir_x = vector_x / direct_distance
            dir_y = vector_y / direct_distance
            dir_z = vector_z / direct_distance

            # Place the approach point most of the way to the destination but keep at least 1m gap
            approach_distance = max(min(direct_distance - 1.0, 20.0), direct_distance * 0.5)

            approach_point = {
                'x': start_point['x'] + dir_x * approach_distance,
                'y': start_point['y'] + dir_y * approach_distance,
                'z': start_point['z'] + dir_z * approach_distance,
                'index': 1,
                'yaw': destination.get('yaw', None),
                'speed_km_h': max(default_speed, 10),
                'auto_generated': True,
                'is_destination': False
            }
            waypoints_to_add.append(approach_point)

    # Always add the final destination at the ACTUAL clicked location (not lane-snapped)
    final_waypoint = {
        'x': destination['x'],  # Use actual destination coordinates
        'y': destination['y'],
        'z': destination['z'],
        'index': len(waypoints_to_add) + 1,
        'yaw': destination.get('yaw', None),
        'speed_km_h': 0,  # Stop at destination
        'auto_generated': True,
        'is_destination': True
    }
    waypoints_to_add.append(final_waypoint)

    # Add all waypoints to the vehicle as a single undoable operation
    waypoint_commands = []
    for waypoint in waypoints_to_add:
        command = PlaceWaypointCommand(
            processor,
            vehicle.id,
            waypoint,
            waypoint['index']
        )
        waypoint_commands.append(command)

    # Execute as composite command (single undo entry)
    composite = CompositeCommand(
        waypoint_commands,
        f"Auto-route {len(waypoints_to_add)} waypoints"
    )
    processor.editor.execute_command(composite)

    print(f"Auto-generated {len(waypoints_to_add)} waypoints for route to destination")

def delete_selected_vehicle(processor):
    """Delete the currently selected vehicle"""
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        print(f"Deleting vehicle at ({processor.selected_vehicle.get_location().x:.2f}, {processor.selected_vehicle.get_location().y:.2f})")
        
        # Get vehicle data for undo command
        vehicle_transform = processor.selected_vehicle.get_transform()
        vehicle_type = processor.selected_vehicle.type_id
        waypoints = copy.deepcopy(processor.get_vehicle_waypoints(processor.selected_vehicle.id))
        vehicle_speed = processor.get_vehicle_speed(processor.selected_vehicle.id, 50)
        
        # Create and execute delete vehicle command
        command = DeleteVehicleCommand(
            processor, 
            processor.selected_vehicle, 
            vehicle_type, 
            vehicle_transform, 
            waypoints, 
            vehicle_speed
        )
        processor.editor.execute_command(command)
        
        # Clear selection
        processor.clear_vehicle_selection()

def handle_mouse_click(processor, screen_x, screen_y, move_camera=False):
    """Handle mouse clicks that should move the camera."""
    if move_camera:
        processor.coordinate_detector.move_camera_to_screen_position(
            screen_x, screen_y, processor.screen_width, processor.screen_height
        )

def _hit_test_displayed_waypoint(processor, screen_x: int, screen_y: int):
    """Return hit waypoint metadata when clicking near a rendered marker."""
    vehicle_id = processor.waypoint_display_vehicle_id
    if not vehicle_id:
        return None
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints:
        return None

    for index, waypoint in enumerate(waypoints):
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            waypoint['x'], waypoint['y'], waypoint['z'] + WAYPOINT_MARKER_Z_OFFSET
        )
        if not screen_pos.get('success'):
            continue
        marker_x = int(screen_pos['x'])
        marker_y = int(screen_pos['y'])
        distance = math.hypot(marker_x - screen_x, marker_y - screen_y)
        if distance <= processor.waypoint_marker_radius + 5:
            return {
                'vehicle_id': vehicle_id,
                'index': index,
                'waypoint': waypoint,
            }
    return None

def _toggle_waypoint_in_group(processor, vehicle_id, waypoint_index):
    """Shift+click on a marker toggles it in/out of the waypoint group
    (an existing single waypoint selection seeds the group)."""
    group = processor.selected_waypoint_group
    indices = set()
    if group and group.get('vehicle_id') == vehicle_id:
        indices = set(group.get('indices', ()))
    if (processor.selected_waypoint_vehicle_id == vehicle_id and
            processor.selected_waypoint_index is not None):
        indices.add(processor.selected_waypoint_index)

    if waypoint_index in indices:
        indices.discard(waypoint_index)
    else:
        indices.add(waypoint_index)

    processor.selected_waypoint_vehicle_id = None
    processor.selected_waypoint_index = None
    processor.waypoint_drag_armed = False
    processor.pending_waypoint_click_action = None
    panel = getattr(processor.editor, 'info_panel', None) if processor.editor else None

    if not indices:
        processor.selected_waypoint_group = None
        if panel and panel.visible and panel.object_type in ('waypoint', 'waypoint_group'):
            panel.hide()
        print("Waypoint selection cleared")
        return
    if len(indices) == 1:
        processor.selected_waypoint_group = None
        processor.select_single_waypoint(vehicle_id, next(iter(indices)))
        return
    processor.selected_waypoint_group = {'vehicle_id': vehicle_id, 'indices': indices}
    if panel and processor.editor:
        panel.show(
            processor.selected_waypoint_group, 'waypoint_group',
            processor.editor.screen_width, processor.editor.screen_height, vehicle_id,
        )
    print(f"Selected {len(indices)} waypoint(s)")

def handle_waypoint_click(processor, screen_x, screen_y):
    """Handle click on waypoint markers for selection and movement"""
    hit = processor._hit_test_displayed_waypoint(screen_x, screen_y)
    if not hit:
        # Misses only disarm; the selection is forgiving (deselect via
        # Esc / right-click tap or by selecting something else).
        processor.waypoint_drag_armed = False
        return False

    vehicle_id = hit['vehicle_id']
    waypoint_index = hit['index']
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints or waypoint_index >= len(waypoints):
        return False

    # Shift+press = pending gesture: a tap (release under the threshold)
    # toggles group membership on mouse-up; a drag past it free-moves the
    # waypoint without lane snap (promotion in app/events.py).
    if is_shift_pressed(pygame.key.get_pressed()) and processor.editor:
        processor.editor._shift_press = {
            'kind': 'waypoint',
            'vehicle_id': vehicle_id,
            'index': waypoint_index,
            'pos': (screen_x, screen_y),
        }
        return True

    _select_and_arm_waypoint(processor, vehicle_id, waypoint_index, screen_x, screen_y)
    return True

def _select_and_arm_waypoint(processor, vehicle_id, waypoint_index, screen_x, screen_y):
    """Select a waypoint marker and arm a potential drag (shared by the plain
    click path and the Shift+drag promotion)."""
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    waypoint = waypoints[waypoint_index]
    # Reset pending click action; will re-arm if this click targets the last waypoint
    processor.pending_waypoint_click_action = None
    # Arm dragging only because this click hit a waypoint
    processor.waypoint_drag_armed = True
    # Waypoint was clicked - only select it, don't start moving yet
    # (a single waypoint selection replaces any marquee waypoint group)
    processor.selected_waypoint_group = None
    processor.selected_waypoint_vehicle_id = vehicle_id
    processor.selected_waypoint_index = waypoint_index
    # Store the click position for potential drag detection
    processor.waypoint_movement_start_pos = (screen_x, screen_y)
    # Store the complete waypoint state for undo operations
    processor.waypoint_movement_start_coords = {
        'x': waypoint['x'],
        'y': waypoint['y'],
        'z': waypoint['z'],
        'yaw': waypoint.get('yaw', None),
        'speed_km_h': waypoint.get('speed_km_h', 50),
        'idle_time_s': waypoint.get('idle_time_s', 0.0)
    }
    # Store original height for potential movement
    processor.waypoint_original_height = waypoint['z']
    # Arm pending action so a simple click on the last waypoint can extend the path
    if processor._is_last_waypoint(vehicle_id, waypoint_index):
        processor.pending_waypoint_click_action = {
            'vehicle_id': vehicle_id,
            'index': waypoint_index
        }
    # Don't set moving_waypoint = True here - wait for mouse movement
    print(f"Selected waypoint {waypoint_index + 1}")

    # Show info panel for selected waypoint
    if processor.editor and hasattr(processor.editor, 'info_panel'):
        processor.editor.info_panel.show(
            waypoint,
            'waypoint',
            processor.editor.screen_width,
            processor.editor.screen_height,
            vehicle_id,
            waypoint_index,
        )

def split_waypoint_at_click(processor, screen_x: int, screen_y: int) -> bool:
    """Split the waypoint hit by the cursor into two offset points."""
    hit = processor._hit_test_displayed_waypoint(screen_x, screen_y)
    if not hit:
        return False
    vehicle_id = hit['vehicle_id']
    waypoint_index = hit['index']
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints or waypoint_index >= len(waypoints):
        return False

    waypoint_data = waypoints[waypoint_index]
    offset = getattr(processor, "waypoint_split_offset", 1.5)
    command = SplitWaypointCommand(
        processor,
        vehicle_id,
        waypoint_index,
        waypoint_data,
        offset_distance=offset,
    )
    if processor.editor:
        success = processor.editor.execute_command(command)
    else:
        success = command.execute()
    if success is False:
        return False

    updated_waypoints = processor.get_vehicle_waypoints(vehicle_id)
    target_index = min(waypoint_index, len(updated_waypoints) - 1) if updated_waypoints else waypoint_index
    processor.selected_waypoint_vehicle_id = vehicle_id
    processor.selected_waypoint_index = target_index
    processor.waypoint_original_height = updated_waypoints[target_index].get('z', 0.0) if updated_waypoints else None
    processor.moving_waypoint = False
    processor.pending_waypoint_click_action = None

    if (processor.editor and hasattr(processor.editor, 'info_panel') and updated_waypoints
            and target_index < len(updated_waypoints)):
        processor.editor.info_panel.show(
            updated_waypoints[target_index],
            'waypoint',
            processor.editor.screen_width,
            processor.editor.screen_height,
            vehicle_id,
            target_index,
        )

    processor.refresh_waypoints_carla_debug()
    print(f"Split waypoint {waypoint_index + 1} for actor {vehicle_id}")
    return True

def handle_waypoint_mouse_movement(processor, screen_x, screen_y):
    """Handle mouse movement for waypoint dragging"""
    # Do not allow dragging while placing new waypoints
    if processor.creating_waypoint:
        return

    # Only process drag logic when armed by a recent waypoint click
    if not processor.waypoint_drag_armed and not processor.moving_waypoint:
        return

    # Check if we have a selected waypoint (but not necessarily moving yet)
    if (processor.selected_waypoint_vehicle_id is not None and 
        processor.selected_waypoint_index is not None):
        
        # Check if the waypoint list and waypoint still exist
        waypoints = processor.get_vehicle_waypoints(processor.selected_waypoint_vehicle_id)
        if (not waypoints or
            processor.selected_waypoint_index >= len(waypoints)):
            processor.selected_waypoint_vehicle_id = None
            processor.selected_waypoint_index = None
            processor.moving_waypoint = False
            processor.waypoint_drag_armed = False
            return
        
        # If we're not moving yet, check if we should start moving based on drag threshold
        if not processor.moving_waypoint:
            # Calculate distance from initial click position
            dx = screen_x - processor.waypoint_movement_start_pos[0]
            dy = screen_y - processor.waypoint_movement_start_pos[1]
            drag_distance = (dx * dx + dy * dy) ** 0.5
            
            # Start moving only if we've exceeded the drag threshold
            if drag_distance > processor.waypoint_drag_threshold:
                processor.moving_waypoint = True
                # Cancel pending click action because we're dragging instead of tapping
                processor.pending_waypoint_click_action = None
                print(f"Started dragging waypoint {processor.selected_waypoint_index + 1}")
            else:
                # Still within threshold, don't start moving
                return
        
        # Now handle the actual movement (only if moving_waypoint is True)
        if processor.moving_waypoint:
            # Check if we have a valid original height
            if processor.waypoint_original_height is None:
                # Fallback to current waypoint height if original height is missing
                if (waypoints and
                    processor.selected_waypoint_index < len(waypoints)):
                    waypoint = waypoints[processor.selected_waypoint_index]
                    processor.waypoint_original_height = waypoint.get('z', 0.0)
                else:
                    processor.waypoint_original_height = 0.0  # Safe fallback
            
            # Check if Shift is held for free movement during waypoint drag
            keys = pygame.key.get_pressed()
            snap_to_lane = not is_shift_pressed(keys)  # Snap by default, shift for free placement
            if processor._is_pedestrian_actor(processor.selected_waypoint_vehicle_id):
                snap_to_lane = False

            # Always use fast coordinate conversion and keep original Z during movement
            world_coords = processor.coordinate_detector.screen_to_world_coordinates_no_raycast(
                screen_x, screen_y, processor.screen_width, processor.screen_height, processor.waypoint_original_height
            )

            if world_coords and world_coords.get('success', False):
                waypoint = waypoints[processor.selected_waypoint_index]
                if snap_to_lane:
                    # Snap to lane, but keep original Z (use fast method, no raycast)
                    lane_result = processor.coordinate_detector.find_closest_lane_point_fast(
                        world_coords['x'], world_coords['y'], processor.waypoint_original_height
                    )
                    if lane_result['success']:
                        waypoint['x'] = lane_result['x']
                        waypoint['y'] = lane_result['y']
                        waypoint['z'] = processor.waypoint_original_height
                        waypoint['yaw'] = lane_result.get('yaw', None)
                    else:
                        waypoint['x'] = world_coords['x']
                        waypoint['y'] = world_coords['y']
                        waypoint['z'] = processor.waypoint_original_height
                else:
                    # Free placement, keep original Z
                    waypoint['x'] = world_coords['x']
                    waypoint['y'] = world_coords['y']
                    waypoint['z'] = processor.waypoint_original_height
            else:
                print("Failed to convert screen coordinates to world coordinates")

def handle_waypoint_mouse_release(processor):
    """Handle mouse release to stop waypoint movement and apply final positioning with raycasting"""
    # If there was no drag, a simple tap on the final waypoint should extend the path
    if (not processor.moving_waypoint and
        processor.pending_waypoint_click_action and
        processor.selected_waypoint_vehicle_id == processor.pending_waypoint_click_action.get('vehicle_id') and
        processor.selected_waypoint_index == processor.pending_waypoint_click_action.get('index') and
        processor._is_last_waypoint(processor.selected_waypoint_vehicle_id, processor.selected_waypoint_index)):

        if getattr(processor.camera_controller, "view_mode", "topdown") == "orbit":
            # 3D view is view + select: a tap must not enter extension mode (it also
            # mutates is_destination/speed). The selection already happened on press;
            # skip silently -- the user asked to select, not to edit.
            processor.pending_waypoint_click_action = None
        else:
            processor._start_waypoint_extension_mode(processor.selected_waypoint_vehicle_id)

    if processor.moving_waypoint and processor.selected_waypoint_index is not None:
        # Store original position for undo command
        original_position = None
        if processor.waypoint_movement_start_coords:
            # waypoint_movement_start_coords now contains the complete waypoint state
            original_position = copy.deepcopy(processor.waypoint_movement_start_coords)
        
        # We were actually moving a waypoint, so do the expensive raycast for final positioning
        waypoints = processor.get_vehicle_waypoints(processor.selected_waypoint_vehicle_id)
        if not waypoints or processor.selected_waypoint_index >= len(waypoints):
            processor.moving_waypoint = False
            processor.waypoint_original_height = None
            processor.waypoint_movement_start_coords = None
            processor.pending_waypoint_click_action = None
            return
        waypoint = waypoints[processor.selected_waypoint_index]
        
        # Check if Shift is held for free movement on release
        keys = pygame.key.get_pressed()
        snap_to_lane = not is_shift_pressed(keys)  # Inverted: snap by default, shift for free movement
        if processor._is_pedestrian_actor(processor.selected_waypoint_vehicle_id):
            snap_to_lane = False

        # Get current mouse position for final raycast
        mouse_pos = pygame.mouse.get_pos()
        final_coords = processor.coordinate_detector.screen_to_world_coordinates(
            mouse_pos[0], mouse_pos[1], processor.screen_width, processor.screen_height
        )
        
        if final_coords and final_coords.get('success', False):
            # Apply lane snapping if requested (Shift = free placement)
            if snap_to_lane:
                lane_result = processor.coordinate_detector.find_closest_lane_point(
                    final_coords['x'], final_coords['y'], final_coords['z']
                )
                if lane_result['success']:
                    waypoint['x'] = lane_result['x']
                    waypoint['y'] = lane_result['y']
                    waypoint['z'] = lane_result['z']  # Use proper raycasted height on release
                    waypoint['yaw'] = lane_result.get('yaw', None)
                    print(f"Snapped waypoint to lane during movement")
                else:
                    # If snapping failed, use regular coordinates with raycasted height
                    waypoint['x'] = final_coords['x']
                    waypoint['y'] = final_coords['y']
                    waypoint['z'] = final_coords['z']
            else:
                # Update with raycasted height (no lane snapping)
                waypoint['x'] = final_coords['x']
                waypoint['y'] = final_coords['y']
                waypoint['z'] = final_coords['z']

        # Create undo command if waypoint was actually moved
        if original_position:
            new_position = {
                'x': waypoint['x'],
                'y': waypoint['y'],
                'z': waypoint['z'],
                'yaw': waypoint.get('yaw', None),
                'speed_km_h': waypoint.get('speed_km_h', 50),
                'idle_time_s': waypoint.get('idle_time_s', 0.0)
            }
            
            # Only create command if position actually changed
            if (abs(original_position['x'] - new_position['x']) > 0.01 or
                abs(original_position['y'] - new_position['y']) > 0.01 or
                abs(original_position['z'] - new_position['z']) > 0.01):
                
                command = MoveWaypointCommand(
                    processor,
                    processor.selected_waypoint_vehicle_id,
                    processor.selected_waypoint_index,
                    original_position,
                    new_position
                )
                processor.editor.execute_command(command)
        
        waypoint_num = processor.selected_waypoint_index + 1
        print(f"Finished moving waypoint {waypoint_num}")
        # If the first waypoint moved for a pedestrian, reorient to it
        if processor._is_pedestrian_actor(processor.selected_waypoint_vehicle_id) and processor.selected_waypoint_index == 0:
            processor._adjust_pedestrian_spawn_orientation(processor.selected_waypoint_vehicle_id)
        
    # Always reset movement state on mouse release
    processor.moving_waypoint = False
    processor.waypoint_original_height = None
    processor.waypoint_movement_start_coords = None  # Clear original coordinates
    processor.pending_waypoint_click_action = None
    processor.waypoint_drag_armed = False

def _cache_destination_speed(processor, vehicle_id):
    """Update cached destination speed for a vehicle based on its waypoints."""
    if vehicle_id is None:
        return

    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints:
        processor.clear_vehicle_destination_speed(vehicle_id)
        return

    for waypoint in reversed(waypoints):
        if waypoint.get('is_destination'):
            speed = waypoint.get(
                'speed_km_h',
                processor.get_vehicle_speed(vehicle_id, 50)
            )
            processor.set_vehicle_destination_speed(vehicle_id, speed)
            return

    # No destination waypoint flagged - fall back to last waypoint speed
    last_speed = waypoints[-1].get('speed_km_h', processor.get_vehicle_speed(vehicle_id, 50))
    processor.set_vehicle_destination_speed(vehicle_id, last_speed)

def _start_waypoint_extension_mode(processor, vehicle_id):
    """Enter waypoint creation mode starting from the current final waypoint."""
    if vehicle_id is None:
        return False

    # Already in creation mode for this vehicle – nothing to change
    if (processor.creating_waypoint and processor.waypoint_vehicle and processor.waypoint_vehicle.is_alive and
            processor.waypoint_vehicle.id == vehicle_id):
        return False

    # If we're creating waypoints for another vehicle, stop that session first
    if processor.creating_waypoint:
        processor.stop_waypoint_creation()

    vehicle = next((v for v in processor.spawned_vehicles if v.id == vehicle_id and v.is_alive), None)
    if not vehicle:
        print("Cannot extend waypoints: vehicle is no longer available.")
        return False

    # Ensure the previous destination waypoint resumes normal behaviour before extending
    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if waypoints:
        last_index = len(waypoints) - 1
        last_waypoint = waypoints[last_index]
        default_speed = processor.get_vehicle_speed(vehicle_id, 50)

        # Prepare property adjustments if needed
        adjustments = []
        if last_waypoint.get('is_destination', False):
            adjustments.append(('is_destination', last_waypoint.get('is_destination'), False))
        if last_waypoint.get('speed_km_h', 50) <= 0:
            # Avoid zero target speed that would make the vehicle stop prematurely
            adjustments.append(('speed_km_h', last_waypoint.get('speed_km_h', 0), max(default_speed, 10)))

        for property_name, old_value, new_value in adjustments:
            if old_value == new_value:
                continue

            if processor.editor:
                command = UpdateWaypointPropertyCommand(
                    processor,
                    vehicle_id,
                    last_index,
                    property_name,
                    old_value,
                    new_value
                )
                processor.editor.execute_command(command)
            else:
                # fallback without command system (should not generally happen)
                waypoints[last_index][property_name] = new_value

    # Reuse the existing creation workflow by selecting the vehicle and invoking start without clearing
    processor.selected_vehicle = vehicle
    processor.selected_vehicle_is_pedestrian = vehicle.type_id.startswith('walker.')
    processor.start_waypoint_creation(reset_existing=False)
    # --- Keep info panel visible for last waypoint during extension mode ---
    if waypoints:
        last_index = len(waypoints) - 1
        last_waypoint = waypoints[last_index]
        if processor.editor and hasattr(processor.editor, 'info_panel'):
            processor.editor.info_panel.show(last_waypoint, 'waypoint', processor.editor.screen_width, processor.editor.screen_height, vehicle_id, last_index)
    print("Additional waypoint mode activated. Click to add waypoints, press ESC or right-click to finish.")
    return True

def _is_last_waypoint(processor, vehicle_id, waypoint_index):
    """Check whether the given waypoint is currently the final waypoint for the vehicle."""
    if vehicle_id is None or waypoint_index is None:
        return False

    waypoints = processor.get_vehicle_waypoints(vehicle_id)
    if not waypoints:
        return False

    if waypoint_index < 0 or waypoint_index >= len(waypoints):
        return False

    return waypoint_index == len(waypoints) - 1

def delete_selected_waypoint(processor):
    """Delete the currently selected waypoint and reconnect the path"""
    if (processor.selected_waypoint_vehicle_id is None or 
        processor.selected_waypoint_index is None or
        not processor.get_vehicle_waypoints(processor.selected_waypoint_vehicle_id)):
        print("No waypoint selected for deletion")
        return False
    
    waypoints = processor.get_vehicle_waypoints(processor.selected_waypoint_vehicle_id)
    
    if processor.selected_waypoint_index >= len(waypoints):
        print("Invalid waypoint index")
        return False
    
    # Get waypoint info before deletion
    waypoint_num = processor.selected_waypoint_index + 1
    waypoint_data = waypoints[processor.selected_waypoint_index]
    
    # Create and execute delete waypoint command
    command = DeleteWaypointCommand(
        processor,
        processor.selected_waypoint_vehicle_id,
        processor.selected_waypoint_index,
        waypoint_data
    )
    success = processor.editor.execute_command(command)
    
    if success:
        print(f"Deleted waypoint {waypoint_num} (was at {waypoint_data['x']:.1f}, {waypoint_data['y']:.1f})")
        
        # Update selection after deletion
        waypoints = processor.get_vehicle_waypoints(processor.selected_waypoint_vehicle_id)
        
        # Handle the case where we deleted the last waypoint
        if processor.selected_waypoint_index >= len(waypoints):
            if len(waypoints) > 0:
                # Select the new last waypoint
                processor.selected_waypoint_index = len(waypoints) - 1
                print(f"Selected previous waypoint (now waypoint {processor.selected_waypoint_index + 1})")
            else:
                # No waypoints left, clear selection
                print("No waypoints remaining for this vehicle")
                processor.selected_waypoint_vehicle_id = None
                processor.selected_waypoint_index = None
        else:
            # We deleted a waypoint in the middle, keep current index
            print(f"Path automatically reconnected. Selected waypoint {processor.selected_waypoint_index + 1}")
        
        # Clear movement state if we were moving the deleted waypoint
        if processor.moving_waypoint:
            processor.moving_waypoint = False
            processor.waypoint_original_height = None
        
        print(f"Waypoints remaining: {len(waypoints)}")
    
    return success

def select_vehicle_actor(
    processor,
    actor: Optional[carla.Actor],
    *,
    focus_camera: bool = False,
    fallback_screen_pos: Optional[Tuple[int, int]] = None,
) -> bool:
    """Select a vehicle or pedestrian actor and optionally move the camera."""
    if not actor or not actor.is_alive:
        return False

    if processor.selected_trigger_index is not None or processor.trigger_action_menu_position:
        processor.selected_trigger_index = None
        processor.trigger_action_menu_position = None
    if processor.selected_personal_trigger:
        processor.clear_personal_trigger_selection()
    processor.clear_traffic_light_selection()
    processor.selected_actor_ids = set()
    processor.selected_waypoint_group = None

    processor.selected_vehicle = actor
    processor.selected_vehicle_is_pedestrian = actor.type_id.startswith('walker.')
    processor.waypoint_display_vehicle_id = actor.id

    actor_location: Optional[carla.Location] = None
    if processor.large_map_active:
        cached = processor._actor_location_cache.get(actor.id)
        if cached:
            actor_location = cached[0]
        if actor_location is None:
            if focus_camera:
                processor._pending_large_map_focus_actor_id = int(actor.id)
                processor._pending_large_map_focus_next_attempt = 0.0
                processor._pending_large_map_focus_failures = 0
            else:
                actor_location = processor._get_actor_location_fast(actor.id, max_age_s=0.25)
    else:
        cached = processor._actor_location_cache.get(actor.id) if focus_camera else None
        if cached:
            actor_location = cached[0]
        if actor_location is None:
            try:
                actor_location = actor.get_location()
            except Exception:
                actor_location = None
    if actor_location is not None:
        processor._cache_actor_location(actor.id, actor_location)

    screen_pos = None
    if actor_location:
        screen_pos = processor.coordinate_detector.world_to_screen_coordinates(
            actor_location.x, actor_location.y, actor_location.z + 2.0
        )

    if screen_pos and screen_pos.get('success'):
        processor.vehicle_menu_position = (int(screen_pos['x']), int(screen_pos['y']))
    elif fallback_screen_pos:
        processor.vehicle_menu_position = (int(fallback_screen_pos[0]), int(fallback_screen_pos[1]))
    else:
        processor.vehicle_menu_position = None

    skip_expensive_ui = bool(focus_camera and processor.large_map_active)

    if processor.editor and hasattr(processor.editor, 'info_panel') and not skip_expensive_ui:
        obj_type = 'pedestrian' if processor.selected_vehicle_is_pedestrian else 'vehicle'
        processor.editor.info_panel.show(actor, obj_type, processor.editor.screen_width, processor.editor.screen_height)

    if focus_camera and actor_location:
        processor.focus_camera_on_location(actor_location)

    if not skip_expensive_ui:
        processor.refresh_selected_vehicle_ui()

    if actor_location:
        print(f"Selected vehicle at ({actor_location.x:.2f}, {actor_location.y:.2f})")
    else:
        print("Selected vehicle.")
    return True

def _toggle_actor_in_group(processor, actor):
    """Shift+click on an actor body toggles it in/out of the actor group
    (an existing single actor selection seeds the group)."""
    ids = set(processor.selected_actor_ids or ())
    if processor.selected_vehicle and processor.selected_vehicle.is_alive:
        ids.add(processor.selected_vehicle.id)

    if actor.id in ids:
        ids.discard(actor.id)
    else:
        ids.add(actor.id)

    if not ids:
        processor.clear_vehicle_selection()
        processor.selected_actor_ids = set()
        print("Selection cleared")
        return
    if len(ids) == 1:
        target_id = next(iter(ids))
        target = next(
            (a for a in processor.spawned_vehicles if a and a.is_alive and a.id == target_id),
            None,
        )
        processor.clear_vehicle_selection()
        processor.selected_actor_ids = set()
        if target is not None:
            processor.select_vehicle_actor(target)
        return
    # Group of 2+: mutually exclusive with every single selection kind
    processor.clear_vehicle_selection()
    processor.clear_traffic_light_selection()
    processor.clear_personal_trigger_selection()
    processor.selected_trigger_index = None
    processor.trigger_action_menu_position = None
    processor.selected_waypoint_group = None
    processor.selected_actor_ids = ids
    print(f"Selected {len(ids)} actor(s)")

def _select_and_arm_actor(processor, actor, screen_x, screen_y):
    """Select an actor and arm a potential grab-to-move drag (shared by the
    plain click path and the Shift+drag promotion)."""
    if processor.select_vehicle_actor(actor, fallback_screen_pos=(screen_x, screen_y)):
        # Arm here, not in select_vehicle_actor: the actor-list side menus
        # select programmatically and must never arm a drag.
        processor.vehicle_drag_armed = True
        processor.movement_start_mouse_pos = (screen_x, screen_y)

def handle_vehicle_click(processor, screen_x, screen_y):
    """Handle click on a vehicle to show action menu"""
    clicked_vehicle = processor.actor_under_click(screen_x, screen_y)

    if clicked_vehicle:
        # Shift+press = pending gesture: a tap (release under the threshold)
        # toggles group membership on mouse-up; a drag past it free-moves the
        # actor without lane snap (promotion in app/events.py).
        if is_shift_pressed(pygame.key.get_pressed()) and processor.editor:
            processor.editor._shift_press = {
                'kind': 'actor',
                'actor_id': clicked_vehicle.id,
                'pos': (screen_x, screen_y),
            }
            return True
        _select_and_arm_actor(processor, clicked_vehicle, screen_x, screen_y)
        return True

    # No vehicle clicked; the left-click fall-through in app/events.py decides
    # whether the current selection is kept (forgiving empty click) or cleared.
    return False

def spawn_vehicle_at_marker(processor, vehicle_type, coordinates, *, role="npc"):
    """Spawn a vehicle at the given coordinates"""
    try:
        if role == "ego" and processor.is_ego_vehicle_active():
            print("An ego vehicle already exists. Delete the current ego vehicle before placing a new one.")
            return False

        blueprint_library = processor.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(vehicle_type)

        # Raycast ground height and add small buffer so actors rest on uneven terrain
        base_location = carla.Location(
            coordinates['x'],
            coordinates['y'],
            coordinates.get('z', 0.0),
        )
        debug_mode = getattr(processor, 'debug_raycast', False)
        cached_map = processor._get_cached_map(refresh=False)
        ground_sample = get_ground_height(
            processor.world,
            base_location,
            debug=debug_mode,
            cached_map=cached_map,
            exclude_actors=processor.spawned_vehicles,
            return_metadata=True,
        )
        # Buffer only on a real hit; a miss echoes the marker Z back, and
        # buffering on top of it drifts the requested height (fix-03).
        if ground_sample.get('source') == 'raycast' and ground_sample.get('height') is not None:
            spawn_height = ground_sample['height'] + 0.2
        else:
            spawn_height = base_location.z

        # Set up spawn transform using raycast height
        location = carla.Location(coordinates['x'], coordinates['y'], spawn_height)

        # Use lane direction if available (snapped to lane)
        if coordinates.get('snapped', False) and 'yaw' in coordinates:
            rotation = carla.Rotation(pitch=0.0, yaw=coordinates['yaw'], roll=0.0)
            spawn_type = "lane-aligned"
        else:
            rotation = carla.Rotation()
            spawn_type = "default orientation"

        transform = carla.Transform(location, rotation)

        # Create and execute spawn vehicle command
        command = SpawnVehicleCommand(processor, vehicle_type, transform, role=role)
        success = processor.editor.execute_command(command)

        if success:
            # Update the command with the spawned vehicle ID for future reference
            if command.spawned_vehicle:
                command.original_vehicle_id = command.spawned_vehicle.id
            print(f"Spawned {vehicle_type} at ({coordinates['x']:.2f}, {coordinates['y']:.2f}, {spawn_height:.2f}) - {spawn_type} (physics disabled)")
            return True
        else:
            print(f"Failed to spawn {vehicle_type} - location might be blocked")
            return False

    except Exception as e:
        print(f"Error spawning vehicle: {e}")
        return False
