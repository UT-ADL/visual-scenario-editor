"""Undo/redo command pattern (moved verbatim from vse.py). Commands take the
camera_processor and mutate its scene state directly. Commands do NOT touch
the info panel: targeted field refreshes are dispatched by the editor's
CommandHistory on_change handler to InfoPanel.refresh_* (step-21).
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import carla

from vse_common.geometry import get_ground_height
from vse_editor.carla_io.spawning import (
    PEDESTRIAN_PLACEMENT_CLEARANCE_M,
    VEHICLE_PLACEMENT_CLEARANCE_M,
    lowest_spawnable_z,
    spawn_vehicle_grounded,
)
from vse_editor.constants import DEFAULT_PERSONAL_TRIGGER_RADIUS, MIN_PERSONAL_TRIGGER_RADIUS
from vse_editor.scene_types import (
    VehicleIgnoreFlags,
    WaypointData,
    clone_waypoint_data,
    clone_waypoint_sequence,
    ensure_ignore_flags,
)


def _clone_transform(transform: carla.Transform) -> carla.Transform:
    """Copy a CARLA transform without retaining mutable location objects."""
    return carla.Transform(
        carla.Location(
            float(transform.location.x),
            float(transform.location.y),
            float(transform.location.z),
        ),
        carla.Rotation(
            float(transform.rotation.pitch),
            float(transform.rotation.yaw),
            float(transform.rotation.roll),
        ),
    )


class Command:
    """
    Base class for undoable commands in the Visual Scenario Editor.
    Each command encapsulates an action (e.g., place/move/delete waypoint or vehicle)
    and provides execute, undo, and redo methods for history management.
    """
    def execute(self):
        """
        Execute the command.
        Should be overridden by subclasses to perform the action.
        """
        pass

    def undo(self):
        """
        Undo the command.
        Should be overridden by subclasses to revert the action.
        """
        pass

    def get_description(self):
        """
        Get a human-readable description of the command for history display.
        """
        return "Unknown command"

class WaypointCommandMixin:
    """Shared helpers for commands that manipulate waypoint lists."""



    def _assert_waypoint_index(self, vehicle_id, waypoint_index, context):
        """Validate waypoint index before mutating the sequence."""
        waypoints = self.camera_processor.get_vehicle_waypoints(vehicle_id)
        if not waypoints:
            print(f"[WARN] {context}: vehicle {vehicle_id} has no waypoints.")
            return False
        if waypoint_index < 0 or waypoint_index >= len(waypoints):
            print(f"[WARN] {context}: waypoint index {waypoint_index} invalid for vehicle {vehicle_id} (len={len(waypoints)}).")
            return False
        return True
class PlaceWaypointCommand(WaypointCommandMixin, Command):
    """
    Command to place a waypoint for a vehicle. Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle_id, waypoint_data, waypoint_index):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.waypoint_data = clone_waypoint_data(waypoint_data)
        self.waypoint_index = waypoint_index
    
    def execute(self):
        self.camera_processor.append_waypoint_data(self.vehicle_id, self.waypoint_data)
        self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def undo(self):
        waypoints = self.camera_processor.get_vehicle_waypoints(self.vehicle_id)
        if not waypoints:
            return False
        self.camera_processor.remove_waypoint_data(self.vehicle_id, len(waypoints) - 1)
        self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True
    
    def redo(self):
        return self.execute()
    
    def get_description(self):
        return f"Place waypoint #{self.waypoint_index}"

class CompositeCommand(Command):
    """
    Groups multiple commands into a single undoable unit.
    Used for operations that create multiple items at once (e.g., auto-route).
    """
    def __init__(self, commands, description="Composite command"):
        self.commands = commands  # List of Command objects
        self.description = description

    def execute(self):
        for cmd in self.commands:
            cmd.execute()
        return True

    def undo(self):
        # Undo in reverse order
        for cmd in reversed(self.commands):
            cmd.undo()
        return True

    def redo(self):
        return self.execute()

    def get_description(self):
        return self.description

class ClearWaypointsCommand(Command):
    """Clear an actor's whole waypoint path so a fresh one can start; undo restores it.

    Snapshot is taken at construction: the path-replacing flows (auto-route, restarting
    waypoint creation) build this right before executing it.
    """
    def __init__(self, camera_processor, vehicle_id: int):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.old_waypoints = clone_waypoint_sequence(
            camera_processor.get_vehicle_waypoints(vehicle_id))
        self.old_destination_speed = camera_processor.get_vehicle_destination_speed(vehicle_id)

    def execute(self) -> bool:
        self.camera_processor.set_vehicle_waypoints(self.vehicle_id, [])
        self.camera_processor.clear_vehicle_destination_speed(self.vehicle_id)
        return True

    def undo(self) -> bool:
        self.camera_processor.set_vehicle_waypoints(
            self.vehicle_id, clone_waypoint_sequence(self.old_waypoints))
        self.camera_processor.set_vehicle_destination_speed(
            self.vehicle_id, self.old_destination_speed)
        return True

    def redo(self) -> bool:
        return self.execute()

    def get_description(self) -> str:
        return f"Clear {len(self.old_waypoints)} waypoints"

class DeleteWaypointCommand(WaypointCommandMixin, Command):
    """
    Command to delete a waypoint from a vehicle's path. Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle_id, waypoint_index, waypoint_data):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.waypoint_index = waypoint_index
        self.waypoint_data = clone_waypoint_data(waypoint_data)
    
    def execute(self):
        context = "DeleteWaypointCommand.execute"
        if not self._assert_waypoint_index(self.vehicle_id, self.waypoint_index, context):
            return False
        removed = self.camera_processor.remove_waypoint_data(self.vehicle_id, self.waypoint_index)
        if removed is None:
            return False
        remaining = self.camera_processor.get_vehicle_waypoints(self.vehicle_id)
        for index, wp in enumerate(remaining):
            wp['index'] = index + 1
        self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def undo(self):
        container = self.camera_processor.get_vehicle_waypoints(self.vehicle_id)
        insert_index = min(self.waypoint_index, len(container))
        inserted = self.camera_processor.insert_waypoint_data(self.vehicle_id, insert_index, self.waypoint_data)
        if inserted is None:
            return False
        waypoints = self.camera_processor.get_vehicle_waypoints(self.vehicle_id)
        for index, wp in enumerate(waypoints):
            wp['index'] = index + 1
        self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def redo(self):
        return self.execute()
    
    def get_description(self):
        return f"Delete waypoint #{self.waypoint_index + 1}"

class SpawnVehicleCommand(Command):
    """
    Command to spawn a vehicle at a given location and orientation. Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle_type, transform, vehicle_id=None, role="npc"):
        self.camera_processor = camera_processor
        self.vehicle_type = vehicle_type
        self.transform = transform
        self.spawned_vehicle = None
        self.original_vehicle_id = None  # Will be set after first spawn
        self.current_vehicle_id = None   # Track current ID for updates
        self.role = role
   
    def execute(self):
        # Adjust spawn location to proper ground height
        adjusted_transform = carla.Transform(self.transform.location, self.transform.rotation)
        debug_mode = getattr(self.camera_processor, 'debug_raycast', False)
        cached_map = self.camera_processor._get_cached_map(refresh=False)
        world = self.camera_processor.world
        ground_sample = get_ground_height(
            world,
            self.transform.location,
            debug=debug_mode,
            cached_map=cached_map,
            exclude_actors=self.camera_processor.spawned_vehicles,
            return_metadata=True,
        )
        ground_z = ground_sample['height'] if ground_sample.get('height') is not None else self.transform.location.z
        ground_hit = ground_sample.get('source') == 'raycast'

        # Spawn the vehicle using CARLA API
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.vehicle_type)

        is_walker = self.vehicle_type.startswith('walker.')
        # Apply random color to vehicle (not pedestrians)
        if not is_walker:
            if vehicle_bp.has_attribute('color'):
                color_attr = vehicle_bp.get_attribute('color')
                if color_attr.recommended_values:
                    color = random.choice(color_attr.recommended_values)
                    vehicle_bp.set_attribute('color', color)
                    # Store the selected color for saving
                    if not hasattr(self, 'assigned_color'):
                        self.assigned_color = color

        x = self.transform.location.x
        y = self.transform.location.y
        rotation = self.transform.rotation
        if is_walker:
            # bbox is unknown until spawned: spawn clear of the ground, then drop
            # to ground + bbox half-height + pedestrian clearance (transform is
            # body-center anchored).
            self.spawned_vehicle = world.try_spawn_actor(
                vehicle_bp, carla.Transform(carla.Location(x, y, ground_z + 1.5), rotation)
            )
            if self.spawned_vehicle:
                try:
                    bbox_z = float(getattr(self.spawned_vehicle.bounding_box.extent, "z", 1.0))
                except Exception:
                    bbox_z = 1.0
                # Offset only on a real raycast hit; a miss echoes the
                # requested Z back as "ground", and stacking bbox+clearance on
                # that floats the walker (fix-03, mirrors the drag-end gate).
                if ground_hit:
                    adjusted_transform.location.z = ground_z + bbox_z + PEDESTRIAN_PLACEMENT_CLEARANCE_M
                else:
                    adjusted_transform.location.z = self.transform.location.z
                self.spawned_vehicle.set_transform(
                    carla.Transform(carla.Location(x, y, adjusted_transform.location.z), rotation)
                )
        else:
            # Spawn at the lowest collision-free height so the saved Z survives
            # the post-Play restore (which respawns via try_spawn_actor).
            self.spawned_vehicle, veh_z = spawn_vehicle_grounded(
                world, vehicle_bp, x, y, rotation, ground_z
            )
            if self.spawned_vehicle and veh_z is not None:
                adjusted_transform.location.z = veh_z

        if self.spawned_vehicle:
            # Disable physics for scenario editing
            self.spawned_vehicle.set_simulate_physics(False)
            self.camera_processor.spawned_vehicles.append(self.spawned_vehicle)
            default_speed = 5 if self.vehicle_type.startswith('walker.') else 50
            if self.role == "ego" and not self.vehicle_type.startswith('walker.'):
                default_speed = 40
            color = getattr(self, 'assigned_color', None)
            self.camera_processor.initialize_vehicle_metadata(
                self.spawned_vehicle.id,
                speed=default_speed,
                destination_speed=None,
                idle_time=0.0,
                color=color,
                ignore_flags=None,
                max_lat_acc=3.0,
            )
            self.camera_processor.vehicle_transforms[self.spawned_vehicle.id] = carla.Transform(
                adjusted_transform.location, adjusted_transform.rotation
            )

            if self.role == "ego":
                self.camera_processor.register_ego_vehicle(
                    self.spawned_vehicle,
                    adjusted_transform,
                    getattr(self, 'assigned_color', None)
                )

            # Track original and current vehicle IDs for undo/redo
            if self.original_vehicle_id is None:
                self.original_vehicle_id = self.spawned_vehicle.id
            self.current_vehicle_id = self.spawned_vehicle.id
            print(f"Vehicle spawned successfully at adjusted height: {adjusted_transform.location.z:.2f}")
        else:
            print(f"Failed to spawn vehicle {self.vehicle_type} - location might still be blocked")

        return self.spawned_vehicle is not None
    
    def undo(self):
        # Remove the vehicle
        if self.spawned_vehicle and self.spawned_vehicle.is_alive:
            if self.spawned_vehicle in self.camera_processor.spawned_vehicles:
                self.camera_processor.spawned_vehicles.remove(self.spawned_vehicle)
            self.camera_processor.clear_vehicle_metadata(self.spawned_vehicle.id, clear_waypoints=True)
            if self.role == "ego":
                self.camera_processor.clear_ego_vehicle(self.spawned_vehicle.id)
            self.spawned_vehicle.destroy()
            self.spawned_vehicle = None
    
    def redo(self):
        # Re-spawn the vehicle (same as execute but track ID changes)
        old_id = self.current_vehicle_id

        # Adjust spawn location to proper ground height
        adjusted_transform = carla.Transform(self.transform.location, self.transform.rotation)
        debug_mode = getattr(self.camera_processor, 'debug_raycast', False)
        cached_map = self.camera_processor._get_cached_map(refresh=False)
        world = self.camera_processor.world
        ground_sample = get_ground_height(
            world,
            self.transform.location,
            debug=debug_mode,
            cached_map=cached_map,
            exclude_actors=self.camera_processor.spawned_vehicles,
            return_metadata=True,
        )
        ground_z = ground_sample['height'] if ground_sample.get('height') is not None else self.transform.location.z
        ground_hit = ground_sample.get('source') == 'raycast'

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.vehicle_type)

        is_walker = self.vehicle_type.startswith('walker.')
        # Apply the same color that was assigned during execute
        if not is_walker and hasattr(self, 'assigned_color'):
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', self.assigned_color)

        x = self.transform.location.x
        y = self.transform.location.y
        rotation = self.transform.rotation
        if is_walker:
            self.spawned_vehicle = world.try_spawn_actor(
                vehicle_bp, carla.Transform(carla.Location(x, y, ground_z + 1.5), rotation)
            )
            if self.spawned_vehicle:
                try:
                    bbox_z = float(getattr(self.spawned_vehicle.bounding_box.extent, "z", 1.0))
                except Exception:
                    bbox_z = 1.0
                # Offset only on a real raycast hit; a miss echoes the
                # requested Z back as "ground", and stacking bbox+clearance on
                # that floats the walker (fix-03, mirrors the drag-end gate).
                if ground_hit:
                    adjusted_transform.location.z = ground_z + bbox_z + PEDESTRIAN_PLACEMENT_CLEARANCE_M
                else:
                    adjusted_transform.location.z = self.transform.location.z
                self.spawned_vehicle.set_transform(
                    carla.Transform(carla.Location(x, y, adjusted_transform.location.z), rotation)
                )
        else:
            self.spawned_vehicle, veh_z = spawn_vehicle_grounded(
                world, vehicle_bp, x, y, rotation, ground_z
            )
            if self.spawned_vehicle and veh_z is not None:
                adjusted_transform.location.z = veh_z

        if self.spawned_vehicle:
            self.spawned_vehicle.set_simulate_physics(False)
            self.camera_processor.spawned_vehicles.append(self.spawned_vehicle)
            default_speed = 5 if self.vehicle_type.startswith('walker.') else 50
            if self.role == "ego" and not self.vehicle_type.startswith('walker.'):
                default_speed = 40
            color = getattr(self, 'assigned_color', None)
            self.camera_processor.initialize_vehicle_metadata(
                self.spawned_vehicle.id,
                speed=default_speed,
                destination_speed=None,
                idle_time=0.0,
                color=color,
                ignore_flags=None,
                max_lat_acc=3.0,
            )
            self.camera_processor.vehicle_transforms[self.spawned_vehicle.id] = carla.Transform(
                adjusted_transform.location, adjusted_transform.rotation
            )

            if self.role == "ego":
                self.camera_processor.register_ego_vehicle(
                    self.spawned_vehicle,
                    adjusted_transform,
                    getattr(self, 'assigned_color', None)
                )

            # Update current vehicle ID and references if changed
            self.current_vehicle_id = self.spawned_vehicle.id
            if old_id and old_id != self.current_vehicle_id:
                self.camera_processor.update_vehicle_id_references(old_id, self.current_vehicle_id)
            print(f"Vehicle re-spawned successfully at adjusted height: {adjusted_transform.location.z:.2f}")
        else:
            print(f"Failed to re-spawn vehicle {self.vehicle_type} - location might still be blocked")
            
        return self.spawned_vehicle is not None
    
    def get_description(self):
        vehicle_label = self.vehicle_type.split('.')[-1] if self.vehicle_type else "unknown"
        if self.current_vehicle_id:
            return f"Spawn vehicle ({vehicle_label})"
        else:
            return f"Spawn vehicle ({vehicle_label})"

class DeleteVehicleCommand(Command):
    """
    Command to delete a vehicle and its waypoints. Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle, vehicle_type, transform, waypoints=None, speed=50):
        self.camera_processor = camera_processor
        self.vehicle = vehicle
        self.vehicle_type = vehicle_type
        self.transform = transform
        self.waypoints = clone_waypoint_sequence(waypoints) if waypoints else []
        self.speed = speed
        self.vehicle_id = vehicle.id if vehicle else None
        self.idle_time = camera_processor.get_actor_idle_time(vehicle.id, 0.0) if vehicle else 0.0
        self.destination_speed = camera_processor.get_vehicle_destination_speed(vehicle.id) if vehicle else None
        self.color = camera_processor.get_vehicle_color(vehicle.id) if vehicle else None
        self.ignore_flags = camera_processor.get_vehicle_ignore_flags(vehicle.id) if vehicle else ensure_ignore_flags(None)
        self.was_ego = camera_processor.is_ego_vehicle(self.vehicle_id) if vehicle else False
        self.saved_trigger_center = None
        self.saved_trigger_radius = None
        self.saved_trigger_kind = None
        if vehicle:
            if vehicle.type_id.startswith('walker.'):
                center = camera_processor.pedestrian_trigger_centers.get(vehicle.id)
                radius = camera_processor.pedestrian_trigger_radii.get(vehicle.id)
                if center and radius is not None:
                    self.saved_trigger_center = dict(center)
                    self.saved_trigger_radius = float(radius)
                    self.saved_trigger_kind = 'pedestrian'
            else:
                center = camera_processor.vehicle_trigger_centers.get(vehicle.id)
                radius = camera_processor.vehicle_trigger_radii.get(vehicle.id)
                if center and radius is not None:
                    self.saved_trigger_center = dict(center)
                    self.saved_trigger_radius = float(radius)
                    self.saved_trigger_kind = 'vehicle'
    
    def execute(self):
        # Delete the vehicle
        if self.vehicle and self.vehicle.is_alive:
            self.camera_processor.assert_spawned_vehicle(self.vehicle_id, "DeleteVehicleCommand.execute")
            # Capture latest metadata before removal
            self.speed = self.camera_processor.get_vehicle_speed(self.vehicle.id, self.speed)
            self.idle_time = self.camera_processor.get_actor_idle_time(self.vehicle.id, self.idle_time)
            self.destination_speed = self.camera_processor.get_vehicle_destination_speed(self.vehicle.id)
            self.color = self.camera_processor.get_vehicle_color(self.vehicle.id)
            self.ignore_flags = self.camera_processor.get_vehicle_ignore_flags(self.vehicle.id)
            self.waypoints = clone_waypoint_sequence(self.camera_processor.get_vehicle_waypoints(self.vehicle.id))
            if self.vehicle.type_id.startswith('walker.'):
                center = self.camera_processor.pedestrian_trigger_centers.get(self.vehicle.id)
                radius = self.camera_processor.pedestrian_trigger_radii.get(self.vehicle.id)
                if center and radius is not None:
                    self.saved_trigger_center = dict(center)
                    self.saved_trigger_radius = float(radius)
                    self.saved_trigger_kind = 'pedestrian'
            else:
                center = self.camera_processor.vehicle_trigger_centers.get(self.vehicle.id)
                radius = self.camera_processor.vehicle_trigger_radii.get(self.vehicle.id)
                if center and radius is not None:
                    self.saved_trigger_center = dict(center)
                    self.saved_trigger_radius = float(radius)
                    self.saved_trigger_kind = 'vehicle'

            if self.camera_processor.is_manual_control_actor(self.vehicle.id):
                self.camera_processor.disable_manual_control()

            if self.vehicle in self.camera_processor.spawned_vehicles:
                self.camera_processor.spawned_vehicles.remove(self.vehicle)
            self.camera_processor.clear_vehicle_metadata(self.vehicle.id, clear_waypoints=True)
            if self.was_ego:
                self.camera_processor.clear_ego_vehicle(self.vehicle.id)
            self.vehicle.destroy()

    def undo(self):
        # Respawn the vehicle with adjusted height
        adjusted_transform = carla.Transform(self.transform.location, self.transform.rotation)
        debug_mode = getattr(self.camera_processor, 'debug_raycast', False)
        cached_map = getattr(self.camera_processor, 'cached_map', None)
        world = self.camera_processor.world
        ground_sample = get_ground_height(
            world,
            self.transform.location,
            debug=debug_mode,
            cached_map=cached_map,
            exclude_actors=self.camera_processor.spawned_vehicles,
            return_metadata=True,
        )
        ground_z = ground_sample['height'] if ground_sample.get('height') is not None else self.transform.location.z
        ground_hit = ground_sample.get('source') == 'raycast'

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.vehicle_type)

        x = self.transform.location.x
        y = self.transform.location.y
        rotation = self.transform.rotation
        if self.vehicle_type.startswith('walker.'):
            self.vehicle = world.try_spawn_actor(
                vehicle_bp, carla.Transform(carla.Location(x, y, ground_z + 1.5), rotation)
            )
            if self.vehicle:
                try:
                    bbox_z = float(getattr(self.vehicle.bounding_box.extent, "z", 1.0))
                except Exception:
                    bbox_z = 1.0
                # Offset only on a real raycast hit; a miss echoes the
                # requested Z back as "ground", and stacking bbox+clearance on
                # that floats the walker (fix-03, mirrors the drag-end gate).
                if ground_hit:
                    adjusted_transform.location.z = ground_z + bbox_z + PEDESTRIAN_PLACEMENT_CLEARANCE_M
                else:
                    adjusted_transform.location.z = self.transform.location.z
                self.vehicle.set_transform(
                    carla.Transform(carla.Location(x, y, adjusted_transform.location.z), rotation)
                )
        else:
            self.vehicle, veh_z = spawn_vehicle_grounded(
                world, vehicle_bp, x, y, rotation, ground_z
            )
            if self.vehicle and veh_z is not None:
                adjusted_transform.location.z = veh_z

        if self.vehicle:
            self.vehicle.set_simulate_physics(False)
            self.camera_processor.spawned_vehicles.append(self.vehicle)
            old_id = self.vehicle_id
            self.camera_processor.initialize_vehicle_metadata(
                self.vehicle.id,
                speed=self.speed,
                destination_speed=self.destination_speed,
                idle_time=self.idle_time,
                color=self.color,
                ignore_flags=self.ignore_flags,
            )
            self.camera_processor.vehicle_transforms[self.vehicle.id] = carla.Transform(
                adjusted_transform.location, adjusted_transform.rotation
            )
            new_id = self.vehicle.id
            if self.vehicle.type_id.startswith('walker.'):
                if self.saved_trigger_kind == 'pedestrian' and self.saved_trigger_center and self.saved_trigger_radius is not None:
                    self.camera_processor.pedestrian_trigger_centers[new_id] = dict(self.saved_trigger_center)
                    self.camera_processor.pedestrian_trigger_radii[new_id] = max(MIN_PERSONAL_TRIGGER_RADIUS, float(self.saved_trigger_radius))
            else:
                if not self.camera_processor.is_ego_vehicle(new_id):
                    if self.saved_trigger_kind == 'vehicle' and self.saved_trigger_center and self.saved_trigger_radius is not None:
                        self.camera_processor.vehicle_trigger_centers[new_id] = dict(self.saved_trigger_center)
                        self.camera_processor.vehicle_trigger_radii[new_id] = max(MIN_PERSONAL_TRIGGER_RADIUS, float(self.saved_trigger_radius))

            if self.waypoints:
                self.camera_processor.set_vehicle_waypoints(self.vehicle.id, clone_waypoint_sequence(self.waypoints))
                self.camera_processor._cache_destination_speed(self.vehicle.id)
            else:
                self.camera_processor.clear_vehicle_waypoints(self.vehicle.id)
            if self.was_ego:
                self.camera_processor.register_ego_vehicle(self.vehicle, adjusted_transform)
            if old_id and old_id != self.vehicle.id:
                self.camera_processor.update_vehicle_id_references(old_id, self.vehicle.id)
            self.vehicle_id = self.vehicle.id
            print(f"Vehicle restored at adjusted height: {adjusted_transform.location.z:.2f}")
        else:
            print(f"Failed to restore vehicle {self.vehicle_type} - location might be blocked")
    
    def redo(self):
        # Re-execute the deletion
        if self.vehicle and self.vehicle.is_alive:
            if self.vehicle in self.camera_processor.spawned_vehicles:
                self.camera_processor.spawned_vehicles.remove(self.vehicle)
            if self.camera_processor.is_manual_control_actor(self.vehicle.id):
                self.camera_processor.disable_manual_control()
            self.camera_processor.clear_vehicle_metadata(self.vehicle.id, clear_waypoints=True)
            if self.was_ego:
                self.camera_processor.clear_ego_vehicle(self.vehicle.id)
            self.vehicle.destroy()
    
    def get_description(self):
        return f"Delete vehicle ({self.vehicle_type.split('.')[-1]})"


class ReplaceActorBlueprintCommand(Command):
    """Replace one non-ego actor's blueprint while retaining authored state."""

    def __init__(self, camera_processor, vehicle_id: int, new_type_id: str):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.new_type_id = new_type_id
        self._current_actor = camera_processor.get_spawned_vehicle(vehicle_id)
        self.original_type_id = (
            self._current_actor.type_id if self._current_actor else None
        )
        self.current_type_id = self.original_type_id
        self._snapshot = (
            self._capture_snapshot(self._current_actor)
            if self._current_actor else None
        )

    @staticmethod
    def _is_pedestrian_type(type_id: str) -> bool:
        """Return whether a blueprint belongs to the pedestrian category."""
        return type_id.startswith('walker.')

    def _capture_snapshot(self, actor) -> Dict[str, object]:
        processor = self.camera_processor
        actor_id = actor.id
        stored_transform = processor.vehicle_transforms.get(actor_id)
        transform = stored_transform if stored_transform is not None else actor.get_transform()
        is_pedestrian = self._is_pedestrian_type(actor.type_id)
        if is_pedestrian:
            trigger_center = processor.pedestrian_trigger_centers.get(actor_id)
            trigger_radius = processor.pedestrian_trigger_radii.get(actor_id)
            trigger_kind = 'pedestrian'
        else:
            trigger_center = processor.vehicle_trigger_centers.get(actor_id)
            trigger_radius = processor.vehicle_trigger_radii.get(actor_id)
            trigger_kind = 'vehicle'

        return {
            'transform': _clone_transform(transform),
            'waypoints': clone_waypoint_sequence(processor.get_vehicle_waypoints(actor_id)),
            'speed': processor.get_vehicle_speed(
                actor_id,
                5 if is_pedestrian else 50,
            ),
            'destination_speed': processor.get_vehicle_destination_speed(actor_id),
            'idle_time': processor.get_actor_idle_time(actor_id, 0.0),
            'color': processor.get_vehicle_color(actor_id),
            'ignore_flags': copy.deepcopy(processor.get_vehicle_ignore_flags(actor_id)),
            'max_lat_acc': processor.get_vehicle_max_lat_acc(actor_id, 3.0),
            'trigger_kind': trigger_kind,
            'trigger_center': copy.deepcopy(trigger_center) if trigger_center else None,
            'trigger_radius': float(trigger_radius) if trigger_radius is not None else None,
        }

    def _spawn_actor(self, type_id: str) -> Optional[object]:
        processor = self.camera_processor
        world = processor.world
        snapshot = self._snapshot
        if snapshot is None:
            return None

        actor = None
        try:
            blueprint = world.get_blueprint_library().find(type_id)
            color = snapshot.get('color')
            if color and not self._is_pedestrian_type(type_id) and blueprint.has_attribute('color'):
                blueprint.set_attribute('color', color)
            requested_transform = _clone_transform(snapshot['transform'])
            debug_mode = getattr(processor, 'debug_raycast', False)
            cached_map = processor._get_cached_map(refresh=False)
            ground_sample = get_ground_height(
                world,
                requested_transform.location,
                debug=debug_mode,
                cached_map=cached_map,
                exclude_actors=processor.spawned_vehicles,
                return_metadata=True,
            )
            ground_z = (
                ground_sample['height']
                if ground_sample.get('height') is not None
                else requested_transform.location.z
            )
            ground_hit = ground_sample.get('source') == 'raycast'
            x = requested_transform.location.x
            y = requested_transform.location.y
            rotation = requested_transform.rotation
            if self._is_pedestrian_type(type_id):
                actor = world.try_spawn_actor(
                    blueprint,
                    carla.Transform(carla.Location(x, y, ground_z + 1.5), rotation),
                )
                if actor:
                    try:
                        bbox_z = float(getattr(actor.bounding_box.extent, 'z', 1.0))
                    except Exception:
                        bbox_z = 1.0
                    replacement_z = (
                        ground_z + bbox_z + PEDESTRIAN_PLACEMENT_CLEARANCE_M
                        if ground_hit else requested_transform.location.z
                    )
                    self._last_spawn_transform = carla.Transform(
                        carla.Location(x, y, replacement_z),
                        rotation,
                    )
                    actor.set_transform(_clone_transform(self._last_spawn_transform))
            else:
                actor, vehicle_z = spawn_vehicle_grounded(
                    world,
                    blueprint,
                    x,
                    y,
                    rotation,
                    ground_z,
                )
                if actor and vehicle_z is not None:
                    self._last_spawn_transform = carla.Transform(
                        carla.Location(x, y, vehicle_z),
                        rotation,
                    )
            if actor is None:
                return None
            actor.set_simulate_physics(False)
            return actor
        except Exception as exc:
            print(f"Failed to spawn replacement actor {type_id}: {exc}")
            if actor is not None and getattr(actor, 'is_alive', False):
                try:
                    actor.destroy()
                except Exception as destroy_exc:
                    print(f"Failed to clean up replacement actor: {destroy_exc}")
            return None

    def _destroy_actor(self, actor) -> bool:
        try:
            actor.destroy()
        except Exception as exc:
            print(f"Failed to remove actor {getattr(actor, 'id', 'unknown')}: {exc}")
            return False

        if getattr(actor, 'is_alive', False):
            print(f"Failed to remove actor {getattr(actor, 'id', 'unknown')}")
            return False
        if actor in self.camera_processor.spawned_vehicles:
            self.camera_processor.spawned_vehicles.remove(actor)
        return True

    def _restore_metadata(self, actor_id: int, type_id: str) -> None:
        processor = self.camera_processor
        snapshot = self._snapshot
        if snapshot is None:
            return

        processor.initialize_vehicle_metadata(
            actor_id,
            speed=snapshot['speed'],
            destination_speed=snapshot['destination_speed'],
            idle_time=snapshot['idle_time'],
            color=snapshot['color'],
            ignore_flags=snapshot['ignore_flags'],
            max_lat_acc=snapshot['max_lat_acc'],
        )
        processor.vehicle_transforms[actor_id] = _clone_transform(
            getattr(self, '_last_spawn_transform', snapshot['transform'])
        )

        waypoints = clone_waypoint_sequence(snapshot['waypoints'])
        if waypoints:
            processor.set_vehicle_waypoints(actor_id, waypoints)
        else:
            processor.clear_vehicle_waypoints(actor_id)

        if self._is_pedestrian_type(type_id):
            centers = processor.pedestrian_trigger_centers
            radii = processor.pedestrian_trigger_radii
        else:
            centers = processor.vehicle_trigger_centers
            radii = processor.vehicle_trigger_radii

        centers.pop(actor_id, None)
        radii.pop(actor_id, None)
        if snapshot['trigger_center'] and snapshot['trigger_radius'] is not None:
            centers[actor_id] = copy.deepcopy(snapshot['trigger_center'])
            radii[actor_id] = max(
                MIN_PERSONAL_TRIGGER_RADIUS,
                float(snapshot['trigger_radius']),
            )

    def _replace_current_actor(self, target_type_id: str) -> bool:
        processor = self.camera_processor
        actor = self._current_actor
        if not actor or not getattr(actor, 'is_alive', False):
            actor = processor.get_spawned_vehicle(self.vehicle_id)
        if not actor or not getattr(actor, 'is_alive', False):
            print(f"Cannot replace missing actor {self.vehicle_id}")
            return False
        if processor.is_ego_vehicle(actor.id):
            print("Cannot replace the ego vehicle blueprint from the Info panel")
            return False
        if self._is_pedestrian_type(actor.type_id) != self._is_pedestrian_type(target_type_id):
            print("Actor blueprint replacement must stay within the same category")
            return False
        if actor.type_id == target_type_id:
            return False

        old_id = actor.id
        old_type_id = actor.type_id
        if not self._destroy_actor(actor):
            return False

        replacement = self._spawn_actor(target_type_id)
        if replacement is None:
            print(f"Failed to replace actor with {target_type_id}; restoring {old_type_id}")
            replacement = self._spawn_actor(old_type_id)
            if replacement is None:
                print(f"Failed to restore actor {old_type_id} after replacement failure")
                self._current_actor = None
                return False
            processor.spawned_vehicles.append(replacement)
            processor.update_vehicle_id_references(old_id, replacement.id, replacement)
            self._restore_metadata(replacement.id, old_type_id)
            self._current_actor = replacement
            self.vehicle_id = replacement.id
            self.current_type_id = old_type_id
            return False

        processor.spawned_vehicles.append(replacement)
        processor.update_vehicle_id_references(old_id, replacement.id, replacement)
        self._restore_metadata(replacement.id, target_type_id)
        self._current_actor = replacement
        self.vehicle_id = replacement.id
        self.current_type_id = target_type_id
        return True

    def execute(self):
        if not self._snapshot or not self.original_type_id:
            return False
        return self._replace_current_actor(self.new_type_id)

    def undo(self):
        if not self._snapshot or not self.original_type_id:
            return False
        return self._replace_current_actor(self.original_type_id)

    def redo(self):
        return self.execute()

    def get_description(self):
        old_type = self.current_type_id or 'unknown'
        new_type = self.new_type_id or 'unknown'
        return f"Change actor blueprint ({old_type.split('.')[-1]} -> {new_type.split('.')[-1]})"


class MoveWaypointCommand(WaypointCommandMixin, Command):
    """
    Command to move a waypoint to a new position. Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle_id, waypoint_index, old_position, new_position):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.waypoint_index = waypoint_index
        self.old_position = copy.deepcopy(old_position)
        self.new_position = copy.deepcopy(new_position)
    
    def execute(self):
        # Move waypoint to new position
        context = "MoveWaypointCommand.execute"
        if not self._assert_waypoint_index(self.vehicle_id, self.waypoint_index, context):
            return False
        previous = self.camera_processor.update_waypoint_fields(
            self.vehicle_id,
            self.waypoint_index,
            self.new_position,
        )
        if previous is None:
            return False
        if self.waypoint_index == 0:
            self.camera_processor._adjust_pedestrian_spawn_orientation(self.vehicle_id)
        return True

    def undo(self):
        # Move waypoint back to old position
        context = "MoveWaypointCommand.undo"
        if not self._assert_waypoint_index(self.vehicle_id, self.waypoint_index, context):
            return False
        updated = self.camera_processor.update_waypoint_fields(
            self.vehicle_id,
            self.waypoint_index,
            self.old_position,
        )
        if updated is None:
            return False
        if self.waypoint_index == 0:
            self.camera_processor._adjust_pedestrian_spawn_orientation(self.vehicle_id)
        return True

    def redo(self):
        # Re-execute the move (same as execute)
        context = "MoveWaypointCommand.redo"
        if not self._assert_waypoint_index(self.vehicle_id, self.waypoint_index, context):
            return False
        previous = self.camera_processor.update_waypoint_fields(
            self.vehicle_id,
            self.waypoint_index,
            self.new_position,
        )
        if previous is None:
            return False
        if self.waypoint_index == 0:
            self.camera_processor._adjust_pedestrian_spawn_orientation(self.vehicle_id)
        return True
    
    def get_description(self):
        return f"Move waypoint #{self.waypoint_index + 1}"

class SplitWaypointCommand(WaypointCommandMixin, Command):
    """Command that splits a waypoint into two offset copies."""

    def __init__(self, camera_processor, vehicle_id, waypoint_index, original_waypoint, offset_distance=1.5):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.waypoint_index = waypoint_index
        self.original_waypoint = clone_waypoint_data(original_waypoint)
        self.offset_distance = offset_distance
        self.generated_waypoints: Optional[List[WaypointData]] = None

    @staticmethod
    def _direction_and_distance(source: WaypointData, target: Optional[WaypointData]):
        if not target:
            return None, None
        dx = target['x'] - source['x']
        dy = target['y'] - source['y']
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            return None, distance
        return (dx / distance, dy / distance), distance

    def _resolve_offset(self, distance: Optional[float]) -> float:
        if distance is None or distance <= 1e-3:
            return self.offset_distance
        limited = min(self.offset_distance, distance * 0.45)
        return max(limited, 0.05)

    def _build_split_pair(self, prev_wp: Optional[WaypointData], base_wp: WaypointData, next_wp: Optional[WaypointData]):
        dir_prev, dist_prev = self._direction_and_distance(base_wp, prev_wp)
        dir_next, dist_next = self._direction_and_distance(base_wp, next_wp)

        if dir_prev is None and dir_next is not None:
            dir_prev = (-dir_next[0], -dir_next[1])
            dist_prev = dist_next
        if dir_next is None and dir_prev is not None:
            dir_next = (-dir_prev[0], -dir_prev[1])
            dist_next = dist_prev
        if dir_prev is None:
            dir_prev = (-1.0, 0.0)
        if dir_next is None:
            dir_next = (1.0, 0.0)

        offset_prev = self._resolve_offset(dist_prev)
        offset_next = self._resolve_offset(dist_next)
        origin_z = base_wp.get('z', 0.0)

        prev_split = clone_waypoint_data(base_wp)
        prev_split['x'] = base_wp['x'] + dir_prev[0] * offset_prev
        prev_split['y'] = base_wp['y'] + dir_prev[1] * offset_prev
        prev_split['z'] = origin_z
        prev_split['is_destination'] = False

        next_split = clone_waypoint_data(base_wp)
        next_split['x'] = base_wp['x'] + dir_next[0] * offset_next
        next_split['y'] = base_wp['y'] + dir_next[1] * offset_next
        next_split['z'] = origin_z
        next_split['is_destination'] = base_wp.get('is_destination', False)
        return prev_split, next_split

    def _apply_split(self, *, reuse_pair=False):
        waypoints = self.camera_processor.get_vehicle_waypoints(self.vehicle_id)
        context = "SplitWaypointCommand"
        if not waypoints or self.waypoint_index < 0 or self.waypoint_index >= len(waypoints):
            print(f"[WARN] {context}: invalid waypoint index {self.waypoint_index} for vehicle {self.vehicle_id}")
            return False

        if reuse_pair:
            if not self.generated_waypoints:
                return False
            split_pair = [clone_waypoint_data(wp) for wp in self.generated_waypoints]
        else:
            prev_wp = waypoints[self.waypoint_index - 1] if self.waypoint_index > 0 else None
            next_wp = waypoints[self.waypoint_index + 1] if self.waypoint_index + 1 < len(waypoints) else None
            split_pair = list(self._build_split_pair(prev_wp, waypoints[self.waypoint_index], next_wp))
            self.generated_waypoints = [clone_waypoint_data(wp) for wp in split_pair]

        waypoints.pop(self.waypoint_index)
        waypoints.insert(self.waypoint_index, split_pair[0])
        waypoints.insert(self.waypoint_index + 1, split_pair[1])

        for index, waypoint in enumerate(waypoints):
            waypoint['index'] = index + 1
        self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def execute(self):
        return self._apply_split(reuse_pair=False)

    def undo(self):
        waypoints = self.camera_processor.get_vehicle_waypoints(self.vehicle_id)
        if not waypoints:
            return False
        removal_count = len(self.generated_waypoints or [])
        for _ in range(removal_count):
            if self.waypoint_index < len(waypoints):
                waypoints.pop(self.waypoint_index)
        waypoints.insert(self.waypoint_index, clone_waypoint_data(self.original_waypoint))
        for index, waypoint in enumerate(waypoints):
            waypoint['index'] = index + 1
        self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def redo(self):
        return self._apply_split(reuse_pair=True)

    def get_description(self):
        return f"Split waypoint #{self.waypoint_index + 1}"

class MoveVehicleCommand(Command):
    """
    Command to move a vehicle to a new position/rotation. Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle, old_transform, new_transform):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle.id if vehicle else None
        self.old_transform = old_transform
        self.new_transform = new_transform
    
    def get_current_vehicle(self):
        """Get the current vehicle object by ID"""
        return self.camera_processor.get_spawned_vehicle(self.vehicle_id)

    def execute(self):
        # Move vehicle to new position
        vehicle = self.get_current_vehicle()
        if vehicle and self.camera_processor.assert_spawned_vehicle(self.vehicle_id, "MoveVehicleCommand.execute"):
            vehicle.set_transform(self.new_transform)
            self.camera_processor.vehicle_transforms[vehicle.id] = carla.Transform(
                self.new_transform.location, self.new_transform.rotation
            )
            # Pedestrians always face their first waypoint; re-derive heading after the move
            # (no-op for vehicles and for pedestrians without waypoints).
            self.camera_processor._adjust_pedestrian_spawn_orientation(vehicle.id)
            if self.camera_processor.is_ego_vehicle(vehicle.id):
                self.camera_processor.update_editor_ego_transform(self.new_transform)
            if vehicle is self.camera_processor.selected_vehicle:
                self.camera_processor.refresh_selected_vehicle_ui()

    def undo(self):
        # Move vehicle back to old position
        vehicle = self.get_current_vehicle()
        if vehicle and self.camera_processor.assert_spawned_vehicle(self.vehicle_id, "MoveVehicleCommand.undo"):
            vehicle.set_transform(self.old_transform)
            self.camera_processor.vehicle_transforms[vehicle.id] = carla.Transform(
                self.old_transform.location, self.old_transform.rotation
            )
            # Re-derive pedestrian heading at the restored position (no-op for vehicles).
            self.camera_processor._adjust_pedestrian_spawn_orientation(vehicle.id)
            if self.camera_processor.is_ego_vehicle(vehicle.id):
                self.camera_processor.update_editor_ego_transform(self.old_transform)
            if vehicle is self.camera_processor.selected_vehicle:
                self.camera_processor.refresh_selected_vehicle_ui()

    def redo(self):
        # Re-execute the move (same as execute)
        vehicle = self.get_current_vehicle()
        if vehicle and self.camera_processor.assert_spawned_vehicle(self.vehicle_id, "MoveVehicleCommand.redo"):
            vehicle.set_transform(self.new_transform)
            self.camera_processor.vehicle_transforms[vehicle.id] = carla.Transform(
                self.new_transform.location, self.new_transform.rotation
            )
            # Re-derive pedestrian heading at the restored position (no-op for vehicles).
            self.camera_processor._adjust_pedestrian_spawn_orientation(vehicle.id)
            if self.camera_processor.is_ego_vehicle(vehicle.id):
                self.camera_processor.update_editor_ego_transform(self.new_transform)
            if vehicle is self.camera_processor.selected_vehicle:
                self.camera_processor.refresh_selected_vehicle_ui()

    def get_description(self):
        return "Move vehicle"

class UpdateWaypointPropertyCommand(WaypointCommandMixin, Command):
    """
    Command to update a property of a waypoint (e.g., position, speed). Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle_id, waypoint_index, property_name, old_value, new_value):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.waypoint_index = waypoint_index
        self.property_name = property_name
        self.old_value = old_value
        self.new_value = new_value

    def execute(self):
        context = f"UpdateWaypointPropertyCommand.execute({self.property_name})"
        if not self._assert_waypoint_index(self.vehicle_id, self.waypoint_index, context):
            return False
        previous = self.camera_processor.update_waypoint_fields(
            self.vehicle_id,
            self.waypoint_index,
            {self.property_name: self.new_value},
        )
        if previous is None:
            return False
        if self.waypoint_index == 0 and self.property_name in ('x', 'y', 'z'):
            self.camera_processor._adjust_pedestrian_spawn_orientation(self.vehicle_id)
        if self.property_name in ('speed_km_h', 'is_destination'):
            self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def undo(self):
        context = f"UpdateWaypointPropertyCommand.undo({self.property_name})"
        if not self._assert_waypoint_index(self.vehicle_id, self.waypoint_index, context):
            return False
        previous = self.camera_processor.update_waypoint_fields(
            self.vehicle_id,
            self.waypoint_index,
            {self.property_name: self.old_value},
        )
        if previous is None:
            return False
        if self.waypoint_index == 0 and self.property_name in ('x', 'y', 'z'):
            self.camera_processor._adjust_pedestrian_spawn_orientation(self.vehicle_id)
        if self.property_name in ('speed_km_h', 'is_destination'):
            self.camera_processor._cache_destination_speed(self.vehicle_id)
        return True

    def redo(self):
        # Re-execute the update
        return self.execute()

    def get_description(self):
        return f"Update waypoint {self.property_name}: {self.old_value} -> {self.new_value}"

class UpdateVehiclePropertyCommand(Command):
    """
    Command to update a property of a vehicle (e.g., position, speed, yaw). Supports undo/redo.
    """
    def __init__(self, camera_processor, vehicle_id, property_name, old_value, new_value, old_transform=None, new_transform=None):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.property_name = property_name
        self.old_value = old_value
        self.new_value = new_value
        self.old_transform = old_transform
        self.new_transform = new_transform
    
    def get_current_vehicle(self):
        """Get the current vehicle object by ID"""
        return self.camera_processor.get_spawned_vehicle(self.vehicle_id)
    
    def execute(self):
        # Update vehicle property
        vehicle = self.get_current_vehicle()
        if vehicle and self.camera_processor.assert_spawned_vehicle(self.vehicle_id, f"UpdateVehiclePropertyCommand.execute({self.property_name})"):
            if self.property_name == 'speed_km_h':
                self.camera_processor.set_vehicle_speed(vehicle.id, self.new_value)
            elif self.property_name == 'idle_time_s':
                self.camera_processor.set_actor_idle_time(vehicle.id, self.new_value)
            elif self.property_name == 'max_lat_acc':
                self.camera_processor.set_vehicle_max_lat_acc(vehicle.id, self.new_value)
            elif self.new_transform:
                vehicle.set_transform(self.new_transform)
                self.camera_processor.vehicle_transforms[vehicle.id] = carla.Transform(
                    self.new_transform.location, self.new_transform.rotation
                )
                # Pedestrians always face their first waypoint after a position edit
                # (no-op for vehicles and for pedestrians without waypoints).
                self.camera_processor._adjust_pedestrian_spawn_orientation(vehicle.id)
                if self.camera_processor.is_ego_vehicle(vehicle.id):
                    self.camera_processor.update_editor_ego_transform(self.new_transform)
                if vehicle is self.camera_processor.selected_vehicle:
                    self.camera_processor.refresh_selected_vehicle_ui()

    def undo(self):
        # Restore old value
        vehicle = self.get_current_vehicle()
        if vehicle and self.camera_processor.assert_spawned_vehicle(self.vehicle_id, f"UpdateVehiclePropertyCommand.undo({self.property_name})"):
            if self.property_name == 'speed_km_h':
                self.camera_processor.set_vehicle_speed(vehicle.id, self.old_value)
            elif self.property_name == 'idle_time_s':
                self.camera_processor.set_actor_idle_time(vehicle.id, self.old_value)
            elif self.property_name == 'max_lat_acc':
                self.camera_processor.set_vehicle_max_lat_acc(vehicle.id, self.old_value)
            elif self.old_transform:
                vehicle.set_transform(self.old_transform)
                self.camera_processor.vehicle_transforms[vehicle.id] = carla.Transform(
                    self.old_transform.location, self.old_transform.rotation
                )
                # Re-derive pedestrian heading at the restored position (no-op for vehicles).
                self.camera_processor._adjust_pedestrian_spawn_orientation(vehicle.id)
                if self.camera_processor.is_ego_vehicle(vehicle.id):
                    self.camera_processor.update_editor_ego_transform(self.old_transform)
                if vehicle is self.camera_processor.selected_vehicle:
                    self.camera_processor.refresh_selected_vehicle_ui()

    def redo(self):
        # Re-execute the update
        self.execute()
    
    def get_description(self):
        if self.property_name == 'speed_km_h':
            return f"Update vehicle speed: {self.old_value}km/h -> {self.new_value}km/h"
        else:
            return f"Update vehicle {self.property_name}: {self.old_value} -> {self.new_value}"

class MovePersonalTriggerCommand(Command):
    """Command to move a personal trigger (pedestrian, vehicle, or traffic-light)."""
    def __init__(self, camera_processor, selection, old_center, new_center, radius=None):
        self.camera_processor = camera_processor
        self.selection = dict(selection) if selection else None
        self.old_center = dict(old_center) if old_center else None
        self.new_center = dict(new_center) if new_center else None
        self.radius = float(radius) if radius is not None else None

    def _apply_center(self, center):
        if not center or not self.selection or not self.camera_processor:
            return False
        cp = self.camera_processor
        kind = self.selection.get('kind')
        if kind == 'pedestrian':
            actor_id = self.selection.get('id')
            if actor_id is None:
                return False
            cp.pedestrian_trigger_centers[actor_id] = dict(center)
        elif kind == 'vehicle':
            actor_id = self.selection.get('id')
            if actor_id is None:
                return False
            cp.vehicle_trigger_centers[actor_id] = dict(center)
        elif kind == 'traffic_light':
            group = self.selection.get('group')
            key = self.selection.get('key')
            radius = self.radius
            if radius is None:
                _, radius = cp._get_personal_trigger_payload(self.selection)
            if radius is None:
                radius = DEFAULT_PERSONAL_TRIGGER_RADIUS
            resolved_key = cp._set_traffic_light_trigger_data(
                dict(center),
                radius,
                key=key,
                group=group,
            )
            if resolved_key and resolved_key != key:
                self.selection['key'] = resolved_key
            if not group and resolved_key:
                found_group = cp._find_traffic_light_group_by_key(resolved_key)
                if found_group:
                    self.selection['group'] = found_group
        else:
            return False
        cp.update_personal_trigger_menu_position(force=True)
        return True

    def execute(self):
        return self._apply_center(self.new_center)

    def undo(self):
        return self._apply_center(self.old_center)

    def redo(self):
        return self.execute()

    def get_description(self):
        if not self.selection:
            return "Move personal trigger"
        kind = self.selection.get('kind')
        if kind in ('pedestrian', 'vehicle'):
            actor_id = self.selection.get('id', 'unknown')
            label = "pedestrian" if kind == 'pedestrian' else "vehicle"
            return f"Move {label} trigger ({actor_id})"
        if kind == 'traffic_light':
            return "Move traffic-light trigger"
        return "Move personal trigger"

class SetPersonalTriggerCommand(Command):
    """Command to create, update, or delete a personal trigger with undo/redo support."""

    def __init__(self, camera_processor, selection, new_center, new_radius, old_center=None, old_radius=None):
        self.camera_processor = camera_processor
        self.selection = dict(selection) if selection else None
        self.new_center = dict(new_center) if new_center is not None else None
        self.new_radius = float(new_radius) if new_radius is not None else None
        self.old_center = dict(old_center) if old_center is not None else None
        self.old_radius = float(old_radius) if old_radius is not None else None

    def _apply(self, center, radius):
        if not self.selection or not self.camera_processor:
            return False

        cp = self.camera_processor
        kind = self.selection.get('kind')
        radius_value = None if radius is None else max(MIN_PERSONAL_TRIGGER_RADIUS, float(radius))

        if kind == 'pedestrian':
            actor_id = self.selection.get('id')
            if actor_id is None:
                return False
            if center is None or radius_value is None:
                cp.pedestrian_trigger_centers.pop(actor_id, None)
                cp.pedestrian_trigger_radii.pop(actor_id, None)
                return True
            if not cp._ensure_pedestrian_trigger(actor_id):
                return False
            cp.pedestrian_trigger_centers[actor_id] = dict(center)
            cp.pedestrian_trigger_radii[actor_id] = radius_value
            return True

        if kind == 'vehicle':
            actor_id = self.selection.get('id')
            if actor_id is None:
                return False
            if center is None or radius_value is None:
                cp.vehicle_trigger_centers.pop(actor_id, None)
                cp.vehicle_trigger_radii.pop(actor_id, None)
                return True
            if not cp._ensure_vehicle_trigger(actor_id):
                return False
            cp.vehicle_trigger_centers[actor_id] = dict(center)
            cp.vehicle_trigger_radii[actor_id] = radius_value
            return True

        if kind == 'traffic_light':
            group = self.selection.get('group')
            key = self.selection.get('key')
            if center is None or radius_value is None:
                return cp._delete_traffic_light_trigger_data(key=key, group=group)
            resolved_key = cp._set_traffic_light_trigger_data(
                dict(center),
                radius_value,
                key=key,
                group=group,
            )
            if resolved_key and resolved_key != key:
                self.selection['key'] = resolved_key
            return True

        return False

    def execute(self):
        return self._apply(self.new_center, self.new_radius)

    def undo(self):
        return self._apply(self.old_center, self.old_radius)

    def redo(self):
        return self.execute()

    def get_description(self):
        if not self.selection:
            return "Update trigger"

        kind = self.selection.get('kind')
        action = "Update"
        if self.new_center is None or self.new_radius is None:
            action = "Delete"
        elif self.old_center is None and self.old_radius is None:
            action = "Place"

        if kind == 'pedestrian':
            actor_id = self.selection.get('id', 'unknown')
            return f"{action} pedestrian trigger ({actor_id})"
        if kind == 'vehicle':
            actor_id = self.selection.get('id', 'unknown')
            return f"{action} vehicle trigger ({actor_id})"
        if kind == 'traffic_light':
            return f"{action} traffic-light trigger"
        return f"{action} trigger"

class SetGlobalTriggerCommand(Command):
    """Command to create/update/delete the single global trigger with undo/redo support."""

    def __init__(self, camera_processor, old_trigger, new_trigger, *, preserve_selection=False, description=None):
        self.camera_processor = camera_processor
        self.old_trigger = copy.deepcopy(old_trigger) if old_trigger else None
        self.new_trigger = copy.deepcopy(new_trigger) if new_trigger else None
        self.preserve_selection = preserve_selection
        self.description = description or "Update trigger"

    def _apply(self, trigger_payload):
        cp = self.camera_processor
        if not cp:
            return False
        cp.triggers.clear()
        cp.selected_trigger_index = None
        cp.trigger_action_menu_position = None
        cp.trigger_menu_hidden_for_camera_pan = False

        if trigger_payload:
            cp.triggers.append(copy.deepcopy(trigger_payload))
            if self.preserve_selection:
                cp.selected_trigger_index = 0
                screen_pos = cp.coordinate_detector.world_to_screen_coordinates(
                    trigger_payload['x'],
                    trigger_payload['y'],
                    trigger_payload['z'],
                )
                if screen_pos.get('success'):
                    cp.trigger_action_menu_position = (int(screen_pos['x']), int(screen_pos['y']))
        return True

    def execute(self):
        return self._apply(self.new_trigger)

    def undo(self):
        return self._apply(self.old_trigger)

    def redo(self):
        return self.execute()

    def get_description(self):
        return self.description

class WeatherEditCommand(Command):
    """Command to mutate weather keyframes/state with undo/redo support."""

    def __init__(
        self,
        editor,
        old_keyframes,
        new_keyframes,
        old_active_index,
        new_active_index,
        old_pending_pct,
        new_pending_pct,
        description: str = "Update weather",
    ):
        self.editor = editor
        self.old_keyframes = copy.deepcopy(old_keyframes) if old_keyframes is not None else []
        self.new_keyframes = copy.deepcopy(new_keyframes) if new_keyframes is not None else []
        self.old_active_index = int(old_active_index) if old_active_index is not None else 0
        self.new_active_index = int(new_active_index) if new_active_index is not None else 0
        self.old_pending_pct = float(old_pending_pct) if old_pending_pct is not None else 0.0
        self.new_pending_pct = float(new_pending_pct) if new_pending_pct is not None else 0.0
        self.description = description
        self._noop = (
            self.old_keyframes == self.new_keyframes
            and self.old_active_index == self.new_active_index
            and abs(self.old_pending_pct - self.new_pending_pct) < 1e-6
        )

    def _apply(self, keyframes, active_index, pending_pct):
        ed = self.editor
        if not ed:
            return False
        try:
            sanitized = ed._sanitize_weather_keyframes_payload(keyframes)
        except Exception:
            sanitized = []
        if not sanitized:
            return False

        active_index = max(0, min(active_index, len(sanitized) - 1))
        ed._weather_keyframes = sanitized
        ed._active_weather_index = active_index
        ed._pending_weather_pct = float(max(0.0, min(100.0, pending_pct)))

        weather = None
        try:
            weather = ed._weather_params_from_dict(sanitized[active_index])
            if ed.world:
                ed.world.set_weather(weather)
            ed._update_weather_state(weather, keyframe_index=active_index)
        except Exception as exc:
            print(f"[Weather] Failed to apply weather during undo/redo: {exc}")
        ed._refresh_weather_window_metadata()
        if ed.weather_window and ed.weather_window.alive():
            try:
                ed.weather_window.apply_weather(ed.world.get_weather() if ed.world else weather)
            except Exception:
                pass
        return True

    def execute(self):
        if self._noop:
            return False
        return self._apply(self.new_keyframes, self.new_active_index, self.new_pending_pct)

    def undo(self):
        return self._apply(self.old_keyframes, self.old_active_index, self.old_pending_pct)

    def redo(self):
        return self.execute()

    def get_description(self):
        return self.description

class UpdateIgnoreFlagsCommand(Command):
    """Command to toggle a vehicle's ignore flag (traffic lights / stop signs / vehicles)."""

    FLAG_LABELS = {
        'traffic_lights': 'Ignore Traffic Lights',
        'stop_signs': 'Ignore Stop Signs',
        'vehicles': 'Ignore Vehicles',
    }

    def __init__(self, camera_processor, vehicle_id, flag_key, old_value, new_value):
        self.camera_processor = camera_processor
        self.vehicle_id = vehicle_id
        self.flag_key = flag_key        # e.g. 'traffic_lights'
        self.old_value = old_value
        self.new_value = new_value

    def _apply(self, value):
        flags = self.camera_processor.get_vehicle_ignore_flags(self.vehicle_id)
        flags[self.flag_key] = value
        self.camera_processor.set_vehicle_ignore_flags(self.vehicle_id, flags)

    def execute(self):
        self._apply(self.new_value)

    def undo(self):
        self._apply(self.old_value)

    def redo(self):
        self.execute()

    def get_description(self):
        label = self.FLAG_LABELS.get(self.flag_key, self.flag_key)
        state = "on" if self.new_value else "off"
        return f"Toggle {label} {state}"


class UpdateTrafficLightSequenceCommand(Command):
    """Command to update a traffic-light group's sequence with undo/redo support."""

    def __init__(self, camera_processor, group, old_sequence, new_sequence):
        self.camera_processor = camera_processor
        self.old_sequence = copy.deepcopy(old_sequence) if old_sequence is not None else []
        self.new_sequence = copy.deepcopy(new_sequence) if new_sequence is not None else []
        self.group_key = None
        self.group_ids = None
        if group:
            self.group_key = camera_processor._traffic_light_trigger_key(group=group)
            try:
                self.group_ids = tuple(sorted(group.ids))
            except Exception:
                self.group_ids = None

    def _resolve_group(self):
        cp = self.camera_processor
        if not cp:
            return None
        if self.group_key:
            group = cp._find_traffic_light_group_by_key(self.group_key)
            if group:
                return group
        if self.group_ids:
            for group in cp.traffic_light_groups:
                try:
                    if tuple(sorted(group.ids)) == self.group_ids:
                        return group
                except Exception:
                    continue
        return None

    def _apply(self, sequence_payload):
        group = self._resolve_group()
        if not group:
            return False
        group.sequence = copy.deepcopy(sequence_payload)
        if self.camera_processor:
            self.camera_processor._cache_traffic_light_sequence(group)
        return True

    def execute(self):
        return self._apply(self.new_sequence)

    def undo(self):
        return self._apply(self.old_sequence)

    def redo(self):
        return self.execute()

    def get_description(self):
        return "Update traffic-light sequence"
