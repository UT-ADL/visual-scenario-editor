"""SceneModel: the scenario document + editing/selection state.

Every field here was moved verbatim from CameraImageProcessor.__init__
(step-17). CameraImageProcessor keeps @property forwarders for all of them
(install_scene_forwarders), so `cp.<field>` reads/writes still work unchanged
while callers migrate.

Lifetime contract (matches pre-refactor behavior exactly):
- the editor owns one SceneModel for its whole life;
- CameraImageProcessor.__init__ calls scene.reset(), so scene state is
  re-initialized whenever a processor is (re)constructed - exactly when the
  old per-processor attributes were re-created;
- weather fields (added in a later step) are initialized in __init__ only and
  deliberately survive reset(), mirroring their editor-owned lifetime.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from vse_editor.constants import DEFAULT_PERSONAL_TRIGGER_RADIUS
from vse_editor.scene_types import (
    TrafficLightGroupData,
    VehicleIgnoreFlags,
    clone_waypoint_data,
    ensure_ignore_flags,
)


class SceneModel:
    def __init__(self):
        # Weather scene state (statements verbatim from the editor's old
        # storage init). Initialized here, NOT in reset(): weather keyframes
        # survive processor reconstruction (world reset) because their owner
        # was the editor — reset() covers only the per-processor fields.
        self._weather_state: Dict[str, float] = {}
        self._weather_keyframes: List[Dict[str, float]] = []
        self._active_weather_index: int = 0
        self._pending_weather_pct: float = 0.0
        self.reset()

    def reset(self):
        """Re-initialize all per-scenario scene state.

        Statement bodies are verbatim from CameraImageProcessor.__init__.
        """
        self.vehicle_control_mode = "basic_agent"  # "basic_agent" (simulated) or "velocity" (scripted)
        self.spawned_vehicles = []  # Track spawned vehicles
        self.vehicle_speeds = {}  # Track vehicle speeds by vehicle ID
        self.vehicle_destination_speeds = {}  # Track destination speeds separately
        self.vehicle_colors = {}  # Track vehicle colors by vehicle ID
        self.vehicle_transforms = {}  # Authoritative transforms by vehicle ID (for save)
        self.selected_vehicle = None  # Currently selected vehicle
        self.selected_vehicle_is_pedestrian = False  # Track if current selection is a pedestrian
        self.selected_actor_ids = set()  # Group (marquee) selection, actor ids
        self.selected_waypoint_group = None  # {'vehicle_id', 'indices'} when a marquee caught waypoints (no actors)
        self.ego_vehicle_id = None
        self.ego_vehicle_transform = None
        self.ego_vehicle_blueprint = None
        self.ego_vehicle_color = None
        self.actor_idle_times = {}  # Track idle times for actors (pedestrians)
        self.vehicle_ignore_traffic_lights = {}  # Track ignore_traffic_lights by vehicle ID
        self.vehicle_ignore_stop_signs = {}     # Track ignore_stop_signs by vehicle ID
        self.vehicle_ignore_vehicles = {}        # Track ignore_vehicles by vehicle ID
        self.vehicle_max_lat_acc = {}            # Track per-vehicle lateral acceleration caps
        self.pedestrian_trigger_centers = {}  # Track trigger centers by pedestrian ID {x, y, z}
        self.pedestrian_trigger_radii = {}    # Track trigger radii by pedestrian ID (float)
        self.scaling_pedestrian_trigger = False  # Track if we're currently scaling a pedestrian trigger
        self._pedestrian_scaling_id = None  # ID of pedestrian whose trigger is being scaled
        self.pedestrian_trigger_scale_start_pos = (0, 0)  # Starting mouse position for scaling
        self.pedestrian_trigger_scale_start_radius = 0.0  # Starting radius for scaling
        self.vehicle_trigger_centers = {}   # Track trigger centers by vehicle ID {x, y, z}
        self.vehicle_trigger_radii = {}     # Track trigger radii by vehicle ID (float)
        self.scaling_vehicle_trigger = False  # Track if we're currently scaling a vehicle trigger
        self._vehicle_scaling_id = None     # ID of vehicle whose trigger is being scaled
        self.vehicle_trigger_scale_start_pos = (0, 0)  # Starting mouse position for scaling
        self.vehicle_trigger_scale_start_radius = 0.0  # Starting radius for scaling
        self.rotating_vehicle = False  # Track if we're currently rotating a vehicle
        self.rotation_start_mouse_y = 0  # Starting mouse Y position for rotation
        self.rotation_start_yaw = 0  # Starting vehicle yaw for rotation
        self.moving_vehicle = False  # Track if we're currently moving a vehicle
        self.vehicle_drag_armed = False  # Last click hit an actor body; drag may promote to a move
        self.movement_start_mouse_pos = (0, 0)  # Starting mouse position for movement
        self.movement_start_location = None  # Starting vehicle location for movement
        self.creating_destination = False  # Track if we're in destination placement mode
        self.destination_marker_coordinates = None  # Store placed destination marker
        self.creating_waypoint = False  # Track if we're in waypoint creation mode
        self.waypoint_vehicle = None  # Vehicle for which we're creating waypoints
        self.waypoint_list = {}  # Dictionary mapping vehicle IDs to their waypoint lists
        self.selected_waypoint_vehicle_id = None  # Vehicle ID of selected waypoint
        self.waypoint_display_vehicle_id = None  # Vehicle ID whose waypoints should be displayed
        self.moving_waypoint = False  # Track if we're moving a waypoint
        self.selected_waypoint_index = None  # Index of currently selected waypoint for moving
        self.waypoint_movement_start_pos = (0, 0)  # Starting mouse position for movement
        self.waypoint_movement_start_coords = None  # Starting waypoint coordinates for movement
        self.waypoint_original_height = None  # Original Z height when starting to move waypoint
        self.waypoint_drag_armed = False  # Only allow dragging when the last click hit a waypoint
        self.pending_waypoint_click_action = None  # Track simple clicks on waypoints (no drag)
        self.preserve_info_panel_on_waypoint_cancel = False  # Keep waypoint info panel visible after extension cancel
        self.waypoint_hover_index = None  # Index of waypoint being hovered over
        self.triggers = []  # List containing at most one trigger: [{'x': float, 'y': float, 'z': float, 'radius': float}]
        self.selected_trigger_index = None  # Index of currently selected trigger (always 0 if exists)
        self.placing_trigger = False  # Track if we're placing a new trigger
        self.moving_trigger = False  # Track if we're moving a trigger
        self.trigger_drag_armed = False  # Last click hit the global trigger; drag may promote to a move
        self.scaling_trigger = False  # Track if we're scaling a trigger
        self.trigger_movement_start_pos = (0, 0)  # Starting mouse position for trigger movement
        self.trigger_movement_start_coords = None  # Starting trigger coordinates
        self.trigger_original_height = None  # Original Z height when starting to move trigger
        self.trigger_scale_start_pos = (0, 0)  # Starting mouse position for scaling
        self.trigger_scale_start_radius = 0  # Starting radius for scaling
        self._trigger_move_start_snapshot = None  # Snapshot for undo/redo when moving global trigger
        self._trigger_scale_start_snapshot = None  # Snapshot for undo/redo when scaling global trigger
        self.pending_personal_trigger = None  # Track placement mode for personal triggers
        self.personal_trigger_preview_radius = DEFAULT_PERSONAL_TRIGGER_RADIUS  # Default placement radius for personal triggers
        self.selected_personal_trigger = None  # {'type': str, 'id': int} or {'type': 'traffic_light', 'group': TrafficLightGroupData}
        self.moving_personal_trigger = False
        self.personal_trigger_drag_armed = False  # Last click hit a personal trigger; drag may promote to a move
        self.personal_trigger_original_height = None
        self.personal_trigger_movement_start_pos = (0, 0)
        self._personal_trigger_move_target = None  # Cache for currently moving personal trigger selection
        self._personal_trigger_move_start_center = None  # Track original center for undo support
        self.traffic_light_trigger_centers: Dict[Tuple[str, Tuple], Dict[str, float]] = {}
        self.traffic_light_trigger_radii: Dict[Tuple[str, Tuple], float] = {}
        self._last_visible_traffic_light_trigger_key: Optional[Tuple[str, Tuple]] = None
        self.selected_traffic_light_group: Optional[TrafficLightGroupData] = None
        self._selected_traffic_light_group_ids: Set[int] = set()
        self._selected_traffic_light_group_fingerprint: Optional[Tuple[Tuple[int, int, int], ...]] = None
        self.scaling_traffic_light_trigger = False
        self._traffic_light_scaling_group: Optional[TrafficLightGroupData] = None
        self.traffic_light_scale_start_pos: Tuple[int, int] = (0, 0)
        self.traffic_light_scale_start_radius: float = 0.0
        self._scenario_active_traffic_light_trigger: Optional[Dict[str, object]] = None
        self.traffic_light_sequences: Dict[frozenset, List[Dict[str, Union[str, float, int]]]] = {}
        # Last-loaded scenario JSON payload. Pre-refactor this attribute was
        # created lazily outside __init__ (hasattr-guarded); initializing it to
        # None preserves every guard (`not hasattr(...) or not value`).
        self.loaded_scenario_data = None

    def is_ego_vehicle(self, vehicle_id: Optional[int]) -> bool:
        return vehicle_id is not None and self.ego_vehicle_id == vehicle_id

    def get_vehicle_waypoints(self, vehicle_id):
        """Return the waypoint list for a vehicle (empty list if not present)."""
        return self.waypoint_list.get(vehicle_id, [])

    def set_vehicle_waypoints(self, vehicle_id, waypoints):
        """Assign a waypoint list for a vehicle."""
        self.waypoint_list[vehicle_id] = waypoints

    def clear_vehicle_waypoints(self, vehicle_id):
        """Remove stored waypoints for a vehicle."""
        if vehicle_id in self.waypoint_list:
            del self.waypoint_list[vehicle_id]

    def get_vehicle_speed(self, vehicle_id, default=50):
        if self.is_ego_vehicle(vehicle_id) and default == 50:
            default = 40
        return self.vehicle_speeds.get(vehicle_id, default)

    def set_vehicle_speed(self, vehicle_id, value):
        self.vehicle_speeds[vehicle_id] = value

    def clear_vehicle_speed(self, vehicle_id):
        self.vehicle_speeds.pop(vehicle_id, None)

    def get_vehicle_destination_speed(self, vehicle_id):
        return self.vehicle_destination_speeds.get(vehicle_id)

    def set_vehicle_destination_speed(self, vehicle_id, value):
        if value is None:
            self.vehicle_destination_speeds.pop(vehicle_id, None)
        else:
            self.vehicle_destination_speeds[vehicle_id] = value

    def clear_vehicle_destination_speed(self, vehicle_id):
        self.vehicle_destination_speeds.pop(vehicle_id, None)

    def get_vehicle_max_lat_acc(self, vehicle_id, default=3.0):
        return self.vehicle_max_lat_acc.get(vehicle_id, default)

    def set_vehicle_max_lat_acc(self, vehicle_id, value):
        if value is None:
            self.vehicle_max_lat_acc.pop(vehicle_id, None)
        else:
            self.vehicle_max_lat_acc[vehicle_id] = value

    def clear_vehicle_max_lat_acc(self, vehicle_id):
        self.vehicle_max_lat_acc.pop(vehicle_id, None)

    def get_actor_idle_time(self, actor_id, default=0.0):
        return self.actor_idle_times.get(actor_id, default)

    def set_actor_idle_time(self, actor_id, value):
        self.actor_idle_times[actor_id] = value

    def clear_actor_idle_time(self, actor_id):
        self.actor_idle_times.pop(actor_id, None)

    def get_vehicle_color(self, vehicle_id):
        return self.vehicle_colors.get(vehicle_id)

    def set_vehicle_color(self, vehicle_id, color):
        if color is None:
            self.vehicle_colors.pop(vehicle_id, None)
        else:
            self.vehicle_colors[vehicle_id] = color

    def clear_vehicle_color(self, vehicle_id):
        self.vehicle_colors.pop(vehicle_id, None)

    def get_vehicle_ignore_flags(self, vehicle_id) -> VehicleIgnoreFlags:
        raw = {
            'traffic_lights': self.vehicle_ignore_traffic_lights.get(vehicle_id, False),
            'stop_signs': self.vehicle_ignore_stop_signs.get(vehicle_id, False),
            'vehicles': self.vehicle_ignore_vehicles.get(vehicle_id, False),
        }
        return ensure_ignore_flags(raw)

    def set_vehicle_ignore_flags(self, vehicle_id, flags):
        normalized = ensure_ignore_flags(flags)
        self.vehicle_ignore_traffic_lights[vehicle_id] = normalized['traffic_lights']
        self.vehicle_ignore_stop_signs[vehicle_id] = normalized['stop_signs']
        self.vehicle_ignore_vehicles[vehicle_id] = normalized['vehicles']

    def clear_vehicle_ignore_flags(self, vehicle_id):
        self.vehicle_ignore_traffic_lights.pop(vehicle_id, None)
        self.vehicle_ignore_stop_signs.pop(vehicle_id, None)
        self.vehicle_ignore_vehicles.pop(vehicle_id, None)

    def _ensure_waypoint_container(self, vehicle_id):
        waypoints = self.waypoint_list.get(vehicle_id)
        if waypoints is None:
            waypoints = []
            self.waypoint_list[vehicle_id] = waypoints
        return waypoints

    def append_waypoint_data(self, vehicle_id, waypoint):
        waypoints = self._ensure_waypoint_container(vehicle_id)
        waypoint_copy = clone_waypoint_data(waypoint)
        waypoints.append(waypoint_copy)
        return waypoint_copy

    def update_waypoint_fields(self, vehicle_id, index, updates: Dict[str, Union[float, bool, str, None]]):
        waypoints = self.waypoint_list.get(vehicle_id)
        if not waypoints or index < 0 or index >= len(waypoints):
            return None
        waypoint = waypoints[index]
        previous = clone_waypoint_data(waypoint)
        waypoint.update(updates)
        return previous

    def initialize_vehicle_metadata(
        self,
        vehicle_id,
        *,
        speed,
        destination_speed=None,
        idle_time=0.0,
        color=None,
        ignore_flags=None,
        max_lat_acc=3.0,
    ):
        self.set_vehicle_speed(vehicle_id, speed)
        self.set_vehicle_destination_speed(vehicle_id, destination_speed)
        self.set_actor_idle_time(vehicle_id, idle_time)
        self.set_vehicle_color(vehicle_id, color)
        self.set_vehicle_ignore_flags(vehicle_id, ignore_flags or {})
        if max_lat_acc is None:
            max_lat_acc = 3.0
        self.set_vehicle_max_lat_acc(vehicle_id, max_lat_acc)

    def clear_vehicle_metadata(self, vehicle_id, *, clear_waypoints=False):
        self.clear_vehicle_speed(vehicle_id)
        self.clear_vehicle_destination_speed(vehicle_id)
        self.clear_actor_idle_time(vehicle_id)
        self.clear_vehicle_color(vehicle_id)
        self.clear_vehicle_ignore_flags(vehicle_id)
        self.clear_vehicle_max_lat_acc(vehicle_id)
        # Clear pedestrian trigger data
        self.pedestrian_trigger_centers.pop(vehicle_id, None)
        self.pedestrian_trigger_radii.pop(vehicle_id, None)
        # Clear vehicle trigger data
        self.vehicle_trigger_centers.pop(vehicle_id, None)
        self.vehicle_trigger_radii.pop(vehicle_id, None)
        self.vehicle_transforms.pop(vehicle_id, None)
        if clear_waypoints:
            self.clear_vehicle_waypoints(vehicle_id)

    def clear_all_vehicle_metadata(self):
        """Clear all stored vehicle metadata collections."""
        self.waypoint_list.clear()
        self.vehicle_speeds.clear()
        self.vehicle_destination_speeds.clear()
        self.actor_idle_times.clear()
        self.vehicle_colors.clear()
        self.vehicle_ignore_traffic_lights.clear()
        self.vehicle_ignore_stop_signs.clear()
        self.vehicle_ignore_vehicles.clear()
        self.vehicle_max_lat_acc.clear()
        # Clear pedestrian trigger data
        self.pedestrian_trigger_centers.clear()
        self.pedestrian_trigger_radii.clear()
        # Clear vehicle trigger data
        self.vehicle_trigger_centers.clear()
        self.vehicle_trigger_radii.clear()
        self.vehicle_transforms.clear()


# Every SceneModel field, for forwarder installation and completeness tests.
SCENE_FIELDS = (
    "vehicle_control_mode",
    "spawned_vehicles",
    "vehicle_speeds",
    "vehicle_destination_speeds",
    "vehicle_colors",
    "vehicle_transforms",
    "selected_vehicle",
    "selected_vehicle_is_pedestrian",
    "selected_actor_ids",
    "selected_waypoint_group",
    "ego_vehicle_id",
    "ego_vehicle_transform",
    "ego_vehicle_blueprint",
    "ego_vehicle_color",
    "actor_idle_times",
    "vehicle_max_lat_acc",
    "pedestrian_trigger_centers",
    "pedestrian_trigger_radii",
    "scaling_pedestrian_trigger",
    "_pedestrian_scaling_id",
    "pedestrian_trigger_scale_start_pos",
    "pedestrian_trigger_scale_start_radius",
    "vehicle_trigger_centers",
    "vehicle_trigger_radii",
    "scaling_vehicle_trigger",
    "_vehicle_scaling_id",
    "vehicle_trigger_scale_start_pos",
    "vehicle_trigger_scale_start_radius",
    "rotating_vehicle",
    "rotation_start_mouse_y",
    "rotation_start_yaw",
    "moving_vehicle",
    "vehicle_drag_armed",
    "movement_start_mouse_pos",
    "movement_start_location",
    "creating_destination",
    "destination_marker_coordinates",
    "creating_waypoint",
    "waypoint_vehicle",
    "waypoint_list",
    "selected_waypoint_vehicle_id",
    "waypoint_display_vehicle_id",
    "moving_waypoint",
    "selected_waypoint_index",
    "waypoint_movement_start_pos",
    "waypoint_movement_start_coords",
    "waypoint_original_height",
    "waypoint_drag_armed",
    "pending_waypoint_click_action",
    "preserve_info_panel_on_waypoint_cancel",
    "waypoint_hover_index",
    "triggers",
    "selected_trigger_index",
    "placing_trigger",
    "moving_trigger",
    "trigger_drag_armed",
    "scaling_trigger",
    "trigger_movement_start_pos",
    "trigger_movement_start_coords",
    "trigger_original_height",
    "trigger_scale_start_pos",
    "trigger_scale_start_radius",
    "_trigger_move_start_snapshot",
    "_trigger_scale_start_snapshot",
    "pending_personal_trigger",
    "personal_trigger_preview_radius",
    "selected_personal_trigger",
    "moving_personal_trigger",
    "personal_trigger_drag_armed",
    "personal_trigger_original_height",
    "personal_trigger_movement_start_pos",
    "_personal_trigger_move_target",
    "_personal_trigger_move_start_center",
    "traffic_light_trigger_centers",
    "traffic_light_trigger_radii",
    "_last_visible_traffic_light_trigger_key",
    "selected_traffic_light_group",
    "_selected_traffic_light_group_ids",
    "_selected_traffic_light_group_fingerprint",
    "scaling_traffic_light_trigger",
    "_traffic_light_scaling_group",
    "traffic_light_scale_start_pos",
    "traffic_light_scale_start_radius",
    "_scenario_active_traffic_light_trigger",
    "traffic_light_sequences",
    "loaded_scenario_data",
)


# Weather fields: editor-lifetime scene state (see SceneModel.__init__).
# Forwarded onto BOTH CameraImageProcessor and VisualScenarioEditor — the
# historical duplication collapsed into one storage.
SCENE_WEATHER_FIELDS = (
    "_weather_state",
    "_weather_keyframes",
    "_active_weather_index",
    "_pending_weather_pct",
)


def install_scene_forwarders(cls, fields=SCENE_FIELDS, source_attr="scene"):
    """Install read/write @property forwarders for every scene field on cls.

    `obj.<field>` then reads/writes `obj.<source_attr>.<field>`. Keeps all
    pre-move call sites working unchanged; deleted field-by-field once callers
    are retargeted at the scene directly.
    """
    for name in fields:
        def getter(self, _n=name):
            return getattr(getattr(self, source_attr), _n)

        def setter(self, value, _n=name):
            setattr(getattr(self, source_attr), _n, value)

        setattr(cls, name, property(getter, setter))
