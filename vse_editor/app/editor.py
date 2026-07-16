"""VisualScenarioEditor: the app-shell class (moved verbatim from vse.py,
step-43, Phase 7).

The class now holds __init__ (construction/wiring), run() (main loop),
cleanup(), the CommandHistory on_change callback, the culling_enabled
property and the delegate table into vse_editor/app/* module functions.
"""

import json
import os
import select
import threading
import time

from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

import carla
import pygame

from pygame_gui import UIManager
from pygame_gui.elements.ui_window import UIWindow

from vse_common.env import env_float
from vse_editor.app import (
    chrome,
    events,
    modals,
    playback_launch,
    relaunch,
    scenario_files,
    startup,
    stream_settings,
    world_lifecycle,
)
from vse_editor.app import settings as settings_app
from vse_editor.carla_io.profiles import ConnectionProfile
from vse_editor.carla_io.server_manager import CarlaServerManager
from vse_editor.commands import (
    UpdateIgnoreFlagsCommand,
    UpdateVehiclePropertyCommand,
    UpdateWaypointPropertyCommand,
)
from vse_editor.constants import (
    REMOTE_STREAM_RESOLUTIONS,
    WEATHER_PARAMETER_SPECS,
    WeatherParameterSpec,
)
from vse_editor.controllers import weather as weather_ctl
from vse_editor.history import CommandHistory
from vse_editor.scene.model import (
    SCENE_WEATHER_FIELDS,
    SceneModel,
    install_scene_forwarders,
)
from vse_editor.session import SessionState
from vse_editor.ui.dialogs import ResultWindow
from vse_editor.ui.menus import (
    EgoVehicleSelectionMenu,
    PedestrianSelectionMenu,
    PlacementMode,
    TrafficLightGroupSelectionMenu,
    VehicleSelectionMenu,
)
from vse_editor.ui.weather_window import WeatherControlWindow
from vse_editor.ui.widgets import TooltipManager


class VisualScenarioEditor:
    """
    Main application class for the Visual Scenario Editor.
    Orchestrates UI, CARLA interaction, event loop, undo/redo, and scenario management.
    """

    def _handle_quit_event(self):
        return events._handle_quit_event(self)

    def _handle_keydown(self, event, scenario_running):
        return events._handle_keydown(self, event, scenario_running)

    def _handle_keyup(self, event):
        return events._handle_keyup(self, event)

    def _handle_left_click(self, pos, scenario_running):
        return events._handle_left_click(self, pos, scenario_running)

    def _begin_mouse_pan(self, pos, button):
        return events._begin_mouse_pan(self, pos, button)

    def _end_mouse_pan(self):
        return events._end_mouse_pan(self)

    def _handle_right_click_down(self, event):
        return events._handle_right_click_down(self, event)

    def _is_locked_view_camera_active(self) -> bool:
        return events._is_locked_view_camera_active(self)

    def _handle_mouse_button_down(self, event, scenario_running):
        return events._handle_mouse_button_down(self, event, scenario_running)

    def _handle_mouse_button_up(self, event):
        return events._handle_mouse_button_up(self, event)

    def _handle_mouse_motion(self, event):
        return events._handle_mouse_motion(self, event)

    def _handle_mouse_wheel(self, event, scenario_running):
        return events._handle_mouse_wheel(self, event, scenario_running)

    def _handle_resize(self, event):
        return events._handle_resize(self, event)

    def toggle_orbit_view(self):
        return events.toggle_orbit_view(self)

    def show_status_hint(self, text: str, duration_s: float = 2.5) -> None:
        """Show a short transient hint line in the top bar (blocked-action feedback)."""
        self._status_hint_text = text
        self._status_hint_until = time.time() + duration_s
        print(f"[Hint] {text}")

    def _center_dialog_rect(self, width: int, height: int) -> pygame.Rect:
        return modals._center_dialog_rect(self, width, height)

    def _render_dialog_background(self) -> None:
        return modals._render_dialog_background(self)

    def _run_modal_window(self, dialog: UIWindow, event_handler) -> Optional[Union[bool, str, Dict[str, Union[str, int]]]]:
        return modals._run_modal_window(self, dialog, event_handler)

    def _prompt_text_input(self, title: str, prompt: str, default: str = "") -> Optional[str]:
        return modals._prompt_text_input(self, title, prompt, default)

    def _ask_use_running_server(self, port: int) -> bool:
        return modals._ask_use_running_server(self, port)

    def __init__(self, carla_path=None, port=2000, *, debug: bool = False):
        self.keep_server_running_on_exit = False
        # Initialize Pygame
        pygame.init()

        # Screen settings
        self.windowed_width = 1200
        self.windowed_height = 800
        self.screen_width = self.windowed_width
        self.screen_height = self.windowed_height
        self.maximized = False
        self.stream_resolution = (1280, 720)
        self.camera_stream_enabled = True
        base_resolution_options = list(REMOTE_STREAM_RESOLUTIONS)
        self.remote_resolution_options: List[Optional[Tuple[int, int]]] = [None] + base_resolution_options
        if self.stream_resolution in base_resolution_options:
            self.remote_resolution_index = self.remote_resolution_options.index(self.stream_resolution)
        elif base_resolution_options:
            first_resolution = base_resolution_options[0]
            self.stream_resolution = first_resolution
            self.remote_resolution_index = self.remote_resolution_options.index(first_resolution)
        else:
            self.remote_resolution_index = 0
            self.camera_stream_enabled = False
        self.stream_fps = 10
        self.remote_fps_min = 1
        self.remote_fps_max = 20
        self.resolution_menu_open = False
        self.fps_menu_open = False
        self.pending_menu_icon = None  # Click-type menu icon awaiting mouse-up dispatch
        self.play_camera_menu_open = False
        self.play_camera_button_rect: Optional[pygame.Rect] = None
        self.play_camera_option_rects: List[Tuple[pygame.Rect, str]] = []
        self.resolution_option_rects: List[Tuple[pygame.Rect, Optional[Tuple[int, int]]]] = []
        self.fps_option_rects: List[Tuple[pygame.Rect, int]] = []
        self._dropdown_draw_ops: List[Tuple[pygame.Rect, Tuple[int, int, int], pygame.Surface]] = []
        self._dropdown_capture_rects: List[pygame.Rect] = []
        self._dropdown_mouse_captured = False
        self.result_window: Optional[ResultWindow] = None
        self._pending_result_dialog: Optional[Dict[str, object]] = None
        self._no_camera_resolution_override: Optional[Tuple[int, int]] = None
        self._no_camera_saved_pose: Optional[Tuple[float, float, float]] = None

        env_windowed_width = os.environ.get('VSE_WINDOWED_WIDTH')
        env_windowed_height = os.environ.get('VSE_WINDOWED_HEIGHT')
        env_screen_width = os.environ.get('VSE_SCREEN_WIDTH')
        env_screen_height = os.environ.get('VSE_SCREEN_HEIGHT')
        env_maximized = os.environ.get('VSE_WINDOW_MAXIMIZED')

        try:
            if env_windowed_width:
                windowed_width = int(env_windowed_width)
                if windowed_width > 0:
                    self.windowed_width = windowed_width
        except ValueError:
            pass

        try:
            if env_windowed_height:
                windowed_height = int(env_windowed_height)
                if windowed_height > 0:
                    self.windowed_height = windowed_height
        except ValueError:
            pass

        try:
            if env_screen_width:
                screen_width = int(env_screen_width)
                if screen_width > 0:
                    self.screen_width = screen_width
        except ValueError:
            pass

        try:
            if env_screen_height:
                screen_height = int(env_screen_height)
                if screen_height > 0:
                    self.screen_height = screen_height
        except ValueError:
            pass

        if env_maximized:
            self.maximized = env_maximized == '1'

        # Check for pending scenario load from map change
        pending_scenario = os.environ.get('VSE_PENDING_SCENARIO')
        if pending_scenario:
            self.pending_scenario_load = pending_scenario
            print(f"Restored pending scenario from environment: {pending_scenario}")

        os.environ.pop('VSE_WINDOWED_WIDTH', None)
        os.environ.pop('VSE_WINDOWED_HEIGHT', None)
        os.environ.pop('VSE_SCREEN_WIDTH', None)
        os.environ.pop('VSE_SCREEN_HEIGHT', None)
        os.environ.pop('VSE_WINDOW_MAXIMIZED', None)
        os.environ.pop('VSE_PENDING_SCENARIO', None)

        # Create windowed screen initially
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Visual Scenario Editor")

        # pygame_gui manager
        self.ui_manager = UIManager((self.screen_width, self.screen_height))

        # Tooltip manager for hover help
        self.tooltip_manager = TooltipManager(self.screen_width, self.screen_height)

        # Colors
        self.colors = {
            'background': (50, 50, 50),
            'text': (255, 255, 255),
            'ui_bg': (30, 30, 30, 180),
            'loading': (100, 100, 255),
            'error': (255, 100, 100),
            'coordinates': (255, 255, 0),  # Yellow overlays
            'fps': (255, 255, 0),          # Yellow FPS meter
            'toggle_on': (80, 150, 80),
            'toggle_off': (70, 70, 70),
            'button': (70, 70, 70),
            'button_hover': (90, 90, 90),
        }

        self.top_ui_height = 90
        self.side_panel_top = self.top_ui_height + 60
        self.mode_button_base_x = 10
        self.mode_button_width = 140
        self.mode_button_height = 28
        self.mode_button_spacing = 10

        # Font for UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)
        self.micro_font = pygame.font.Font(None, 12)
        self.fps_font = pygame.font.Font(None, 18)
        self._fps_display = 0.0
        self._fps_smoothing = 0.85
        self._world_tick_times: Deque[float] = deque()
        self._world_tick_window = 3.0
        self._world_tick_fps = 0.0
        self.rendering_enabled = False
        self._rendering_desired_enabled = False
        self.rendering_toggle_rect: Optional[pygame.Rect] = None
        self._rendering_apply_in_progress = False
        # Ego collision toggle (session-only; read at Play time). On by default = collision enabled.
        self.ego_collision_enabled = True
        self.ego_collision_toggle_rect: Optional[pygame.Rect] = None
        # Ego physics toggle (session-only; read at Play time). On by default = physics enabled.
        # Untick for external/VIL ego so VSE stops re-enabling CARLA physics (which fights awmini's teleport).
        self.ego_physics_enabled = True
        self.ego_physics_toggle_rect: Optional[pygame.Rect] = None
        self._fps_text_rect: Optional[pygame.Rect] = None
        self.large_map_active = False
        self.large_map_ui_delay = env_float("VSE_LARGE_MAP_UI_DELAY", 0.2)

        # UI integration state
        self.open_map_button_rect = None
        self.connection_button_rect = None
        self.gpu_button_rect = None
        self.manual_tick_button_rect = None
        self.resolution_button_rect = None
        self.fps_button_rect = None
        # "Cull" dropdown (WorldSettings.max_culling_distance, metres; 0 = Off). Editor-only;
        # remembered in settings.json. culling_distance_m is (re)set by _load_culling() below.
        self.culling_distance_m = self.CULLING_DEFAULT_M
        self._culling_button_rect: Optional[pygame.Rect] = None
        self._culling_option_rects: List[Tuple[pygame.Rect, float]] = []
        self._culling_menu_open = False
        self.open_scenario_button_rect = None
        self.last_map_directory = self._resolve_default_map_directory()
        
        # CARLA components
        self.camera_processor = None
        self.info_panel = None
        self.server_manager = CarlaServerManager(carla_path, port)
        self.local_port = port
        self._default_engine_ini_backup_path: Optional[Path] = None
        self._restore_default_engine_ini_backup_if_present()

        profile_override_raw = os.environ.pop('VSE_PROFILE_JSON', None)
        self._launched_with_profile_override = bool(profile_override_raw)
        override_profile = None
        override_label = None
        override_port = None
        if profile_override_raw:
            try:
                profile_data = json.loads(profile_override_raw)
                override_profile = ConnectionProfile(
                    name=profile_data.get('name', 'remote'),
                    host=profile_data.get('host', '127.0.0.1'),
                    port=int(profile_data.get('port', port)),
                    manage_server=bool(profile_data.get('manage_server', False)),
                    display_name=profile_data.get('display_name', profile_data.get('host', 'Remote CARLA')),
                    description=profile_data.get('description', ''),
                    map_hint=profile_data.get('map_hint')
                )
                override_label = profile_data.get('label')
                override_port = int(profile_data.get('port', port))
            except Exception as exc:
                print(f"Failed to parse connection profile override: {exc}")
                override_profile = None

        self.connection_profile = override_profile or self._create_local_profile()
        self.server_manager.set_port(self.local_port)
        self._previous_connection_profile = None
        self.remote_connection_active = self.connection_profile.is_remote
        self.cached_map = None
        self._map_refresh_disabled = False
        self.remote_map_hint = "tartu_demo"
        self._migrate_legacy_settings()  # fold legacy last_*.json into settings.json (once), before any load
        self._load_last_remote_cache()
        self._load_last_agent_cache()
        self._load_last_play_camera_cache()
        self._load_culling()
        if override_profile and override_profile.is_remote:
            self.active_remote_label = override_label or f"{override_profile.host}:{override_profile.port}"
            self.active_remote_port = override_port
            if not getattr(self, "last_remote_host", None):
                self.last_remote_host = override_profile.host
            if override_port is not None and not getattr(self, "last_remote_port", None):
                self.last_remote_port = str(override_port)
        else:
            self.active_remote_label = None
            self.active_remote_port = None
        self.pending_profile_switch = None
        self.pending_profile_switch_label = None
        self.pending_map_switch = None
        self.pending_remote_map = os.environ.pop('VSE_PENDING_MAP', None)
        self.client = None
        self.world = None
        self.world_map = None
        self.manual_tick_required = False
        self.manual_tick_interval = 0.05
        self.manual_tick_accumulator = 0.0
        self.manual_tick_enabled = False
        self.manual_tick_recommendation = False
        self.vehicle_control_mode = "basic_agent"  # "basic_agent" (simulated) or "velocity" (scripted)
        self._preserved_vehicle_control_mode = None  # Preserved across play/stop cycle
        self.npc_dropdown_open = False
        self.npc_button_rect = None
        self._npc_dropdown_item_rects = []
        self._large_map_bootstrap_ticking = False  # Temporary ticking until external ego connects on large maps
        self._bootstrap_tick_warned = False  # Suppress repeated bootstrap tick warnings
        self._pending_external_ego_data = None  # Saved ego data to apply to external ego on large maps
        self._bootstrap_original_camera_transform = None  # Save camera position before moving to spawn point
        self._external_ego_present_last_check = False
        self._last_external_ego_scan_time = 0.0
        self._manual_tick_last_required_time = 0.0
        self._external_ego_scenario_reloaded = False
        self._external_ego_prompt_pending_id = None
        # When True, external ego detection should never auto-adopt/move the actor.
        # Used for "New Scenario" resets where the external ego must remain untouched.
        self._suppress_external_ego_adoption = False
        self._scenario_run_in_progress = False  # Track scenario launch phase to prevent bootstrap during playback
        self._external_swap_active = False
        self._external_swap_current_id = None
        self._external_swap_last_transform = None
        self._external_swap_last_blueprint = None
        self._external_swap_last_color = None
        self._external_swap_pre_run_transform = None
        self._external_swap_overlay = False
        self._external_swap_overlay_message: Optional[str] = None
        self._external_swap_overlay_started_at: float = 0.0
        self._external_swap_overlay_keep_until: float = 0.0
        # Camera systems
        self.camera_controller = None
        self.camera_processor = None
        self.camera_debug_enabled = bool(debug)
        self.debug_mode = bool(debug)
        self._pending_camera_focus_target: Optional[Tuple[float, float, float]] = None
        self._pending_camera_focus_not_before: float = 0.0
        self._pending_camera_focus_until: float = 0.0
        self._pending_camera_focus_last_attempt: float = 0.0
        self._pending_camera_focus_reason: Optional[str] = None
        self._pending_camera_focus_debug_logged: bool = False
        
        # Input states
        self.keys_pressed = set()
        self.mouse_dragging = False
        self.mouse_drag_button = None
        self._right_click_press_pos = None  # Right-tap = Esc-style cancel on release
        self._marquee_press_pos = None  # Left press on empty ground; may become a marquee
        self.marquee_rect = None  # Active marquee rectangle (x, y, w, h) while dragging
        self._shift_press = None  # Shift+press on an object; tap toggles group, drag free-moves
        self._orbit_left_press_pos = None  # Left press in the 3D view; hints on drag attempts
        self._orbit_press_was_armed = False  # That press had armed an edit gesture (latched)
        self.last_mouse_pos = (0, 0)

        # Transient status hint (blocked-action feedback, e.g. editing in the 3D view)
        self._status_hint_text = None
        self._status_hint_until = 0.0

        # Camera movement tracking for dynamic min height
        self.camera_panned_with_mouse = False  # Track if mouse (right/middle) pan occurred
        self.camera_moved_with_wasd = False    # Track if WASD movement occurred
        
        # Key acceleration tracking
        self.key_hold_times = {}
        self.max_acceleration_time = 2.0
        self.acceleration_curve = 3.0
        
        # Startup state
        self.loading_stage = "Checking for existing CARLA server..."
        self.ready = False
        self.startup_error = None
        self.start_screen_active = False
        self.start_screen_open_scenario_rect: Optional[pygame.Rect] = None
        self.start_screen_open_map_rect: Optional[pygame.Rect] = None
        self.start_screen_connect_remote_rect: Optional[pygame.Rect] = None
        self.start_screen_recent_scenario_rects: List[Tuple[pygame.Rect, Optional[str]]] = []
        self._startup_requested = False
        self._startup_thread_started = False
        self._startup_selected_map_name: Optional[str] = None
        self._startup_selected_map_package: Optional[str] = None
        self._startup_selected_scenario_path: Optional[str] = None

        # Check for startup map package from environment (for INI patching before CARLA starts)
        # This is used when reloading after external ego removal on large maps
        startup_map_package = os.environ.get('VSE_STARTUP_MAP_PACKAGE')
        if startup_map_package:
            self._startup_selected_map_package = startup_map_package
            self._startup_selected_map_name = startup_map_package.split('/')[-1]
            print(f"[Startup] Restored startup map package from environment: {startup_map_package}")
            os.environ.pop('VSE_STARTUP_MAP_PACKAGE', None)

        self.restart_thread = None
        self.restart_in_progress = False
        self.relaunch_process = None
        self.restart_start_time = None
        self.restart_wait_duration = 30.0
        self.pending_exit = False
        self.handoff_read_fd = None
        self.wait_for_child = False
        handoff_fd_env = os.environ.get('VSE_HANDOFF_FD')
        if handoff_fd_env is not None:
            try:
                self.handoff_write_fd = int(handoff_fd_env)
            except ValueError:
                self.handoff_write_fd = None
            os.environ.pop('VSE_HANDOFF_FD', None)
        else:
            self.handoff_write_fd = None

        auto_start = bool(
            self.handoff_write_fd is not None
            or getattr(self, "pending_scenario_load", None)
            or self.pending_remote_map
            or self._launched_with_profile_override
            or self._startup_selected_map_package is not None
            or getattr(self.server_manager, "assume_existing_server_pid", None) is not None
        )
        if auto_start:
            self.start_screen_active = False
            self._startup_requested = True
        else:
            existing_server = False
            try:
                existing_server = self.server_manager.check_existing_carla_server(
                    self.server_manager.port
                )
            except Exception as exc:
                print(f"[Startup] Warning: failed to probe for existing CARLA server: {exc}")

            if existing_server and self._ask_use_running_server(self.server_manager.port):
                # User chose to reuse the running server: connect to it as-is
                # (the start-screen map picker is skipped) and leave it running on exit.
                print("[Startup] Connecting to already-running CARLA server.")
                self.start_screen_active = False
                self._startup_requested = True
                self.server_manager.use_existing_server = True
                self.keep_server_running_on_exit = True
            else:
                self.start_screen_active = True
                self._startup_requested = False
                try:
                    self.server_manager.kill_existing_carla_processes()
                except Exception as exc:
                    print(f"[Startup] Warning: failed to kill existing CARLA processes: {exc}")
        
        # Selection menus
        self.vehicle_menu = VehicleSelectionMenu()
        self.selected_vehicle_type = None
        self.pedestrian_menu = PedestrianSelectionMenu()
        self.selected_pedestrian_type = None
        self.ego_vehicle_menu = EgoVehicleSelectionMenu()
        self.selected_ego_vehicle_type = None
        self.traffic_light_group_menu = TrafficLightGroupSelectionMenu()
        self.placement_mode = PlacementMode.VEHICLE
        self.mode_button_rects = {}
        # One SceneModel per editor life; every CameraImageProcessor binds and
        # resets it in its constructor (see vse_editor/scene/model.py).
        self.scene = SceneModel()
        self.session = SessionState(self)
        self.weather_button_rect = None
        self.weather_button_enabled = False
        self.weather_window: Optional[WeatherControlWindow] = None
        # Weather scene state (_weather_state/_weather_keyframes/
        # _active_weather_index/_pending_weather_pct) lives on SceneModel —
        # single source of truth; @property forwarders installed below keep
        # self.<field> working. Drag snapshot + baseline stay editor-local
        # (UI gesture / session state, not scenario content).
        self._weather_spec_lookup: Dict[str, WeatherParameterSpec] = {
            spec.name: spec for spec in WEATHER_PARAMETER_SPECS
        }
        self._weather_drag_snapshot: Optional[Dict[str, object]] = None
        self._baseline_weather_state: Optional[Dict[str, float]] = None

        self.vehicle_menu.set_vertical_offset(self.side_panel_top)
        self.pedestrian_menu.set_vertical_offset(self.side_panel_top)
        self.ego_vehicle_menu.set_vertical_offset(self.side_panel_top)
        self.traffic_light_group_menu.set_vertical_offset(self.side_panel_top)
        self.vehicle_menu.set_camera_processor(None)
        self.pedestrian_menu.set_camera_processor(None)
        self.ego_vehicle_menu.set_camera_processor(None)
        self.traffic_light_group_menu.set_camera_processor(None)
        for _menu in (self.vehicle_menu, self.pedestrian_menu, self.ego_vehicle_menu):
            _menu.on_dropdown_open = (lambda menu_ref=_menu: self._close_all_dropdowns(except_menu=menu_ref))

        # Map selection menu
        self.map_menu = None  # Will be initialized after world is created
        self.map_menu_visible = False

        # Scenario selection menu
        self.scenario_menu = None  # Will be initialized after world is created
        self.scenario_menu_visible = False

        # Pure-camera view: backtick (`) toggles hiding all overlays/UI in run()
        self.hide_all_ui = False

        # Keyboard & mouse shortcuts help overlay (F1 / H toggles; ? button opens)
        self.keyboard_help_visible = False
        self.help_button_rect = None        # the "?" button hit-rect (set in render_ui)
        self.help_close_button_rect = None  # the panel's X hit-rect (set when panel drawn)
        self.current_scenario_name = None  # Name of currently loaded scenario
        self.current_scenario_path = None  # Full path to currently loaded scenario JSON
        self.recent_scenarios: List[Dict[str, str]] = []  # Cached recent scenarios (most recent first)
        self.last_scenario_path: Optional[str] = None  # Cached recent scenario path (first entry)
        self.last_scenario_name: Optional[str] = None  # Cached recent scenario name (first entry)
        self.last_scenario_map: Optional[str] = None   # Cached map name (first entry)
        # pending_scenario_load is set earlier from environment variable if present
        if not hasattr(self, 'pending_scenario_load'):
            self.pending_scenario_load = None  # Scenario to load after map change
        self._load_last_scenario_cache()

        # Scenario runner state (for Play button functionality)
        self.scenario_running = False
        self.scenario_process = None
        self.scenario_stop_requested = False
        self.saved_scene_vehicles = []
        self.external_ego_actor = None
        self.external_ego_actor_id = None
        self.scene_preview_destroyed = False
        self._world_settings_signature = None
        self._scenario_preserved_waypoint_vehicle_id = None
        self._scenario_preserved_waypoints = None
        self._scenario_preserved_actor_blueprint = None
        self._scenario_preserved_actor_location = None
        self._scenario_preserved_actor_is_pedestrian = False
        self._scenario_preserved_info_panel_visible = False
        self._scenario_preserved_waypoint_signature = None
        self._scenario_preserved_traffic_light_ids: Optional[Tuple[int, ...]] = None
        self._scenario_preserved_traffic_light_info_visible = False
        self._scenario_active_traffic_light_trigger: Optional[Dict[str, object]] = None
        self._scenario_preserved_traffic_lights_visible = False
        self._scenario_preserved_lane_overlay_enabled = False
        self._scenario_pending_actor_reselect = False
        self._scenario_pending_actor_reselect_attempts = 0
        self._scenario_pending_actor_reselect = False
        self._scenario_pending_actor_reselect_attempts = 0
        self._scenario_pending_actor_reselect_max_attempts = 120
        self._camera_restore_state: Optional[Tuple[float, float, float]] = None
        self._restore_in_progress: bool = False
        self._restore_invoked_for_run: bool = False
        self._restore_once_lock = threading.Lock()
        self._debug_last_pose_label: Optional[str] = None
        self._scenario_active_override: bool = False
        self.loaded_scenario_data: Optional[dict] = None

        # Scene change tracking
        self._scene_dirty_hint = False
        self._saved_scene_signature: Optional[str] = None  # legacy snapshot (deprecated)
        self._saved_disk_signature: Optional[str] = None   # legacy snapshot (deprecated)

        # World reset detection and recovery
        self._world_reset_candidate_world = None
        self._world_tick_subscription = None
        self._world_last_episode = None
        self._world_last_frame = None
        # world.id baseline: a reload (e.g. by an external stack) changes world.id. We compare
        # the live world.id against this baseline to detect externally-initiated world reloads.
        self._expected_world_id = None
        # Latest world snapshot, stored by _on_world_tick (O(1), no RPC). Used as a cheap
        # per-tick signal for external-ego spawn/leave detection (len()/has_actor()).
        self._latest_world_snapshot = None
        self._last_seen_actor_count = None
        self._world_reset_lock = threading.Lock()
        self._world_reset_pending = False
        self._world_reset_detected_at = 0.0
        self._world_reset_in_progress = False
        self._world_reset_wait_thread = None
        self._world_reset_ready_world = None
        self._world_reset_finalize_pending = False
        self._world_reset_failed = False
        # Bound on the "Waiting for CARLA..." recovery wait (fix-06); on
        # expiry the server-lost overlay takes over. Env override exists for
        # the server-death smoke (tartu-class reloads need the full default).
        self._world_recovery_deadline_s = env_float('VSE_WORLD_RECOVERY_DEADLINE_S', 120.0)

        # Server-liveness watchdog state (Phase 8, fix-05; tunables are the
        # _SERVER_* class attrs)
        self._server_lost = False
        self._server_poll_last = 0.0   # last liveness retry stamp (while lost)
        self._server_rpc_last = 0.0    # last world-reset-detection RPC stamp
        self._server_restart_button_rect = None

        # Info panel for waypoint/vehicle editing will be initialized in startup_sequence
        
        # Undo/Redo system
        # Undo/redo stacks + edit-position counters live on CommandHistory;
        # its on_change callback is the single wiring point back into the
        # editor (dirty-hint + info-panel refresh). The old write-only
        # command_history list is gone (never read).
        self.history = CommandHistory(25, on_change=self._on_history_change)

    def _camera_debug(self, message: str) -> None:
        return stream_settings._camera_debug(self, message)

    def _debug_camera_pose(self, label: str):
        return stream_settings._debug_camera_pose(self, label)

    def _auto_camera_allowed(self) -> bool:
        return stream_settings._auto_camera_allowed(self)


    def _process_pending_camera_focus(self) -> None:
        return stream_settings._process_pending_camera_focus(self)


    def connect_to_remote(self):
        return relaunch.connect_to_remote(self)

    def _show_remote_connection_dialog(self):
        return relaunch._show_remote_connection_dialog(self)

    def _using_remote_server(self) -> bool:
        return stream_settings._using_remote_server(self)

    def _apply_stream_settings(self):
        return stream_settings._apply_stream_settings(self)

    def _cache_camera_pose_for_no_camera(self) -> None:
        return stream_settings._cache_camera_pose_for_no_camera(self)

    def _apply_camera_pose(self, pose: Tuple[float, float, float]) -> None:
        return stream_settings._apply_camera_pose(self, pose)

    def _jump_camera_to_no_camera_pose(self) -> None:
        return stream_settings._jump_camera_to_no_camera_pose(self)

    def _restore_camera_pose_after_no_camera(self) -> None:
        return stream_settings._restore_camera_pose_after_no_camera(self)

    def _set_stream_resolution(self, resolution: Optional[Tuple[int, int]]):
        return stream_settings._set_stream_resolution(self, resolution)

    def _set_stream_fps(self, fps: int):
        return stream_settings._set_stream_fps(self, fps)



    def _capture_world_settings_signature(self, settings) -> Optional[Tuple[bool, bool, Optional[float], bool]]:
        return world_lifecycle._capture_world_settings_signature(self, settings)

    def _apply_rendering_mode(
        self,
        desired_enabled: Optional[bool] = None,
        *,
        reason: Optional[str] = None,
        force: bool = False,
        world=None,
    ) -> bool:
        return world_lifecycle._apply_rendering_mode(self, desired_enabled, reason=reason, force=force, world=world)

    def _check_world_replacement(self):
        return world_lifecycle._check_world_replacement(self)

    def _register_world_tick_handler(self):
        return world_lifecycle._register_world_tick_handler(self)

    def _handle_world_settings_change(self, settings):
        return world_lifecycle._handle_world_settings_change(self, settings)

    def _switch_world_to_async_if_safe(self, reason: Optional[str] = None, *, force: bool = False) -> bool:
        return world_lifecycle._switch_world_to_async_if_safe(self, reason, force=force)

    def _set_large_map_bootstrap_ticking(
        self,
        enabled: bool,
        *,
        log_message: Optional[str] = None,
        disable_manual_tick: bool = False,
        clear_pending_external_ego: bool = False,
    ) -> None:
        return world_lifecycle._set_large_map_bootstrap_ticking(self, enabled, log_message=log_message, disable_manual_tick=disable_manual_tick, clear_pending_external_ego=clear_pending_external_ego)

    def _external_ego_can_prompt(self) -> bool:
        return world_lifecycle._external_ego_can_prompt(self)

    def _show_external_swap_overlay(self, message: Optional[str] = None) -> None:
        return world_lifecycle._show_external_swap_overlay(self, message)

    def _clear_external_swap_overlay(self) -> None:
        return world_lifecycle._clear_external_swap_overlay(self)

    def _show_error_overlay(self, message: str, duration: float = 4.0) -> None:
        return chrome._show_error_overlay(self, message, duration)

    def _render_error_overlay(self) -> None:
        return chrome._render_error_overlay(self)

    def _render_server_lost_overlay(self) -> None:
        return chrome._render_server_lost_overlay(self)

    def _restart_after_server_crash(self):
        return relaunch._restart_after_server_crash(self)

    def _perform_external_ego_swap(self, actor: Optional[carla.Actor]) -> bool:
        return world_lifecycle._perform_external_ego_swap(self, actor)

    def _update_external_swap_state(self, actor: Optional[carla.Actor]) -> None:
        return world_lifecycle._update_external_swap_state(self, actor)

    def _handle_external_ego_disconnect(self) -> None:
        return world_lifecycle._handle_external_ego_disconnect(self)

    def _maybe_prompt_external_ego_swap(self, actor: Optional[carla.Actor]) -> None:
        return world_lifecycle._maybe_prompt_external_ego_swap(self, actor)

    def _monitor_external_ego_status(self):
        return world_lifecycle._monitor_external_ego_status(self)

    def _unregister_world_tick_handler(self):
        return world_lifecycle._unregister_world_tick_handler(self)

    def _on_world_tick(self, snapshot):
        return world_lifecycle._on_world_tick(self, snapshot)

    def _process_world_reset_events(self):
        return world_lifecycle._process_world_reset_events(self)

    def _begin_world_reset_recovery(self):
        return world_lifecycle._begin_world_reset_recovery(self)

    def _teardown_world_resources_for_reset(self):
        return world_lifecycle._teardown_world_resources_for_reset(self)

    def _wait_for_world_recovery(self):
        return world_lifecycle._wait_for_world_recovery(self)

    def _finalize_world_reset(self):
        return world_lifecycle._finalize_world_reset(self)

    def _confirm_switch_to_local(self):
        return relaunch._confirm_switch_to_local(self)

    def _clone_profile(self, profile):
        return relaunch._clone_profile(self, profile)

    def _safe_get_world_map(self, *, refresh=True):
        return world_lifecycle._safe_get_world_map(self, refresh=refresh)

    def request_remote_map_change(self, map_name):
        return relaunch.request_remote_map_change(self, map_name)


    def execute_command(self, command):
        """Execute a command and add it to the undo stack"""
        return self.history.execute(command)

    def undo_last_command(self):
        """Undo the last command (Ctrl+Z)"""
        return self.history.undo()

    def redo_last_command(self):
        """Redo the last undone command (Ctrl+Y)"""
        return self.history.redo()

    def _on_history_change(self, command, action):
        """Single CommandHistory callback: dirty-hint on new edits, targeted
        info-panel field updates for the three property commands (bodies
        moved from their editor reach-ins to InfoPanel in step-21), then the
        editing-gated full refresh. The targeted updates deliberately run
        even while the panel is mid-edit — committing a field edit must
        update snap_status and sibling fields immediately.
        """
        if action == "execute":
            self._scene_dirty_hint = True
        panel = getattr(self, "info_panel", None)
        cp = getattr(command, "camera_processor", None)
        if panel is not None and cp is not None:
            applying = action in ("execute", "redo")
            if isinstance(command, UpdateWaypointPropertyCommand):
                panel.refresh_waypoint_property_fields(
                    cp, command.vehicle_id, command.waypoint_index)
            elif isinstance(command, UpdateVehiclePropertyCommand):
                panel.refresh_vehicle_property_fields(
                    cp, command.vehicle_id, command.property_name,
                    command.new_value if applying else command.old_value,
                    None if applying else command.old_transform)
            elif isinstance(command, UpdateIgnoreFlagsCommand):
                panel.set_ignore_flag_field(
                    command.flag_key,
                    command.new_value if applying else command.old_value)
        self._refresh_active_info_panel()

    def _refresh_active_info_panel(self):
        """Refresh the info panel after history mutations to reflect live state."""
        panel = getattr(self, "info_panel", None)
        if panel and panel.visible and not panel.editing:
            panel._refresh_fields()

    # Scenario runner methods (Play button functionality)
    def _expected_ego_roles(self):
        return world_lifecycle._expected_ego_roles(self)

    def _scenario_ego_type(self):
        return world_lifecycle._scenario_ego_type(self)

    def _get_current_world(self):
        return world_lifecycle._get_current_world(self)

    def _refresh_external_ego_actor_reference(self):
        return world_lifecycle._refresh_external_ego_actor_reference(self)

    def _detect_external_ego_vehicle(self, *, silent: bool = False):
        return world_lifecycle._detect_external_ego_vehicle(self, silent=silent)

    def _resolve_external_ego_actor(
        self,
        *,
        silent: bool,
        force_scan: bool = False,
        allow_scan: bool = False,
    ) -> Optional[carla.Actor]:
        return world_lifecycle._resolve_external_ego_actor(self, silent=silent, force_scan=force_scan, allow_scan=allow_scan)

    def _focus_camera_on_ego_vehicle(self) -> bool:
        return world_lifecycle._focus_camera_on_ego_vehicle(self)

    def _close_result_window(self):
        return playback_launch._close_result_window(self)

    def _enqueue_result_dialog(self, reason: str):
        return playback_launch._enqueue_result_dialog(self, reason)

    def _copy_result_text(self, text: str):
        return playback_launch._copy_result_text(self, text)

    def _save_result_text(self, text: str, default_path: Optional[str]):
        return playback_launch._save_result_text(self, text, default_path)

    def _maybe_open_result_window(self):
        return playback_launch._maybe_open_result_window(self)

    def _reap_finished_runner(self):
        return playback_launch._reap_finished_runner(self)

    def _capture_playback_preserved_state(
        self,
    ) -> Tuple[Optional[Dict[str, object]], Optional[int]]:
        return playback_launch._capture_playback_preserved_state(self)

    def _detect_scenario_ego_flags(self) -> Tuple[bool, bool]:
        return playback_launch._detect_scenario_ego_flags(self)

    def _prepare_external_ego_for_playback(self, scenario_has_ego: bool) -> Tuple[bool, bool]:
        return playback_launch._prepare_external_ego_for_playback(self, scenario_has_ego)

    def _restore_preserved_waypoints_for_playback(self, ego_override_id: Optional[int]) -> None:
        return playback_launch._restore_preserved_waypoints_for_playback(self, ego_override_id)

    def _apply_playback_ui_state(
        self,
        *,
        active_trigger_snapshot: Optional[Dict[str, object]],
        scenario_has_ego: bool,
        external_ego_present: bool,
        ego_has_route: bool,
    ) -> None:
        return playback_launch._apply_playback_ui_state(self, active_trigger_snapshot=active_trigger_snapshot, scenario_has_ego=scenario_has_ego, external_ego_present=external_ego_present, ego_has_route=ego_has_route)

    def run_scenario(self):
        return playback_launch.run_scenario(self)

    def _discard_unsaved_changes_for_play(self):
        return playback_launch._discard_unsaved_changes_for_play(self)

    def stop_scenario(self):
        return playback_launch.stop_scenario(self)


    def _record_scenario_start_markers(self):
        return playback_launch._record_scenario_start_markers(self)

    def _clear_scenario_start_markers(self):
        return playback_launch._clear_scenario_start_markers(self)
    
    def _reset_scenario_traffic_light_preserve(self):
        return playback_launch._reset_scenario_traffic_light_preserve(self)

    def _reset_scenario_waypoint_preserve(self, *, include_traffic: bool = True):
        return playback_launch._reset_scenario_waypoint_preserve(self, include_traffic=include_traffic)

    def _restore_preserved_traffic_light_selection(self) -> bool:
        return playback_launch._restore_preserved_traffic_light_selection(self)

    def _compute_waypoint_signature(self, waypoints: Optional[List[dict]]) -> Optional[Tuple]:
        return playback_launch._compute_waypoint_signature(self, waypoints)

    def _restore_preserved_overlay_visibility(self) -> None:
        return playback_launch._restore_preserved_overlay_visibility(self)

    def _reselect_preserved_actor(self) -> bool:
        return playback_launch._reselect_preserved_actor(self)

    def _process_pending_actor_reselect(self):
        return playback_launch._process_pending_actor_reselect(self)

    def _save_all_scene_vehicles(self):
        return playback_launch._save_all_scene_vehicles(self)

    def _restore_scene_preview_if_destroyed(self, focus_fn: Callable[[], None]) -> bool:
        return playback_launch._restore_scene_preview_if_destroyed(self, focus_fn)

    def _restore_without_saved_actors(self, focus_fn: Callable[[], None]) -> bool:
        return playback_launch._restore_without_saved_actors(self, focus_fn)

    def _restore_all_scene_vehicles_once(self):
        return playback_launch._restore_all_scene_vehicles_once(self)

    def _restore_all_scene_vehicles(self):
        return playback_launch._restore_all_scene_vehicles(self)

    def _get_map_display_name(self):
        return scenario_files._get_map_display_name(self)

    def _resolve_default_map_directory(self):
        return scenario_files._resolve_default_map_directory(self)

    def show_open_map_dialog(self):
        return scenario_files.show_open_map_dialog(self)

    def show_open_scenario_dialog(self):
        return scenario_files.show_open_scenario_dialog(self)

    def _scene_has_content(self) -> bool:
        return scenario_files._scene_has_content(self)

    def _scene_signature_from_payload(self, payload: Optional[dict], *, context: str = "") -> Optional[str]:
        return scenario_files._scene_signature_from_payload(self, payload, context=context)

    def _compute_scene_signature(self) -> Optional[str]:
        return scenario_files._compute_scene_signature(self)

    def _compute_disk_scene_signature(self) -> Optional[str]:
        return scenario_files._compute_disk_scene_signature(self)

    def _mark_scene_saved(
        self,
        live_signature: Optional[str] = None,
        disk_signature: Optional[str] = None,
    ):
        return scenario_files._mark_scene_saved(self, live_signature, disk_signature)

    def _has_unsaved_changes(self) -> bool:
        return scenario_files._has_unsaved_changes(self)

    def _unsaved_changes_handler(self, dialog):
        return scenario_files._unsaved_changes_handler(self, dialog)

    def _confirm_new_scenario(self) -> str:
        return scenario_files._confirm_new_scenario(self)

    def _confirm_load_scenario_discard(self) -> str:
        return scenario_files._confirm_load_scenario_discard(self)

    def _confirm_play_discard_changes(self) -> str:
        return scenario_files._confirm_play_discard_changes(self)

    def _reset_camera_view_to_origin(self):
        return scenario_files._reset_camera_view_to_origin(self)

    def reset_current_scenario(self) -> bool:
        return scenario_files.reset_current_scenario(self)

    def _scenario_has_playable_content(self) -> bool:
        return scenario_files._scenario_has_playable_content(self)


    def _get_vse_cache_dir(self) -> Path:
        return settings_app._get_vse_cache_dir(self)

    # Legacy per-feature cache files, folded into settings.json on first launch (then removed).
    _LEGACY_SETTINGS_FILES = {
        'remote': 'last_remote.json',
        'agent': 'last_agent.json',
        'scenario': 'last_scenario.json',
        'play_camera': 'last_play_camera.json',
    }

    def _get_settings_path(self) -> Path:
        return settings_app._get_settings_path(self)

    def _load_settings(self) -> Dict[str, Any]:
        return settings_app._load_settings(self)

    def _read_settings_section(self, key: str) -> Dict[str, Any]:
        return settings_app._read_settings_section(self, key)

    def _write_settings_section(self, key: str, payload: Any) -> None:
        return settings_app._write_settings_section(self, key, payload)

    def _migrate_legacy_settings(self) -> None:
        return settings_app._migrate_legacy_settings(self)

    def _load_last_remote_cache(self) -> None:
        return settings_app._load_last_remote_cache(self)

    def _remember_last_remote(self, host: str, port: Union[str, int, None]) -> None:
        return settings_app._remember_last_remote(self, host, port)

    def _load_last_agent_cache(self) -> None:
        return settings_app._load_last_agent_cache(self)

    def _remember_last_agent(self, agent_path: Optional[str] = None) -> None:
        return settings_app._remember_last_agent(self, agent_path)

    def _clear_last_agent_cache(self, *, remove_file: bool = False) -> None:
        return settings_app._clear_last_agent_cache(self, remove_file=remove_file)

    def _get_last_agent_directory(self) -> Optional[str]:
        return settings_app._get_last_agent_directory(self)

    def _load_last_scenario_cache(self) -> None:
        return settings_app._load_last_scenario_cache(self)

    def _remember_last_scenario(self, file_path: str, scenario_name: Optional[str], scenario_map: Optional[str]) -> None:
        return settings_app._remember_last_scenario(self, file_path, scenario_name, scenario_map)

    def _clear_last_scenario_cache(self, *, remove_file: bool = False) -> None:
        return settings_app._clear_last_scenario_cache(self, remove_file=remove_file)

    # Server-liveness watchdog tunables (Phase 8, fix-05). The poll runs every
    # _SERVER_POLL_INTERVAL_S while healthy (was every frame pre-Phase-8);
    # a segfaulted server refuses TCP instantly, so 2 consecutive failures
    # declare it dead in ~1-2 s (a wedged-but-listening server takes 2 RPC
    # timeouts instead). While lost, retries run every _SERVER_RETRY_INTERVAL_S
    # and auto-clear the state when the server answers again.
    _server_poll_interval_s = 0.5
    # The RPC port must be closed across this many back-to-back probes
    # (~0.2 s apart) to declare the server dead — a closed port is the ONLY
    # death signal (a live-but-laggy server keeps its port open, so it is never
    # falsely declared dead). 3 probes ≈ 0.4 s of continuous closure.
    _server_dead_after_failures = 3
    _server_retry_interval_s = 5.0

    # Playback camera modes the Play Cam dropdown / persistence accept.
    PLAY_CAMERA_MODES = ("topdown", "chase", "cockpit")

    def _load_last_play_camera_cache(self) -> None:
        return settings_app._load_last_play_camera_cache(self)

    def _remember_last_play_camera(self) -> None:
        return settings_app._remember_last_play_camera(self)

    # Culling distance presets (metres) for the "Cull" dropdown; 0.0 = Off (engine default).
    CULLING_PRESETS = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)
    # Out-of-box default when nothing is remembered (no `culling` section) and no env/CLI override: Off.
    CULLING_DEFAULT_M = 0.0

    def _load_culling(self) -> None:
        return settings_app._load_culling(self)

    def _remember_last_culling(self) -> None:
        return settings_app._remember_last_culling(self)

    @property
    def culling_enabled(self) -> bool:
        return settings_app.culling_enabled(self)

    def _culling_apply_safe(self) -> bool:
        return settings_app._culling_apply_safe(self)

    def _apply_culling(self, *, world=None, force: bool = False, reason: str = "") -> bool:
        return settings_app._apply_culling(self, world=world, force=force, reason=reason)

    def _set_culling_distance(self, meters: float) -> None:
        return settings_app._set_culling_distance(self, meters)

    def _set_play_camera_mode(self, mode: str, *, persist: bool = True, apply_live: bool = True) -> None:
        return settings_app._set_play_camera_mode(self, mode, persist=persist, apply_live=apply_live)

    def _load_scenario_from_path(self, file_path: str, *, prompt_unsaved: bool = True) -> bool:
        return scenario_files._load_scenario_from_path(self, file_path, prompt_unsaved=prompt_unsaved)

    def _open_last_scenario_from_cache(self) -> bool:
        return scenario_files._open_last_scenario_from_cache(self)

    def _prompt_scenario_file_path(self, *, initial_path: Optional[str] = None) -> Optional[str]:
        return scenario_files._prompt_scenario_file_path(self, initial_path=initial_path)

    def _prompt_agent_file_path(self, *, initial_path: Optional[str] = None) -> Optional[str]:
        return scenario_files._prompt_agent_file_path(self, initial_path=initial_path)

    def _prompt_startup_map_choice(self, options: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        return scenario_files._prompt_startup_map_choice(self, options)
    
    def load_scenario_with_dialog(self):
        return scenario_files.load_scenario_with_dialog(self)

    def _save_current_scenario(self) -> bool:
        return scenario_files._save_current_scenario(self)

    def save_scenario_with_dialog(self):
        return scenario_files.save_scenario_with_dialog(self)

    def export_scenario_as_xosc_with_dialog(self):
        return scenario_files.export_scenario_as_xosc_with_dialog(self)

    def _prepare_for_map_change(self):
        return relaunch._prepare_for_map_change(self)

    def _wait_for_world_tick(
        self,
        world,
        *,
        timeout: float = 30.0,
        min_ticks: int = 1,
        label: str = "world",
    ) -> bool:
        return startup._wait_for_world_tick(self, world, timeout=timeout, min_ticks=min_ticks, label=label)

    def _wait_for_remote_server(self, host, port, timeout=60):
        return startup._wait_for_remote_server(self, host, port, timeout)

    def _probe_remote_connection(self, host, port, *, max_attempts=3, timeout=3.0):
        return startup._probe_remote_connection(self, host, port, max_attempts=max_attempts, timeout=timeout)

    def _create_local_profile(self):
        return relaunch._create_local_profile(self)

    def _create_remote_profile(self, host, port, label=None):
        return relaunch._create_remote_profile(self, host, port, label)


    def _restore_start_weather_from_presets(self):
        return weather_ctl._restore_start_weather_from_presets(self)

    def _restart_with_profile(self, profile, *, label=None, target_map=None):
        return relaunch._restart_with_profile(self, profile, label=label, target_map=target_map)

    def _perform_profile_restart(self, profile, label, target_map):
        return relaunch._perform_profile_restart(self, profile, label, target_map)

    def _handle_profile_restart_failure(self, profile_name, error):
        return relaunch._handle_profile_restart_failure(self, profile_name, error)

    def switch_to_local_server(self):
        return relaunch.switch_to_local_server(self)

    def _enter_restart_wait_state(self):
        return relaunch._enter_restart_wait_state(self)

    def _reset_restart_flags(self):
        return relaunch._reset_restart_flags(self)

    def _manual_world_tick(self, dt):
        return world_lifecycle._manual_world_tick(self, dt)

    def _load_remote_map(self, map_name):
        return relaunch._load_remote_map(self, map_name)

    def _handle_map_restart_failure(self, map_name, error):
        return relaunch._handle_map_restart_failure(self, map_name, error)

    def load_map(self, map_name):
        return relaunch.load_map(self, map_name)

    def _cleanup_for_map_restart(self):
        return relaunch._cleanup_for_map_restart(self)

    def _notify_handoff_ready(self):
        return relaunch._notify_handoff_ready(self)

    def _perform_map_restart(self, map_name):
        return relaunch._perform_map_restart(self, map_name)

    def _force_reload_current_map_for_external_ego(self):
        return relaunch._force_reload_current_map_for_external_ego(self)

    def _perform_fast_map_reload(self, map_name):
        return relaunch._perform_fast_map_reload(self, map_name)

    def _extract_scenario_map_name(self, file_path: str) -> Optional[str]:
        return startup._extract_scenario_map_name(self, file_path)

    def _resolve_carla_maps_root(self) -> Path:
        return startup._resolve_carla_maps_root(self)

    def _resolve_map_package_path(self, map_name: str) -> Optional[str]:
        return startup._resolve_map_package_path(self, map_name)

    def _list_local_startup_maps(self) -> List[Dict[str, str]]:
        return startup._list_local_startup_maps(self)

    def _patch_default_engine_ini(self, ini_path: Path, map_value: str) -> None:
        return startup._patch_default_engine_ini(self, ini_path, map_value)

    def _get_default_engine_ini_path(self) -> Optional[Path]:
        return startup._get_default_engine_ini_path(self)

    def _get_default_engine_ini_backup_path(self, ini_path: Path) -> Path:
        return startup._get_default_engine_ini_backup_path(self, ini_path)

    def _restore_default_engine_ini_backup_if_present(self) -> None:
        return startup._restore_default_engine_ini_backup_if_present(self)

    def _backup_and_patch_default_engine_ini_for_startup(self, map_value: str) -> None:
        return startup._backup_and_patch_default_engine_ini_for_startup(self, map_value)

    def _restore_default_engine_ini_backup(self) -> None:
        return startup._restore_default_engine_ini_backup(self)

    def _begin_startup_for_scenario(self, scenario_path: str, map_name: str, map_value: str) -> None:
        return startup._begin_startup_for_scenario(self, scenario_path, map_name, map_value)

    def _begin_startup_for_map(self, map_name: str, map_value: str) -> None:
        return startup._begin_startup_for_map(self, map_name, map_value)

    def _startup_entrypoint(self) -> None:
        return startup._startup_entrypoint(self)
        
    def startup_sequence(self):
        return startup.startup_sequence(self)
    
    def toggle_maximize(self):
        return chrome.toggle_maximize(self)
    
    def calculate_acceleration_factor(self, hold_time):
        return events.calculate_acceleration_factor(self, hold_time)
    
    def show_exit_confirmation(self, **kwargs) -> str:
        return events.show_exit_confirmation(self, **kwargs)
    

    def handle_events(self, dt):
        return events.handle_events(self, dt)


    def render_start_screen(self):
        return chrome.render_start_screen(self)

    def _handle_start_screen_click(self, mouse_pos: Tuple[int, int]) -> bool:
        return startup._handle_start_screen_click(self, mouse_pos)

    def _start_screen_open_scenario(self) -> None:
        return startup._start_screen_open_scenario(self)

    def _start_screen_open_map(self) -> None:
        return startup._start_screen_open_map(self)

    def _start_screen_launch_scenario(self, scenario_path: str) -> None:
        return startup._start_screen_launch_scenario(self, scenario_path)

    def render_loading_screen(self):
        return chrome.render_loading_screen(self)

    def _render_external_swap_overlay(self):
        return chrome._render_external_swap_overlay(self)

    def _render_external_swap_overlay_immediate(self):
        return chrome._render_external_swap_overlay_immediate(self)

    def _truncate_text_to_width(self, font, text: str, max_width: int) -> str:
        return chrome._truncate_text_to_width(self, font, text, max_width)

    def _draw_keycap(self, surface, label, left, center_y, font):
        return chrome._draw_keycap(self, surface, label, left, center_y, font)

    def _draw_mouse_glyph(self, surface, left, center_y, highlight=None):
        return chrome._draw_mouse_glyph(self, surface, left, center_y, highlight)

    def _draw_key_combo(self, surface, tokens, left, center_y, font):
        return chrome._draw_key_combo(self, surface, tokens, left, center_y, font)

    def render_keyboard_help_overlay(self, screen):
        return chrome.render_keyboard_help_overlay(self, screen)

    def render_ui(self):
        return chrome.render_ui(self)

    def _close_all_dropdowns(self, *, except_menu=None, keep_info_panel: bool = False) -> None:
        return chrome._close_all_dropdowns(self, except_menu=except_menu, keep_info_panel=keep_info_panel)

    def get_active_selection_menu(self):
        return chrome.get_active_selection_menu(self)

    def _get_mode_button_rects(self):
        return chrome._get_mode_button_rects(self)

    def _get_weather_button_rect(self) -> pygame.Rect:
        return chrome._get_weather_button_rect(self)

    def handle_mode_toggle_click(self, mouse_pos):
        return chrome.handle_mode_toggle_click(self, mouse_pos)

    def set_placement_mode(self, mode):
        return chrome.set_placement_mode(self, mode)

    def render_mode_toggle(self):
        return chrome.render_mode_toggle(self)

    def render_view3d_button(self):
        return chrome.render_view3d_button(self)

    def _weather_dict_from_params(self, weather: "carla.WeatherParameters") -> Dict[str, float]:
        return weather_ctl._weather_dict_from_params(self, weather)

    def _capture_baseline_weather(self, weather: Optional["carla.WeatherParameters"]) -> None:
        return weather_ctl._capture_baseline_weather(self, weather)

    def _reset_weather_to_baseline(self) -> None:
        return weather_ctl._reset_weather_to_baseline(self)

    def _weather_params_from_dict(self, values: Dict[str, float]) -> "carla.WeatherParameters":
        return weather_ctl._weather_params_from_dict(self, values)

    def _sanitize_weather_keyframes_payload(self, keyframes: Optional[Iterable[Dict[str, float]]]) -> List[Dict[str, float]]:
        return weather_ctl._sanitize_weather_keyframes_payload(self, keyframes)

    def _ensure_weather_keyframes(self) -> None:
        return weather_ctl._ensure_weather_keyframes(self)

    def _update_weather_state(self, weather: "carla.WeatherParameters", *, keyframe_index: Optional[int] = None) -> None:
        return weather_ctl._update_weather_state(self, weather, keyframe_index=keyframe_index)


    def _on_weather_window_closed(self) -> None:
        return weather_ctl._on_weather_window_closed(self)

    def _on_weather_slider_preview(self, parameter: str, value: float) -> None:
        return weather_ctl._on_weather_slider_preview(self, parameter, value)

    def _on_weather_slider_commit(self, parameter: str, value: float) -> None:
        return weather_ctl._on_weather_slider_commit(self, parameter, value)

    def _refresh_weather_window_metadata(self) -> None:
        return weather_ctl._refresh_weather_window_metadata(self)

    def _set_active_weather_index(self, index: int) -> None:
        return weather_ctl._set_active_weather_index(self, index)

    def _set_active_weather_percentage(self, percentage: float) -> None:
        return weather_ctl._set_active_weather_percentage(self, percentage)

    def _add_weather_keyframe_at(self, percentage: float) -> None:
        return weather_ctl._add_weather_keyframe_at(self, percentage)

    def _delete_active_weather_keyframe(self) -> None:
        return weather_ctl._delete_active_weather_keyframe(self)

    def _on_weather_keyframe_percentage_change(self, value: float) -> None:
        return weather_ctl._on_weather_keyframe_percentage_change(self, value)

    def _on_weather_add_keyframe(self) -> None:
        return weather_ctl._on_weather_add_keyframe(self)

    def _on_weather_delete_keyframe(self) -> None:
        return weather_ctl._on_weather_delete_keyframe(self)

    def _on_weather_prev_keyframe(self) -> None:
        return weather_ctl._on_weather_prev_keyframe(self)

    def _on_weather_next_keyframe(self) -> None:
        return weather_ctl._on_weather_next_keyframe(self)

    def _apply_weather_from_json_data(self, scenario_data: Optional[dict]) -> bool:
        return weather_ctl._apply_weather_from_json_data(self, scenario_data)

    def toggle_weather_window(self) -> None:
        return weather_ctl.toggle_weather_window(self)

    def render_crosshair(self):
        return chrome.render_crosshair(self)
    
    def _update_fps_meter(self, raw_fps: float) -> None:
        return chrome._update_fps_meter(self, raw_fps)

    def render_fps_meter(self) -> Optional[pygame.Rect]:
        return chrome.render_fps_meter(self)

    def render_rendering_toggle(self, anchor_rect: Optional[pygame.Rect]) -> None:
        return chrome.render_rendering_toggle(self, anchor_rect)

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")

        self._close_result_window()
        self._world_reset_in_progress = False
        self._world_reset_finalize_pending = False
        self._world_reset_ready_world = None
        self._world_reset_candidate_world = None
        self._world_reset_pending = False
        self._unregister_world_tick_handler()

        try:
            scenario_proc = getattr(self, "scenario_process", None)
            mini_runner = getattr(self, "_mini_runner", None)
            scenario_active = bool(
                getattr(self, "scenario_running", False)
                or (scenario_proc and scenario_proc.poll() is None)
                or (mini_runner and getattr(mini_runner, "is_running", False))
            )
            if scenario_active:
                self.stop_scenario()
        except Exception as exc:
            print(f"[Cleanup] Warning: failed to stop scenario: {exc}")
        
        # Clean up vehicles first. When the server is known dead (fix-05/06),
        # skip the RPC-heavy teardown: destroy()/batch-destroy would each block
        # on the client timeout against a corpse, hanging exit. The actors and
        # sensor die with the server anyway; just drop the references.
        if self.camera_processor:
            if getattr(self, "_server_lost", False):
                try:
                    self.camera_processor._unregister_pedestrian_highlight_tick()
                except Exception:
                    pass
                self.camera_processor.camera_sensor = None
                self.camera_processor.spawned_vehicles = []
            else:
                self.camera_processor.cleanup_all_vehicles()
                self.camera_processor.cleanup()
        
        if self.vehicle_menu:
            self.vehicle_menu.cleanup()
        if self.pedestrian_menu:
            self.pedestrian_menu.cleanup()

        if self.keep_server_running_on_exit:
            self.server_manager.set_auto_stop_enabled(False)
        else:
            self.server_manager.set_auto_stop_enabled(True)
            self.server_manager.stop_server(force=True)
        if self.handoff_read_fd is not None:
            try:
                os.close(self.handoff_read_fd)
            except OSError:
                pass
            self.handoff_read_fd = None
        if self.handoff_write_fd is not None:
            try:
                os.close(self.handoff_write_fd)
            except OSError:
                pass
            self.handoff_write_fd = None
        pygame.quit()
        self.keep_server_running_on_exit = False
    
    def run(self):
        """
        Main application loop for the Visual Scenario Editor.
        Handles startup, event processing, rendering, and cleanup.
        """
        clock = pygame.time.Clock()
        running = True

        # --- Startup Banner ---
        print("\nVisual Scenario Editor")
        print("======================")
        print("Click waypoints to select, drag to move")
        print("Press 'O' to toggle OpenDRIVE overlay, 'T' to toggle traffic light stop lines (both start disabled)")
        print("Ctrl+Click to spawn selected entity, Ctrl+S to save, Ctrl+L to load")
        print("ESC cancels operations - Alt+F4 or close button (X) to exit")

        try:
            while running:
                dt = clock.tick(60) / 1000.0  # --- Timing: Delta time in seconds ---
                self._update_fps_meter(clock.get_fps())

                # --- Startup Sequence ---
                if (
                    self._startup_requested
                    and not self._startup_thread_started
                    and not self.ready
                    and not self.startup_error
                    and not self.start_screen_active
                ):
                    startup_thread = threading.Thread(target=self._startup_entrypoint)
                    startup_thread.daemon = True
                    startup_thread.start()
                    self._startup_thread_started = True

                # --- Event Handling ---
                running = self.handle_events(dt)

                if self.ui_manager:
                    self.ui_manager.update(dt)

                # --- Rendering ---
                self.screen.fill(self.colors['background'])  # Clear screen

                # --- Handle world reset lifecycle ---
                self._process_world_reset_events()

                if not self.ready:
                    if self.start_screen_active:
                        self.render_start_screen()
                    else:
                        self.render_loading_screen()
                    self._dropdown_draw_ops = []
                else:
                    self._maybe_open_result_window()
                    self._reap_finished_runner()

                    # --- Update Simulation State ---
                    # While the server is lost (fix-05) the per-frame RPC users
                    # are gated: they only swallow errors against a dead
                    # server and each failed call burns a client timeout.
                    if not self._server_lost:
                        self._monitor_external_ego_status()

                        if self.manual_tick_required:
                            self._manual_world_tick(dt)

                        self._process_pending_camera_focus()

                        if self.camera_processor:
                            self.camera_processor.update(dt)
                            self._process_pending_actor_reselect()

                    # --- Render Camera Feed ---
                    if self.camera_processor:
                        latest_image = self.camera_processor.get_latest_image()
                        if self.camera_stream_enabled and latest_image:
                            self.screen.blit(latest_image, (0, 0))
                        elif not self.camera_stream_enabled:
                            self.screen.fill((0, 0, 0))

                    # --- Clear tooltip hover state for this frame ---
                    if self.tooltip_manager:
                        self.tooltip_manager.clear_hover()

                    # --- Render Overlays and UI ---
                    # `hide_all_ui` (backtick toggle) suppresses every overlay/UI draw,
                    # leaving only the camera image; the camera blit and the state logic
                    # below keep running, so input stays live while hidden.
                    if not self.hide_all_ui:
                        self.render_crosshair()  # Center crosshair
                        # Render action menus BEFORE UI panels so they appear behind
                        if self.camera_processor:
                            self.camera_processor.render_action_menus(self.screen)
                        self.render_ui()         # Top UI bar
                        self.render_mode_toggle()
                        self.render_view3d_button()  # Centered top-bar 3D toggle
                        active_menu = self.get_active_selection_menu()
                        if (active_menu and not self.scenario_running):
                            active_menu.render(self.screen, tooltip_manager=self.tooltip_manager)

                    # Sync vehicle_control_mode from camera_processor (menu pills write there)
                    # Skip during restore to avoid race with the MiniRunner thread
                    if self.camera_processor and not getattr(self, '_restore_in_progress', False):
                        self.vehicle_control_mode = getattr(
                            self.camera_processor, 'vehicle_control_mode', 'basic_agent')

                    # --- Render Vehicle and Waypoint Overlays ---
                    # Skip when the server is lost (fix-05b): render_all_overlays
                    # makes CARLA RPCs (e.g. selected-actor/traffic-light
                    # transforms) that would block on the shared client behind
                    # the MiniRunner thread's dead-server RPC — the stall that
                    # delayed the connection-lost overlay during playback.
                    if not self.hide_all_ui and not self._server_lost:
                        if self.camera_processor:
                            self.camera_processor.render_all_overlays(self.screen)

                        # --- Render Info Panel ---
                        self.info_panel.render(self.screen, tooltip_manager=self.tooltip_manager)

                        # --- Render Map Selection Menu ---
                        if self.map_menu_visible and self.map_menu:
                            mouse_pos = pygame.mouse.get_pos()
                            self.map_menu.render(self.screen, mouse_pos)

                        # --- Render Scenario Selection Menu ---
                        if self.scenario_menu_visible and self.scenario_menu:
                            mouse_pos = pygame.mouse.get_pos()
                            self.scenario_menu.render(self.screen, mouse_pos)

                if not self.hide_all_ui:
                    if self.ui_manager:
                        self.ui_manager.draw_ui(self.screen)

                    self._render_external_swap_overlay()
                    self._render_error_overlay()

                    fps_rect = self.render_fps_meter()
                    self.render_rendering_toggle(fps_rect)

                    # Deferred dropdown rendering (must be after FPS meter to overlay on top)
                    if getattr(self, "_dropdown_draw_ops", None):
                        for rect, fill_color, text_surface in self._dropdown_draw_ops:
                            if fill_color is not None:
                                pygame.draw.rect(self.screen, fill_color, rect, border_radius=4)
                                pygame.draw.rect(self.screen, self.colors['text'], rect, 1, border_radius=4)
                            text_rect = text_surface.get_rect(center=rect.center)
                            self.screen.blit(text_surface, text_rect)

                    # --- Render Tooltip (must be last, on top of everything) ---
                    if self.tooltip_manager:
                        self.tooltip_manager.update()
                        self.tooltip_manager.render(self.screen)

                    # --- Keyboard shortcuts help modal (drawn last, over everything) ---
                    if self.keyboard_help_visible:
                        self.render_keyboard_help_overlay(self.screen)

                # Server-lost overlay (fix-05) renders regardless of
                # hide_all_ui: playback hides the UI, and a mid-run server
                # crash must still surface visibly.
                self._render_server_lost_overlay()

                # --- Display Update ---
                pygame.display.flip()

                if self.pending_exit:
                    if self.handoff_read_fd is not None:
                        ready, _, _ = select.select([self.handoff_read_fd], [], [], 0)
                        if ready:
                            try:
                                os.read(self.handoff_read_fd, 1)
                            except OSError:
                                pass
                            try:
                                os.close(self.handoff_read_fd)
                            except OSError:
                                pass
                            self.handoff_read_fd = None
                            if self.pending_profile_switch:
                                if (self.connection_profile.manage_server and
                                        self.pending_profile_switch.is_remote):
                                    print("[Connection] Remote session confirmed; local CARLA server will be stopped on exit.")
                                    self.keep_server_running_on_exit = False
                                self.pending_profile_switch = None
                                self.pending_profile_switch_label = None
                            self.pending_exit = False
                            running = False
                            continue
                    if self.relaunch_process and self.relaunch_process.poll() is not None:
                        return_code = self.relaunch_process.returncode
                        print(f"Map change failed (code {return_code}). Staying in current session.")
                        self.loading_stage = f"Map change failed (code {return_code})."
                        self.startup_error = f"Map change failed (code {return_code})"
                        self.pending_exit = False
                        self.keep_server_running_on_exit = False
                        self.relaunch_process = None
                        self.restart_start_time = None
                        self.ready = False
                        self.server_manager.set_auto_stop_enabled(True)
                        self.wait_for_child = False
                        self.pending_profile_switch = None
                        self.pending_profile_switch_label = None
                    elif self.restart_start_time and (time.time() - self.restart_start_time) >= self.restart_wait_duration:
                        if self.handoff_read_fd is not None:
                            try:
                                os.close(self.handoff_read_fd)
                            except OSError:
                                pass
                            self.handoff_read_fd = None
                        self.pending_exit = False
                        running = False
                        self.pending_profile_switch = None
                        self.pending_profile_switch_label = None

        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
        finally:
            self.cleanup()
            if self.wait_for_child and self.relaunch_process:
                try:
                    self.relaunch_process.wait()
                except Exception:
                    pass
                finally:
                    self.relaunch_process = None
                    self.wait_for_child = False
                    self.server_manager.set_auto_stop_enabled(True)


install_scene_forwarders(VisualScenarioEditor, SCENE_WEATHER_FIELDS)
