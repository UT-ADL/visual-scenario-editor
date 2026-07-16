"""Startup: server boot + DefaultEngine.ini patching, map package resolution,
startup_sequence (runs on the startup thread; constructs CameraImageProcessor,
InfoPanel and menus), start-screen begin flows (duck-typed module functions,
first arg `editor`). Moved verbatim from vse.py (step-41, Phase 7).
"""

import json
import os
import shutil
import time
import traceback

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import carla

from vse_common.geometry import is_large_map as is_large_map_name
from vse_editor.carla_io.camera import TopDownCamera
from vse_editor.carla_io.camera_stream import CameraImageProcessor
from vse_editor.ui.info_panel import InfoPanel
from vse_editor.ui.menus import MapSelectionMenu, ScenarioMenu


def _wait_for_world_tick(
    editor,
    world,
    *,
    timeout: float = 30.0,
    min_ticks: int = 1,
    label: str = "world",
) -> bool:
    """Best-effort wait for CARLA to start ticking (reduces startup race conditions)."""
    if not world or not hasattr(world, "wait_for_tick"):
        return False

    try:
        timeout_s = float(timeout)
    except Exception:
        timeout_s = 0.0
    timeout_s = max(0.0, timeout_s)

    try:
        min_ticks_int = int(min_ticks)
    except Exception:
        min_ticks_int = 1
    min_ticks_int = max(1, min_ticks_int)

    deadline = time.time() + timeout_s
    ticks_seen = 0
    last_exc: Optional[Exception] = None

    while ticks_seen < min_ticks_int and time.time() < deadline:
        try:
            world.wait_for_tick(1.0)
            ticks_seen += 1
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)

    if ticks_seen >= min_ticks_int:
        return True

    if last_exc is not None:
        print(f"[Startup] Timed out waiting for world tick ({label}): {last_exc}")
    else:
        print(f"[Startup] Timed out waiting for world tick ({label}).")
    return False

def _wait_for_remote_server(editor, host, port, timeout=60):
    """Poll a remote CARLA endpoint until it responds or timeout expires."""
    start_time = time.time()
    last_error = None
    try:
        client = carla.Client(host, port)
        client.set_timeout(5.0)
    except Exception as exc:
        print(f"[Remote Wait] Failed to create client for {host}:{port}: {exc}")
        return False

    attempt = 0

    while time.time() - start_time < timeout:
        try:
            attempt += 1
            print(f"[Remote Wait] Attempt {attempt} connecting to {host}:{port}...")
            version = client.get_server_version()
            print(f"[Remote Wait] Remote CARLA server ready! Version: {version}")
            return True
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)

    print(f"[Remote Wait] Timeout waiting for remote CARLA server at {host}:{port}")
    if last_error:
        print(f"[Remote Wait] Last error: {last_error}")
    return False

def _probe_remote_connection(editor, host, port, *, max_attempts=3, timeout=3.0):
    """Quickly verify a remote CARLA endpoint before committing to a profile switch."""
    print(f"[Remote Probe] Probing {host}:{port} before switching...")
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
    except Exception as exc:
        print(f"[Remote Probe] Failed to create client for {host}:{port}: {exc}")
        return False

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[Remote Probe] Attempt {attempt}/{max_attempts}...")
            version = client.get_server_version()
            print(f"[Remote Probe] Success. Remote CARLA version: {version}")
            return True
        except Exception as exc:
            last_error = exc
            print(f"[Remote Probe] Attempt {attempt} failed: {exc}")
            time.sleep(1.0)

    print(f"[Remote Probe] Unable to reach {host}:{port} after {max_attempts} attempts.")
    if last_error:
        print(f"[Remote Probe] Last error: {last_error}")
    return False

def _extract_scenario_map_name(editor, file_path: str) -> Optional[str]:
    """Return the scenario's required map name (short form) or None."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read scenario JSON: {exc}") from exc

    if not isinstance(data, dict):
        return None

    raw = data.get('map_name')
    if raw is None:
        return None
    map_name = str(raw).strip()
    if not map_name or map_name.lower() == 'unknown':
        return None
    return map_name

def _resolve_carla_maps_root(editor) -> Path:
    """Return the CARLA content maps root directory."""
    carla_root = os.environ.get('CARLA_ROOT')
    if not carla_root:
        raise RuntimeError("CARLA_ROOT is not set.")
    maps_root = Path(carla_root) / 'CarlaUE4' / 'Content' / 'Carla' / 'Maps'
    if not maps_root.is_dir():
        raise RuntimeError(f"CARLA maps directory not found: {maps_root}")
    return maps_root

def _resolve_map_package_path(editor, map_name: str) -> Optional[str]:
    """
    Resolve a scenario map name to a CARLA package map path (e.g. /Game/Carla/Maps/Town03).
    Returns None if no matching .umap exists under CARLA_ROOT/CarlaUE4/Content/Carla/Maps.
    """
    map_name = (map_name or '').strip()
    if not map_name or map_name.lower() == 'unknown':
        return None
    # Large maps are composed of streamed tiles named <MapName>_Tile_X_Y. Always resolve
    # to the root <MapName> level so CARLA can stream tiles automatically.
    last_segment = map_name.split('/')[-1]
    if "_Tile_" in last_segment:
        map_name = last_segment.split("_Tile_")[0].strip()
        if not map_name:
            return None
    if map_name.startswith('/Game/'):
        return map_name

    maps_root = editor._resolve_carla_maps_root()
    direct = maps_root / f"{map_name}.umap"
    if direct.is_file():
        rel = direct.relative_to(maps_root).with_suffix('')
        return f"/Game/Carla/Maps/{rel.as_posix()}"

    nested = maps_root / map_name / f"{map_name}.umap"
    if nested.is_file():
        rel = nested.relative_to(maps_root).with_suffix('')
        return f"/Game/Carla/Maps/{rel.as_posix()}"

    matches = sorted(maps_root.rglob(f"{map_name}.umap"), key=lambda p: str(p))
    if not matches:
        return None
    rel = matches[0].relative_to(maps_root).with_suffix('')
    return f"/Game/Carla/Maps/{rel.as_posix()}"

def _list_local_startup_maps(editor) -> List[Dict[str, str]]:
    """Return available local CARLA maps (no server required)."""
    maps_root = editor._resolve_carla_maps_root()
    entries = []

    for umap_path in maps_root.rglob("*.umap"):
        try:
            rel = umap_path.relative_to(maps_root)
        except Exception:
            continue
        if any(part.lower() == "sublevels" for part in rel.parts):
            continue
        if any(part.lower() == "testmaps" for part in rel.parts):
            continue

        short_name = umap_path.stem
        # Large Maps (e.g. Town11/12 or RoadRunner imports) generate per-tile levels named
        # <MapName>_Tile_X_Y; only the root <MapName> level should be selectable on startup.
        if "_Tile_" in short_name:
            continue
        if short_name == "OpenDriveMap":
            continue
        if short_name.lower() == "emptymap":
            continue

        rel_noext = rel.with_suffix("").as_posix()
        map_value = f"/Game/Carla/Maps/{rel_noext}"
        entries.append({
            "short_name": short_name,
            "rel": rel_noext,
            "map_value": map_value,
        })

    entries.sort(key=lambda item: item["map_value"].lower())

    name_counts: Dict[str, int] = {}
    for entry in entries:
        name_counts[entry["short_name"]] = name_counts.get(entry["short_name"], 0) + 1

    options: List[Dict[str, str]] = []
    for entry in entries:
        short_name = entry["short_name"]
        rel_noext = entry["rel"]
        if name_counts.get(short_name, 0) > 1:
            label = f"{short_name} ({rel_noext})"
        else:
            label = short_name
        options.append({
            "label": label,
            "map_name": short_name,
            "map_value": entry["map_value"],
        })

    options.sort(key=lambda item: item["label"].lower())
    return options

def _patch_default_engine_ini(editor, ini_path: Path, map_value: str) -> None:
    """Patch GameMapsSettings default maps inside a DefaultEngine.ini."""
    target_section = "[/Script/EngineSettings.GameMapsSettings]"
    keys = ("EditorStartupMap", "GameDefaultMap", "ServerDefaultMap")

    text = ini_path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    seen = {k: False for k in keys}
    out: List[str] = []
    in_section = False

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            if in_section:
                for k in keys:
                    if not seen[k]:
                        out.append(f"{k}={map_value}")
                in_section = False

            if line.strip() == target_section:
                in_section = True
                out.append(line)
                continue

        if in_section:
            replaced = False
            for k in keys:
                if line.startswith(k + "="):
                    out.append(f"{k}={map_value}")
                    seen[k] = True
                    replaced = True
                    break
            if not replaced:
                out.append(line)
        else:
            out.append(line)

    if in_section:
        for k in keys:
            if not seen[k]:
                out.append(f"{k}={map_value}")

    patched = "\n".join(out) + "\n"
    tmp_path = ini_path.with_suffix(ini_path.suffix + ".tmp")
    tmp_path.write_text(patched, encoding='utf-8')
    os.replace(tmp_path, ini_path)

def _get_default_engine_ini_path(editor) -> Optional[Path]:
    """Return CARLA's DefaultEngine.ini path, or None when CARLA_ROOT is unset."""
    carla_root = os.environ.get('CARLA_ROOT')
    if not carla_root:
        return None
    return Path(carla_root) / 'CarlaUE4' / 'Config' / 'DefaultEngine.ini'

def _get_default_engine_ini_backup_path(editor, ini_path: Path) -> Path:
    """Return VSE's crash-safe backup path for DefaultEngine.ini."""
    return ini_path.with_suffix(ini_path.suffix + ".vse_runtime_backup")

def _restore_default_engine_ini_backup_if_present(editor) -> None:
    """Restore DefaultEngine.ini if a previous crash left a backup behind."""
    ini_path = editor._get_default_engine_ini_path()
    if not ini_path:
        return
    backup_path = editor._get_default_engine_ini_backup_path(ini_path)
    if not backup_path.is_file():
        return
    try:
        print(f"[CARLA INI] Restoring DefaultEngine.ini from backup: {backup_path}")
        os.replace(backup_path, ini_path)
        print("[CARLA INI] DefaultEngine.ini restored.")
    except Exception as exc:
        print(f"[CARLA INI] Failed to restore DefaultEngine.ini backup: {exc}")

def _backup_and_patch_default_engine_ini_for_startup(editor, map_value: str) -> None:
    """
    Temporarily patch CARLA_ROOT/CarlaUE4/Config/DefaultEngine.ini to boot into the requested map.
    The original file is saved as a backup and restored once startup completes.
    """
    ini_path = editor._get_default_engine_ini_path()
    if not ini_path:
        raise RuntimeError("CARLA_ROOT is not set.")
    if not ini_path.is_file():
        raise RuntimeError(f"DefaultEngine.ini not found: {ini_path}")

    backup_path = editor._get_default_engine_ini_backup_path(ini_path)
    if backup_path.exists():
        # Leftover from crash: try to restore first so we snapshot a clean original.
        try:
            os.replace(backup_path, ini_path)
            print("[CARLA INI] Restored leftover DefaultEngine.ini backup before patching.")
        except Exception as exc:
            raise RuntimeError(f"Failed to restore leftover DefaultEngine.ini backup: {exc}") from exc

    shutil.copy2(ini_path, backup_path)
    editor._default_engine_ini_backup_path = backup_path
    try:
        editor._patch_default_engine_ini(ini_path, map_value)
    except Exception:
        # Ensure we don't leave a partially patched ini behind.
        editor._restore_default_engine_ini_backup()
        raise
    print(f"[CARLA INI] Patched DefaultEngine.ini for startup map: {map_value}")

def _restore_default_engine_ini_backup(editor) -> None:
    """Restore DefaultEngine.ini from the active backup, if any."""
    backup_path = getattr(editor, "_default_engine_ini_backup_path", None)
    if not backup_path:
        return
    editor._default_engine_ini_backup_path = None
    if not isinstance(backup_path, Path):
        return
    if not backup_path.is_file():
        return
    ini_path = backup_path.with_suffix("")
    try:
        os.replace(backup_path, ini_path)
        print("[CARLA INI] DefaultEngine.ini restored after startup.")
    except Exception as exc:
        print(f"[CARLA INI] Failed to restore DefaultEngine.ini after startup: {exc}")

def _begin_startup_for_scenario(editor, scenario_path: str, map_name: str, map_value: str) -> None:
    """Transition out of the start screen and begin startup for a local scenario."""
    editor.startup_error = None
    editor.loading_stage = f"Preparing map '{map_name}'..."
    editor._startup_selected_scenario_path = scenario_path
    editor._startup_selected_map_name = map_name
    editor._startup_selected_map_package = map_value
    editor.pending_scenario_load = scenario_path
    editor.start_screen_active = False
    editor.ready = False
    editor._startup_thread_started = False
    editor._startup_requested = True

def _begin_startup_for_map(editor, map_name: str, map_value: str) -> None:
    """Transition out of the start screen and begin startup for a local map."""
    editor.startup_error = None
    editor.loading_stage = f"Preparing map '{map_name}'..."
    editor._startup_selected_scenario_path = None
    editor._startup_selected_map_name = map_name
    editor._startup_selected_map_package = map_value
    editor.pending_scenario_load = None
    editor.start_screen_active = False
    editor.ready = False
    editor._startup_thread_started = False
    editor._startup_requested = True

def _startup_entrypoint(editor) -> None:
    """Wrapper that runs the normal startup sequence."""
    try:
        editor.startup_sequence()
    finally:
        # Always restore DefaultEngine.ini after attempting startup so we don't leave CARLA_ROOT modified.
        editor._restore_default_engine_ini_backup()
        editor._startup_requested = False
        if editor.startup_error and not editor.ready:
            editor.start_screen_active = True
            editor._startup_thread_started = False
            editor._startup_selected_scenario_path = None
            editor._startup_selected_map_name = None
            editor._startup_selected_map_package = None
            editor.pending_scenario_load = None

def startup_sequence(editor):
    """Complete startup sequence with a single managed retry on failure."""
    max_attempts = 2
    attempt = 0

    while attempt < max_attempts:
        profile = editor.connection_profile
        host = profile.host
        port = profile.port
        try:
            print(f"[Startup] Beginning startup for profile '{profile.name}' ({host}:{port}), manage_server={profile.manage_server}")

            if profile.manage_server:
                startup_map_value = None
                if editor._startup_requested and editor._startup_selected_map_package:
                    startup_map_value = editor._startup_selected_map_package

                if startup_map_value:
                    editor.loading_stage = (
                        f"Configuring CARLA startup map "
                        f"'{editor._startup_selected_map_name or startup_map_value}'..."
                    )
                    editor._backup_and_patch_default_engine_ini_for_startup(startup_map_value)

                editor.loading_stage = "Starting CARLA server..."
                editor.server_manager.set_port(port)
                try:
                    editor.server_manager.start_server()
                finally:
                    if startup_map_value:
                        editor._restore_default_engine_ini_backup()

                updated_port = editor.server_manager.port
                if updated_port != port:
                    port = updated_port
                    profile.port = updated_port
                    editor.local_port = updated_port
                    print(f"[Startup] Updated managed port to {updated_port}")

                editor.loading_stage = "Waiting for server to be ready..."
                if not editor.server_manager.wait_for_server():
                    raise Exception("CARLA server failed to start within timeout period")
                print("[Startup] Local CARLA server is ready.")
            else:
                editor.loading_stage = f"Waiting for remote server ({host}:{port})..."
                if not editor._wait_for_remote_server(host, port):
                    raise Exception("Remote CARLA server failed to respond within timeout period")
                print(f"[Startup] Remote CARLA endpoint at {host}:{port} confirmed reachable.")

            # Stage 3: Connect client
            editor.loading_stage = "Connecting to CARLA..."
            editor.client = carla.Client(host, port)
            editor.client.set_timeout(10.0)
            print("[Startup] CARLA client established.")

            requested_remote_map = None
            if profile.is_remote and editor.pending_map_switch:
                requested_remote_map = editor.pending_map_switch
                editor.pending_map_switch = None
            elif profile.is_remote and editor.pending_remote_map:
                requested_remote_map = editor.pending_remote_map
                editor.pending_remote_map = None

            # Stage 4: Get world
            editor.loading_stage = "Loading world..."
            editor.world = editor.client.get_world()
            # CARLA RPC can become ready slightly before the world/map finishes loading.
            # Waiting for at least one tick helps avoid rare CARLA Python API crashes on world.get_map().
            is_sync = False
            try:
                settings_probe = editor.world.get_settings()
                is_sync = bool(getattr(settings_probe, "synchronous_mode", False))
            except Exception:
                settings_probe = None

            ticked = False
            if hasattr(editor.world, "wait_for_tick"):
                editor.loading_stage = "Waiting for CARLA world to initialize..."
                tick_timeout = 30.0 if not is_sync else 2.0
                ticked = editor._wait_for_world_tick(
                    editor.world,
                    timeout=tick_timeout,
                    min_ticks=2,
                    label="startup",
                )

            if not ticked:
                if not is_sync and hasattr(editor.world, "wait_for_tick"):
                    raise RuntimeError("CARLA world did not start ticking after connect; map may not be ready.")
                # In synchronous mode we might not receive ticks yet. Give CARLA a moment to settle
                # before querying the map to reduce the risk of CARLA API crashes.
                if is_sync:
                    time.sleep(2.0)
            editor.world_map = editor.world.get_map()
            editor.cached_map = editor.world_map

            if requested_remote_map:
                requested_short = requested_remote_map.split('/')[-1]
                current_short = editor.world_map.name.split('/')[-1] if editor.world_map else ""
                if requested_short != current_short:
                    if not editor._load_remote_map(requested_remote_map):
                        return False
                else:
                    print(f"[Remote Map] '{requested_short}' already active on remote server.")

            print(f"Connected to CARLA world: {editor.world_map.name} ({host}:{port})")
            editor.large_map_active = is_large_map_name(getattr(editor.world_map, "name", None))
            if editor.large_map_active:
                print(f"[Startup] Large map detected; UI settle delay={editor.large_map_ui_delay:.1f}s")

            try:
                settings = editor.world.get_settings()
            except Exception as exc:
                settings = None
                print(f"Unable to retrieve world settings: {exc}")

            # Stage 5: Configure map settings where possible
            if profile.manage_server and settings is not None:
                editor.loading_stage = "Configuring map settings..."
                print("Using default world settings (no large-map streaming overrides).")
            elif profile.manage_server:
                editor.loading_stage = "Configuring map settings..."
                print("Skipping map configuration; unable to retrieve settings from local server.")
            else:
                editor.loading_stage = "Inspecting remote map..."
                if settings is not None:
                    print("Retrieved remote world settings")
                else:
                    print("Unable to inspect remote world settings.")

            editor.manual_tick_required = False
            editor.manual_tick_accumulator = 0.0
            editor.manual_tick_enabled = False
            editor.manual_tick_recommendation = False

            # Default to CARLA's no-rendering mode for higher editor FPS unless the user enables it.
            editor._apply_rendering_mode(
                desired_enabled=editor._rendering_desired_enabled,
                reason="initial configuration",
            )
            # Apply the remembered culling distance (or adopt the live value if unsafe).
            editor._apply_culling(reason="initial configuration")

            if profile.is_remote and settings is not None:
                if getattr(settings, 'synchronous_mode', False):
                    interval = settings.fixed_delta_seconds or 0.05
                    if interval <= 0:
                        interval = 0.05
                    editor.manual_tick_required = True
                    editor.manual_tick_interval = interval
                    editor.manual_tick_accumulator = 0.0
                    editor.manual_tick_enabled = False
                    editor.manual_tick_recommendation = True
                    wait_for_control = getattr(settings, 'synchronous_mode_wait_for_vehicle_control_command', False)
                    if wait_for_control:
                        print("[Startup] Remote server awaits vehicle control; enable 'Drive Clock' only if no other controller is active.")
                    else:
                        print("[Startup] Remote server in synchronous mode; use 'Drive Clock' toggle if frames do not stream.")
                else:
                    print("[Startup] Remote server is asynchronous; no manual ticking needed.")

            # Stage 6: Set up camera at world origin
            editor.loading_stage = "Setting up visual scenario editor..."

            # Start at world origin - safe for all maps including large maps
            center_x, center_y = 0, 0
            print("Starting at world origin (0, 0, 200)")

            # Set up camera system
            editor.camera_controller = TopDownCamera(center_x, center_y, 200)
            editor.camera_controller.world = editor.world  # Set world reference for debug raycast

            # Stage 7: Position spectator (only when VSE started a fresh local
            # server). When connecting to an already-running server (local reuse
            # or remote), leave the spectator wherever it is.
            started_fresh_server = (
                profile.manage_server and not editor.server_manager.use_existing_server
            )
            if started_fresh_server:
                spectator = editor.world.get_spectator()
                spectator_transform = carla.Transform(
                    carla.Location(x=10000, y=10000, z=-10000),
                    carla.Rotation(pitch=-90, yaw=0, roll=0)  # Looking down
                )
                spectator.set_transform(spectator_transform)
            else:
                print("[Startup] Connected to existing server; leaving spectator camera untouched.")

            # Stage 8: Now safe to spawn camera sensor and do raycasting
            editor.loading_stage = "Setting up camera sensor..."
            editor.camera_processor = CameraImageProcessor(
                editor.world,
                editor.camera_controller,
                editor.screen_width,
                editor.screen_height,
                editor,
                stream_resolution=editor.stream_resolution,
                stream_fps=editor.stream_fps if editor._using_remote_server() else 0,
            )
            if editor.vehicle_menu:
                editor.vehicle_menu.set_camera_processor(editor.camera_processor)
            if editor.pedestrian_menu:
                editor.pedestrian_menu.set_camera_processor(editor.camera_processor)
            if editor.ego_vehicle_menu:
                editor.ego_vehicle_menu.set_camera_processor(editor.camera_processor)
            if getattr(editor, "traffic_light_group_menu", None):
                editor.traffic_light_group_menu.set_camera_processor(editor.camera_processor)
            if not editor.camera_stream_enabled:
                editor._apply_stream_settings()
            if editor.cached_map is not None:
                editor.camera_processor.coordinate_detector.world_map = editor.cached_map

            # Pass server details to camera processor for map reload helpers
            editor.camera_processor._server_port = port
            editor.camera_processor._server_host = host
            print("[Startup] Camera processor initialized.")

            # Initialize info panel with camera processor reference
            editor.info_panel = InfoPanel(editor.camera_processor)

            # Initialize vehicle menu
            editor.vehicle_menu.initialize_vehicles(editor.world)
            editor.pedestrian_menu.initialize_pedestrians(editor.world)
            editor.ego_vehicle_menu.initialize_ego_vehicle(editor.world)
            print("[Startup] Actor menus initialized.")

            # Initialize map menu
            editor.map_menu = MapSelectionMenu(editor)
            editor.map_menu.initialize()
            print("[Startup] Map menu initialized.")

            # Initialize scenario menu
            editor.scenario_menu = ScenarioMenu(editor)

            # Precompute OpenDRIVE lane data on startup (even though overlay is disabled)
            # This prevents issues when loading scenarios before toggling the overlay
            print("Precomputing OpenDRIVE lane data...")
            editor.camera_processor.precompute_opendrive_lane_data()
            print("OpenDRIVE lane data ready")

            # Set initial minimum camera height based on terrain
            editor.camera_controller.update_min_height_from_terrain(editor.world, height_buffer=5.0)

            try:
                initial_weather = editor.world.get_weather()
            except Exception as exc:
                print(f"[Weather] Unable to fetch initial weather: {exc}")
            else:
                editor._update_weather_state(initial_weather)
                editor._capture_baseline_weather(initial_weather)
                if editor.weather_window and editor.weather_window.alive():
                    editor.weather_window.apply_weather(initial_weather)

            # Subscribe to world tick events for reset detection
            editor._register_world_tick_handler()

            pending_path = editor.pending_scenario_load
            if pending_path:
                if not os.path.isfile(pending_path):
                    print(f"[WARN] Pending scenario '{pending_path}' no longer exists; clearing handoff.")
                    editor.pending_scenario_load = None
                else:
                    editor.loading_stage = "Loading scenario..."
                    print(f"Loading pending scenario: {pending_path}")
                    editor.pending_scenario_load = None
                    try:
                        editor._load_scenario_from_path(pending_path, prompt_unsaved=False)
                    except Exception as exc:
                        print(f"[Startup] Failed to load pending scenario '{pending_path}': {exc}")
                        traceback.print_exc()
                    if editor.restart_in_progress:
                        return True

            editor.loading_stage = "Ready!"
            editor.ready = True
            editor.remote_connection_active = profile.is_remote
            if profile.is_remote:
                editor.active_remote_port = profile.port
                if not editor.active_remote_label:
                    editor.active_remote_label = f"Port {profile.port}"
            else:
                editor.active_remote_label = None
                editor.active_remote_port = None
            editor._notify_handoff_ready()

            print("Visual Scenario Editor ready!")
            print(f"Connected to CARLA on {host}:{port}")
            print(f"[Startup] Startup sequence finished. Remote mode={editor.remote_connection_active}")
            return True

        except Exception as e:
            print(f"Startup failed: {e}")
            traceback.print_exc()

            attempt += 1
            can_retry = profile.manage_server and attempt < max_attempts
            if not can_retry:
                editor.startup_error = str(e)
                editor.loading_stage = f"Error: {e}"
                return False

            print("[Startup] Attempting managed CARLA restart and retry (1/1)...")
            try:
                editor.server_manager.kill_existing_carla_processes()
            except Exception as kill_exc:
                print(f"[Startup] Warning: Failed to kill existing CARLA processes: {kill_exc}")

            editor.server_manager.use_existing_server = False
            editor.server_manager.process = None
            editor.server_manager.known_server_pid = None

            editor.world = None
            editor.world_map = None
            editor.client = None
            editor.camera_processor = None
            editor.cached_map = None

            try:
                new_port = editor.server_manager.find_available_port(editor.connection_profile.port)
                if new_port != editor.connection_profile.port:
                    print(f"[Startup] Switching to alternative port {new_port} for retry.")
                editor.connection_profile.port = new_port
                editor.local_port = new_port
                editor.server_manager.set_port(new_port)
            except Exception as port_exc:
                print(f"[Startup] Warning: unable to pick alternative port: {port_exc}")

            editor.ready = False
            editor.startup_error = None
            editor.loading_stage = "Retrying startup after restarting CARLA..."
            time.sleep(1.0)
            continue

    return False

def _handle_start_screen_click(editor, mouse_pos: Tuple[int, int]) -> bool:
    """Handle click interactions on the start screen."""
    if editor.start_screen_open_scenario_rect and editor.start_screen_open_scenario_rect.collidepoint(mouse_pos):
        editor._start_screen_open_scenario()
        return True

    if editor.start_screen_open_map_rect and editor.start_screen_open_map_rect.collidepoint(mouse_pos):
        editor._start_screen_open_map()
        return True

    if editor.start_screen_connect_remote_rect and editor.start_screen_connect_remote_rect.collidepoint(mouse_pos):
        editor.startup_error = None
        editor.connect_to_remote()
        return True

    for rect, path in editor.start_screen_recent_scenario_rects:
        if rect.collidepoint(mouse_pos):
            if path and os.path.isfile(path):
                editor._start_screen_launch_scenario(path)
            return True

    return False

def _start_screen_open_scenario(editor) -> None:
    """Prompt for a scenario file and begin startup."""
    editor.startup_error = None
    scenario_path = editor._prompt_scenario_file_path()
    if not scenario_path:
        return
    editor._start_screen_launch_scenario(scenario_path)

def _start_screen_open_map(editor) -> None:
    """Prompt for a map choice and begin startup."""
    editor.startup_error = None
    try:
        options = editor._list_local_startup_maps()
    except Exception as exc:
        editor.startup_error = str(exc)
        return

    if not options:
        editor.startup_error = "No CARLA maps found under CARLA_ROOT."
        return

    picked = editor._prompt_startup_map_choice(options)
    if not picked:
        return

    map_name = (picked.get("map_name") or picked.get("label") or "").strip()
    map_value = (picked.get("map_value") or "").strip()
    if not map_name or not map_value:
        editor.startup_error = "Invalid map selection."
        return

    editor.connection_profile = editor._create_local_profile()
    editor.server_manager.set_port(editor.connection_profile.port)
    editor._begin_startup_for_map(map_name, map_value)

def _start_screen_launch_scenario(editor, scenario_path: str) -> None:
    """Validate scenario, resolve map, and begin local startup."""
    if not scenario_path or not os.path.isfile(scenario_path):
        editor.startup_error = "Scenario file not found."
        return

    try:
        map_name = editor._extract_scenario_map_name(scenario_path)
    except Exception as exc:
        editor.startup_error = str(exc)
        return

    if not map_name:
        editor.startup_error = "Scenario is missing map_name (or map_name is Unknown)."
        return

    try:
        map_value = editor._resolve_map_package_path(map_name)
    except Exception as exc:
        editor.startup_error = str(exc)
        return
    if not map_value:
        editor.startup_error = f"Map '{map_name}' not found under CARLA_ROOT CarlaUE4/Content/Carla/Maps."
        return

    editor.connection_profile = editor._create_local_profile()
    editor.server_manager.set_port(editor.connection_profile.port)
    editor._begin_startup_for_scenario(scenario_path, map_name, map_value)
