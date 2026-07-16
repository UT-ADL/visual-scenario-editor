"""Self-relaunch + map/profile restart: subprocess + VSE_HANDOFF_FD env
handoff, remote connect, switch-to-local (duck-typed module functions, first
arg `editor`). Moved verbatim from vse.py (step-40, Phase 7).

The three `cmd = [sys.executable] + sys.argv` relaunch sites and the fd
open/close ordering are byte-identical; _perform_fast_map_reload keeps its
os._exit(0)-on-daemon-thread semantics.
"""

import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback

import carla
import pygame

from pygame_gui._constants import (
    UI_CONFIRMATION_DIALOG_CONFIRMED,
    UI_WINDOW_CLOSE,
    UI_BUTTON_PRESSED,
    UI_TEXT_ENTRY_FINISHED,
)
from pygame_gui.windows.ui_confirmation_dialog import UIConfirmationDialog

from vse_common.actor_cache import clear_bounding_box_cache
from vse_common.geometry import is_large_map as is_large_map_name
from vse_editor.carla_io.profiles import ConnectionProfile
from vse_editor.rendering.overlays import OpenDriveOverlayRenderer
from vse_editor.ui.dialogs import RemoteConnectionDialog


def connect_to_remote(editor):
    """Prompt for remote host/port and switch to it."""
    editor.map_menu_visible = False
    selection = editor._show_remote_connection_dialog()
    if not selection:
        print("Remote connection cancelled.")
        if editor.start_screen_active:
            editor.startup_error = None
        return False

    action = selection.get('action') or "cancel"
    host = (selection.get('host') or "").strip()
    port_raw = (selection.get('port') or "").strip()

    if action == "local":
        return editor._confirm_switch_to_local()

    if action != "ok":
        print("Remote connection cancelled.")
        if editor.start_screen_active:
            editor.startup_error = None
        return False

    editor._remember_last_remote(host, port_raw)

    if not host:
        print("[Remote] Host is required.")
        if editor.start_screen_active:
            editor.startup_error = "Remote host is required."
        return False
    try:
        port = int(port_raw)
    except ValueError:
        print("[Remote] Port must be a number.")
        if editor.start_screen_active:
            editor.startup_error = "Remote port must be a number."
        return False

    profile = editor._create_remote_profile(host, port)
    if (
        editor.connection_profile
        and editor.connection_profile.is_remote
        and editor.connection_profile.host == profile.host
        and editor.connection_profile.port == profile.port
    ):
        print(f"[Remote] Already connected to {host}:{port}.")
        if editor.start_screen_active:
            editor.startup_error = f"Already connected to {host}:{port}."
        return False

    if not editor._probe_remote_connection(profile.host, profile.port, max_attempts=3, timeout=3.0):
        print(f"[Remote] Cannot reach {profile.host}:{profile.port}. Leaving current session untouched.")
        if editor.start_screen_active:
            editor.startup_error = f"Cannot reach {profile.host}:{profile.port}."
        else:
            editor._show_error_overlay(f"Cannot connect to {profile.host}:{profile.port}")
        return False

    label = f"{host}:{port}"
    print(f"[Remote] Connecting to {label}...")
    editor.active_remote_label = label
    editor.active_remote_port = port
    editor.manual_tick_enabled = False
    editor.manual_tick_accumulator = 0.0
    editor.manual_tick_recommendation = False
    editor.resolution_menu_open = False
    editor.fps_menu_open = False
    started = editor._restart_with_profile(profile, label=label)
    if started and editor.start_screen_active:
        editor.start_screen_active = False
    return started

def _show_remote_connection_dialog(editor):
    """Prompt the user for remote host/port."""
    editor._close_all_dropdowns(keep_info_panel=True)
    default_host = getattr(editor, "last_remote_host", None) or ""
    default_port = getattr(editor, "last_remote_port", None) or ""
    if (not default_host or not default_port) and editor.connection_profile and editor.connection_profile.is_remote:
        default_host = default_host or editor.connection_profile.host
        default_port = default_port or str(editor.connection_profile.port)
    show_local_button = bool(editor.connection_profile and editor.connection_profile.is_remote)
    dialog_rect = editor._center_dialog_rect(420, 220)
    dialog = RemoteConnectionDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        default_host=default_host,
        default_port=default_port,
        show_local_button=show_local_button,
    )

    def handler(event: pygame.event.Event):
        if not dialog.alive():
            return True, dialog.get_values()
        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element in (dialog.ok_button, dialog.cancel_button) or (
                dialog.show_local_button and event.ui_element == dialog.local_button
            ):
                return True, dialog.get_values()
        if event.type == UI_TEXT_ENTRY_FINISHED and event.ui_element in (dialog.host_entry, dialog.port_entry):
            return True, dialog.get_values()
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, dialog.get_values()
        return False, None

    return editor._run_modal_window(dialog, handler)

def _confirm_switch_to_local(editor):
    if not editor.connection_profile or not editor.connection_profile.is_remote:
        print("Already connected to local CARLA server.")
        return False

    editor._close_all_dropdowns(keep_info_panel=True)
    editor.map_menu_visible = False
    dialog_rect = editor._center_dialog_rect(420, 220)
    dialog = UIConfirmationDialog(
        rect=dialog_rect,
        action_long_desc=(
            "Switch back to the local CARLA server?<br><br>"
            "This will terminate the remote session."
        ),
        manager=editor.ui_manager,
        window_title="Switch to Local CARLA",
        action_short_name="Switch",
        blocking=True,
    )

    def handler(event: pygame.event.Event):
        if event.type == UI_CONFIRMATION_DIALOG_CONFIRMED and event.ui_element == dialog:
            return True, True
        if event.type == UI_BUTTON_PRESSED and event.ui_element == dialog.confirm_button:
            return True, True
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, False
        return False, None

    confirmed = editor._run_modal_window(dialog, handler)
    if confirmed:
        editor.manual_tick_enabled = False
        editor.manual_tick_accumulator = 0.0
        editor.manual_tick_recommendation = False
        return editor.switch_to_local_server()
    return False

def _clone_profile(editor, profile):
    return ConnectionProfile(
        name=profile.name,
        host=profile.host,
        port=profile.port,
        manage_server=profile.manage_server,
        display_name=profile.display_name,
        description=profile.description,
        map_hint=profile.map_hint
    )

def request_remote_map_change(editor, map_name):
    if not editor.connection_profile or not editor.connection_profile.is_remote:
        print("Remote map change requested while not connected to a remote server.")
        return False
    if not map_name:
        return False

    target_short = map_name.split('/')[-1]
    current_short = editor._get_map_display_name()
    if current_short == target_short:
        print(f"Remote map '{target_short}' already active.")
        return True

    print(f"[Remote Map] Requesting '{map_name}' on remote server...")
    profile_copy = editor._clone_profile(editor.connection_profile)
    editor.manual_tick_enabled = False
    editor.manual_tick_accumulator = 0.0
    editor.manual_tick_recommendation = False
    return editor._restart_with_profile(profile_copy, label=editor.active_remote_label, target_map=map_name)

def _prepare_for_map_change(editor):
    """Stop conflicting systems before initiating a map change."""
    if editor.scenario_running:
        print("Stopping active scenario before map change...")
        editor.stop_scenario()

    if editor.camera_processor and editor.camera_processor.manual_control_enabled:
        print("Disabling manual control before map change...")
        editor.camera_processor.disable_manual_control()

    if editor.pending_scenario_load and not os.path.isfile(editor.pending_scenario_load):
        print(f"[WARN] Pending scenario '{editor.pending_scenario_load}' no longer exists; clearing handoff.")
        editor.pending_scenario_load = None

def _create_local_profile(editor):
    """Build a fresh local connection profile."""
    return ConnectionProfile(
        name="local",
        host='127.0.0.1',
        port=editor.local_port,
        manage_server=True,
        display_name="Local CARLA",
        description="Simulator started and managed by VSE",
        map_hint=None
    )

def _create_remote_profile(editor, host, port, label=None):
    """Build a connection profile for a remote server."""
    display_label = label or f"{host}:{port}"
    return ConnectionProfile(
        name=f"remote-{host}-{port}",
        host=host,
        port=port,
        manage_server=False,
        display_name=f"Remote {display_label}",
        description=f"Remote CARLA server at {host}:{port}",
        map_hint=editor.remote_map_hint
    )

def _restart_with_profile(editor, profile, *, label=None, target_map=None):
    """Restart the editor into a different connection profile (e.g., remote)."""
    if editor.restart_in_progress:
        print("Profile switch already in progress. Please wait.")
        return False

    editor._prepare_for_map_change()

    editor.pending_profile_switch = profile
    editor.pending_profile_switch_label = label
    editor.pending_map_switch = target_map
    editor.restart_in_progress = True
    editor.ready = False
    editor.startup_error = None
    editor.loading_stage = f"Switching to '{profile.display_name}'..."

    editor.restart_thread = threading.Thread(
        target=editor._perform_profile_restart,
        args=(profile, label, target_map),
        daemon=True
    )
    editor.restart_thread.start()
    return True

def _perform_profile_restart(editor, profile, label, target_map):
    """Launch a new VSE process configured for the requested profile."""
    try:
        editor.loading_stage = f"Preparing to connect to {profile.display_name}..."
        editor._cleanup_for_map_restart()

        profile_payload = {
            'name': profile.name,
            'host': profile.host,
            'port': profile.port,
            'manage_server': profile.manage_server,
            'display_name': profile.display_name,
            'description': profile.description,
            'map_hint': profile.map_hint,
        }
        if label:
            profile_payload['label'] = label
        if target_map:
            profile_payload['pending_map'] = target_map

        env = os.environ.copy()
        env['VSE_WINDOWED_WIDTH'] = str(int(editor.windowed_width))
        env['VSE_WINDOWED_HEIGHT'] = str(int(editor.windowed_height))
        env['VSE_SCREEN_WIDTH'] = str(int(editor.screen_width))
        env['VSE_SCREEN_HEIGHT'] = str(int(editor.screen_height))
        env['VSE_WINDOW_MAXIMIZED'] = '1' if editor.maximized else '0'
        env['VSE_PROFILE_JSON'] = json.dumps(profile_payload)

        env.pop('VSE_PENDING_MAP', None)
        if target_map:
            env['VSE_PENDING_MAP'] = target_map

        if editor.pending_scenario_load:
            env['VSE_PENDING_SCENARIO'] = editor.pending_scenario_load

        cmd = [sys.executable] + sys.argv
        handoff_read_fd, handoff_write_fd = os.pipe()
        env['VSE_HANDOFF_FD'] = str(handoff_write_fd)
        editor.handoff_read_fd = handoff_read_fd
        pass_fds = (handoff_write_fd,)

        try:
            editor.relaunch_process = subprocess.Popen(cmd, env=env, pass_fds=pass_fds)
        finally:
            try:
                os.close(handoff_write_fd)
            except OSError:
                pass

        editor._enter_restart_wait_state()

    except Exception as exc:
        editor._handle_profile_restart_failure(profile.display_name, exc)
    finally:
        editor.restart_in_progress = False
        editor.restart_thread = None

def _handle_profile_restart_failure(editor, profile_name, error):
    """Handle errors while attempting to restart into a different profile."""
    print(f"Failed to switch to profile '{profile_name}': {error}")
    traceback.print_exc()
    editor.startup_error = str(error)
    editor.loading_stage = f"Error: {error}"
    editor.ready = False
    editor.pending_exit = False
    editor.keep_server_running_on_exit = False
    editor.relaunch_process = None
    editor.wait_for_child = False
    if editor.handoff_read_fd is not None:
        try:
            os.close(editor.handoff_read_fd)
        except OSError:
            pass
        editor.handoff_read_fd = None
    editor.pending_profile_switch = None
    editor.pending_profile_switch_label = None
    editor.pending_map_switch = None
    editor._reset_restart_flags()

def switch_to_local_server(editor):
    """Return to the locally managed CARLA server."""
    print("[Connection] Switching back to local CARLA server...")
    local_profile = editor._create_local_profile()
    started = editor._restart_with_profile(local_profile, label="Local CARLA")
    if not started:
        print("[Connection] Unable to initiate restart into local server.")
    return started

def _enter_restart_wait_state(editor):
    """Put the current process into a waiting state while the new instance starts."""
    editor.loading_stage = "Completing map change..."
    editor.restart_start_time = time.time()
    editor.pending_exit = True
    editor.keep_server_running_on_exit = True
    editor.server_manager.set_auto_stop_enabled(False)
    editor.wait_for_child = True

def _reset_restart_flags(editor):
    """Clear restart-related flags and ensure server auto-stop is restored."""
    editor.restart_in_progress = False
    editor.restart_thread = None
    editor.restart_start_time = None
    editor.server_manager.set_auto_stop_enabled(True)

def _load_remote_map(editor, map_name):
    """Request the remote CARLA server to load a specific map."""
    if not editor.client:
        print('[Remote Map] No client available for remote map load.')
        return False

    try:
        print(f"[Remote Map] Loading '{map_name}' on remote server...")
        editor.loading_stage = f"Loading remote map '{map_name}'..."
        editor.client.load_world(map_name, map_layers=carla.MapLayer.NONE)
        clear_bounding_box_cache()  # actor ids restart with the new episode

        editor.loading_stage = "Waiting for remote map..."
        editor.world = editor.client.get_world()
        # This is a VSE-initiated, in-process reload (no process reboot), so re-baseline
        # world.id here; otherwise the new world.id would look like an external reset.
        try:
            editor._expected_world_id = editor.world.id
        except Exception:
            editor._expected_world_id = None
        editor._latest_world_snapshot = None
        editor._last_seen_actor_count = None
        is_sync = False
        try:
            settings_probe = editor.world.get_settings()
            is_sync = bool(getattr(settings_probe, "synchronous_mode", False))
        except Exception:
            settings_probe = None

        ticked = False
        if hasattr(editor.world, "wait_for_tick"):
            tick_timeout = 60.0 if not is_sync else 2.0
            ticked = editor._wait_for_world_tick(
                editor.world,
                timeout=tick_timeout,
                min_ticks=2,
                label="remote-map-load",
            )

        if not ticked:
            if not is_sync and hasattr(editor.world, "wait_for_tick"):
                raise RuntimeError("Remote world did not start ticking after map load; map may not be ready.")
            if is_sync:
                time.sleep(2.0)

        editor.world_map = editor.world.get_map()
        editor.cached_map = editor.world_map

        print(f"[Remote Map] Remote map ready: {editor.world_map.name}")
        return True
    except Exception as exc:
        print(f"[Remote Map] Failed to load '{map_name}': {exc}")
        editor.startup_error = str(exc)
        editor.loading_stage = f"Error: {exc}"
        return False

def _handle_map_restart_failure(editor, map_name, error):
    """Handle errors during the map restart process."""
    print(f"Failed to change map '{map_name}': {error}")
    traceback.print_exc()
    editor.startup_error = str(error)
    editor.loading_stage = f"Error: {error}"
    editor.ready = False
    editor.pending_exit = False
    editor.keep_server_running_on_exit = False
    editor.relaunch_process = None
    editor.wait_for_child = False
    if editor.handoff_read_fd is not None:
        try:
            os.close(editor.handoff_read_fd)
        except OSError:
            pass
        editor.handoff_read_fd = None
    editor._reset_restart_flags()

def load_map(editor, map_name):
    """Load a different CARLA map while keeping the current window visible."""
    # Large maps are streamed from a root level; avoid loading individual tiles directly.
    requested_name = (map_name or "").strip()
    if requested_name:
        last_segment = requested_name.split('/')[-1]
        if "_Tile_" in last_segment:
            base = last_segment.split("_Tile_")[0].strip()
            if base:
                if "/" in requested_name:
                    parts = requested_name.split("/")
                    parts[-1] = base
                    requested_name = "/".join(parts)
                else:
                    requested_name = base
                map_name = requested_name

    if editor.connection_profile and editor.connection_profile.is_remote:
        return editor.request_remote_map_change(map_name)

    if not editor.client:
        print("CARLA client not initialized yet; cannot change map.")
        return False

    if editor.restart_in_progress:
        print("Map change already in progress. Please wait for it to finish.")
        return False

    # Check if OpenDRIVE overlay is enabled - disable automatically before changing maps
    if editor.camera_processor:
        OpenDriveOverlayRenderer.disable_overlay(editor.camera_processor, silent=True)

    current_map = editor._get_map_display_name()
    # Extract short name from map_name for comparison (e.g., "/Game/Carla/Maps/tartu_large/tartu_large" -> "tartu_large")
    requested_map_short = map_name.split('/')[-1]
    if current_map == requested_map_short:
        print(f"Map '{requested_map_short}' is already loaded.")
        return True

    editor._prepare_for_map_change()

    print(f"Changing map: {current_map} → {map_name}")

    editor.restart_in_progress = True
    editor.ready = False
    editor.startup_error = None
    editor.loading_stage = f"Changing map to '{map_name}'..."

    editor.restart_thread = threading.Thread(
        target=editor._perform_map_restart,
        args=(map_name,),
        daemon=True
    )
    editor.restart_thread.start()
    return True

def _cleanup_for_map_restart(editor):
    """Release editor resources prior to launching the new process."""
    print("[Cleanup] Releasing editor resources before reconnect/restart...")

    if editor.camera_processor:
        print("[Cleanup] Disabling overlays and destroying camera processor...")
        OpenDriveOverlayRenderer.disable_overlay(editor.camera_processor, silent=True)
        if hasattr(editor.camera_processor, 'opendrive_lane_data'):
            editor.camera_processor.opendrive_lane_data = None
        if hasattr(editor.camera_processor, 'opendrive_segment_grid'):
            editor.camera_processor.opendrive_segment_grid = None
        if hasattr(editor.camera_processor, 'opendrive_segment_bboxes'):
            editor.camera_processor.opendrive_segment_bboxes = None
        editor.camera_processor.cleanup_all_vehicles()
        editor.camera_processor.cleanup()
        editor.camera_processor = None
        if editor.vehicle_menu:
            editor.vehicle_menu.set_camera_processor(None)
        if editor.pedestrian_menu:
            editor.pedestrian_menu.set_camera_processor(None)
        if editor.ego_vehicle_menu:
            editor.ego_vehicle_menu.set_camera_processor(None)

    if editor.info_panel:
        editor.info_panel.hide()
        editor.info_panel = None

    print("[Cleanup] Clearing world/client references and command stacks...")
    editor.camera_controller = None
    editor._no_camera_saved_pose = None
    editor.history.clear()
    editor._scene_dirty_hint = False
    editor._saved_scene_signature = None
    editor._saved_disk_signature = None
    editor.keys_pressed.clear()
    editor.key_hold_times.clear()
    editor.selected_vehicle_type = None
    editor.selected_pedestrian_type = None
    editor.selected_ego_vehicle_type = None

    editor.world_map = None
    editor.world = None
    editor.client = None
    editor.manual_tick_required = False
    editor.manual_tick_accumulator = 0.0
    editor.manual_tick_enabled = False
    editor.manual_tick_recommendation = False
    editor.manual_tick_button_rect = None
    editor.pending_map_switch = None
    editor.pending_remote_map = None

    gc.collect()
    time.sleep(1.0)
    print("[Cleanup] Resource cleanup complete.")

def _notify_handoff_ready(editor):
    """Notify the previous VSE instance that takeover is complete."""
    if editor.handoff_write_fd is not None:
        try:
            os.write(editor.handoff_write_fd, b'1')
        except OSError:
            pass
        try:
            os.close(editor.handoff_write_fd)
        except OSError:
            pass
        editor.handoff_write_fd = None

def _perform_map_restart(editor, map_name):
    """Background sequence that prepares the server and launches a fresh editor."""
    try:
        editor.loading_stage = "Loading map..."
        editor._cleanup_for_map_restart()

        # Large maps can't be loaded via load_world (times out on tiled maps).
        # Kill CARLA and restart with INI patching so the target map is the startup map.
        target_short = map_name.split('/')[-1] if map_name else ""
        if is_large_map_name(target_short):
            # Resolve short name to full package path for INI patching
            full_path = editor._resolve_map_package_path(map_name)
            reload_name = full_path or map_name
            print(f"[Map Switch] Target '{target_short}' is a large map; using full server restart ({reload_name}).")
            editor._perform_fast_map_reload(reload_name)
            return

        editor.loading_stage = f"Loading {map_name}..."
        temp_client = carla.Client('127.0.0.1', editor.server_manager.port)
        temp_client.set_timeout(30.0)
        # Force async mode before load_world -- the large-map skip doesn't
        # apply when we're about to destroy the entire world. Verify the
        # apply actually took (read back + one retry): a world left in sync
        # mode with nobody ticking can stall the load_world below.
        try:
            pre_world = temp_client.get_world()
            is_sync = bool(getattr(pre_world.get_settings(), 'synchronous_mode', False))
            for _ in range(2):
                if not is_sync:
                    break
                pre_settings = pre_world.get_settings()
                pre_settings.synchronous_mode = False
                pre_settings.fixed_delta_seconds = 0.0
                pre_world.apply_settings(pre_settings)
                is_sync = bool(getattr(pre_world.get_settings(), 'synchronous_mode', False))
            if is_sync:
                print("[Map Switch] WARNING: world is still in synchronous mode after two "
                      "attempts to disable it; load_world may stall.")
        except Exception as exc:
            print(f"[Map Switch] WARNING: failed to force async mode before load_world ({exc}); "
                  "proceeding with the map load.")
        temp_client.load_world(map_name, map_layers=carla.MapLayer.NONE)
        clear_bounding_box_cache()  # actor ids restart with the new episode

        editor.loading_stage = "Finalizing map load..."
        world = None
        try:
            world = temp_client.get_world()
        except Exception:
            world = None

        is_sync = False
        if world is not None:
            try:
                settings_probe = world.get_settings()
                is_sync = bool(getattr(settings_probe, "synchronous_mode", False))
            except Exception:
                settings_probe = None

        ticked = False
        if world is not None and hasattr(world, "wait_for_tick"):
            tick_timeout = 60.0 if not is_sync else 2.0
            ticked = editor._wait_for_world_tick(
                world,
                timeout=tick_timeout,
                min_ticks=3,
                label="map-restart",
            )

        if not ticked:
            if not is_sync and world is not None and hasattr(world, "wait_for_tick"):
                raise RuntimeError("CARLA world did not start ticking after load_world; map may not be ready.")
            time.sleep(2.0)

        editor.loading_stage = "Reloading editor..."
        cmd = [sys.executable] + sys.argv
        env = os.environ.copy()
        known_pid = editor.server_manager.get_known_server_pid()
        if known_pid:
            env['VSE_SERVER_PID'] = str(known_pid)
        elif 'VSE_SERVER_PID' in env:
            del env['VSE_SERVER_PID']

        env['VSE_WINDOWED_WIDTH'] = str(int(editor.windowed_width))
        env['VSE_WINDOWED_HEIGHT'] = str(int(editor.windowed_height))
        env['VSE_SCREEN_WIDTH'] = str(int(editor.screen_width))
        env['VSE_SCREEN_HEIGHT'] = str(int(editor.screen_height))
        env['VSE_WINDOW_MAXIMIZED'] = '1' if editor.maximized else '0'

        if editor.pending_scenario_load:
            env['VSE_PENDING_SCENARIO'] = editor.pending_scenario_load
            print(f"Passing pending scenario through environment: {editor.pending_scenario_load}")

        handoff_read_fd, handoff_write_fd = os.pipe()
        env['VSE_HANDOFF_FD'] = str(handoff_write_fd)
        editor.handoff_read_fd = handoff_read_fd
        pass_fds = (handoff_write_fd,)
        try:
            editor.relaunch_process = subprocess.Popen(cmd, env=env, pass_fds=pass_fds)
        except Exception as launch_error:
            raise RuntimeError(f"Failed to reload editor: {launch_error}")
        finally:
            try:
                os.close(handoff_write_fd)
            except OSError:
                pass

        editor._enter_restart_wait_state()

    except Exception as e:
        editor._handle_map_restart_failure(map_name, e)
    finally:
        editor.restart_in_progress = False
        editor.restart_thread = None

def _force_reload_current_map_for_external_ego(editor):
    """Force reload current map without cleanup - used when external ego removed on large map."""
    if editor.restart_in_progress:
        return False

    # If the external ego vanished because the SERVER died (VIL + server
    # crash), an automatic relaunch would come up against the dead server,
    # re-trigger this path in the child and chain relaunches (observed live:
    # three stacked VSE processes). Route to the server-lost overlay instead —
    # the user restarts deliberately from there. awmini exiting cleanly on a
    # live server still reloads as before.
    from vse_editor.app.world_lifecycle import _server_confirmed_dead
    if editor._server_lost or _server_confirmed_dead(editor):
        if not editor._server_lost:
            editor._server_lost = True
            print("[Server Watchdog] CARLA server unreachable")
        print("[External Ego] Server is down — skipping auto map reload "
              "(use the overlay's Restart button)")
        return False

    # Save scenario for reload
    if editor.current_scenario_path:
        editor.pending_scenario_load = editor.current_scenario_path

    # Get current map name
    current_map = None
    if editor.world_map:
        current_map = editor.world_map.name
    if not current_map and editor.cached_map:
        current_map = editor.cached_map.name

    if not current_map:
        print("[External Ego] Cannot determine current map for reload")
        return False

    print(f"[External Ego] Force reloading map '{current_map}' to clear state...")

    editor.restart_in_progress = True
    editor.ready = False
    editor.loading_stage = "Reloading map after external ego removal..."

    # Run in thread
    editor.restart_thread = threading.Thread(
        target=editor._perform_fast_map_reload,
        args=(current_map,),
        daemon=True
    )
    editor.restart_thread.start()
    return True

def _restart_after_server_crash(editor):
    """'Restart server & reload scene' from the server-lost overlay (fix-05).

    Reloads the scenario FILE the user had open (no scratch/recovery file — a
    mystery file confuses people). Unsaved edits are handled with the usual
    Save / Don't Save / Cancel prompt, saving to the user's own file (Save As
    if it was never saved). Then reuse the fast-reload flow: kill every CARLA
    process (same method as startup), relaunch vse.py as a child exactly like
    a map change, boot a fresh server and auto-load the file.
    """
    if editor.restart_in_progress:
        return False

    # Ask about unsaved changes BEFORE committing to the restart (so restart_
    # in_progress isn't set yet — Cancel leaves the overlay button clickable).
    # During playback the scene placeholders are gone and saving is blocked, so
    # skip the prompt and just reload the last-saved file.
    playback_active = bool(editor.scenario_running
                           or getattr(editor, '_mini_runner', None))
    if not playback_active:
        choice = editor.show_exit_confirmation(
            window_title="Restart Required",
            action_long_desc=(
                "The CARLA server was lost and VSE needs to restart.<br><br>"
                "Save your changes first? They are reloaded after the restart. "
                "Unsaved changes are lost if you don't save."
            ),
        )
        if choice == "cancel":
            return False  # stay on the overlay
        if choice == "save":
            # Saves to the current file, or opens Save As if never saved.
            # A cancelled Save As aborts the restart.
            if not editor._save_current_scenario():
                return False

    # Commit: the overlay button now greys to "Restarting...".
    editor.restart_in_progress = True
    editor.loading_stage = "Restarting after server crash..."

    # Reload the file we had open. None (never saved + declined Save As) ->
    # restart into an empty scene on the same map.
    pending = editor.current_scenario_path or None
    if pending:
        editor.pending_scenario_load = pending

    # Remote profile: the editor cannot reboot a remote machine — instead
    # restart VSE against the same remote profile (the existing profile-switch
    # relaunch), which reconnects and reloads the pending scenario once the
    # remote server is back.
    if editor._using_remote_server():
        editor.restart_in_progress = False  # _restart_with_profile re-arms it
        print("[Server Watchdog] restarting VSE against remote profile "
              f"'{getattr(editor.connection_profile, 'display_name', 'remote')}' ...")
        return editor._restart_with_profile(editor.connection_profile)

    current_map = None
    if editor.world_map:
        current_map = editor.world_map.name
    if not current_map and editor.cached_map:
        current_map = editor.cached_map.name
    if not current_map:
        print("[Server Watchdog] cannot determine current map for restart")
        editor.restart_in_progress = False
        editor.loading_stage = ""
        return False

    print(f"[Server Watchdog] restarting CARLA + editor on '{current_map}' "
          f"(reload: {pending or 'empty scene'}) ...")
    editor.ready = False

    editor.restart_thread = threading.Thread(
        target=editor._perform_fast_map_reload,
        args=(current_map,),
        daemon=True,
    )
    editor.restart_thread.start()
    return True

def _perform_fast_map_reload(editor, map_name):
    """Kill CARLA server and spawn fresh VSE - used when CARLA is frozen."""
    try:
        # Unregister CARLA tick callbacks BEFORE killing server (prevents crash in old process)
        editor._unregister_world_tick_handler()
        if editor.camera_processor:
            editor.camera_processor._manual_control_unregister_tick_callback()

        # Just clear references, don't try to destroy (CARLA is frozen)
        if editor.camera_processor:
            editor.camera_processor.spawned_vehicles = []
            editor.camera_processor = None

        editor.external_ego_actor = None
        editor.external_ego_actor_id = None
        editor.world = None
        editor.client = None

        # Kill ALL CARLA processes - there may be zombie processes
        editor.loading_stage = "Killing ALL CARLA processes..."
        print("[External Ego] Killing ALL CARLA processes...")
        editor.server_manager.kill_existing_carla_processes()

        # Spawn new editor process - it will start fresh CARLA
        editor.loading_stage = "Spawning new editor..."
        cmd = [sys.executable] + sys.argv
        env = os.environ.copy()

        # Don't pass server PID - we killed it, new VSE should start fresh
        env.pop('VSE_SERVER_PID', None)

        env['VSE_WINDOWED_WIDTH'] = str(int(editor.windowed_width))
        env['VSE_WINDOWED_HEIGHT'] = str(int(editor.windowed_height))
        env['VSE_SCREEN_WIDTH'] = str(int(editor.screen_width))
        env['VSE_SCREEN_HEIGHT'] = str(int(editor.screen_height))
        env['VSE_WINDOW_MAXIMIZED'] = '1' if editor.maximized else '0'

        if editor.pending_scenario_load:
            env['VSE_PENDING_SCENARIO'] = editor.pending_scenario_load
            print(f"[External Ego] Passing pending scenario: {editor.pending_scenario_load}")

        # Pass map package so new VSE can do INI patching before starting CARLA
        if map_name:
            startup_map = map_name if map_name.startswith('/Game/') else f'/Game/{map_name}'
            env['VSE_STARTUP_MAP_PACKAGE'] = startup_map
            print(f"[External Ego] Passing startup map: {startup_map}")

        handoff_read_fd, handoff_write_fd = os.pipe()
        env['VSE_HANDOFF_FD'] = str(handoff_write_fd)
        editor.handoff_read_fd = handoff_read_fd
        pass_fds = (handoff_write_fd,)
        try:
            editor.relaunch_process = subprocess.Popen(cmd, env=env, pass_fds=pass_fds)
        except Exception as launch_error:
            raise RuntimeError(f"Failed to reload editor: {launch_error}")
        finally:
            try:
                os.close(handoff_write_fd)
            except OSError:
                pass

        # Hard exit immediately - don't wait for child
        # The normal wait state causes CARLA C++ threads to crash when the connection dies
        # os._exit() skips all Python cleanup and C++ destructors that would timeout
        print("[External Ego] New editor spawned - exiting old process immediately")
        os._exit(0)

    except Exception as e:
        print(f"[External Ego] Fast map reload failed: {e}")
        traceback.print_exc()
        editor.startup_error = str(e)
        editor.loading_stage = f"Error: {e}"
    finally:
        editor.restart_in_progress = False
        editor.restart_thread = None
