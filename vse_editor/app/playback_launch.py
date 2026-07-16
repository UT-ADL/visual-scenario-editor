"""Playback launch: Play/Stop, preserve/restore around a run, result windows
(duck-typed module functions, first arg `editor`). Moved verbatim from vse.py
(step-38, Phase 7).

The lazy `from vse_play import MiniRunner` import inside run_scenario stays
function-local together with the VSE_PLAY_INSTALL_HANDLERS env dance; the
_on_finish closure runs on the MiniRunner thread and re-enters the editor
exactly as before. _restore_once_lock stays on the editor instance.
"""

import faulthandler
import json
import locale
import math
import os
import subprocess
import traceback
import warnings

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import carla
import pygame

from pygame_gui._constants import (
    UI_FILE_DIALOG_PATH_PICKED,
    UI_WINDOW_CLOSE,
    UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION,
    UI_BUTTON_PRESSED,
)

from vse_editor.rendering.overlays import OpenDriveOverlayRenderer
from vse_editor.scene_types import clone_waypoint_sequence
from vse_editor.ui.dialogs import EnhancedFileDialog, ResultWindow


def _close_result_window(editor):
    """Kill any open result window and clear pending state."""
    if editor.result_window:
        try:
            editor.result_window.kill()
        except Exception:
            pass
    editor.result_window = None
    editor._pending_result_dialog = None

def _enqueue_result_dialog(editor, reason: str):
    """Prepare result text to be shown once the scenario stops."""
    text = ""
    result_path: Optional[str] = None
    try:
        if editor.current_scenario_path:
            result_path = str(Path(editor.current_scenario_path).with_suffix(".txt"))
            try:
                with open(result_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception as exc:
                print(f"[Results] Unable to read result file: {exc}")
    except Exception as exc:
        print(f"[Results] Failed to resolve result file path: {exc}")

    if not text:
        text = f"Scenario finished ({reason})"

    editor._pending_result_dialog = {
        "text": text,
        "path": result_path,
        "reason": reason,
    }

def _copy_result_text(editor, text: str):
    if not text:
        return
    copied = False
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*scrap.*deprecated.*",
                category=DeprecationWarning,
            )
            if not pygame.scrap.get_init():
                pygame.scrap.init()
            pygame.scrap.put_text(text)
        copied = True
    except Exception as exc:
        print(f"[Results] Clipboard copy via pygame failed: {exc}")

    if not copied:
        try:
            import tkinter  # type: ignore

            root = tkinter.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
            copied = True
        except Exception as exc:
            print(f"[Results] Clipboard copy via tkinter failed: {exc}")

    if not copied:
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            copied = True
        except Exception as exc:
            print(f"[Results] Clipboard copy fallback failed: {exc}")

    if copied:
        print("[Results] Copied scenario results to clipboard.")

def _save_result_text(editor, text: str, default_path: Optional[str]):
    if not text:
        print("[Results] Nothing to save.")
        return
    if not editor.ui_manager:
        print("[Results] UI not ready; cannot open save dialog.")
        return

    initial_path = default_path
    if not initial_path:
        if editor.current_scenario_path:
            initial_path = str(Path(editor.current_scenario_path).with_suffix(".txt"))
        else:
            initial_path = os.getcwd()
    if os.path.isdir(initial_path):
        initial_path = os.path.join(initial_path, "results.txt")

    dialog_rect = editor._center_dialog_rect(640, 480)
    dialog = EnhancedFileDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        window_title="Save Results",
        allowed_suffixes={".txt", ".TXT"},
        initial_file_path=initial_path,
        allow_existing_files_only=False,
        always_on_top=True,
    )
    dialog.set_blocking(True)
    confirm_button = getattr(dialog, "ok_button", getattr(dialog, "confirm_button", None))
    if confirm_button:
        confirm_button.set_text("Save")
    file_entry = getattr(dialog, "file_path_text_line", None)
    if file_entry and initial_path:
        file_entry.set_text(initial_path)
        filename = os.path.basename(initial_path)
        name_part = filename[:-4] if filename.lower().endswith(".txt") else filename
        start_index = len(initial_path) - len(filename)
        end_index = start_index + len(name_part)
        file_entry.select_range = [start_index, end_index]
        file_entry.cursor_has_moved_recently = True
        file_entry.edit_position = end_index
        try:
            file_entry.focus()
        except AttributeError:
            pass

    def handler(event: pygame.event.Event):
        if event.type == UI_FILE_DIALOG_PATH_PICKED and event.ui_element == dialog:
            picked = getattr(event, "text", None)
            if picked and not os.path.isdir(picked):
                return True, picked
            return False, None
        if event.type == pygame.USEREVENT:
            picked_path = getattr(event, "text", None)
            ui_element = getattr(event, "ui_element", None)
            if picked_path and ui_element == dialog and not os.path.isdir(picked_path):
                return True, picked_path
        if event.type == UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION and event.ui_element == getattr(dialog, "file_selection_list", None):
            if Path(dialog.current_directory_path).name == event.text and Path(dialog.current_directory_path).is_dir():
                return False, None
            path = dialog.current_file_path
            if path and not os.path.isdir(path):
                return True, str(path)
            return False, None
        if (
            event.type == UI_BUTTON_PRESSED
            and hasattr(dialog, "new_folder_button")
            and event.ui_element == dialog.new_folder_button
        ):
            folder_name = editor._prompt_text_input("Create Folder", "Folder name:", "New Folder")
            if folder_name:
                base_dir = Path(dialog.current_directory_path)
                new_path = base_dir / folder_name
                try:
                    new_path.mkdir()
                except FileExistsError:
                    print(f"[File Dialog] Folder '{folder_name}' already exists.")
                except Exception as exc:
                    print(f"[File Dialog] Unable to create folder '{folder_name}': {exc}")
                else:
                    dialog._change_directory_path(new_path)
            return False, None
        if (
            event.type == UI_BUTTON_PRESSED
            and confirm_button is not None
            and event.ui_element == confirm_button
        ):
            path = dialog.current_file_path
            if path and not os.path.isdir(path):
                return True, str(path)
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, None
        return False, None

    file_path = editor._run_modal_window(dialog, handler)
    if not file_path:
        return

    file_path = os.path.abspath(file_path)
    if not file_path.lower().endswith(".txt"):
        file_path += ".txt"

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[Results] Saved results to {file_path}")
    except Exception as exc:
        print(f"[Results] Failed to save results: {exc}")

_RESULT_WRAP_MIN_COLS: int = 80  # chars; width floor so degenerate texts don't wrap to slivers

# The result writer puts each of these headers on its own line and joins EVERY actor in that
# group onto the single following line — the only unbounded lines in the result text.
# "> Other actors:" is the stock scenario_runner header, kept for result files predating the
# grouped "NPC vehicles" / "Pedestrians" output (_GroupedResultOutput in vse_playback/runner.py).
_RESULT_ACTOR_LIST_HEADERS: Tuple[str, ...] = (
    "> Ego vehicles:",
    "> NPC vehicles:",
    "> Pedestrians:",
    "> Other actors:",
)


def _wrap_result_text_for_display(text: str) -> str:
    """Reflow the unbounded actor-list lines so the window width is set by the criteria table.

    Display-only: Copy/Save and the <scenario>.txt on disk keep the original single-line text.
    The text is rendered monospace, so character counts are a faithful width proxy.
    """
    lines = text.split("\n")
    wrappable = set()
    for i in range(len(lines) - 1):
        if lines[i].strip() in _RESULT_ACTOR_LIST_HEADERS and lines[i + 1].strip():
            wrappable.add(i + 1)
    if not wrappable:
        return text
    target = max(
        [len(l) for i, l in enumerate(lines) if i not in wrappable] + [_RESULT_WRAP_MIN_COLS]
    )
    out: List[str] = []
    for i, line in enumerate(lines):
        if i not in wrappable or len(line) <= target:
            out.append(line)
            continue
        # Greedily repack "Actor(...); " chunks; an oversized chunk stays whole on its own line.
        current = ""
        for chunk in (c + "; " for c in line.split("; ") if c.strip()):
            if current and len(current) + len(chunk) > target:
                out.append(current.rstrip())
                current = chunk
            else:
                current += chunk
        if current:
            out.append(current.rstrip())
    return "\n".join(out)


def _maybe_open_result_window(editor):
    """Show the result window if a scenario just finished."""
    if editor.result_window and not editor.result_window.alive():
        editor.result_window = None

    scenario_active = bool(editor.scenario_running or (editor.scenario_process and editor.scenario_process.poll() is None))
    if scenario_active or not editor.ready:
        return
    if not editor._pending_result_dialog or not editor.ui_manager:
        return

    payload = editor._pending_result_dialog
    editor._pending_result_dialog = None

    text = str(payload.get("text", "") or "")
    default_path = payload.get("path")

    # The result dialog is set_blocking(True), so once it opens it swallows all input. It must
    # therefore be visible. If the run was stopped from Chase/Cockpit (which auto-hides the UI),
    # hide_all_ui is True and the per-frame ui_manager.draw_ui() is skipped -> the dialog would
    # be invisible AND eat every keystroke (including ` to un-hide), freezing the app. Force the
    # UI back on so the dialog is shown and closable, matching a Top-Down (Stop button) stop.
    editor.hide_all_ui = False

    rect_width = max(640, int(editor.screen_width * 0.6))
    rect_height = max(420, int(editor.screen_height * 0.6))
    dialog_rect = editor._center_dialog_rect(rect_width, rect_height)

    editor.result_window = ResultWindow(
        rect=dialog_rect,
        manager=editor.ui_manager,
        text=_wrap_result_text_for_display(text),
        on_copy=lambda: editor._copy_result_text(text),
        on_save=lambda: editor._save_result_text(text, default_path if isinstance(default_path, str) else None),
        on_close=editor._close_result_window,
    )

def _reap_finished_runner(editor):
    """Clear the MiniRunner handle once its thread (and ROS-agent subprocess) is fully gone.

    on_finish fires from MiniRunner._cleanup *before* _stop_ros_agent_process(), so the
    runner thread is still alive (and the subprocess still shutting down) when the results
    appear. We keep self._mini_runner set until that thread dies — which only happens after
    _cleanup completes — so the Play button stays disabled ("Finishing…") until a new run is
    safe to start. This is the moment the subprocess is guaranteed dead.
    """
    runner = getattr(editor, "_mini_runner", None)
    if not runner or editor.scenario_running:
        return
    thread = getattr(runner, "_thread", None)
    if not thread or not thread.is_alive():
        editor._mini_runner = None

def _capture_playback_preserved_state(
    editor,
) -> Tuple[Optional[Dict[str, object]], Optional[int]]:
    """Capture selection/waypoint state before playback and stash it on the editor."""
    preserved_display_id = None
    preserved_waypoints = None
    preserved_signature = None
    preserved_blueprint = None
    preserved_location = None
    preserved_is_pedestrian = False
    panel_visible = False
    preserved_traffic_light_ids: Optional[Tuple[int, ...]] = None
    traffic_light_panel_visible = False
    ego_override_id = None

    cp = editor.camera_processor
    actor_ref = None
    if cp:
        candidate_id = None
        if cp.selected_vehicle and cp.selected_vehicle.is_alive:
            candidate_id = cp.selected_vehicle.id
            actor_ref = cp.selected_vehicle
        elif cp.waypoint_display_vehicle_id:
            candidate_id = cp.waypoint_display_vehicle_id

        if candidate_id is not None:
            preserved_display_id = candidate_id

            waypoints = cp.get_vehicle_waypoints(candidate_id)
            if waypoints:
                preserved_waypoints = clone_waypoint_sequence(waypoints)
                preserved_signature = editor._compute_waypoint_signature(preserved_waypoints)

            if actor_ref is None:
                actor_ref = next(
                    (
                        actor
                        for actor in cp.spawned_vehicles
                        if actor and actor.is_alive and actor.id == candidate_id
                    ),
                    None,
                )

            if actor_ref:
                preserved_blueprint = actor_ref.type_id
                preserved_is_pedestrian = actor_ref.type_id.startswith('walker.')
                try:
                    transform = actor_ref.get_transform()
                    preserved_location = {
                        'x': transform.location.x,
                        'y': transform.location.y,
                        'z': transform.location.z,
                        'yaw': transform.rotation.yaw,
                    }
                except Exception:
                    preserved_location = None

        panel = getattr(editor, "info_panel", None)
        if (
            panel
            and panel.visible
            and panel.object_type in ('vehicle', 'pedestrian')
            and actor_ref
            and getattr(panel, "selected_object", None) is actor_ref
        ):
            panel_visible = True

        selected_group = cp.selected_traffic_light_group
        if selected_group:
            preserved_traffic_light_ids = tuple(sorted(selected_group.ids))
            tl_panel = getattr(editor, "info_panel", None)
            if (
                tl_panel
                and tl_panel.visible
                and tl_panel.object_type == 'traffic_light'
                and getattr(tl_panel, "selected_object", None) is selected_group
            ):
                traffic_light_panel_visible = True

        if candidate_id is not None and cp.is_ego_vehicle(candidate_id):
            ego_override_id = candidate_id

    active_trigger_snapshot: Optional[Dict[str, object]] = None

    editor._scenario_preserved_waypoint_vehicle_id = preserved_display_id
    editor._scenario_preserved_waypoints = preserved_waypoints
    editor._scenario_preserved_waypoint_signature = preserved_signature
    editor._scenario_preserved_actor_blueprint = preserved_blueprint
    editor._scenario_preserved_actor_location = preserved_location
    editor._scenario_preserved_actor_is_pedestrian = preserved_is_pedestrian
    editor._scenario_preserved_info_panel_visible = panel_visible
    editor._scenario_preserved_traffic_light_ids = preserved_traffic_light_ids
    editor._scenario_preserved_traffic_light_info_visible = traffic_light_panel_visible
    if preserved_traffic_light_ids and cp:
        group = cp.selected_traffic_light_group
        if group and group.has_trigger() and tuple(sorted(group.ids)) == preserved_traffic_light_ids:
            center = group.trigger_center or {}
            radius = group.trigger_radius
            if center and radius is not None:
                active_trigger_snapshot = {
                    'ids_live': list(preserved_traffic_light_ids),
                    'ids_reference': sorted(group.reference_ids or group.ids),
                    'center': {
                        'x': float(center.get('x', 0.0)),
                        'y': float(center.get('y', 0.0)),
                        'z': float(center.get('z', 0.0)),
                    },
                    'radius': float(radius),
                }

    # Preserve overlay visibility state
    preserved_traffic_lights_visible = False
    preserved_lane_overlay_enabled = False
    if cp:
        preserved_traffic_lights_visible = getattr(cp, 'traffic_lights_visible', False)
        preserved_lane_overlay_enabled = getattr(cp, 'lane_overlay_enabled', False)

    editor._scenario_preserved_traffic_lights_visible = preserved_traffic_lights_visible
    editor._scenario_preserved_lane_overlay_enabled = preserved_lane_overlay_enabled
    print(f"[OVERLAY PRESERVE] Traffic={preserved_traffic_lights_visible}, OpenDRIVE={preserved_lane_overlay_enabled}")

    return active_trigger_snapshot, ego_override_id

def _detect_scenario_ego_flags(editor) -> Tuple[bool, bool]:
    """Return (scenario_has_ego, ego_has_route) based on current scenario data."""
    scenario_has_ego = False
    ego_has_route = False
    try:
        scenario_data = getattr(editor.camera_processor, "loaded_scenario_data", None)
        if not scenario_data and editor.current_scenario_path and os.path.isfile(editor.current_scenario_path):
            with open(editor.current_scenario_path, "r", encoding="utf-8") as fh:
                scenario_data = json.load(fh)
        if scenario_data:
            ego_entries = []
            ego_entry = scenario_data.get("ego_vehicle")
            if isinstance(ego_entry, dict):
                ego_entries = [ego_entry]
            elif isinstance(ego_entry, list):
                ego_entries = [entry for entry in ego_entry if isinstance(entry, dict)]
            if not ego_entries:
                for entry in scenario_data.get("vehicles", []):
                    if str(entry.get("role", "")).lower() == "ego_vehicle":
                        ego_entries = [entry]
                        break
            if ego_entries:
                scenario_has_ego = True
                for entry in ego_entries:
                    waypoints = entry.get("waypoints", [])
                    if isinstance(waypoints, list) and waypoints:
                        ego_has_route = True
                        break
    except Exception as exc:
        print(f"[Scenario] Unable to determine ego presence: {exc}")
        ego_has_route = False
    return scenario_has_ego, ego_has_route

def _prepare_external_ego_for_playback(editor, scenario_has_ego: bool) -> Tuple[bool, bool]:
    """Detect and configure external ego handling for playback."""
    external_ego_identified = False
    external_actor = None
    external_ego_present = False
    if scenario_has_ego:
        external_actor = editor._resolve_external_ego_actor(silent=False, force_scan=True)
        external_ego_present = bool(external_actor)
        if external_actor:
            external_ego_identified = True
            try:
                role_name = external_actor.attributes.get('role_name', '<unknown>')
            except Exception:
                role_name = '<unknown>'
            try:
                editor._external_swap_pre_run_transform = external_actor.get_transform()
            except Exception:
                editor._external_swap_pre_run_transform = None
            print(
                f"Detected external ego vehicle {external_actor.id} "
                f"with role '{role_name}'; scenario will attach to it."
            )
            try:
                # Honor the Ego Physics toggle: don't force physics on an externally-controlled
                # (e.g. awmini VIL) ego, which disables physics and drives by teleport.
                external_actor.set_simulate_physics(editor.ego_physics_enabled)
            except Exception:
                pass
        else:
            editor._switch_world_to_async_if_safe(reason="scenario launch without external ego")
    else:
        # Ignore any lingering external ego when the scenario has no ego entry.
        editor.external_ego_actor = None
        editor.external_ego_actor_id = None
        editor._switch_world_to_async_if_safe(reason="scenario launch without ego vehicle")
    return external_ego_present, external_ego_identified

def _restore_preserved_waypoints_for_playback(editor, ego_override_id: Optional[int]) -> None:
    if (
        editor.camera_processor
        and editor._scenario_preserved_waypoint_vehicle_id is not None
        and editor._scenario_preserved_waypoints
    ):
        cp = editor.camera_processor
        if ego_override_id is not None:
            cp._waypoint_ego_override_id = ego_override_id
        cp.waypoint_list[editor._scenario_preserved_waypoint_vehicle_id] = clone_waypoint_sequence(
            editor._scenario_preserved_waypoints
        )
        cp.waypoint_display_vehicle_id = editor._scenario_preserved_waypoint_vehicle_id
        cp.selected_vehicle = None
        cp.selected_vehicle_is_pedestrian = False
        cp.vehicle_menu_position = None
        cp.selected_waypoint_vehicle_id = None
        cp.selected_waypoint_index = None

        info_panel = getattr(editor, "info_panel", None)
        if info_panel:
            markers = getattr(info_panel, "scenario_starting_points", {}) or {}
            marker_key = editor._scenario_preserved_waypoint_vehicle_id
            if marker_key not in markers and editor._scenario_preserved_actor_location:
                loc = editor._scenario_preserved_actor_location
                markers = dict(markers)
                markers[marker_key] = {
                    "x": loc.get("x", 0.0),
                    "y": loc.get("y", 0.0),
                    "z": loc.get("z", 0.0),
                    "vehicle_id": marker_key,
                }
                info_panel.scenario_starting_points = markers

        cp.refresh_waypoints_carla_debug()

def _apply_playback_ui_state(
    editor,
    *,
    active_trigger_snapshot: Optional[Dict[str, object]],
    scenario_has_ego: bool,
    external_ego_present: bool,
    ego_has_route: bool,
) -> None:
    """Update UI/selection state at scenario start."""
    # Update state to indicate scenario is running
    editor.scenario_running = True
    # Clear contextual menus while Scenario Runner takes control
    if editor.camera_processor:
        selected_display_id = editor._scenario_preserved_waypoint_vehicle_id

        editor.camera_processor.clear_vehicle_selection(
            keep_waypoints=selected_display_id is not None,
            hide_info_panel=True,
            clear_traffic_light=False,
        )
        editor.camera_processor.selected_trigger_index = None
        editor.camera_processor.trigger_action_menu_position = None
        if active_trigger_snapshot:
            editor._scenario_active_traffic_light_trigger = active_trigger_snapshot
            editor.camera_processor._scenario_active_traffic_light_trigger = active_trigger_snapshot
        if selected_display_id is not None:
            editor.camera_processor.waypoint_display_vehicle_id = selected_display_id
            editor.camera_processor.refresh_waypoints_carla_debug()
        else:
            editor.camera_processor.waypoint_display_vehicle_id = None
    if hasattr(editor, 'info_panel') and editor.info_panel:
        editor.info_panel.hide()
    # Always enable manual control for local ego (camera follows regardless of route)
    should_enable_manual_control = scenario_has_ego and not external_ego_present
    starting_ego_camera = False
    if editor.camera_processor and should_enable_manual_control:
        editor.camera_processor.request_manual_control()
        editor.camera_processor.playback_camera_mode = editor.play_camera_mode
        starting_ego_camera = True
    elif editor.camera_processor and scenario_has_ego and external_ego_present:
        editor.camera_processor.request_playback_camera_follow()
        editor.camera_processor.playback_camera_mode = editor.play_camera_mode
        starting_ego_camera = True
    if starting_ego_camera:
        # Auto-hide the overlays for Chase/Cockpit; keep them for Top-Down. The `~` key
        # toggles `hide_all_ui` manually afterwards. See _set_play_camera_mode().
        editor.hide_all_ui = (editor.play_camera_mode != "topdown")

    # Hide overlays during playback for a clean viewing experience
    if editor.camera_processor:
        # Hide traffic light overlay
        if editor.camera_processor.traffic_lights_visible:
            editor.camera_processor.traffic_lights_visible = False
            editor.camera_processor.clear_traffic_light_selection()
            if (editor.camera_processor.selected_personal_trigger
                    and editor.camera_processor.selected_personal_trigger.get('kind') == 'traffic_light'):
                editor.camera_processor.clear_personal_trigger_selection()
            editor.camera_processor._last_visible_traffic_light_trigger_key = None

        # Hide OpenDRIVE overlay
        if editor.camera_processor.lane_overlay_enabled:
            OpenDriveOverlayRenderer.disable_overlay(editor.camera_processor, silent=True)

def run_scenario(editor):
    """Run the currently loaded scenario"""
    if not editor.current_scenario_path or not os.path.isfile(editor.current_scenario_path):
        print("Scenario has not been saved yet. Please choose a file to save before playback.")
        editor.save_scenario_with_dialog()
        if not editor.current_scenario_path or not os.path.isfile(editor.current_scenario_path):
            print("Scenario run cancelled: scenario must be saved before playback.")
            return

    if editor.scenario_running:
        editor.stop_scenario()
        return

    debug_enabled = bool(getattr(editor, "debug_mode", False))
    if debug_enabled:
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass

    def _play_diag(message: str) -> None:
        if debug_enabled:
            print(f"[PlayDiag] {message}")

    _play_diag("run_scenario: enter")

    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
    except Exception as exc:
        _play_diag(f"Locale warning: failed to force LC_NUMERIC=C before playback: {exc}")

    # Set flag to prevent bootstrap mode during scenario launch
    editor._scenario_run_in_progress = True

    try:
        editor._restore_invoked_for_run = False
        editor._close_result_window()
        if editor._has_unsaved_changes():
            choice = editor._confirm_play_discard_changes()
            if choice == "cancel":
                print("Scenario run cancelled by user.")
                return

            if choice == "save":
                if not editor._save_current_scenario():
                    print("Scenario run cancelled: save was cancelled or failed.")
                    return
                # On-disk now matches in-memory edits; no discard needed.
            else:
                # "discard" -- reload the last-saved version from disk.
                if not editor.camera_processor:
                    print("Scenario run cancelled: camera processor not initialized.")
                    return

                if not os.path.isfile(editor.current_scenario_path):
                    print("Scenario run cancelled: scenario file missing. Save before playback.")
                    return

                # Preserve vehicle control mode across play/stop (reload from JSON would overwrite it)
                editor._preserved_vehicle_control_mode = editor.vehicle_control_mode

                try:
                    editor._discard_unsaved_changes_for_play()
                    print(f"[Scenario] Discarded unsaved changes; running '{editor.current_scenario_path}' from disk.")
                except Exception as exc:
                    print(f"Scenario run cancelled: failed to discard unsaved changes ({exc})")
                    return

        editor.scenario_stop_requested = False
        # Playback drives the camera through top-down fields (follow writes center/height
        # directly), so a 3D orbit view must exit before the first camera touch below.
        # Placed AFTER the unsaved-changes dialog: a cancelled Play must not yank the
        # user out of the 3D view.
        if (editor.camera_controller
                and getattr(editor.camera_controller, "view_mode", "topdown") == "orbit"):
            editor.camera_controller.exit_orbit()
            if editor.camera_processor:
                editor.camera_processor.update_camera_position()
            print("Camera: top-down view (playback)")
        if editor.camera_processor:
            editor.camera_processor.disable_manual_control()
            editor._debug_camera_pose("pre-run focus")
            if editor._auto_camera_allowed():
                editor._focus_camera_on_ego_vehicle()
        editor._reset_scenario_waypoint_preserve()
        editor._scenario_pending_actor_reselect = False

        active_trigger_snapshot, ego_override_id = editor._capture_playback_preserved_state()

        scenario_has_ego, ego_has_route = editor._detect_scenario_ego_flags()

        external_ego_present, external_ego_identified = editor._prepare_external_ego_for_playback(scenario_has_ego)

        # Arm manual-control arrow routing as early as the run mode is known, before the slow
        # startup work below (vehicle relocation, marker recording). This keeps arrow keys
        # reserved for the ego instead of leaking to the camera during the startup window.
        if editor.camera_processor:
            editor.camera_processor.manual_control_armed = scenario_has_ego and not external_ego_present
            if editor.camera_processor.manual_control_armed:
                for _arrow in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    editor.keys_pressed.discard(_arrow)

        # Record start markers before relocating vehicles so UI can display placeholders
        editor._record_scenario_start_markers()

        # Save and move ALL vehicles in the scene to avoid conflicts
        editor._save_all_scene_vehicles()
        editor._restore_preserved_waypoints_for_playback(ego_override_id)

        if scenario_has_ego:
            if (
                external_ego_present
                and not external_ego_identified
            ):
                try:
                    role_name = editor.external_ego_actor.attributes.get('role_name', '<unknown>')
                except Exception:
                    role_name = '<unknown>'
                print(
                    f"Detected external ego vehicle {editor.external_ego_actor.id} "
                    f"with role '{role_name}'; scenario will attach to it."
                )
                external_ego_identified = True

        # Validate scenario JSON
        if not os.path.isfile(editor.current_scenario_path):
            print(f"Error: Scenario JSON '{editor.current_scenario_path}' not found. Save your scenario before running.")
            editor._restore_all_scene_vehicles_once()
            return
        else:
            print(f"[Scenario] Using existing scenario data from {editor.current_scenario_path}")

        # If a runner is already active, treat Play as stop
        if getattr(editor, "_mini_runner", None) and editor._mini_runner.is_running:
            print("[MiniRunner] Stop requested by user.")
            editor.scenario_stop_requested = True
            editor._mini_runner.request_stop()
            return

        env_timeout = os.environ.get("VSE_SCENARIO_RUNNER_TIMEOUT")
        try:
            timeout_seconds = float(env_timeout) if env_timeout else 18000.0
        except (TypeError, ValueError):
            timeout_seconds = 18000.0
        timeout_seconds = max(timeout_seconds, 30.0)

        tick_mode = "ros" if (scenario_has_ego and external_ego_present) else "own"
        wait_for_ego = tick_mode == "ros"
        agent_path = None
        if tick_mode == "ros":
            candidate = getattr(editor, "agent_path", None)
            if candidate and not os.path.isfile(candidate):
                editor._clear_last_agent_cache(remove_file=True)
                candidate = None
            if not candidate:
                candidate = editor._prompt_agent_file_path()
                if not candidate:
                    print("[Agent] Scenario run cancelled: external ego playback requires an agent file.")
                    editor._restore_all_scene_vehicles_once()
                    return
                editor.agent_mode = "custom"
                editor._remember_last_agent(candidate)
            agent_path = candidate
        fixed_delta = 0.05
        try:
            settings = editor.world.get_settings() if editor.world else None
            if settings and settings.fixed_delta_seconds:
                fixed_delta = float(settings.fixed_delta_seconds)
        except Exception:
            fixed_delta = 0.05

        # Preserve current world weather so we can restore it after playback.
        try:
            editor._scenario_restore_weather = editor.world.get_weather() if editor.world else None
        except Exception:
            editor._scenario_restore_weather = None

        print(f"Running scenario via MiniRunner (mode={tick_mode}, agent={getattr(editor, 'agent_mode', 'autopilot')}, timeout={timeout_seconds}s)")
        _play_diag("run_scenario: printed runner start banner")

        editor._apply_playback_ui_state(
            active_trigger_snapshot=active_trigger_snapshot,
            scenario_has_ego=scenario_has_ego,
            external_ego_present=external_ego_present,
            ego_has_route=ego_has_route,
        )

        def _log_fn(message: str):
            print(f"[MiniRunner] {message}")

        def _on_finish(reason: str):
            editor._debug_camera_pose("pre-restore-finally")
            editor._restore_all_scene_vehicles_once()
            if editor.camera_processor:
                editor.camera_processor.disable_manual_control()
            editor._restore_start_weather_from_presets()
            editor.scenario_running = False
            editor.scenario_stop_requested = False
            # NOTE: do NOT clear editor._mini_runner here. on_finish fires from
            # MiniRunner._cleanup *before* _stop_ros_agent_process() tears down the
            # ROS-agent subprocess, and the runner thread is still alive at this point.
            # Keep the handle so the Play button stays disabled ("Finishing…") until the
            # thread (and thus the subprocess) is fully torn down; _reap_finished_runner()
            # clears it once editor._mini_runner._thread is no longer alive.
            editor._debug_camera_pose("post-run restore")
            if editor.camera_processor:
                try:
                    editor._debug_camera_pose("post-run restore after disable")
                except Exception:
                    pass
            print(f"[MiniRunner] Scenario finished ({reason})")
            try:
                editor._enqueue_result_dialog(reason or "")
            except Exception as exc:
                print(f"[Results] Failed to prepare result dialog: {exc}")

        try:
            _play_diag("run_scenario: importing MiniRunner")
            prev_install_handlers = os.environ.get("VSE_PLAY_INSTALL_HANDLERS")
            os.environ["VSE_PLAY_INSTALL_HANDLERS"] = "0"
            from vse_play import MiniRunner
            _play_diag("run_scenario: MiniRunner import OK")
        except Exception as import_err:
            print(f"Error importing MiniRunner: {import_err}")
            editor._restore_all_scene_vehicles_once()
            # Undo the Chase/Cockpit auto-hide from _apply_playback_ui_state: the run
            # never started, so no result dialog will force the UI back on.
            editor.hide_all_ui = False
            editor.scenario_running = False
            return
        finally:
            if 'prev_install_handlers' in locals():
                if prev_install_handlers is None:
                    os.environ.pop("VSE_PLAY_INSTALL_HANDLERS", None)
                else:
                    os.environ["VSE_PLAY_INSTALL_HANDLERS"] = prev_install_handlers

        external_ego_id = None
        try:
            if scenario_has_ego and editor.external_ego_actor and editor.external_ego_actor.is_alive:
                external_ego_id = editor.external_ego_actor.id
        except Exception:
            external_ego_id = None

        # Disable bootstrap mode if it was active from preview
        if getattr(editor, '_large_map_bootstrap_ticking', False):
            editor._set_large_map_bootstrap_ticking(
                False,
                log_message="[Scenario Launch] Disabling bootstrap mode for scenario playback",
                disable_manual_tick=True,
                clear_pending_external_ego=True,
            )

        ego_agent_mode = getattr(editor, "agent_mode", "autopilot")
        ego_agent_behavior = getattr(editor, "agent_behavior", "normal")
        editor._mini_runner = MiniRunner(
            client=editor.client,
            world=editor.world,
            json_path=editor.current_scenario_path,
            tick_mode=tick_mode,
            fixed_delta=fixed_delta,
            wait_for_ego=wait_for_ego,
            ego_role_name="ego_vehicle",
            timeout_s=timeout_seconds,
            external_ego_actor_id=external_ego_id,
            log_fn=_log_fn,
            debug=getattr(editor, "debug_mode", False),
            on_finish=_on_finish,
            agent_path=agent_path,
            agent_mode=ego_agent_mode,
            agent_behavior=ego_agent_behavior,
            vehicle_control_mode=editor.vehicle_control_mode,
            disable_ego_collision=not editor.ego_collision_enabled,
            disable_ego_physics=not editor.ego_physics_enabled,
            ignore_actor_ids=getattr(editor, "_playback_removed_preview_ids", None),
        )
        _play_diag("run_scenario: MiniRunner init OK")
        _play_diag("run_scenario: MiniRunner start()")
        editor._mini_runner.start()
        _play_diag("run_scenario: MiniRunner start() returned")

        runner_thread = getattr(editor._mini_runner, "_thread", None)
        if not runner_thread or not runner_thread.is_alive():
            print("[MiniRunner] Start failed; restoring scene.")
            editor._restore_all_scene_vehicles_once()
            if editor.camera_processor:
                editor.camera_processor.disable_manual_control()
            editor._restore_start_weather_from_presets()
            # Undo the Chase/Cockpit auto-hide from _apply_playback_ui_state: the run
            # never started, so no result dialog will force the UI back on.
            editor.hide_all_ui = False
            editor.scenario_running = False
            editor.scenario_stop_requested = False
            editor._mini_runner = None
            return

    except Exception as e:
        print(f"Error starting scenario: {e}")
        editor._restore_all_scene_vehicles_once()
        if editor.camera_processor:
            editor.camera_processor.disable_manual_control()
        # Undo the Chase/Cockpit auto-hide (see failed-start path above).
        editor.hide_all_ui = False
        editor.scenario_running = False
    finally:
        # Always clear the flag when exiting run_scenario (success, error, or early return)
        editor._scenario_run_in_progress = False

def _discard_unsaved_changes_for_play(editor):
    """Drop in-memory edits before playback without reloading the preview."""
    cp = editor.camera_processor
    external_id = getattr(editor, "external_ego_actor_id", None)
    preserve_ids = {external_id} if external_id is not None else set()

    if cp:
        try:
            cp.cleanup_all_vehicles(preserve_ids=preserve_ids, preserve_ego=True)
        except Exception:
            pass

        if hasattr(cp, "spawned_vehicles"):
            cp.spawned_vehicles.clear()
        if hasattr(cp, "waypoint_list"):
            cp.waypoint_list.clear()
        if hasattr(cp, "triggers"):
            cp.triggers.clear()
        for attr in (
            "traffic_light_trigger_centers",
            "traffic_light_trigger_radii",
            "traffic_light_sequences",
            "_traffic_light_group_snapshots",
        ):
            if hasattr(cp, attr):
                getattr(cp, attr).clear()
        if hasattr(cp, "_scenario_active_traffic_light_trigger"):
            cp._scenario_active_traffic_light_trigger = None
        if hasattr(cp, "_last_visible_traffic_light_trigger_key"):
            cp._last_visible_traffic_light_trigger_key = None
        if hasattr(cp, "scaling_traffic_light_trigger"):
            cp.scaling_traffic_light_trigger = False
        if hasattr(cp, "_traffic_light_scaling_group"):
            cp._traffic_light_scaling_group = None
        if hasattr(cp, "traffic_light_menu_position"):
            cp.traffic_light_menu_position = None
        if hasattr(cp, "traffic_light_groups"):
            try:
                for group in cp.traffic_light_groups:
                    group.trigger_center = None
                    group.trigger_radius = None
            except Exception:
                pass
        if hasattr(cp, "clear_traffic_light_selection"):
            try:
                cp.clear_traffic_light_selection()
            except Exception:
                pass

        # Reset selection and ego metadata
        for attr in (
            "selected_vehicle",
            "vehicle_menu_position",
            "selected_waypoint_vehicle_id",
            "selected_waypoint_index",
            "waypoint_display_vehicle_id",
            "selected_trigger_index",
        ):
            if hasattr(cp, attr):
                setattr(cp, attr, None)
        if hasattr(cp, "selected_vehicle_is_pedestrian"):
            cp.selected_vehicle_is_pedestrian = False
        if hasattr(cp, "moving_waypoint"):
            cp.moving_waypoint = False
        if hasattr(cp, "ego_vehicle_id"):
            cp.ego_vehicle_id = None
        if hasattr(cp, "ego_vehicle_transform"):
            cp.ego_vehicle_transform = None
        if hasattr(cp, "ego_vehicle_blueprint"):
            cp.ego_vehicle_blueprint = None
        if hasattr(cp, "ego_vehicle_color"):
            cp.ego_vehicle_color = None

        if hasattr(cp, "loaded_scenario_data"):
            cp.loaded_scenario_data = None

        if hasattr(cp, "disable_manual_control"):
            try:
                cp.disable_manual_control()
            except Exception:
                pass

    # Reset editor/UI state
    editor._reset_scenario_waypoint_preserve()
    editor._clear_scenario_start_markers()
    if hasattr(editor, "info_panel") and editor.info_panel:
        try:
            editor.info_panel.hide()
        except Exception:
            pass

    editor.history.clear()

    editor.saved_scene_vehicles = []
    editor.scene_preview_destroyed = True

    disk_sig = editor._compute_disk_scene_signature()
    editor._saved_scene_signature = disk_sig
    editor._saved_disk_signature = disk_sig
    editor._scene_dirty_hint = False

def stop_scenario(editor):
    """Stop the running scenario and restore original state"""
    try:
        print("Stopping scenario...")
        editor.scenario_stop_requested = True

        mini_runner_active = False
        # Stop in-process MiniRunner if active
        mini_runner = getattr(editor, "_mini_runner", None)
        if mini_runner and getattr(mini_runner, "is_running", False):
            print("[MiniRunner] Stop requested by user.")
            try:
                mini_runner.request_stop()
            except Exception:
                pass

            thread = getattr(mini_runner, "_thread", None)
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=5.0)
                except Exception:
                    pass
            mini_runner_active = getattr(mini_runner, "is_running", False)
            if not mini_runner_active:
                editor._mini_runner = None
            else:
                print("[MiniRunner] Runner is still stopping in background.")
                # Avoid forcing cleanup from the UI thread while the runner thread is
                # still active; CARLA API calls (apply_settings, map queries) are not
                # thread-safe and can segfault during Play/Stop races.
                mini_runner_active = True
        else:
            mini_runner_active = False

        # Terminate the scenario process if it exists
        if editor.scenario_process and editor.scenario_process.poll() is None:
            editor.scenario_process.terminate()
            print("Scenario process terminated")

            # Wait a bit for graceful termination, then force kill if needed
            try:
                editor.scenario_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                editor.scenario_process.kill()
                print("Scenario process force killed")

        # Restore all scene vehicles (MiniRunner will handle restore via on_finish)
        if not mini_runner_active:
            editor._restore_all_scene_vehicles_once()
            if editor.camera_processor:
                editor.camera_processor.disable_manual_control()
            editor._restore_start_weather_from_presets()

        # Clear scenario state
        editor.scenario_process = None
        editor.scenario_running = bool(mini_runner_active and getattr(mini_runner, "is_running", False))
        if editor.scenario_running:
            print("Scenario stop requested (MiniRunner still shutting down).")
        else:
            print("Scenario stopped")

    except Exception as e:
        print(f"Error stopping scenario: {e}")
        # Still try to restore
        editor._restore_all_scene_vehicles_once()
        if editor.camera_processor:
            editor.camera_processor.disable_manual_control()
        runner = getattr(editor, "_mini_runner", None)
        editor.scenario_running = bool(runner and getattr(runner, "is_running", False))
        editor.scenario_process = None

def _record_scenario_start_markers(editor):
    """Capture marker positions for spawned scenario actors before relocation."""
    info_panel = getattr(editor, "info_panel", None)
    if not info_panel:
        return

    markers = {}
    for actor in getattr(editor, "spawned_vehicles", []):
        if not actor or not actor.is_alive:
            continue
        try:
            loc = actor.get_location()
        except RuntimeError:
            continue
        markers[actor.id] = {
            "x": loc.x,
            "y": loc.y,
            "z": loc.z,
            "vehicle_id": actor.id,
        }

    info_panel.scenario_starting_points = markers
    if markers:
        first_marker = next(iter(markers.values()))
        info_panel.scenario_starting_point = dict(first_marker)
    else:
        info_panel.scenario_starting_point = None

def _clear_scenario_start_markers(editor):
    """Remove any stored scenario start markers from the UI."""
    info_panel = getattr(editor, "info_panel", None)
    if not info_panel:
        return
    info_panel.scenario_starting_points = {}
    info_panel.scenario_starting_point = None

def _reset_scenario_traffic_light_preserve(editor):
    """Clear any cached traffic light selection data preserved during ScenarioRunner playback."""
    editor._scenario_preserved_traffic_light_ids = None
    editor._scenario_preserved_traffic_light_info_visible = False
    editor._scenario_active_traffic_light_trigger = None
    if editor.camera_processor:
        editor.camera_processor._scenario_active_traffic_light_trigger = None

def _reset_scenario_waypoint_preserve(editor, *, include_traffic: bool = True):
    """Clear any cached waypoint data preserved during ScenarioRunner playback."""
    editor._scenario_preserved_waypoint_vehicle_id = None
    editor._scenario_preserved_waypoints = None
    editor._scenario_preserved_actor_blueprint = None
    editor._scenario_preserved_actor_location = None
    editor._scenario_preserved_actor_is_pedestrian = False
    editor._scenario_preserved_info_panel_visible = False
    editor._scenario_preserved_waypoint_signature = None
    editor._scenario_preserved_traffic_lights_visible = False
    editor._scenario_preserved_lane_overlay_enabled = False
    if include_traffic:
        editor._reset_scenario_traffic_light_preserve()
    cp = getattr(editor, "camera_processor", None)
    if cp and hasattr(cp, "_waypoint_ego_override_id"):
        cp._waypoint_ego_override_id = None

def _restore_preserved_traffic_light_selection(editor) -> bool:
    """Reapply traffic light selection after ScenarioRunner playback."""
    cp = editor.camera_processor
    if not cp:
        return False

    ids = editor._scenario_preserved_traffic_light_ids
    if not ids:
        editor._reset_scenario_traffic_light_preserve()
        return False

    ids_set = set(ids)
    group = cp.selected_traffic_light_group

    if not group or group.ids != ids_set:
        group = next((g for g in cp.traffic_light_groups if g.ids == ids_set), None)
        if group is None:
            try:
                cp._refresh_traffic_lights()
            except Exception:
                pass
            group = next((g for g in cp.traffic_light_groups if g.ids == ids_set), None)
            if group is None:
                return False
        cp.select_traffic_light_group(group)

    cp._update_traffic_light_menu_anchor(group)

    if editor._scenario_preserved_traffic_light_info_visible:
        panel = getattr(editor, "info_panel", None)
        if panel:
            panel.show(group, 'traffic_light', editor.screen_width, editor.screen_height)

    # Move the camera to the traffic light focus point, mirroring vehicle selection behavior.
    focus_location: Optional[carla.Location] = None

    def _valid(loc: Optional[carla.Location]) -> bool:
        if loc is None:
            return False
        try:
            if not (math.isfinite(loc.x) and math.isfinite(loc.y) and math.isfinite(loc.z)):
                return False
            if abs(loc.x) + abs(loc.y) + abs(loc.z) < 0.5:
                return False
        except Exception:
            return False
        return True

    # Prefer trigger center, then cached centroid, then recomputed trigger center.
    center = getattr(group, "trigger_center", None) or {}
    try:
        loc = carla.Location(
            float(center.get("x", 0.0)),
            float(center.get("y", 0.0)),
            float(center.get("z", 0.0)),
        )
        if _valid(loc):
            focus_location = loc
    except Exception:
        focus_location = None

    if focus_location is None:
        centroid = getattr(group, "center_location", None)
        if centroid and len(centroid) == 3:
            try:
                loc = carla.Location(float(centroid[0]), float(centroid[1]), float(centroid[2]))
                if _valid(loc):
                    focus_location = loc
            except Exception:
                focus_location = None

    if focus_location is None:
        try:
            recomputed = cp._compute_traffic_light_group_trigger_center(group)
            if _valid(recomputed):
                focus_location = recomputed
        except Exception:
            pass

    if focus_location and editor._auto_camera_allowed():
        try:
            cp.focus_camera_on_location(focus_location)
        except Exception:
            pass

    editor._reset_scenario_traffic_light_preserve()
    return True

def _compute_waypoint_signature(editor, waypoints: Optional[List[dict]]) -> Optional[Tuple]:
    """Create a coarse signature for a waypoint list for matching after reload."""
    if not waypoints:
        return None
    length = len(waypoints)
    first = waypoints[0]
    last = waypoints[-1]
    def _rounded(point, key):
        return round(float(point.get(key, 0.0)), 1)
    return (
        length,
        _rounded(first, 'x'), _rounded(first, 'y'), _rounded(first, 'z'),
        _rounded(last, 'x'), _rounded(last, 'y'), _rounded(last, 'z'),
    )

def _restore_preserved_overlay_visibility(editor) -> None:
    """Restore traffic light and OpenDRIVE overlay visibility after playback."""
    cp = editor.camera_processor
    if not cp:
        print("[OVERLAY RESTORE] No camera processor")
        return

    # Restore traffic lights visibility
    traffic_was_visible = editor._scenario_preserved_traffic_lights_visible
    print(f"[OVERLAY RESTORE] Traffic: was_visible={traffic_was_visible}, current={cp.traffic_lights_visible}")
    if traffic_was_visible and not cp.traffic_lights_visible:
        cp.traffic_lights_visible = True
        print("Traffic light overlay restored")
    elif traffic_was_visible and cp.traffic_lights_visible:
        print("[OVERLAY RESTORE] Traffic light overlay already visible")

    # Restore OpenDRIVE overlay visibility
    lane_was_enabled = editor._scenario_preserved_lane_overlay_enabled
    print(f"[OVERLAY RESTORE] OpenDRIVE: was_enabled={lane_was_enabled}, current={cp.lane_overlay_enabled}")
    if lane_was_enabled and not cp.lane_overlay_enabled:
        cp.lane_overlay_enabled = True
        print("OpenDRIVE overlay restored")
        if not getattr(cp, 'opendrive_lane_data', None):
            OpenDriveOverlayRenderer.precompute_lane_data(cp)
        OpenDriveOverlayRenderer.invalidate_cache(cp, drop_surfaces=True)
    elif lane_was_enabled and cp.lane_overlay_enabled:
        print("[OVERLAY RESTORE] OpenDRIVE overlay already enabled")

def _reselect_preserved_actor(editor) -> bool:
    """Attempt to restore the actor selection after ScenarioRunner finishes."""
    cp = editor.camera_processor
    if not cp:
        return False

    old_id = editor._scenario_preserved_waypoint_vehicle_id
    preserved_waypoints = editor._scenario_preserved_waypoints or []
    target_signature = editor._scenario_preserved_waypoint_signature
    matched_actor = None
    matched_vehicle_id = None
    candidates = [
        actor for actor in cp.spawned_vehicles
        if actor and actor.is_alive
    ]

    def _valid_focus_location(loc: Optional[carla.Location], reference: Optional[Dict[str, float]] = None) -> bool:
        if loc is None:
            return False
        try:
            if not (math.isfinite(loc.x) and math.isfinite(loc.y) and math.isfinite(loc.z)):
                return False
        except Exception:
            return False
        # Reject origin-ish locations that come from proto actors before AWmini ticks
        if abs(loc.x) + abs(loc.y) + abs(loc.z) < 0.5:
            return False
        if reference:
            try:
                rx = float(reference.get("x", 0.0))
                ry = float(reference.get("y", 0.0))
                rz = float(reference.get("z", 0.0))
                dx = loc.x - rx
                dy = loc.y - ry
                dz = loc.z - rz
                if math.sqrt(dx * dx + dy * dy + dz * dz) < 0.25:
                    # Treat locations indistinguishable from the preserved point as still valid.
                    return True
            except Exception:
                pass
        return True

    matched_from_signature = False
    if target_signature:
        for vehicle_id, waypoints in cp.waypoint_list.items():
            signature = editor._compute_waypoint_signature(waypoints)
            if signature != target_signature:
                continue

            candidate_actor = next(
                (
                    actor
                    for actor in candidates
                    if actor.id == vehicle_id
                ),
                None,
            )

            matched_vehicle_id = vehicle_id
            if candidate_actor:
                matched_actor = candidate_actor
                matched_from_signature = True
                break

    def waypoints_match(sequence_a, sequence_b, tol=1.0):
        if len(sequence_a) != len(sequence_b):
            return False
        for wp_a, wp_b in zip(sequence_a, sequence_b):
            if (
                abs(float(wp_a.get('x', 0.0)) - float(wp_b.get('x', 0.0))) > tol
                or abs(float(wp_a.get('y', 0.0)) - float(wp_b.get('y', 0.0))) > tol
                or abs(float(wp_a.get('z', 0.0)) - float(wp_b.get('z', 0.0))) > tol
            ):
                return False
        return True

    if preserved_waypoints and not matched_from_signature:
        for vehicle_id, waypoints in cp.waypoint_list.items():
            if not waypoints:
                continue

            if not waypoints_match(preserved_waypoints, waypoints):
                continue

            candidate_actor = next(
                (
                    actor
                    for actor in cp.spawned_vehicles
                    if actor and actor.is_alive and actor.id == vehicle_id
                ),
                None,
            )

            if candidate_actor:
                matched_vehicle_id = vehicle_id
                matched_actor = candidate_actor
                break

            if matched_vehicle_id is None and vehicle_id != old_id:
                matched_vehicle_id = vehicle_id

    target_bp = editor._scenario_preserved_actor_blueprint
    target_loc = editor._scenario_preserved_actor_location

    if not matched_actor and target_bp:
        best_actor = None
        best_distance = float("inf")
        for actor in candidates:
            if actor.type_id != target_bp:
                continue
            distance = float("inf")
            if target_loc:
                try:
                    loc = actor.get_location()
                    dx = loc.x - target_loc.get('x', 0.0)
                    dy = loc.y - target_loc.get('y', 0.0)
                    dz = loc.z - target_loc.get('z', 0.0)
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                except Exception:
                    distance = float("inf")
            else:
                distance = 0.0

            if distance < best_distance:
                best_distance = distance
                best_actor = actor

        if best_actor:
            matched_actor = best_actor
            matched_vehicle_id = best_actor.id

    if matched_actor:
        if cp.restore_actor_selection(matched_actor):
            focus_location: Optional[carla.Location] = None
            live_location_valid = False
            editor._camera_debug(f"[CameraDebug] Restoring selection for actor {matched_actor.id} (old_id={old_id})")
            try:
                loc = matched_actor.get_location()
                if _valid_focus_location(loc, editor._scenario_preserved_actor_location):
                    focus_location = loc
                    live_location_valid = True
                else:
                    editor._camera_debug(
                        f"[CameraDebug] Actor {matched_actor.id} location invalid/placeholder "
                        f"({getattr(loc, 'x', '?')}, {getattr(loc, 'y', '?')}, {getattr(loc, 'z', '?')}); "
                        "will use preserved position until a valid tick arrives."
                    )
            except Exception:
                focus_location = None
            if focus_location is None and editor._scenario_preserved_actor_location:
                loc = editor._scenario_preserved_actor_location
                try:
                    focus_location = carla.Location(
                        float(loc.get("x", 0.0)),
                        float(loc.get("y", 0.0)),
                        float(loc.get("z", 0.0)),
                    )
                except Exception:
                    focus_location = None
            if focus_location is None and editor.camera_processor:
                try:
                    original = editor.camera_processor.get_actor_original_json_position(
                        matched_actor.type_id,
                        "vehicle" if not editor._scenario_preserved_actor_is_pedestrian else "pedestrian",
                    )
                    if original:
                        focus_location = original.location
                except Exception:
                    focus_location = None

            if focus_location:
                if editor._auto_camera_allowed():
                    cp.focus_camera_on_location(focus_location)  # Snap camera to the restored selection
                    try:
                        editor._camera_debug(
                            "[CameraDebug] Focused on restored actor "
                            f"{matched_actor.id} at ({focus_location.x:.2f}, {focus_location.y:.2f}, {focus_location.z:.2f})"
                        )
                    except Exception:
                        editor._camera_debug("[CameraDebug] Focused on restored actor (coords unavailable)")
                cp.refresh_selected_vehicle_ui()
                if old_id and old_id != matched_actor.id:
                    cp.waypoint_list.pop(old_id, None)
                panel = getattr(editor, "info_panel", None)
                if (
                    panel
                    and editor._scenario_preserved_info_panel_visible
                ):
                    obj_type = (
                        'pedestrian'
                        if editor._scenario_preserved_actor_is_pedestrian
                        else 'vehicle'
                    )
                    panel.show(matched_actor, obj_type, editor.screen_width, editor.screen_height)
                if live_location_valid:
                    # Selection fully restored with a valid actor location; clear preserved state.
                    editor._reset_scenario_waypoint_preserve(include_traffic=False)
                    return True
                # Keep preserved state so we can retry once AWmini provides a real transform.
                editor._scenario_pending_actor_reselect = True
                return False
            # If we get here, keep preserved state so we can retry when the actor/location becomes available.
            editor._camera_debug(
                "[CameraDebug] Restored actor selection but no focus location yet; "
                "will keep retrying."
            )
            editor._scenario_pending_actor_reselect_attempts = 0
            return False

    if matched_vehicle_id:
        cp.waypoint_display_vehicle_id = matched_vehicle_id
        if old_id and old_id != matched_vehicle_id:
            cp.waypoint_list.pop(old_id, None)
        try:
            cp.refresh_waypoints_carla_debug()
        except Exception:
            pass

    return False

def _process_pending_actor_reselect(editor):
    """Retry actor reselection across frames when immediate restoration fails."""
    if not editor._scenario_pending_actor_reselect:
        return
    if not editor.camera_processor:
        return

    editor._reselect_preserved_actor()
    editor._restore_preserved_traffic_light_selection()

    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if not actor_pending and not traffic_pending:
        editor._reset_scenario_waypoint_preserve()
        editor._scenario_pending_actor_reselect = False
        editor._scenario_pending_actor_reselect_attempts = 0
        return

    editor._scenario_pending_actor_reselect_attempts += 1
    if (
        editor._scenario_pending_actor_reselect_attempts
        >= editor._scenario_pending_actor_reselect_max_attempts
    ):
        print("[Scenario] Gave up restoring previous selection after ScenarioRunner playback.")
        editor._reset_scenario_waypoint_preserve()
        editor._scenario_pending_actor_reselect = False

def _save_all_scene_vehicles(editor):
    """Save and move all vehicles and pedestrians in the scene to avoid conflicts"""
    try:
        editor.saved_scene_vehicles = []

        if not editor.camera_processor or not hasattr(editor.camera_processor, 'world'):
            print("ERROR: Cannot access world through camera_processor")
            return

        world = editor.camera_processor.world

        # Get all vehicles and pedestrians currently in the scene
        all_vehicles = world.get_actors().filter('vehicle.*')
        all_walkers = world.get_actors().filter('walker.*')
        all_actors = list(all_vehicles) + list(all_walkers)
        print(f"\n=== SCENE ACTOR MANAGEMENT ===")
        print(f"Found {len(all_vehicles)} vehicles and {len(all_walkers)} pedestrians in scene to save (total: {len(all_actors)})")

        ego_actor = editor._refresh_external_ego_actor_reference()
        ego_actor_id = ego_actor.id if ego_actor else None

        # If an external ego is present, save its pre-run transform so we can
        # teleport it back after the scenario ends.  Then teleport it to the
        # placeholder ego position for scenario start.
        if ego_actor_id is not None and ego_actor and ego_actor.is_alive:
            try:
                editor._external_swap_pre_run_transform = ego_actor.get_transform()
            except Exception:
                editor._external_swap_pre_run_transform = editor._external_swap_last_transform

            # Teleport external ego to placeholder ego position for scenario start
            cp = editor.camera_processor
            placeholder_tf = None
            if cp:
                ego_id = getattr(cp, 'ego_vehicle_id', None)
                if ego_id is not None:
                    placeholder_tf = cp.vehicle_transforms.get(ego_id) or getattr(cp, 'ego_vehicle_transform', None)
            if placeholder_tf:
                try:
                    ego_actor.set_transform(placeholder_tf)
                    ego_actor.set_target_velocity(carla.Vector3D())
                    ego_actor.set_target_angular_velocity(carla.Vector3D())
                    # Honor the Ego Physics toggle (see _prepare_external_ego_for_playback).
                    ego_actor.set_simulate_physics(editor.ego_physics_enabled)
                    print(f"[External Ego] Teleported to placeholder position for scenario start.")
                except Exception as exc:
                    print(f"[External Ego] Failed to teleport to placeholder position: {exc}")

        # Record the preview actors about to be destroyed. DestroyActor only lands on
        # the next tick, and with an external tick source idle between runs (awmini
        # sync mode) the placeholder ego still looks alive on the spawn point when
        # MiniRunner runs its spawn-clear check — aborting the run. The runner gets
        # this set so it can skip exactly these pending-destroy actors.
        removed_ids = set()
        if editor.camera_processor and hasattr(editor.camera_processor, "spawned_vehicles"):
            for preview_actor in editor.camera_processor.spawned_vehicles:
                try:
                    if preview_actor is not None:
                        removed_ids.add(int(preview_actor.id))
                except Exception:
                    continue
        if ego_actor_id is not None:
            removed_ids.discard(int(ego_actor_id))  # preserved by cleanup, not destroyed
        editor._playback_removed_preview_ids = removed_ids

        # Drop all preview/placeholder actors; ScenarioRunner will spawn fresh ones.
        # The external ego is NOT in spawned_vehicles, so cleanup won't touch it.
        if editor.camera_processor:
            editor.camera_processor.cleanup_all_vehicles()
            if hasattr(editor.camera_processor, "spawned_vehicles"):
                editor.camera_processor.spawned_vehicles.clear()
        editor.saved_scene_vehicles = []
        editor.scene_preview_destroyed = True
        print("Preview actors removed. ScenarioRunner will spawn fresh actors.")
        print(f"=== END SCENE ACTOR MANAGEMENT ===\n")

    except Exception as e:
        print(f"ERROR in _save_all_scene_vehicles: {e}")
        traceback.print_exc()
        editor.saved_scene_vehicles = []

def _restore_scene_preview_if_destroyed(editor, focus_fn: Callable[[], None]) -> bool:
    if not editor.scene_preview_destroyed:
        return False
    print("Rebuilding scenario preview from saved file...")
    editor.scene_preview_destroyed = False
    editor.saved_scene_vehicles = []
    if (
        editor.current_scenario_path
        and os.path.isfile(editor.current_scenario_path)
        and editor.camera_processor
    ):
        preserve_camera = editor._scenario_preserved_waypoint_vehicle_id is not None
        preserve_reason = "actor preserved" if preserve_camera else ""
        if not preserve_camera:
            try:
                if editor.camera_processor.manual_control_enabled:
                    preserve_camera = True
                    preserve_reason = "manual control enabled"
                elif editor.camera_processor.manual_control_pending:
                    preserve_camera = True
                    preserve_reason = "manual control pending"
                elif getattr(editor.camera_processor, "ego_vehicle_transform", None) is not None:
                    preserve_camera = True
                    preserve_reason = "ego transform cached"
            except Exception as exc:
                editor._camera_debug(f"[CameraDebug] Failed to evaluate preserve_camera heuristics: {exc}")
        editor._debug_camera_pose("restore-preload")
        editor._camera_debug(
            f"[CameraDebug] Reload preview preserve_camera={preserve_camera} reason={preserve_reason or 'none'}"
        )
        # Save in-memory control mode before JSON reload overwrites it
        _saved_control_mode = getattr(editor.camera_processor, 'vehicle_control_mode', None)
        editor.camera_processor.load_waypoint_data_from_file(
            editor.current_scenario_path,
            preserve_camera=preserve_camera,
        )
        # Restore in-memory control mode — JSON is only a fallback
        if _saved_control_mode is not None:
            editor.camera_processor.vehicle_control_mode = _saved_control_mode
        try:
            scenario_data = getattr(editor.camera_processor, "loaded_scenario_data", None)
            if not scenario_data and os.path.isfile(editor.current_scenario_path):
                with open(editor.current_scenario_path, "r", encoding="utf-8") as fh:
                    scenario_data = json.load(fh)
            if scenario_data:
                editor._apply_weather_from_json_data(scenario_data)
        except Exception as exc:
            print(f"[Weather] Failed to restore weather from scenario: {exc}")
        selection_restored = editor._reselect_preserved_actor()
        traffic_preserve = bool(editor._scenario_preserved_traffic_light_ids)
        traffic_restored = (
            traffic_preserve and editor._restore_preserved_traffic_light_selection()
        )
    else:
        print("No scenario file available to reload preview.")
        selection_restored = False
        traffic_restored = False
    if editor.camera_processor:
        editor.camera_processor.disable_manual_control()
        # Restore overlay visibility state
        editor._restore_preserved_overlay_visibility()
    # Remove any preserved start markers so the placeholder square disappears after playback.
    editor._clear_scenario_start_markers()
    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if (selection_restored or traffic_restored) and not actor_pending:
        editor._reset_scenario_waypoint_preserve(include_traffic=False)

    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if (
        not selection_restored
        and not traffic_restored
        and not actor_pending
        and not traffic_pending
    ):
        editor._debug_camera_pose("restore-pre-ego")
        focus_fn()
    else:
        editor._debug_camera_pose("restore-skip-ego")

    # Clear pre-run transform — external ego is left wherever it ended up.
    editor._external_swap_pre_run_transform = None

    if actor_pending or traffic_pending:
        editor._scenario_pending_actor_reselect = True
        editor._scenario_pending_actor_reselect_attempts = 0
    else:
        editor._scenario_pending_actor_reselect = False
        editor._scenario_pending_actor_reselect_attempts = 0
        editor._reset_scenario_waypoint_preserve()
    return True

def _restore_without_saved_actors(editor, focus_fn: Callable[[], None]) -> bool:
    if editor.saved_scene_vehicles:
        return False
    print("No saved actors to restore")
    if editor.camera_processor:
        selection_restored = editor._reselect_preserved_actor()
        traffic_preserve = bool(editor._scenario_preserved_traffic_light_ids)
        traffic_restored = (
            traffic_preserve and editor._restore_preserved_traffic_light_selection()
        )
        editor.camera_processor.disable_manual_control()
        # Restore overlay visibility state
        editor._restore_preserved_overlay_visibility()
    else:
        selection_restored = False
        traffic_restored = False

    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if (selection_restored or traffic_restored) and not actor_pending:
        editor._reset_scenario_waypoint_preserve(include_traffic=False)

    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if (
        not selection_restored
        and not traffic_restored
        and not actor_pending
        and not traffic_pending
    ):
        editor._debug_camera_pose("restore-pre-ego-nosave2")
        focus_fn()

    print(f"Attempting to restore {len(editor.saved_scene_vehicles)} actors (vehicles and pedestrians)")
    restored_count = 0
    for actor_data in editor.saved_scene_vehicles:
        try:
            actor = actor_data['actor']
            if actor and actor.is_alive:
                original_transform = actor_data['transform']
                actor_type = "Vehicle" if actor.type_id.startswith('vehicle.') else "Pedestrian"
                print(f"Restoring {actor_type.lower()} {actor.id} to ({original_transform.location.x:.1f}, {original_transform.location.y:.1f}, {original_transform.location.z:.1f})")

                # Restore position
                actor.set_transform(original_transform)
                physics_enabled = actor_data.get('physics_enabled', True)
                try:
                    actor.set_simulate_physics(physics_enabled)
                except Exception:
                    pass
                if actor.type_id.startswith('vehicle.'):
                    velocity = actor_data.get('velocity')
                    angular_velocity = actor_data.get('angular_velocity')
                    if velocity is not None:
                        try:
                            actor.set_velocity(velocity)
                        except Exception:
                            pass
                    if angular_velocity is not None:
                        try:
                            actor.set_angular_velocity(angular_velocity)
                        except Exception:
                            pass
                restored_count += 1
            else:
                print(f"Warning: Saved actor no longer exists or is not alive")

        except Exception as e:
            print(f"Error restoring individual actor: {e}")

    print(f"Successfully restored {restored_count} actors (vehicles and pedestrians) to original positions")
    print(f"=== END RESTORING SCENE ACTORS ===\n")
    editor.saved_scene_vehicles = []
    editor._clear_scenario_start_markers()
    if editor.camera_processor:
        editor._debug_camera_pose("restore-post-actors")
        # External ego is left wherever it ended up — no teleport back.
        selection_restored = editor._reselect_preserved_actor()
        traffic_preserve = bool(editor._scenario_preserved_traffic_light_ids)
        traffic_restored = (
            traffic_preserve and editor._restore_preserved_traffic_light_selection()
        )
        editor.camera_processor.disable_manual_control()
        # Restore overlay visibility state
        editor._restore_preserved_overlay_visibility()
    else:
        selection_restored = False
        traffic_restored = False
    editor._external_swap_pre_run_transform = None

    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if (selection_restored or traffic_restored) and not actor_pending:
        editor._reset_scenario_waypoint_preserve(include_traffic=False)

    # Recompute pending states after potential reset operations
    actor_pending = editor._scenario_preserved_waypoint_vehicle_id is not None
    traffic_pending = editor._scenario_preserved_traffic_light_ids is not None

    if (
        not selection_restored
        and not traffic_restored
        and not actor_pending
        and not traffic_pending
    ):
        editor._debug_camera_pose("restore-pre-ego-nosave")
        focus_fn()

    if actor_pending or traffic_pending:
        editor._scenario_pending_actor_reselect = True
        editor._scenario_pending_actor_reselect_attempts = 0
    else:
        editor._scenario_pending_actor_reselect = False
        editor._scenario_pending_actor_reselect_attempts = 0
        editor._reset_scenario_waypoint_preserve()
    return True

def _restore_all_scene_vehicles_once(editor):
    """Run restore only once per scenario run to avoid duplicate reloads."""
    with editor._restore_once_lock:
        if getattr(editor, "_restore_invoked_for_run", False):
            editor._debug_camera_pose("restore-skip")
            print("[Restore] Already completed for this run; skipping.")
            return
        editor._restore_invoked_for_run = True
    editor._restore_all_scene_vehicles()

def _restore_all_scene_vehicles(editor):
    """Restore all saved vehicles and pedestrians to their original positions"""
    if getattr(editor, "_restore_in_progress", False):
        editor._debug_camera_pose("restore-nested-skip")
        print("[Restore] Already in progress; skipping nested call.")
        return
    editor._debug_camera_pose("restore-start")
    editor._restore_in_progress = True
    focus_attempted = False

    # Capture the live camera zoom (height) NOW, before the scene reload below resets the
    # controller to (0,0,0). This is the zoom the user had when Stop was pressed; it is
    # reapplied after the reload so the camera stays at the Stop-time zoom (not the Play-time one).
    try:
        editor._restore_camera_height = (
            float(editor.camera_controller.height) if editor.camera_controller else None
        )
    except Exception:
        editor._restore_camera_height = None

    # Pre-position the camera near the last known ego to avoid visible jumps from the manual-control follow offset.
    try:
        if editor._auto_camera_allowed() and editor.camera_controller and editor.camera_processor:
            # Reapply the Stop-time zoom captured just above (preserves the user's current zoom).
            saved_height = getattr(editor, "_restore_camera_height", None)
            if saved_height is not None:
                editor.camera_controller.height = saved_height
            ego_tf = getattr(editor.camera_processor, "ego_vehicle_transform", None)
            if ego_tf:
                editor.camera_controller.center_x = ego_tf.location.x
                editor.camera_controller.center_y = ego_tf.location.y
                editor._debug_camera_pose("restore-prealign-ego")
    except Exception:
        pass

    def _focus_ego_once():
        nonlocal focus_attempted
        if focus_attempted:
            return
        if not editor._auto_camera_allowed():
            return
        focus_attempted = True
        try:
            # The scene reload above resets the camera controller to (0,0,0). Reapply the
            # Stop-time zoom captured at the start of restore so focus only needs to recenter
            # on the ego (focus_camera_on_location preserves height); otherwise it stays at h=0.
            saved_height = getattr(editor, "_restore_camera_height", None)
            if saved_height is not None and editor.camera_controller is not None:
                editor.camera_controller.height = saved_height
            editor._focus_camera_on_ego_vehicle()
        except Exception:
            pass

    try:
        print(f"\n=== RESTORING SCENE ACTORS ===")
        if editor._restore_scene_preview_if_destroyed(_focus_ego_once):
            return

        if editor._restore_without_saved_actors(_focus_ego_once):
            return

    except Exception as e:
        print(f"ERROR in _restore_all_scene_vehicles: {e}")
        traceback.print_exc()
        editor.saved_scene_vehicles = []
        editor._clear_scenario_start_markers()
        if editor.camera_processor:
            editor.camera_processor.disable_manual_control()
        editor.external_ego_actor = None
        editor.external_ego_actor_id = None
        editor._external_swap_active = False
        editor._external_swap_current_id = None
        editor._external_swap_last_transform = None
        editor._external_swap_last_blueprint = None
        editor._external_swap_last_color = None
        editor._reset_scenario_waypoint_preserve()
    finally:
        # Restore preserved vehicle control mode BEFORE clearing _restore_in_progress
        # so the main loop sync doesn't read the stale JSON value
        preserved_mode = getattr(editor, '_preserved_vehicle_control_mode', None)
        if preserved_mode:
            editor.vehicle_control_mode = preserved_mode
            if editor.camera_processor:
                editor.camera_processor.vehicle_control_mode = preserved_mode
        editor._restore_in_progress = False
        editor._debug_camera_pose("restore-end")
