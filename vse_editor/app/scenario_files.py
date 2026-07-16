"""Scenario file flows: open/save/export dialogs, prompts, scene signatures,
unsaved-changes confirms, reset_current_scenario (duck-typed module functions,
first arg `editor`). Moved verbatim from vse.py (step-39, Phase 7) except the
four file-dialog default dirs: `script_dir` now resolves via _SCRIPT_DIR
(repo root) because __file__ moved from vse.py to vse_editor/app/ -- same value
as before the move (documented mechanism delta, D5).
"""

import json
import os

from pathlib import Path
from typing import Dict, List, Optional, cast

import carla
import pygame

from pygame_gui._constants import (
    UI_FILE_DIALOG_PATH_PICKED,
    UI_WINDOW_CLOSE,
    UI_SELECTION_LIST_NEW_SELECTION,
    UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION,
    UI_BUTTON_PRESSED,
)
from pygame_gui.elements.ui_button import UIButton
from pygame_gui.elements.ui_selection_list import UISelectionList
from pygame_gui.elements.ui_window import UIWindow

from vse_editor.ui.dialogs import EnhancedFileDialog
from vse_editor.ui.widgets import UISaveConfirmationDialog

# repo root (the directory that holds vse.py) -- the same value the moved
# bodies used to compute from vse.py's own __file__ (D5 anchor).
_SCRIPT_DIR = str(Path(__file__).resolve().parents[2])


def _get_map_display_name(editor):
    """Return a short map identifier for UI display."""
    if not editor.world_map or not editor.world_map.name:
        return "Unknown"
    return editor.world_map.name.split('/')[-1]

def _resolve_default_map_directory(editor):
    """Determine the default directory used by the map selection dialog."""
    carla_root = os.environ.get('CARLA_ROOT')
    if carla_root:
        candidates = [
            os.path.join(carla_root, 'CarlaUE4', 'Content', 'Carla', 'Maps'),
            os.path.join(carla_root, 'CarlaUE4', 'Content', 'Carla', 'Map'),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
    return os.getcwd()

def show_open_map_dialog(editor):
    """Show the map selection menu to allow the user to select a new CARLA map."""
    if editor.scenario_running:
        print("Cannot change map while a scenario is running. Stop the scenario first.")
        return

    if editor.map_menu:
        was_open = editor.map_menu_visible
        editor._close_all_dropdowns()
        editor.map_menu_visible = not was_open
        if editor.map_menu_visible:
            editor.map_menu.initialize()
            editor.map_menu.scroll_offset = 0
    else:
        print("Map menu not initialized yet.")

def show_open_scenario_dialog(editor):
    """Show the scenario selection menu to allow the user to save or load scenarios."""
    if editor.scenario_running:
        print("Cannot open scenario menu while a scenario is running. Stop the scenario first.")
        return

    # Toggle the scenario menu visibility
    if editor.scenario_menu:
        was_open = editor.scenario_menu_visible
        editor._close_all_dropdowns()
        editor.scenario_menu_visible = not was_open
    else:
        print("Scenario menu not initialized yet.")

def _scene_has_content(editor) -> bool:
    """Return True if the editor currently holds any scenario data."""
    cp = editor.camera_processor
    if not cp:
        return False

    if any(cp.waypoint_list.values()):
        return True

    if getattr(cp, "triggers", None):
        return True

    spawned = getattr(cp, "spawned_vehicles", None)
    if spawned and any(actor for actor in spawned if actor):
        return True

    if getattr(cp, "loaded_scenario_data", None):
        return True

    traffic_triggers = getattr(cp, "traffic_light_groups", None)
    if traffic_triggers:
        if any(
            getattr(group, "trigger_center", None) or getattr(group, "trigger_radius", None)
            for group in traffic_triggers
        ):
            return True

    return False

def _scene_signature_from_payload(editor, payload: Optional[dict], *, context: str = "") -> Optional[str]:
    """Return a deterministic signature for a scenario payload."""
    if not payload or not isinstance(payload, dict):
        return None
    try:
        data = dict(payload)
        data.pop("timestamp", None)
        return json.dumps(data, sort_keys=True, ensure_ascii=True)
    except Exception as exc:
        label = f" [{context}]" if context else ""
        print(f"[Scene] Unable to compute signature{label}: {exc}")
        return None

def _compute_scene_signature(editor) -> Optional[str]:
    """Return a deterministic signature of the current scene for dirty-checking."""
    cp = editor.camera_processor
    if not cp:
        return None

    try:
        map_obj = editor._safe_get_world_map(refresh=False)
        if map_obj:
            full_map_name = map_obj.name
            map_name = full_map_name.split('/')[-1] if '/' in full_map_name else full_map_name
        else:
            map_name = "unknown"

        snapshot, _ego = cp._collect_scenario_snapshot(map_name)
        return editor._scene_signature_from_payload(snapshot, context="current")
    except Exception as exc:
        print(f"[Scene] Unable to compute scene signature: {exc}")
        return None

def _compute_disk_scene_signature(editor) -> Optional[str]:
    """Return the signature of the on-disk scenario JSON, if available."""
    path = editor.current_scenario_path
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return editor._scene_signature_from_payload(payload, context="disk")
    except Exception as exc:
        print(f"[Scene] Unable to read scenario from disk for signature: {exc}")
        return None

def _mark_scene_saved(
    editor,
    live_signature: Optional[str] = None,
    disk_signature: Optional[str] = None,
):
    """Record that the current scene matches saved state (live and disk snapshots)."""
    editor._saved_scene_signature = live_signature if live_signature is not None else editor._compute_scene_signature()
    editor._saved_disk_signature = disk_signature if disk_signature is not None else editor._compute_disk_scene_signature()
    editor._scene_dirty_hint = False
    editor.history.mark_saved()

def _has_unsaved_changes(editor) -> bool:
    """Best-effort detection of scenario edits that haven't been written to disk."""
    if not editor._scene_has_content():
        return False

    # If no file path is set, any content counts as unsaved.
    if not editor.current_scenario_path or not os.path.isfile(editor.current_scenario_path):
        return True

    # Dirty only when the edit pointer diverges from the last saved pointer.
    return editor.history.position != editor.history.saved_position

def _unsaved_changes_handler(editor, dialog):
    """Return a modal event handler for a UISaveConfirmationDialog."""
    def handler(event: pygame.event.Event):
        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element == dialog.save_button:
                return True, "save"
            if event.ui_element == dialog.dont_save_button:
                return True, "discard"
            if event.ui_element == dialog.cancel_button:
                return True, "cancel"
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, "cancel"
        return False, None
    return handler

def _confirm_new_scenario(editor) -> str:
    """Prompt the user before discarding the current scenario.

    Returns ``"save"``, ``"discard"``, or ``"cancel"``.
    """
    if not editor.ui_manager:
        return "discard"

    dialog_rect = editor._center_dialog_rect(460, 220)
    dialog = UISaveConfirmationDialog(
        rect=dialog_rect,
        action_long_desc=(
            "Do you want to save your changes before starting a new scenario?<br><br>"
            "Your unsaved changes will be lost if you don't save them."
        ),
        manager=editor.ui_manager,
        window_title="New Scenario",
        blocking=True,
    )

    result = editor._run_modal_window(dialog, editor._unsaved_changes_handler(dialog))
    return result or "cancel"

def _confirm_load_scenario_discard(editor) -> str:
    """Prompt before loading a scenario that will discard unsaved changes.

    Returns ``"save"``, ``"discard"``, or ``"cancel"``.
    """
    if not editor.ui_manager:
        return "discard"

    editor._close_all_dropdowns(keep_info_panel=True)
    dialog_rect = editor._center_dialog_rect(460, 220)
    dialog = UISaveConfirmationDialog(
        rect=dialog_rect,
        action_long_desc=(
            "Do you want to save your changes before loading a different scenario?<br><br>"
            "Your unsaved changes will be lost if you don't save them."
        ),
        manager=editor.ui_manager,
        window_title="Load Scenario",
        blocking=True,
    )

    result = editor._run_modal_window(dialog, editor._unsaved_changes_handler(dialog))
    return result or "cancel"

def _confirm_play_discard_changes(editor) -> str:
    """Prompt the user before discarding unsaved edits to play the saved scenario.

    Returns ``"save"``, ``"discard"``, or ``"cancel"``.
    """
    if not editor.ui_manager:
        return "discard"

    editor._close_all_dropdowns(keep_info_panel=True)
    dialog_rect = editor._center_dialog_rect(460, 220)
    dialog = UISaveConfirmationDialog(
        rect=dialog_rect,
        action_long_desc=(
            "Do you want to save your changes before playing the scenario?<br><br>"
            "Your unsaved changes will be lost if you don't save them."
        ),
        manager=editor.ui_manager,
        window_title="Play Scenario",
        blocking=True,
    )

    result = editor._run_modal_window(dialog, editor._unsaved_changes_handler(dialog))
    return result or "cancel"

def _reset_camera_view_to_origin(editor):
    """Move the editor camera back to the default world origin."""
    # Leave the 3D view even when the camera reset itself is skipped below (No-Camera
    # mode): a fresh scenario always starts top-down. A bare mode flip is safe because
    # enter_orbit() recomputes the full orbit pose from scratch.
    if editor.camera_controller:
        editor.camera_controller.view_mode = "topdown"
    if not editor._auto_camera_allowed():
        return
    if not (editor.camera_processor and editor.camera_controller):
        return

    default_height = getattr(editor.camera_controller, "height", 200.0)
    try:
        editor.camera_controller.height = default_height
        editor.camera_processor.focus_camera_on_location(carla.Location(x=0.0, y=0.0, z=default_height))
    except Exception as exc:
        print(f"[Scenario] Warning: failed to reset camera position: {exc}")

def reset_current_scenario(editor) -> bool:
    """Clear the active scene and return to a blank scenario canvas."""
    if editor.scenario_running:
        print("Cannot start a new scenario while Scenario Runner is running. Stop it first.")
        return False

    if not editor.camera_processor:
        print("Camera processor not initialized yet.")
        return False

    if editor._has_unsaved_changes():
        choice = editor._confirm_new_scenario()
        if choice == "cancel":
            print("New scenario cancelled by user.")
            return False
        if choice == "save":
            if not editor._save_current_scenario():
                print("New scenario cancelled: save was cancelled or failed.")
                return False

    print("Resetting scenario to a blank state...")

    previous_suppress = bool(getattr(editor, "_suppress_external_ego_adoption", False))
    editor._suppress_external_ego_adoption = True
    preserved_external: Optional[carla.Actor] = None

    try:
        editor.camera_processor._reset_scenario_state()
        editor.camera_processor.disable_manual_control()
        if hasattr(editor.camera_processor, "loaded_scenario_data"):
            editor.camera_processor.loaded_scenario_data = None
        preserved_external = editor._refresh_external_ego_actor_reference()
    except Exception as exc:
        print(f"[Scenario] Warning: failed to clear existing scene: {exc}")
    finally:
        editor._suppress_external_ego_adoption = previous_suppress

    editor.saved_scene_vehicles = []
    if preserved_external and getattr(preserved_external, "is_alive", False):
        editor.external_ego_actor = preserved_external
        editor.external_ego_actor_id = preserved_external.id
    else:
        editor.external_ego_actor = None
        editor.external_ego_actor_id = None
    editor._external_swap_active = False
    editor._external_swap_current_id = None
    editor._external_swap_last_transform = None
    editor._external_swap_last_blueprint = None
    editor._external_swap_last_color = None
    editor._external_ego_prompt_pending_id = None
    editor.pending_scenario_load = None
    editor.current_scenario_name = None
    editor.current_scenario_path = None
    editor.history.clear()
    editor._reset_scenario_waypoint_preserve()
    editor._clear_scenario_start_markers()
    editor._scenario_pending_actor_reselect = False
    editor._scenario_pending_actor_reselect_attempts = 0

    if editor.info_panel:
        editor.info_panel.hide()

    editor.scenario_menu_visible = False

    editor._reset_camera_view_to_origin()
    editor._reset_weather_to_baseline()

    editor._saved_scene_signature = None
    editor._saved_disk_signature = None
    editor._scene_dirty_hint = False

    print("Scenario reset complete.")
    return True

def _scenario_has_playable_content(editor) -> bool:
    """Return True if the scene has any playable route or a manual-control ego."""
    cp = editor.camera_processor
    if not cp:
        return False

    waypoint_map = getattr(cp, "waypoint_list", {}) or {}
    spawned = getattr(cp, "spawned_vehicles", None) or []

    for actor in spawned:
        if not actor or not actor.is_alive:
            continue
        waypoints = waypoint_map.get(actor.id)
        if waypoints:
            return True

    # Allow manual-control runs when an ego exists without a route.
    try:
        if cp.is_ego_vehicle_active() or cp.get_editor_ego_actor():
            return True
    except Exception:
        pass

    # Fallback to scenario JSON for ego presence (e.g., external ego workflows).
    scenario_data = getattr(cp, "loaded_scenario_data", None)
    if not scenario_data and editor.current_scenario_path and os.path.isfile(editor.current_scenario_path):
        try:
            with open(editor.current_scenario_path, "r", encoding="utf-8") as fh:
                scenario_data = json.load(fh)
        except Exception:
            scenario_data = None
    if scenario_data:
        if scenario_data.get("ego_vehicle"):
            return True
        for entry in scenario_data.get("vehicles", []):
            if str(entry.get("role", "")).lower() == "ego_vehicle":
                return True

    return False

def _load_scenario_from_path(editor, file_path: str, *, prompt_unsaved: bool = True) -> bool:
    """Load a scenario from a concrete file path with map handling."""
    if not editor.camera_processor:
        print("Camera processor not initialized yet.")
        return False
    if not file_path:
        return False

    if prompt_unsaved and editor._has_unsaved_changes():
        choice = editor._confirm_load_scenario_discard()
        if choice == "cancel":
            print("Load cancelled: unsaved changes were kept.")
            return False
        if choice == "save":
            if not editor._save_current_scenario():
                print("Load cancelled: save was cancelled or failed.")
                return False

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        print(f"Scenario file not found: {file_path}")
        editor._clear_last_scenario_cache(remove_file=True)
        return False

    print(f"Loading scenario from: {file_path}")

    # Validate BEFORE the loader runs: the loader destroys the current scene
    # (cleanup_all_vehicles) before spawning the new one, so a bad file must
    # be rejected here, while the scene is still intact (fix-13).
    scenario_data = None
    try:
        with open(file_path, 'r') as f:
            scenario_data = json.load(f)
    except Exception as exc:
        print(f"Failed to parse scenario file: {exc}")
        editor._show_error_overlay(
            f"Failed to load '{os.path.basename(file_path)}': not valid JSON — scene left untouched."
        )
        return False
    if not isinstance(scenario_data, dict) or 'vehicles' not in scenario_data:
        print(f"Invalid scenario file (expected a JSON object with a 'vehicles' key): {file_path}")
        editor._show_error_overlay(
            f"Failed to load '{os.path.basename(file_path)}': not a scenario file — scene left untouched."
        )
        return False

    scenario_map = scenario_data.get('map_name', 'Unknown')
    if not isinstance(scenario_map, str) or not scenario_map:
        scenario_map = 'Unknown'
    try:
        current_map = editor._get_map_display_name()

        if scenario_map != 'Unknown' and scenario_map != current_map:
            print(f"Scenario requires map '{scenario_map}' but current map is '{current_map}'")
            print(f"Switching to map '{scenario_map}'...")
            editor.pending_scenario_load = file_path
            editor.load_map(scenario_map)
            return True
    except Exception as exc:
        print(f"Warning: Could not check map compatibility: {exc}")

    # Loaded scenarios start in the default top-down view (same contract as New
    # scenario / Play / world reset -- the loader and its focus poke top-down fields).
    if (editor.camera_controller
            and getattr(editor.camera_controller, "view_mode", "topdown") == "orbit"):
        editor.camera_controller.exit_orbit()
        editor.camera_processor.update_camera_position()

    try:
        load_ok = editor.camera_processor.load_waypoint_data_from_file(file_path)
    except Exception as exc:
        print(f"Failed to load scenario file: {exc}")
        load_ok = False
    if load_ok is False:
        # The loader cleans up the current scene before spawning, so a
        # mid-load failure leaves a partial scene behind (fix-11).
        print(f"Failed to load scenario: {file_path}")
        editor._show_error_overlay(
            f"Failed to load '{os.path.basename(file_path)}' — scene may be partially loaded (see console)."
        )
        return False

    scenario_name = os.path.splitext(os.path.basename(file_path))[0]
    if isinstance(scenario_data, dict):
        editor._apply_weather_from_json_data(scenario_data)

    # Sync vehicle_control_mode from camera_processor after load
    editor.vehicle_control_mode = getattr(editor.camera_processor, 'vehicle_control_mode', 'basic_agent')

    editor.current_scenario_path = file_path
    editor.current_scenario_name = scenario_name
    print(f"Scenario '{editor.current_scenario_name}' loaded successfully")
    live_sig = editor._compute_scene_signature()
    disk_sig = editor._compute_disk_scene_signature()
    editor._mark_scene_saved(live_sig, disk_sig)
    # Load resets the edit counters but deliberately NOT the stacks —
    # frozen pre-refactor behavior (undo history survives a load).
    editor.history.reset_positions()
    editor._remember_last_scenario(file_path, scenario_name, scenario_map)
    return True

def _open_last_scenario_from_cache(editor) -> bool:
    """Open the most recently used scenario if available."""
    for entry in editor.recent_scenarios:
        path = entry.get('path')
        if path and os.path.isfile(path):
            return editor._load_scenario_from_path(path)
    print("No recent scenario available.")
    return False

def _prompt_scenario_file_path(editor, *, initial_path: Optional[str] = None) -> Optional[str]:
    """Prompt the user for a scenario JSON file path using the in-window file dialog."""
    script_dir = _SCRIPT_DIR  # repo root (D5 anchor; __file__ moved in step-39)
    default_path = initial_path or editor.current_scenario_path or editor.last_scenario_path or script_dir

    dialog_rect = editor._center_dialog_rect(640, 480)
    dialog = EnhancedFileDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        window_title="Open Scenario",
        allowed_suffixes={'.json', '.JSON'},
        initial_file_path=default_path,
        allow_existing_files_only=True,
        always_on_top=True,
    )
    dialog.set_blocking(True)
    confirm_button = getattr(dialog, "ok_button", getattr(dialog, "confirm_button", None))
    if confirm_button:
        confirm_button.set_text("Open")

    def handler(event: pygame.event.Event):
        if event.type == UI_FILE_DIALOG_PATH_PICKED and event.ui_element == dialog:
            if event.text and os.path.isfile(event.text):
                return True, event.text
            return False, None
        # Legacy pre-0.8 support: treat user_type events the same way without touching the deprecated attribute directly.
        if event.type == pygame.USEREVENT:
            picked_path = getattr(event, "text", None)
            ui_element = getattr(event, "ui_element", None)
            if picked_path and ui_element == dialog and os.path.isfile(picked_path):
                return True, picked_path
        if event.type == UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION and event.ui_element == getattr(dialog, "file_selection_list", None):
            # If the directory changed (double-clicked folder), ignore.
            if Path(dialog.current_directory_path).name == event.text and Path(dialog.current_directory_path).is_dir():
                return False, None
            path = dialog.current_file_path
            if path and os.path.isfile(path):
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
            if path and os.path.isfile(path):
                return True, str(path)
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, None
        return False, None

    file_path = editor._run_modal_window(dialog, handler)
    if not file_path:
        return None
    return str(file_path)

def _prompt_agent_file_path(editor, *, initial_path: Optional[str] = None) -> Optional[str]:
    """Prompt the user for a ScenarioRunner-compatible agent Python file."""
    if not editor.ui_manager:
        print("[Agent] UI not ready; cannot open agent picker.")
        return None

    editor._close_all_dropdowns(keep_info_panel=True)
    script_dir = _SCRIPT_DIR  # repo root (D5 anchor; __file__ moved in step-39)
    default_path = initial_path or getattr(editor, "agent_path", None) or editor._get_last_agent_directory() or script_dir

    dialog_rect = editor._center_dialog_rect(640, 480)
    dialog = EnhancedFileDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        window_title="Select Agent",
        allowed_suffixes={'.py', '.PY'},
        initial_file_path=default_path,
        allow_existing_files_only=True,
        always_on_top=True,
    )
    dialog.set_blocking(True)
    confirm_button = getattr(dialog, "ok_button", getattr(dialog, "confirm_button", None))
    if confirm_button:
        confirm_button.set_text("Select")

    def handler(event: pygame.event.Event):
        if event.type == UI_FILE_DIALOG_PATH_PICKED and event.ui_element == dialog:
            if event.text and os.path.isfile(event.text):
                return True, event.text
            return False, None
        if event.type == pygame.USEREVENT:
            picked_path = getattr(event, "text", None)
            ui_element = getattr(event, "ui_element", None)
            if picked_path and ui_element == dialog and os.path.isfile(picked_path):
                return True, picked_path
        if event.type == UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION and event.ui_element == getattr(dialog, "file_selection_list", None):
            if Path(dialog.current_directory_path).name == event.text and Path(dialog.current_directory_path).is_dir():
                return False, None
            path = dialog.current_file_path
            if path and os.path.isfile(path):
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
                except Exception:
                    pass
                else:
                    dialog._change_directory_path(new_path)
            return False, None
        if (
            event.type == UI_BUTTON_PRESSED
            and confirm_button is not None
            and event.ui_element == confirm_button
        ):
            path = dialog.current_file_path
            if path and os.path.isfile(path):
                return True, str(path)
        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, None
        return False, None

    file_path = editor._run_modal_window(dialog, handler)
    if not file_path:
        return None
    file_path = str(file_path)
    if not os.path.isfile(file_path):
        return None
    if not file_path.lower().endswith(".py"):
        return None
    return os.path.abspath(file_path)

def _prompt_startup_map_choice(editor, options: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Prompt the user to choose a local CARLA map for startup."""
    if not options:
        return None

    dialog_rect = editor._center_dialog_rect(520, 560)
    dialog = UIWindow(
        rect=dialog_rect,
        manager=editor.ui_manager,
        window_display_title="Open Map",
        resizable=False,
        always_on_top=True,
    )
    dialog.set_blocking(True)

    container_width, container_height = dialog.get_container().get_size()
    list_rect = pygame.Rect(
        10,
        10,
        max(120, container_width - 20),
        max(80, container_height - 70),
    )

    label_to_option = {opt["label"]: opt for opt in options if isinstance(opt, dict) and opt.get("label")}
    selection_list = UISelectionList(
        relative_rect=list_rect,
        item_list=[opt["label"] for opt in options if opt.get("label")],
        manager=editor.ui_manager,
        container=dialog,
        allow_double_clicks=True,
        object_id="#startup_map_selection_list",
    )

    button_width = 120
    button_height = 32
    button_y = container_height - button_height - 10

    cancel_button = UIButton(
        relative_rect=pygame.Rect(
            container_width - button_width - 10,
            button_y,
            button_width,
            button_height,
        ),
        text="Cancel",
        manager=editor.ui_manager,
        container=dialog,
        object_id="#startup_map_cancel_button",
        anchors={
            "left": "right",
            "right": "right",
            "top": "bottom",
            "bottom": "bottom",
        },
    )

    ok_button = UIButton(
        relative_rect=pygame.Rect(
            container_width - (2 * button_width) - 20,
            button_y,
            button_width,
            button_height,
        ),
        text="Start",
        manager=editor.ui_manager,
        container=dialog,
        object_id="#startup_map_start_button",
        anchors={
            "left": "right",
            "right": "right",
            "top": "bottom",
            "bottom": "bottom",
            "right_target": cancel_button,
        },
    )
    ok_button.disable()

    selected_label: Optional[str] = None

    def handler(event: pygame.event.Event):
        nonlocal selected_label

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True, None

        if event.type == UI_SELECTION_LIST_NEW_SELECTION and event.ui_element == selection_list:
            selected_label = event.text
            ok_button.enable()
            return False, None

        if event.type == UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION and event.ui_element == selection_list:
            selected_label = event.text
            return True, label_to_option.get(selected_label) if selected_label else None

        if event.type == UI_BUTTON_PRESSED:
            if event.ui_element == ok_button:
                return True, label_to_option.get(selected_label) if selected_label else None
            if event.ui_element == cancel_button:
                return True, None

        if event.type == UI_WINDOW_CLOSE and event.ui_element == dialog:
            return True, None

        return False, None

    picked = editor._run_modal_window(dialog, handler)
    if isinstance(picked, dict):
        return cast(Dict[str, str], picked)
    return None

def load_scenario_with_dialog(editor):
    """Load a scenario using an in-window file dialog."""
    if not editor.camera_processor:
        print("Camera processor not initialized yet.")
        return

    file_path = editor._prompt_scenario_file_path()
    if not file_path:
        return

    editor._load_scenario_from_path(file_path)

def _save_current_scenario(editor) -> bool:
    """Save the scenario (silently if a path exists, otherwise via file dialog).

    Returns True if the save succeeded, False if it was cancelled or failed.
    """
    if getattr(editor, 'scenario_running', False):
        print("Cannot save while scenario is running.")
        return False
    if not editor.camera_processor:
        return False

    if editor.current_scenario_path and editor.current_scenario_name:
        try:
            result = editor.camera_processor.save_waypoint_data_to_file(editor.current_scenario_path)
            if result is False:
                print(f"Save failed for scenario: {editor.current_scenario_name}")
                editor._show_error_overlay(
                    f"Save FAILED for '{editor.current_scenario_name}' — changes are NOT saved (see console)."
                )
                return False
            live_sig = editor._compute_scene_signature()
            disk_sig = editor._compute_disk_scene_signature()
            editor._mark_scene_saved(live_sig, disk_sig)
            editor.history.mark_saved()
            print(f"Saved scenario: {editor.current_scenario_name}")
            return True
        except Exception as exc:
            print(f"Save failed: {exc}")
            editor._show_error_overlay(
                f"Save FAILED for '{editor.current_scenario_name}' — changes are NOT saved (see console)."
            )
            return False

    # No path yet -- open the save-as dialog.
    editor.save_scenario_with_dialog()
    # If the dialog set a path, the save succeeded.
    return bool(editor.current_scenario_path and os.path.isfile(editor.current_scenario_path))

def save_scenario_with_dialog(editor):
    """Save a scenario using an in-window file dialog."""
    if getattr(editor, 'scenario_running', False):
        print("Cannot save while scenario is running.")
        return
    if not editor.camera_processor:
        print("Camera processor not initialized yet.")
        return

    # Get the script directory as default folder
    script_dir = _SCRIPT_DIR  # repo root (D5 anchor; __file__ moved in step-39)
    initial_path = editor.current_scenario_path or script_dir

    dialog_rect = editor._center_dialog_rect(640, 480)
    dialog = EnhancedFileDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        window_title="Save Scenario",
        allowed_suffixes={'.json', '.JSON'},
        initial_file_path=initial_path,
        allow_existing_files_only=False,
        always_on_top=True,
    )
    dialog.set_blocking(True)
    confirm_button = getattr(dialog, "ok_button", getattr(dialog, "confirm_button", None))
    if confirm_button:
        confirm_button.set_text("Save")
    file_entry = getattr(dialog, "file_path_text_line", None)
    if file_entry and not editor.current_scenario_path:
        default_dir = initial_path if os.path.isdir(initial_path) else os.path.dirname(initial_path)
        if not default_dir:
            default_dir = os.getcwd()
        base_name = editor.current_scenario_name or "scenario"
        candidate_path = os.path.join(default_dir, f"{base_name}.json")
        dialog.current_directory_path = default_dir
        dialog.current_file_path = Path(candidate_path)
        file_entry.set_text(candidate_path)
        filename = os.path.basename(candidate_path)
        name_part = filename[:-5] if filename.lower().endswith(".json") else filename
        start_index = len(candidate_path) - len(filename)
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
            if event.text and not os.path.isdir(event.text):
                return True, event.text
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

    # Ensure .json extension
    if not file_path.endswith('.json'):
        file_path += '.json'

    # Extract scenario name and target directory
    scenario_name = os.path.splitext(os.path.basename(file_path))[0]
    target_dir = os.path.dirname(file_path)

    print(f"Saving scenario '{scenario_name}' to: {target_dir}")
    ego_data = editor.camera_processor.get_ego_vehicle_data() if editor.camera_processor else None

    map_obj = editor._safe_get_world_map(refresh=not editor._map_refresh_disabled)
    if map_obj:
        full_map_name = map_obj.name
        map_name = full_map_name.split('/')[-1] if '/' in full_map_name else full_map_name
    else:
        full_map_name = "unknown"
        map_name = "unknown"

    # 3. Save scenario JSON to target directory
    result = editor.camera_processor.save_waypoint_data_to_file(file_path)
    if result is False:
        print(f"Save failed for scenario: {scenario_name}")
        editor._show_error_overlay(
            f"Save FAILED for '{scenario_name}' — changes are NOT saved (see console)."
        )
        return

    # Update current scenario tracking
    editor.current_scenario_path = file_path
    editor.current_scenario_name = scenario_name
    print(f"Scenario '{scenario_name}' saved successfully")
    live_sig = editor._compute_scene_signature()
    disk_sig = editor._compute_disk_scene_signature()
    editor._mark_scene_saved(live_sig, disk_sig)
    editor.history.mark_saved()
    editor._remember_last_scenario(file_path, scenario_name, map_name)

def export_scenario_as_xosc_with_dialog(editor):
    """Export the current scenario to OpenSCENARIO 1.0 .xosc using a file dialog."""
    if not editor.camera_processor:
        print("Camera processor not initialized yet.")
        return

    script_dir = _SCRIPT_DIR  # repo root (D5 anchor; __file__ moved in step-39)
    if editor.current_scenario_path:
        initial_path = os.path.splitext(editor.current_scenario_path)[0] + '.xosc'
    else:
        initial_path = script_dir

    dialog_rect = editor._center_dialog_rect(640, 480)
    dialog = EnhancedFileDialog(
        rect=dialog_rect,
        manager=editor.ui_manager,
        window_title="Export OpenSCENARIO (.xosc)",
        allowed_suffixes={'.xosc', '.XOSC'},
        initial_file_path=initial_path,
        allow_existing_files_only=False,
        always_on_top=True,
    )
    dialog.set_blocking(True)
    confirm_button = getattr(dialog, "ok_button", getattr(dialog, "confirm_button", None))
    if confirm_button:
        confirm_button.set_text("Export")
    file_entry = getattr(dialog, "file_path_text_line", None)
    if file_entry and editor.current_scenario_path:
        candidate_path = os.path.splitext(editor.current_scenario_path)[0] + '.xosc'
        dialog.current_file_path = Path(candidate_path)
        file_entry.set_text(candidate_path)

    def handler(event: pygame.event.Event):
        if event.type == UI_FILE_DIALOG_PATH_PICKED and event.ui_element == dialog:
            if event.text and not os.path.isdir(event.text):
                return True, event.text
            return False, None
        if event.type == pygame.USEREVENT:
            picked_path = getattr(event, "text", None)
            ui_element = getattr(event, "ui_element", None)
            if picked_path and ui_element == dialog and not os.path.isdir(picked_path):
                return True, picked_path
        if (event.type == UI_SELECTION_LIST_DOUBLE_CLICKED_SELECTION
                and event.ui_element == getattr(dialog, "file_selection_list", None)):
            path = dialog.current_file_path
            if path and not os.path.isdir(path):
                return True, str(path)
            return False, None
        if (event.type == UI_BUTTON_PRESSED
                and confirm_button is not None
                and event.ui_element == confirm_button):
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
    if not file_path.lower().endswith('.xosc'):
        file_path += '.xosc'

    result = editor.camera_processor.export_to_openscenario(file_path)
    if result:
        print(f"Exported OpenSCENARIO to: {file_path}")
    else:
        print("Export to OpenSCENARIO failed.")
        editor._show_error_overlay("OpenSCENARIO export failed (see console).")
