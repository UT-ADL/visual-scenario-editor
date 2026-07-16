"""Event pump + input dispatch (duck-typed module functions, first arg `editor`).

Moved verbatim from vse.py (step-36, Phase 7).
"""

import os

import pygame

from vse_editor.input_helpers import is_shift_pressed, is_ctrl_pressed
from vse_editor.ui.menus import PlacementMode
from vse_editor.ui.widgets import UISaveConfirmationDialog


def _handle_quit_event(editor):
    """Shared quit handler for window close shortcuts."""
    choice = editor.show_exit_confirmation()
    if choice == "cancel":
        return True
    if choice == "save":
        if not editor._save_current_scenario():
            return True  # save cancelled or failed -- don't exit
    # "save" (succeeded) or "discard" -- proceed with exit.
    # Skip the RPC batch-destroy when the server is known dead (fix-05/06):
    # it would stall on the client timeout per actor (the actors die with the
    # server anyway). editor.cleanup() applies the same gate.
    if editor.camera_processor and not getattr(editor, "_server_lost", False):
        editor.camera_processor.cleanup_all_vehicles()
    editor.keep_server_running_on_exit = False
    editor.server_manager.set_auto_stop_enabled(True)
    return False

def _camera_in_orbit(editor) -> bool:
    """True while the editor camera is in the 3D orbit view."""
    cam = editor.camera_controller
    return bool(cam and getattr(cam, "view_mode", "topdown") == "orbit")

def _cancel_gestures_for_orbit(editor):
    """Entering the 3D view cancels in-flight placement/drag gestures but keeps the
    selection (unlike _cancel_editing_step, whose later rungs clear it)."""
    cp = editor.camera_processor
    if not cp:
        return
    if cp.creating_waypoint:
        cp.stop_waypoint_creation()
    if cp.creating_destination:
        cp.stop_destination_creation()
    if cp.placing_trigger:
        cp.stop_trigger_placement()
    if cp.pending_personal_trigger:
        cp.cancel_personal_trigger_placement()
    cp.cancel_active_drag()
    editor._shift_press = None
    editor._marquee_press_pos = None
    editor.marquee_rect = None

def toggle_orbit_view(editor):
    """Toggle the editor camera between top-down and the 3D orbit view (Tab / 3D button)."""
    cam = editor.camera_controller
    cp = editor.camera_processor
    if cam is None or cp is None or not editor.ready:
        print("World is not ready yet. Please wait for map to finish loading.")
        return
    if getattr(cam, "view_mode", "topdown") == "orbit":
        cam.exit_orbit()
        print("Camera: top-down view")
    else:
        _cancel_gestures_for_orbit(editor)
        cp._large_map_travel_target = None  # a pending travel ease must not drive the pivot
        cam.enter_orbit()
        print("Camera: 3D orbit view (right-drag orbit, middle-drag pan, wheel zoom, Tab back)")
    cp.update_camera_position()

def _handle_keydown(editor, event, scenario_running):
    """Process KEYDOWN events. Returns False to exit, True if handled, or None to fall through."""
    if editor.info_panel.handle_key_input(event):
        return True

    # Keyboard shortcuts help overlay: F1 / H toggles it; Esc closes it while open.
    # Placed after the info-panel key capture so H is not hijacked while editing a field,
    # and before the ESCAPE block below so Esc closes the panel instead of stopping a run.
    # Disabled during playback (along with F11 maximize below) so these keys can't interrupt
    # a run -- only C changes the view, and only Esc/Stop ends the run.
    if event.key in (pygame.K_F1, pygame.K_h):
        if scenario_running:
            return True  # Help overlay disabled during playback
        editor.keyboard_help_visible = not editor.keyboard_help_visible
        return True
    if editor.keyboard_help_visible and event.key == pygame.K_ESCAPE:
        editor.keyboard_help_visible = False
        return True

    # Window maximize toggle is disabled during playback so it can't disrupt/end the run.
    if event.key == pygame.K_F11 and scenario_running:
        return True  # Maximize disabled during playback

    manual_arrow_consumed = False
    if editor.camera_processor:
        manual_arrow_consumed = editor.camera_processor.handle_manual_control_key(event.key, True)

    keys = pygame.key.get_pressed()
    ctrl_pressed = is_ctrl_pressed(keys)
    shift_pressed = is_shift_pressed(keys)

    # Check if this key is part of a keyboard shortcut before adding to keys_pressed
    is_keyboard_shortcut = (
        ctrl_pressed and event.key in {pygame.K_s, pygame.K_l, pygame.K_z, pygame.K_y}
    )

    if not manual_arrow_consumed and not is_keyboard_shortcut:
        editor.keys_pressed.add(event.key)
        if event.key not in editor.key_hold_times:
            editor.key_hold_times[event.key] = 0.0

    if event.key == pygame.K_F4 and (keys[pygame.K_LALT] or keys[pygame.K_RALT]):
        if not editor._handle_quit_event():
            return False
        return True

    if event.key == pygame.K_ESCAPE:
        if scenario_running:
            # While a scenario is playing, ESC always means "stop" -- an alternative to the
            # Stop button that works even when the UI is hidden in Chase/Cockpit camera modes.
            print("ESC pressed - stopping scenario")
            editor.stop_scenario()
            return True
        if editor.map_menu_visible:
            editor.map_menu_visible = False
            print("Map menu closed")
        elif editor.scenario_menu_visible:
            editor.scenario_menu_visible = False
            print("Scenario menu closed")
        elif _cancel_editing_step(editor):
            pass  # one cancel level per press (mode -> drag -> selection)
        else:
            print("ESC pressed - Use Alt+F4 or close button to exit")
        return True

    if event.key == pygame.K_o:
        if not editor.ready:
            print("World is not ready yet. Please wait for map to finish loading.")
        elif editor.scenario_running:
            return True  # Disable toggle during playback
        elif editor.camera_processor:
            editor.camera_processor.toggle_opendrive_overlay()
        return True

    if event.key == pygame.K_t:
        if not editor.ready:
            print("World is not ready yet. Please wait for map to finish loading.")
        elif editor.scenario_running:
            return True  # Disable toggle during playback
        elif editor.camera_processor:
            editor.camera_processor.traffic_lights_visible = not editor.camera_processor.traffic_lights_visible
            status = "ENABLED" if editor.camera_processor.traffic_lights_visible else "DISABLED"
            print(f"Traffic light stop lines overlay {status}")
            if not editor.camera_processor.traffic_lights_visible:
                editor.camera_processor.clear_traffic_light_selection()
                if (editor.camera_processor.selected_personal_trigger
                        and editor.camera_processor.selected_personal_trigger.get('kind') == 'traffic_light'):
                    editor.camera_processor.clear_personal_trigger_selection()
                editor.camera_processor._last_visible_traffic_light_trigger_key = None
        return True

    if event.key == pygame.K_c:
        # Cycle the playback camera top-down -> chase -> cockpit -> top-down. Only while a follow
        # is active (a run with an ego, external or local) -- it does nothing while plain editing.
        if editor.camera_processor and (
            editor.camera_processor.playback_camera_follow_enabled
            or editor.camera_processor.manual_control_enabled
        ):
            mode = editor.camera_processor.toggle_playback_camera_mode()
            # Keep the Play Cam dropdown, persistence, and UI-visibility in sync with C.
            editor._set_play_camera_mode(mode, apply_live=False)
            print(f"Playback camera mode: {mode}")
        return True

    if event.key == pygame.K_TAB:
        # Editor-only 3D orbit view toggle. Inert during a run, and while a numeric
        # field is being edited (the info panel does NOT consume Tab: '\t' is not
        # printable, so handle_key_input falls through to the hotkeys).
        if not scenario_running and not editor.info_panel.editing:
            toggle_orbit_view(editor)
        return True

    if event.key == pygame.K_F3 and editor.camera_processor:
        editor.camera_processor.debug_raycast = not editor.camera_processor.debug_raycast
        editor.camera_processor.coordinate_detector.debug_raycast = editor.camera_processor.debug_raycast
        status = "ENABLED" if editor.camera_processor.debug_raycast else "DISABLED"
        print(f"\n[DEBUG MODE] Raycast coordinate debugging {status}")
        print(f"[DEBUG MODE] This will show detailed coordinate information when spawning vehicles")
        if editor.camera_processor.debug_raycast:
            cached = editor._safe_get_world_map(refresh=False)
            map_name = cached.name if cached else "unknown"
            print(f"[DEBUG MODE] Map: {map_name}\n")
        return True

    # Match the physical key left of "1" by scancode (KSCAN_GRAVE) so it works on any
    # keyboard layout -- on non-US layouts (e.g. Estonian) that key is not reported as
    # K_BACKQUOTE.
    if (event.key == pygame.K_BACKQUOTE
            or getattr(event, 'scancode', None) == pygame.KSCAN_GRAVE):
        editor.hide_all_ui = not editor.hide_all_ui
        status = "HIDDEN (pure camera)" if editor.hide_all_ui else "VISIBLE"
        print(f"UI overlays {status} - press ` to toggle")
        return True

    if event.key == pygame.K_s and ctrl_pressed:
        if getattr(editor, 'scenario_running', False):
            print("Cannot save while scenario is running.")
            return True
        if editor.camera_processor:
            if editor.current_scenario_path and editor.current_scenario_name:
                try:
                    result = editor.camera_processor.save_waypoint_data_to_file(editor.current_scenario_path)
                    if result is False:
                        print(f"Save failed for scenario: {editor.current_scenario_name}")
                    else:
                        live_sig = editor._compute_scene_signature()
                        disk_sig = editor._compute_disk_scene_signature()
                        editor._mark_scene_saved(live_sig, disk_sig)
                        print(f"Saved to scenario: {editor.current_scenario_name}")
                except Exception as exc:
                    print(f"Save failed: {exc}")
            else:
                print("No scenario loaded. Opening save dialog...")
                editor.save_scenario_with_dialog()
        return True

    if event.key == pygame.K_l and ctrl_pressed:
        print("Opening load scenario dialog...")
        editor.load_scenario_with_dialog()
        return True

    if event.key == pygame.K_z and ctrl_pressed:
        if shift_pressed:
            editor.redo_last_command()
        else:
            editor.undo_last_command()
        return True

    if event.key == pygame.K_y and ctrl_pressed:
        editor.redo_last_command()
        return True

    if event.key == pygame.K_DELETE and editor.camera_processor:
        if ((editor.camera_processor.selected_actor_ids or
                editor.camera_processor.selected_waypoint_group) and not editor.scenario_running):
            editor.camera_processor.delete_group_selection()
        elif (editor.camera_processor.selected_personal_trigger
                and editor.camera_processor.selected_personal_trigger.get('kind')
                in ('pedestrian', 'vehicle')):
            editor.camera_processor.delete_selected_personal_trigger()
        elif (editor.info_panel.visible and editor.info_panel.selected_object and
                editor.info_panel.object_type in ('vehicle', 'pedestrian')):
            editor.camera_processor.delete_selected_vehicle()
        elif editor.camera_processor.selected_trigger_index is not None:
            editor.camera_processor.delete_selected_trigger()
        else:
            success = editor.camera_processor.delete_selected_waypoint()
            if not success and (editor.camera_processor.selected_waypoint_vehicle_id is None):
                print("No waypoint selected. Click on a waypoint first to select it for deletion.")
        return True

    return True

def _handle_keyup(editor, event):
    """Process KEYUP events."""
    if editor.camera_processor:
        editor.camera_processor.handle_manual_control_key(event.key, False)

    release_keys = {pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d}
    arrow_keys = {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT}
    manual_active = bool(
        editor.camera_processor
        and editor.camera_processor.manual_control_armed
    )
    if event.key in release_keys or (event.key in arrow_keys and not manual_active):
        if event.key in editor.keys_pressed:
            editor.camera_moved_with_wasd = True

    editor.keys_pressed.discard(event.key)
    editor.key_hold_times.pop(event.key, None)

    if (editor.camera_moved_with_wasd and not (editor.keys_pressed &
        {pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d,
         pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT})):
        if editor.camera_controller and editor.world:
            # Prototype: camera navigation performs no terrain raycasts.
            editor.camera_moved_with_wasd = False
            if editor.camera_processor:
                editor.camera_processor.restore_vehicle_menu_after_camera_pan()
        if editor.camera_processor:
            editor.camera_processor.end_manual_camera_free_look("keyboard_pan")

def _handle_left_click(editor, pos, scenario_running):
    """Process left-click (button 1) interactions."""
    # A fresh press always starts unarmed: a stale *_drag_armed flag or
    # pending menu icon from a swallowed MOUSEBUTTONUP (blocking result
    # window, focus loss) must not hijack this click's motion routing.
    editor.pending_menu_icon = None
    editor._marquee_press_pos = None
    editor.marquee_rect = None
    editor._shift_press = None
    editor._orbit_left_press_pos = None
    editor._orbit_press_was_armed = False
    if editor.camera_processor:
        editor.camera_processor.vehicle_drag_armed = False
        editor.camera_processor.trigger_drag_armed = False
        editor.camera_processor.personal_trigger_drag_armed = False
        editor.camera_processor._click_actor_hit_cache = None

    # Server-lost overlay (fix-05) is modal: only its restart button is
    # clickable; every other click is swallowed (the scene below is dead).
    if getattr(editor, "_server_lost", False):
        btn = getattr(editor, "_server_restart_button_rect", None)
        if btn and btn.collidepoint(pos):
            editor._restart_after_server_crash()
        return True

    # Shortcuts help overlay: while open it is modal -- the X closes it and every other
    # click is swallowed so it doesn't fall through to the scene. The "?" button opens it.
    if editor.keyboard_help_visible:
        if editor.help_close_button_rect and editor.help_close_button_rect.collidepoint(pos):
            editor.keyboard_help_visible = False
        return True
    if editor.help_button_rect and editor.help_button_rect.collidepoint(pos):
        if scenario_running:
            return True  # Help overlay disabled during playback
        editor.keyboard_help_visible = True
        return True

    # Dismiss NPC driving mode dropdown if clicking outside it
    if getattr(editor, "npc_dropdown_open", False):
        on_npc_ui = False
        if getattr(editor, "npc_button_rect", None) and editor.npc_button_rect.collidepoint(pos):
            on_npc_ui = True
        for item_rect, _ in getattr(editor, "_npc_dropdown_item_rects", []):
            if item_rect.collidepoint(pos):
                on_npc_ui = True
        if not on_npc_ui:
            editor.npc_dropdown_open = False

    # Dismiss agent dropdown if clicking outside it
    if getattr(editor, "agent_dropdown_open", False):
        on_agent_ui = False
        if getattr(editor, "agent_button_rect", None) and editor.agent_button_rect.collidepoint(pos):
            on_agent_ui = True
        for item_rect, _ in getattr(editor, "_agent_dropdown_item_rects", []):
            if item_rect.collidepoint(pos):
                on_agent_ui = True
        if not on_agent_ui:
            editor.agent_dropdown_open = False

    # Dismiss agent behavior menu if clicking outside it
    if getattr(editor, "agent_behavior_menu_open", False):
        on_behavior_ui = False
        if getattr(editor, "agent_button_rect", None) and editor.agent_button_rect.collidepoint(pos):
            on_behavior_ui = True
        for brect in getattr(editor, "_agent_behavior_item_rects", {}).values():
            if brect.collidepoint(pos):
                on_behavior_ui = True
        if not on_behavior_ui:
            editor.agent_behavior_menu_open = False

    if editor.scenario_menu_visible and editor.scenario_menu and editor.scenario_menu.handle_click(pos):
        return True

    if editor.map_menu_visible and editor.map_menu and editor.map_menu.handle_click(pos):
        return True

    if editor.rendering_toggle_rect and editor.rendering_toggle_rect.collidepoint(pos):
        current_rendering = editor.rendering_enabled if editor.ready and editor.world else editor._rendering_desired_enabled
        target_state = not current_rendering
        if editor.ready and editor.world:
            editor._apply_rendering_mode(desired_enabled=target_state, reason="user toggle", force=True)
        else:
            editor._rendering_desired_enabled = target_state
            editor.rendering_enabled = target_state
            print("[Rendering] Preference updated; will apply once the CARLA world is ready.")
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if editor.ego_collision_toggle_rect and editor.ego_collision_toggle_rect.collidepoint(pos):
        if not editor.ego_physics_enabled:
            # Collision is locked OFF while Ego Physics is off (physics off implies collision off).
            editor.resolution_menu_open = False
            editor.fps_menu_open = False
            return True
        editor.ego_collision_enabled = not editor.ego_collision_enabled
        print(f"[Ego] Collision {'enabled' if editor.ego_collision_enabled else 'disabled'} (applies on next Play).")
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if editor.ego_physics_toggle_rect and editor.ego_physics_toggle_rect.collidepoint(pos):
        editor.ego_physics_enabled = not editor.ego_physics_enabled
        # Couple collision to physics: physics off => collision off; physics on => collision on.
        editor.ego_collision_enabled = editor.ego_physics_enabled
        print(f"[Ego] Physics {'enabled' if editor.ego_physics_enabled else 'disabled'} "
              f"(collision {'enabled' if editor.ego_collision_enabled else 'disabled'}; applies on next Play).")
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if editor.connection_button_rect and editor.connection_button_rect.collidepoint(pos):
        editor.connect_to_remote()
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if (getattr(editor, 'view3d_button_rect', None)
            and not editor.hide_all_ui
            and editor.view3d_button_rect.collidepoint(pos)):
        if not scenario_running:
            toggle_orbit_view(editor)
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if editor.open_map_button_rect and editor.open_map_button_rect.collidepoint(pos):
        editor.show_open_map_dialog()
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if editor.open_scenario_button_rect and editor.open_scenario_button_rect.collidepoint(pos):
        editor.show_open_scenario_dialog()
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    for rect, cam_mode in editor.play_camera_option_rects:
        if rect.collidepoint(pos):
            editor._set_play_camera_mode(cam_mode)
            editor._close_all_dropdowns()
            return True

    if editor.play_camera_button_rect and editor.play_camera_button_rect.collidepoint(pos):
        was_open = editor.play_camera_menu_open
        editor._close_all_dropdowns()
        editor.play_camera_menu_open = not was_open
        return True

    for rect, option in editor.resolution_option_rects:
        if rect.collidepoint(pos):
            editor._set_stream_resolution(option)
            editor._close_all_dropdowns()
            return True

    if editor.resolution_button_rect and editor.resolution_button_rect.collidepoint(pos):
        was_open = editor.resolution_menu_open
        editor._close_all_dropdowns()
        editor.resolution_menu_open = not was_open
        return True

    for rect, preset in editor._culling_option_rects:
        if rect.collidepoint(pos):
            editor._set_culling_distance(preset)  # applies, remembers, and closes dropdowns
            return True

    if editor._culling_button_rect and editor._culling_button_rect.collidepoint(pos):
        # Only editable while VSE solely controls the sim and isn't playing; otherwise consume
        # the click without opening (the button is drawn dimmed in those states).
        if (not editor.scenario_running) and editor._culling_apply_safe():
            was_open = editor._culling_menu_open
            editor._close_all_dropdowns()
            editor._culling_menu_open = not was_open
        return True

    if editor._using_remote_server():
        for rect, fps_value in editor.fps_option_rects:
            if rect.collidepoint(pos):
                editor._set_stream_fps(fps_value)
                editor._close_all_dropdowns()
                return True

        if editor.fps_button_rect and editor.fps_button_rect.collidepoint(pos):
            was_open = editor.fps_menu_open
            editor._close_all_dropdowns()
            editor.fps_menu_open = not was_open
            return True

    if editor.manual_tick_button_rect and editor.manual_tick_button_rect.collidepoint(pos):
        editor.manual_tick_enabled = not editor.manual_tick_enabled
        state = "ENABLED" if editor.manual_tick_enabled else "DISABLED"
        print(f"[Manual Tick] Drive Clock {state} by user request.")
        if editor.manual_tick_enabled:
            if editor.manual_tick_required:
                editor.manual_tick_accumulator = editor.manual_tick_interval
            else:
                editor.manual_tick_accumulator = 0.0
            editor.manual_tick_recommendation = False
            if not editor.manual_tick_required:
                print("[Manual Tick] Waiting for synchronous mode; Drive Clock will advance ticks once required.")
        else:
            editor.manual_tick_accumulator = 0.0
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    # NPC control mode toggle moved to VehicleSelectionMenu pills

    if editor.weather_button_rect and editor.weather_button_rect.collidepoint(pos):
        if editor.weather_button_enabled:
            editor.toggle_weather_window()
        else:
            print("Weather controls will be available once the CARLA world is ready.")
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    # NPC Driving Mode dropdown item clicks
    if getattr(editor, "npc_dropdown_open", False) and getattr(editor, "npc_button_enabled", False):
        for item_rect, mode_key in getattr(editor, "_npc_dropdown_item_rects", []):
            if item_rect.collidepoint(pos):
                if editor.camera_processor:
                    editor.camera_processor.vehicle_control_mode = mode_key
                editor.vehicle_control_mode = mode_key
                mode_label = "Simulated" if mode_key == "basic_agent" else "Scripted"
                print(f"[NPC Control] Mode: {mode_key} ({mode_label})")
                editor.npc_dropdown_open = False
                editor.resolution_menu_open = False
                editor.fps_menu_open = False
                return True

    # NPC Driving Mode button toggle
    if getattr(editor, 'npc_button_rect', None) and editor.npc_button_rect.collidepoint(pos):
        if getattr(editor, "npc_button_enabled", False):
            editor.npc_dropdown_open = not getattr(editor, "npc_dropdown_open", False)
            editor.agent_dropdown_open = False
            editor.agent_behavior_menu_open = False
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    # Agent behavior menu clicks (shown after selecting Autopilot)
    if getattr(editor, "agent_behavior_menu_open", False):
        for bkey, brect in getattr(editor, "_agent_behavior_item_rects", {}).items():
            if brect.collidepoint(pos):
                editor.agent_behavior = bkey
                editor._remember_last_agent(getattr(editor, "agent_path", None))
                print(f"[Agent] Behavior set to: {bkey}")
                editor.agent_behavior_menu_open = False
                return True

    # Agent dropdown item clicks (when dropdown is open)
    if getattr(editor, "agent_dropdown_open", False) and getattr(editor, "agent_button_enabled", False):
        for item_rect, mode_key in getattr(editor, "_agent_dropdown_item_rects", []):
            if item_rect.collidepoint(pos):
                if mode_key == "custom":
                    existing = getattr(editor, "agent_path", None)
                    if existing and os.path.isfile(existing):
                        editor.agent_mode = "custom"
                        editor._remember_last_agent(existing)
                        print(f"[Agent] Custom agent restored: {existing}")
                    else:
                        picked = editor._prompt_agent_file_path()
                        if picked:
                            editor.agent_mode = "custom"
                            editor._remember_last_agent(picked)
                            print(f"[Agent] Custom agent selected: {picked}")
                    editor.agent_dropdown_open = False
                elif mode_key == "autopilot":
                    editor.agent_mode = "autopilot"
                    editor._remember_last_agent()
                    print(f"[Agent] Mode set to: autopilot")
                    editor.agent_dropdown_open = False
                    editor.agent_behavior_menu_open = True
                else:
                    editor.agent_mode = mode_key
                    editor._remember_last_agent()
                    print(f"[Agent] Mode set to: {mode_key}")
                    editor.agent_dropdown_open = False
                editor.resolution_menu_open = False
                editor.fps_menu_open = False
                return True

    # "..." browse button click (change custom agent script)
    if (getattr(editor, 'agent_browse_button_rect', None)
            and editor.agent_browse_button_rect.collidepoint(pos)
            and not getattr(editor, "scenario_running", False)):
        picked = editor._prompt_agent_file_path()
        if picked:
            editor._remember_last_agent(picked)
            print(f"[Agent] Custom agent changed: {picked}")
        editor.agent_dropdown_open = False
        editor.agent_behavior_menu_open = False
        return True

    if getattr(editor, 'agent_button_rect', None) and editor.agent_button_rect and editor.agent_button_rect.collidepoint(pos):
        if getattr(editor, "agent_button_enabled", False):
            was_dropdown = getattr(editor, "agent_dropdown_open", False)
            was_behavior = getattr(editor, "agent_behavior_menu_open", False)
            editor.agent_dropdown_open = not was_dropdown and not was_behavior
            editor.agent_behavior_menu_open = False
            editor.npc_dropdown_open = False
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if hasattr(editor, 'play_button_rect') and editor.play_button_rect and editor.play_button_rect.collidepoint(pos):
        if editor.play_button_enabled:
            scenario_active = bool(
                editor.scenario_running or (editor.scenario_process and editor.scenario_process.poll() is None)
            )
            if scenario_active:
                editor.stop_scenario()
            else:
                editor.run_scenario()
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if editor.info_panel.handle_click(pos):
        return True

    if not scenario_running and editor.handle_mode_toggle_click(pos):
        return True

    active_menu = editor.get_active_selection_menu() if not scenario_running else None
    if active_menu and active_menu.handle_click(pos):
        return True

    if (editor.camera_processor and _camera_in_orbit(editor) and pos[1] > 80 and
            (editor.camera_processor.creating_waypoint or
             editor.camera_processor.creating_destination or
             editor.camera_processor.placing_trigger or
             editor.camera_processor.pending_personal_trigger)):
        # Belt-and-braces: entering orbit cancels placement gestures, but a mode
        # button could re-arm one. Placement is top-down only (view + select).
        editor.show_status_hint("3D view is view & select only - press Tab for top-down to edit")
        return True

    if (editor.camera_processor and editor.camera_processor.creating_waypoint and pos[1] > 80):
        print(f"Placing waypoint at click position: {pos}")
        editor.camera_processor.place_waypoint_at_click(pos[0], pos[1])
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if (editor.camera_processor and editor.camera_processor.creating_destination and pos[1] > 80):
        print(f"Placing destination at click position: {pos}")
        editor.camera_processor.place_destination_at_click(pos[0], pos[1])
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if (editor.camera_processor and editor.camera_processor.placing_trigger and pos[1] > 80):
        print(f"Placing trigger at click position: {pos}")
        editor.camera_processor.place_trigger_at_click(pos[0], pos[1])
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True
    if (editor.camera_processor and editor.camera_processor.pending_personal_trigger and pos[1] > 80):
        placed = editor.camera_processor.place_personal_trigger_at_click(pos[0], pos[1])
        if placed:
            editor.resolution_menu_open = False
            editor.fps_menu_open = False
        return True

    if scenario_running or pos[1] <= 80:
        if not (editor.resolution_button_rect and editor.resolution_button_rect.collidepoint(pos)):
            editor.resolution_menu_open = False
            editor.fps_menu_open = False
        return False

    # 3D orbit view: remember scene-area presses (UI clicks returned above) so the
    # motion handler can hint if the press becomes a blocked drag attempt.
    if _camera_in_orbit(editor):
        editor._orbit_left_press_pos = pos

    if editor.camera_processor and (editor.camera_processor.selected_actor_ids or
                                    editor.camera_processor.selected_waypoint_group):
        group_action = editor.camera_processor.check_group_menu_icon_click(pos[0], pos[1])
        if group_action:
            # Click-type icon: dispatches on mouse-UP (slide off to cancel).
            editor.pending_menu_icon = {'menu': 'group', 'icon': group_action, 'pos': pos}
            return True

    if (editor.camera_processor and editor.camera_processor.selected_vehicle and
            editor.camera_processor.vehicle_menu_position and
            not editor.camera_processor.creating_waypoint):
        menu_action = editor.camera_processor.check_menu_icon_click(pos[0], pos[1])
        if menu_action:
            if menu_action == 'rotate' and not editor.camera_processor.selected_vehicle_is_pedestrian:
                # Rotate is a press-drag gesture: it starts on mouse-down.
                editor.camera_processor.start_vehicle_rotation(pos[1])
            else:
                # Click-type icons dispatch on mouse-UP within the same icon
                # (slide off past the drag threshold to cancel).
                editor.pending_menu_icon = {'menu': 'vehicle', 'icon': menu_action, 'pos': pos}
            return True

    if (editor.camera_processor and pos[1] > 80
            and editor.camera_processor.handle_personal_trigger_click(pos[0], pos[1])):
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    keys = pygame.key.get_pressed()
    snap_to_lane = not is_shift_pressed(keys)
    if editor.placement_mode == PlacementMode.PEDESTRIAN:
        snap_to_lane = False
    ctrl_pressed = is_ctrl_pressed(keys)

    if ctrl_pressed and _camera_in_orbit(editor):
        # Ctrl+click is instant placement (waypoint split / spawn) -- top-down only.
        editor.show_status_hint("3D view is view & select only - press Tab for top-down to edit")
        return True

    if ctrl_pressed and editor.camera_processor:
        if editor.camera_processor.split_waypoint_at_click(pos[0], pos[1]):
            return True

    spawn_vehicle = ctrl_pressed

    if editor.placement_mode == PlacementMode.TRIGGER and spawn_vehicle:
        if not editor.ready:
            print("World is not ready yet. Please wait for map to finish loading.")
            return True
        if editor.camera_processor:
            editor.camera_processor.place_trigger_instantly(pos[0], pos[1])
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    if spawn_vehicle:
        if not editor.ready:
            print("World is not ready yet. Please wait for map to finish loading.")
            return True
        if not editor.camera_processor:
            return True
        detector = editor.camera_processor.coordinate_detector
        print(f"\nVehicle placement click: Screen({pos[0]}, {pos[1]})")
        coordinates = detector.screen_to_world_coordinates(
            pos[0], pos[1], editor.screen_width, editor.screen_height
        )
        if coordinates['success']:
            print(f"Screen-to-world result: World({coordinates['x']:.2f}, {coordinates['y']:.2f}, {coordinates['z']:.2f})")
            if snap_to_lane:
                lane_result = detector.find_closest_lane_point(
                    coordinates['x'], coordinates['y'], coordinates['z']
                )
                if lane_result['success']:
                    coordinates = lane_result
                    coordinates['snapped'] = True
                else:
                    coordinates['snapped'] = False
            else:
                coordinates['snapped'] = False
        else:
            coordinates['snapped'] = False

        spawn_role = "npc"
        if editor.placement_mode == PlacementMode.PEDESTRIAN:
            coordinates['snapped'] = False
            selected_blueprint = editor.pedestrian_menu.get_selected_pedestrian()
            spawn_role = "pedestrian"
        elif editor.placement_mode == PlacementMode.EGO:
            selected_blueprint = editor.ego_vehicle_menu.get_selected_ego_vehicle()
            spawn_role = "ego"
        else:
            selected_blueprint = editor.vehicle_menu.get_selected_vehicle()

        if selected_blueprint:
            editor.camera_processor.spawn_vehicle_at_marker(selected_blueprint, coordinates, role=spawn_role)
        editor.resolution_menu_open = False
        editor.fps_menu_open = False
        return True

    traffic_light_action_clicked = False
    if editor.camera_processor:
        traffic_light_action_clicked = editor.camera_processor.handle_traffic_light_action_click(pos[0], pos[1])

    if traffic_light_action_clicked:
        return True

    trigger_clicked = False
    if editor.camera_processor:
        trigger_clicked = editor.camera_processor.handle_trigger_click(pos[0], pos[1])

    if trigger_clicked:
        return True

    waypoint_clicked = False
    if editor.camera_processor:
        waypoint_clicked = editor.camera_processor.handle_waypoint_click(pos[0], pos[1])

    if waypoint_clicked:
        return True

    if editor.camera_processor and editor.camera_processor.handle_traffic_light_click(pos[0], pos[1]):
        return True

    if editor.camera_processor and editor.camera_processor.handle_vehicle_click(pos[0], pos[1]):
        return True

    if editor.camera_processor:
        # An empty-ground press may become a marquee (group selection) if the
        # mouse travels past the drag threshold before release. Marquee is
        # top-down only (3D view is view + select).
        if not _camera_in_orbit(editor):
            editor._marquee_press_pos = pos
        if ((editor.camera_processor.selected_vehicle and
                editor.camera_processor.selected_vehicle.is_alive) or
                editor.camera_processor.selected_traffic_light_group or
                editor.camera_processor.selected_actor_ids or
                editor.camera_processor.selected_waypoint_group or
                editor.camera_processor.waypoint_display_vehicle_id):
            # Forgiving empty click: while a vehicle/pedestrian, traffic-light
            # group, or group selection is selected, a click on empty ground
            # keeps the selection (and its waypoint path / stop-line overlay)
            # instead of clearing it. Deselect via Esc / right-click tap or by
            # selecting another object.
            editor.resolution_menu_open = False
            editor.fps_menu_open = False
            return True
        editor.camera_processor.clear_vehicle_selection()
        editor.camera_processor.waypoint_display_vehicle_id = None
        editor.camera_processor.selected_trigger_index = None
        editor.camera_processor.trigger_action_menu_position = None
        editor.camera_processor.clear_traffic_light_selection()
    editor.resolution_menu_open = False
    editor.fps_menu_open = False
    return True

def _dispatch_vehicle_menu_action(editor, menu_action):
    """Fire a click-type vehicle-menu icon (called on mouse-up)."""
    cp = editor.camera_processor
    if menu_action == 'delete':
        cp.delete_selected_vehicle()
    elif menu_action == 'ego_destination':
        if cp.selected_vehicle and cp.is_ego_vehicle(cp.selected_vehicle.id):
            # Always start fresh when choosing ego destination placement
            cp.start_waypoint_creation(reset_existing=True)
    elif menu_action == 'waypoint':
        if cp.selected_vehicle and cp.is_ego_vehicle(cp.selected_vehicle.id):
            print("Ego vehicle does not support waypoint creation.")
        else:
            cp.start_waypoint_creation()
    elif menu_action == 'autoroute':
        if cp.selected_vehicle and not cp.selected_vehicle_is_pedestrian:
            cp.start_auto_route_to_destination(cp.selected_vehicle)
    elif menu_action == 'add_trigger':
        selected_actor = cp.selected_vehicle
        if not selected_actor:
            return
        actor_type = 'pedestrian' if cp.selected_vehicle_is_pedestrian else 'vehicle'
        cp.start_personal_trigger_placement(actor_type, actor=selected_actor)
    elif menu_action == 'remove_trigger':
        selected_actor = cp.selected_vehicle
        if not selected_actor:
            return
        actor_id = selected_actor.id
        if cp.selected_vehicle_is_pedestrian:
            cp.delete_pedestrian_trigger(actor_id)
        elif not cp.is_ego_vehicle(actor_id):
            cp.delete_vehicle_trigger(actor_id)

def _promote_shift_press(editor, press, mouse_pos):
    """A Shift+press on an object travelled past the drag threshold: it is a
    move, not a tap. Select+arm exactly like a plain press at the original
    position, then hand off to the normal drag promotion (Shift is held, so
    it picks free placement — no lane snap)."""
    cp = editor.camera_processor
    if press['kind'] == 'actor':
        actor = next(
            (a for a in cp.spawned_vehicles
             if a and a.is_alive and a.id == press['actor_id']),
            None,
        )
        if actor is None:
            return
        cp.select_and_arm_actor(actor, *press['pos'])
        cp.update_vehicle_movement(mouse_pos)
    elif press['kind'] == 'waypoint':
        waypoints = cp.get_vehicle_waypoints(press['vehicle_id'])
        if not waypoints or press['index'] >= len(waypoints):
            return
        cp.select_and_arm_waypoint(press['vehicle_id'], press['index'], *press['pos'])
        cp.update_waypoint_movement(mouse_pos)

def _finalize_shift_tap(editor, press):
    """A Shift+press released under the drag threshold: it is a tap — toggle
    the pressed object in/out of the group selection (deferred from the press
    so a Shift+drag can free-move instead of toggling)."""
    cp = editor.camera_processor
    if press['kind'] == 'actor':
        actor = next(
            (a for a in cp.spawned_vehicles
             if a and a.is_alive and a.id == press['actor_id']),
            None,
        )
        if actor is not None:
            cp.toggle_actor_in_group(actor)
    elif press['kind'] == 'waypoint':
        waypoints = cp.get_vehicle_waypoints(press['vehicle_id'])
        if waypoints and press['index'] < len(waypoints):
            cp.toggle_waypoint_in_group(press['vehicle_id'], press['index'])

def _finalize_marquee(editor, rect):
    """Resolve a finished marquee rectangle.

    Priority: actors, else the displayed route's waypoint markers, else the
    global trigger (center-in-box). Exactly ONE captured item collapses to a
    normal single selection (menu / info panel, same as clicking it); two or
    more form a group. A box that catches nothing changes nothing (forgiving,
    like empty clicks).

    Shift = additive TOGGLE: the box resolves identically, but when its
    result is the SAME KIND as the current selection, each boxed item is
    toggled — unselected items join the group, already-selected items leave
    it (an existing single selection of that kind seeds the toggle).
    """
    cp = editor.camera_processor
    additive = is_shift_pressed(pygame.key.get_pressed())

    actors = cp.coordinate_detector.actors_in_screen_rect(rect)
    if actors:
        caught_ids = {actor.id for actor in actors}
        if additive:
            existing_ids = set(cp.selected_actor_ids or ())
            if cp.selected_vehicle and cp.selected_vehicle.is_alive:
                existing_ids.add(cp.selected_vehicle.id)
            merged_ids = existing_ids ^ caught_ids  # toggle membership
        else:
            merged_ids = caught_ids
        if not merged_ids:
            # Every boxed actor was already selected: the box deselected all.
            cp.clear_vehicle_selection()
            cp.selected_actor_ids = set()
            print("Selection cleared")
            return
        if len(merged_ids) == 1:
            target_id = next(iter(merged_ids))
            target = next(
                (a for a in cp.spawned_vehicles if a and a.is_alive and a.id == target_id),
                None,
            )
            if target is not None:
                cp.select_vehicle_actor(target)
                return
        # Group selection replaces any single selection (mutually exclusive).
        cp.clear_vehicle_selection()
        cp.clear_traffic_light_selection()
        cp.clear_personal_trigger_selection()
        cp.selected_trigger_index = None
        cp.trigger_action_menu_position = None
        cp.selected_waypoint_group = None
        cp.selected_actor_ids = merged_ids
        print(f"Selected {len(merged_ids)} actor(s)")
        return

    vehicle_id, indices = cp.waypoints_in_screen_rect(rect)
    if indices:
        caught_indices = set(indices)
        if additive:
            existing_indices = set()
            group = cp.selected_waypoint_group
            if group and group.get('vehicle_id') == vehicle_id:
                existing_indices = set(group.get('indices', ()))
            if (cp.selected_waypoint_vehicle_id == vehicle_id and
                    cp.selected_waypoint_index is not None):
                existing_indices.add(cp.selected_waypoint_index)
            merged_indices = existing_indices ^ caught_indices  # toggle membership
        else:
            merged_indices = caught_indices
        # Keep the route visible while its waypoints are selected.
        cp.clear_vehicle_selection(keep_waypoints=True)
        cp.clear_traffic_light_selection()
        cp.clear_personal_trigger_selection()
        cp.selected_trigger_index = None
        cp.trigger_action_menu_position = None
        cp.selected_actor_ids = set()
        cp.selected_waypoint_group = None
        if not merged_indices:
            # Every boxed waypoint was already selected: box deselected all.
            cp.selected_waypoint_vehicle_id = None
            cp.selected_waypoint_index = None
            if editor.info_panel.visible and editor.info_panel.object_type in ('waypoint', 'waypoint_group'):
                editor.info_panel.hide()
            print("Waypoint selection cleared")
            return
        if len(merged_indices) == 1:
            cp.select_single_waypoint(vehicle_id, next(iter(merged_indices)))
            return
        cp.selected_waypoint_vehicle_id = None
        cp.selected_waypoint_index = None
        cp.selected_waypoint_group = {'vehicle_id': vehicle_id, 'indices': merged_indices}
        print(f"Selected {len(merged_indices)} waypoint(s)")
        # Group property editing (Speed / Deviation / Idle apply to all)
        editor.info_panel.show(
            cp.selected_waypoint_group, 'waypoint_group',
            editor.screen_width, editor.screen_height, vehicle_id,
        )
        return

    targets = cp.trigger_targets_in_screen_rect(rect)
    if len(targets) > 1:
        # Ambiguous box (several trigger-ish centers): change nothing —
        # a tighter box is better than selecting the wrong thing.
        print(f"Box caught {len(targets)} triggers - draw a tighter box to select one")
        return
    if not targets:
        return
    target = targets[0]
    if target['kind'] == 'personal':
        cp.select_personal_trigger(target['selection'])
        print("Selected personal trigger")
    elif target['kind'] == 'tl_group':
        cp.select_traffic_light_group(target['group'])
    else:  # global trigger
        cp.clear_vehicle_selection()
        cp.clear_traffic_light_selection()
        cp.clear_personal_trigger_selection()
        cp.selected_actor_ids = set()
        cp.selected_waypoint_group = None
        cp.selected_trigger_index = target['index']
        cp.trigger_action_menu_position = target['screen_pos']
        cp.trigger_menu_hidden_for_camera_pan = False
        print(f"Selected trigger {target['index']}")

def _dispatch_pending_menu_icon(editor, pending, release_pos):
    """Fire a click-type menu icon on mouse-up.

    Standard button semantics: dispatch only when the release still hits the
    same icon and the press travelled less than the drag threshold — sliding
    off the icon cancels.
    """
    cp = editor.camera_processor
    if not cp:
        return
    press_pos = pending['pos']
    dx = release_pos[0] - press_pos[0]
    dy = release_pos[1] - press_pos[1]
    if (dx * dx + dy * dy) ** 0.5 > cp.object_drag_threshold:
        return
    menu = pending['menu']
    icon = pending['icon']
    if menu == 'vehicle':
        if cp.check_menu_icon_click(release_pos[0], release_pos[1]) == icon:
            _dispatch_vehicle_menu_action(editor, icon)
    elif menu == 'trigger':
        if cp.check_trigger_menu_icon_click(release_pos[0], release_pos[1]) == icon:
            if icon == 'delete':
                cp.delete_selected_trigger()
    elif menu == 'personal':
        if cp.check_personal_trigger_menu_icon_click(release_pos[0], release_pos[1]) == icon:
            if icon == 'delete':
                cp.delete_selected_personal_trigger()
    elif menu == 'group':
        if cp.check_group_menu_icon_click(release_pos[0], release_pos[1]) == icon:
            if icon == 'delete':
                cp.delete_group_selection()

def _cancel_editing_step(editor):
    """One Esc-style cancel step, shared by the Esc key and a right-click tap:
    cancel the active placement mode, else cancel an in-progress drag/rotate/
    scale (restoring its start state), else clear the current selection.
    Returns True when something was cancelled."""
    cp = editor.camera_processor
    if not cp:
        return False
    if cp.creating_waypoint:
        cp.stop_waypoint_creation()
        print("Waypoint creation cancelled")
        return True
    if cp.creating_destination:
        cp.stop_destination_creation()
        print("Destination placement cancelled")
        return True
    if cp.placing_trigger:
        cp.stop_trigger_placement()
        print("Trigger placement cancelled")
        return True
    if cp.pending_personal_trigger:
        cp.cancel_personal_trigger_placement()
        print("Personal trigger placement cancelled")
        return True
    if getattr(editor, '_shift_press', None) is not None:
        # A held Shift+press is a pending gesture like an armed drag: Esc /
        # right-tap aborts it, so neither the deferred tap-toggle nor a later
        # drag promotion can fire after the cancel.
        editor._shift_press = None
        print("Shift press cancelled")
        return True
    if cp.cancel_active_drag():
        # Restores the recorded start state of whichever gesture was active;
        # selection is kept and no undo entry is pushed.
        print("Drag cancelled")
        return True
    if cp.selected_actor_ids or cp.selected_waypoint_group:
        cp.selected_actor_ids = set()
        cp.selected_waypoint_group = None
        if editor.info_panel.visible and editor.info_panel.object_type == 'waypoint_group':
            editor.info_panel.hide()
        print("Group selection cleared")
        return True
    if cp.selected_vehicle:
        cp.clear_vehicle_selection()
        print("Vehicle selection cleared")
        return True
    if cp.selected_traffic_light_group:
        # Empty clicks no longer deselect a traffic-light group, so Esc /
        # right-click is its deselection path.
        cp.clear_traffic_light_selection()
        print("Traffic light selection cleared")
        return True
    if cp.selected_trigger_index is not None:
        cp.selected_trigger_index = None
        cp.trigger_action_menu_position = None
        print("Trigger selection cleared")
        return True
    if cp.waypoint_display_vehicle_id:
        # A route can stay displayed without its actor being selected (e.g.
        # after box-selecting a single waypoint); empty clicks keep it, so
        # this rung is its deselection path.
        cp.selected_waypoint_vehicle_id = None
        cp.selected_waypoint_index = None
        cp.waypoint_display_vehicle_id = None
        if editor.info_panel.visible and editor.info_panel.object_type in ('waypoint', 'waypoint_group'):
            editor.info_panel.hide()
        print("Waypoint display cleared")
        return True
    return False

def _begin_mouse_pan(editor, pos, button):
    """Start a mouse-drag pan for the given button."""
    editor.mouse_dragging = True
    editor.mouse_drag_button = button
    editor.last_mouse_pos = pos
    if editor.camera_processor:
        editor.camera_processor.begin_manual_camera_free_look("mouse_pan")

def _end_mouse_pan(editor):
    """Finish a mouse-drag pan and restore camera state."""
    editor.mouse_dragging = False
    editor.mouse_drag_button = None
    if editor.camera_panned_with_mouse and editor.camera_controller and editor.world:
        # Prototype: camera navigation performs no terrain raycasts.
        editor.camera_panned_with_mouse = False
        if editor.camera_processor:
            editor.camera_processor.restore_vehicle_menu_after_camera_pan()
    if editor.camera_processor:
        editor.camera_processor.end_manual_camera_free_look("mouse_pan")

def _handle_right_click_down(editor, event):
    """Handle right mouse button down.

    Right-drag pans the camera (with or without Shift); a right TAP (release
    within the drag threshold) acts like Esc-while-editing on release —
    unless Shift is held: shift-taps are inert so an additive-selection
    session can't be nuked by an accidental right click.
    """
    editor._right_click_press_pos = event.pos
    editor._begin_mouse_pan(event.pos, event.button)
    return True

def _is_locked_view_camera_active(editor) -> bool:
    """True while the playback camera is in a locked view (chase or cockpit) tied to the ego."""
    cp = editor.camera_processor
    return bool(cp and getattr(cp, "playback_camera_mode", "topdown") in ("chase", "cockpit"))

def _handle_mouse_button_down(editor, event, scenario_running):
    """Dispatch mouse button down events."""
    # Chase/cockpit lock the view to the ego: disable mouse pan / free-look (middle & right drag,
    # shift+right click-to-pan) so it can't yank the camera back to a free top-down pan. Top-down
    # still pans normally; use C to swap modes, and the Stop button (a left-click) still works.
    if editor._is_locked_view_camera_active() and event.button in (2, 3):
        return True
    if event.button == 1:
        return editor._handle_left_click(event.pos, scenario_running)
    if event.button == 2:
        editor._begin_mouse_pan(event.pos, event.button)
        return True
    if event.button == 3:
        return editor._handle_right_click_down(event)
    return False

def _handle_mouse_button_up(editor, event):
    """Handle mouse button releases."""
    if event.button == 1:
        if editor.weather_window and editor.weather_window.alive():
            try:
                editor.weather_window._commit_drag_if_needed()
            except Exception:
                pass
        pending_icon = getattr(editor, 'pending_menu_icon', None)
        editor.pending_menu_icon = None
        marquee_rect = getattr(editor, 'marquee_rect', None)
        editor.marquee_rect = None
        editor._marquee_press_pos = None
        editor._orbit_left_press_pos = None
        editor._orbit_press_was_armed = False
        # A surviving _shift_press means the press never crossed the drag
        # threshold (promotion clears it): the release is a Shift+tap.
        shift_press = getattr(editor, '_shift_press', None)
        editor._shift_press = None
        if pending_icon is not None and editor.camera_processor:
            _dispatch_pending_menu_icon(editor, pending_icon, event.pos)
        elif shift_press is not None and editor.camera_processor:
            _finalize_shift_tap(editor, shift_press)
        elif marquee_rect is not None and editor.camera_processor:
            _finalize_marquee(editor, marquee_rect)
        elif editor.camera_processor and editor.camera_processor.rotating_vehicle:
            editor.camera_processor.stop_vehicle_rotation()
        elif editor.camera_processor and editor.camera_processor.moving_vehicle:
            editor.camera_processor.stop_vehicle_movement()
        elif editor.camera_processor and editor.camera_processor.moving_waypoint:
            editor.camera_processor.stop_waypoint_movement()
        elif editor.camera_processor and editor.camera_processor.moving_trigger:
            editor.camera_processor.stop_trigger_movement()
        elif editor.camera_processor and editor.camera_processor.moving_personal_trigger:
            editor.camera_processor.stop_personal_trigger_movement()
        elif editor.camera_processor and editor.camera_processor.scaling_traffic_light_trigger:
            editor.camera_processor.stop_traffic_light_trigger_scaling()
        elif editor.camera_processor and editor.camera_processor.scaling_pedestrian_trigger:
            editor.camera_processor.stop_pedestrian_trigger_scaling()
        elif editor.camera_processor and editor.camera_processor.scaling_vehicle_trigger:
            editor.camera_processor.stop_vehicle_trigger_scaling()
        elif editor.camera_processor and editor.camera_processor.scaling_trigger:
            editor.camera_processor.stop_trigger_scaling()
        elif editor.camera_processor:
            editor.camera_processor.handle_waypoint_mouse_release()
        if editor.camera_processor:
            # Plain click (never crossed the drag threshold): disarm.
            editor.camera_processor.vehicle_drag_armed = False
            editor.camera_processor.trigger_drag_armed = False
            editor.camera_processor.personal_trigger_drag_armed = False
    elif event.button == 2:
        if editor.mouse_dragging and editor.mouse_drag_button == event.button:
            editor._end_mouse_pan()
    elif event.button == 3:
        # press_pos survives only if the press never left the drag threshold
        # (the motion handler clears it the moment the drag becomes a pan).
        press_pos = getattr(editor, '_right_click_press_pos', None)
        editor._right_click_press_pos = None
        if editor.mouse_dragging and editor.mouse_drag_button == event.button:
            editor._end_mouse_pan()
        if (press_pos is not None and not editor.scenario_running and
                not is_shift_pressed(pygame.key.get_pressed())):
            # Right-tap = one Esc-style cancel step (inert while Shift is
            # held: release Shift to cancel/deselect).
            _cancel_editing_step(editor)

def _handle_mouse_motion(editor, event):
    """Handle mouse motion. Returns True if the camera moved."""
    camera_moved = False
    if editor.info_panel.visible and editor.info_panel._is_point_in_panel(event.pos):
        # The panel eats motion only while no press-drag gesture is live —
        # otherwise a drag would stall the moment the cursor crosses the
        # docked panel (and an actor under the panel could never be moved).
        cp = editor.camera_processor
        drag_active = bool(cp and (
            cp.moving_vehicle or cp.vehicle_drag_armed or cp.rotating_vehicle or
            cp.moving_waypoint or cp.waypoint_drag_armed or
            cp.moving_trigger or cp.trigger_drag_armed or cp.scaling_trigger or
            cp.moving_personal_trigger or cp.personal_trigger_drag_armed or
            cp.scaling_pedestrian_trigger or cp.scaling_vehicle_trigger or
            cp.scaling_traffic_light_trigger)) or getattr(editor, '_shift_press', None) is not None
        if not drag_active:
            return False

    buttons = pygame.mouse.get_pressed()
    if buttons[0] and editor.camera_processor and _camera_in_orbit(editor):
        # 3D view is view + select: a held left button must never promote into a
        # move/rotate/scale/marquee gesture. Disarm anything a click-select armed
        # (selection itself already happened on the press) and hint once the press
        # clearly became a drag attempt.
        cp = editor.camera_processor
        # Latch whether the press had actually armed an edit gesture BEFORE disarming:
        # the flags are cleared on the first motion event, but the hint threshold is
        # only crossed later. A UI or empty-ground drag never arms, so it never hints.
        if (cp.vehicle_drag_armed or cp.waypoint_drag_armed or
                cp.trigger_drag_armed or cp.personal_trigger_drag_armed or
                getattr(editor, '_shift_press', None) is not None):
            editor._orbit_press_was_armed = True
        cp.vehicle_drag_armed = False
        cp.waypoint_drag_armed = False
        cp.trigger_drag_armed = False
        cp.personal_trigger_drag_armed = False
        editor._shift_press = None
        editor._marquee_press_pos = None
        editor.marquee_rect = None
        press = getattr(editor, '_orbit_left_press_pos', None)
        if press is not None and getattr(editor, '_orbit_press_was_armed', False):
            ddx = event.pos[0] - press[0]
            ddy = event.pos[1] - press[1]
            if (ddx * ddx + ddy * ddy) ** 0.5 > cp.object_drag_threshold:
                editor._orbit_left_press_pos = None  # one hint per press
                editor._orbit_press_was_armed = False
                editor.show_status_hint("3D view is view & select only - press Tab for top-down to edit")
        return False
    if (editor.camera_processor and editor.camera_processor.rotating_vehicle and buttons[0]):
        editor.camera_processor.update_vehicle_rotation(event.pos[1])
        if (editor.info_panel.visible and editor.info_panel.object_type in ('vehicle', 'pedestrian') and
                editor.info_panel.selected_object == editor.camera_processor.selected_vehicle):
            editor.info_panel._refresh_fields()
    elif (editor.camera_processor and buttons[0] and
          (editor.camera_processor.moving_vehicle or editor.camera_processor.vehicle_drag_armed)):
        editor.camera_processor.update_vehicle_movement(event.pos)
        if (editor.info_panel.visible and editor.info_panel.object_type in ('vehicle', 'pedestrian') and
                editor.info_panel.selected_object == editor.camera_processor.selected_vehicle):
            editor.info_panel._refresh_fields()
    elif (editor.camera_processor and buttons[0] and
          not editor.camera_processor.creating_waypoint and
          (editor.camera_processor.moving_waypoint or editor.camera_processor.waypoint_drag_armed)):
        editor.camera_processor.update_waypoint_movement(event.pos)
        if (editor.info_panel.visible and editor.info_panel.object_type == 'waypoint' and
                editor.camera_processor.selected_waypoint_vehicle_id is not None and
                editor.camera_processor.selected_waypoint_index is not None):
            editor.info_panel._refresh_fields()
    elif (editor.camera_processor and buttons[0] and
          (editor.camera_processor.moving_trigger or editor.camera_processor.trigger_drag_armed)):
        editor.camera_processor.update_trigger_movement(event.pos)
    elif (editor.camera_processor and buttons[0] and
          (editor.camera_processor.moving_personal_trigger or
           editor.camera_processor.personal_trigger_drag_armed)):
        editor.camera_processor.update_personal_trigger_movement(event.pos)
    elif (editor.camera_processor and editor.camera_processor.scaling_traffic_light_trigger and buttons[0]):
        editor.camera_processor.update_traffic_light_trigger_scaling(event.pos)
        # Refresh info panel to update trigger radius display in real-time
        if (editor.info_panel.visible and editor.info_panel.object_type == 'traffic_light'):
            panel_group = editor.info_panel.selected_object
            scaling_group = editor.camera_processor._traffic_light_scaling_group
            if panel_group and scaling_group and panel_group.ids == scaling_group.ids:
                editor.info_panel._refresh_fields()
    elif (editor.camera_processor and editor.camera_processor.scaling_pedestrian_trigger and buttons[0]):
        editor.camera_processor.update_pedestrian_trigger_scaling(event.pos)
        # Refresh info panel to update trigger radius display in real-time
        if (editor.info_panel.visible and editor.info_panel.object_type == 'pedestrian'):
            panel_actor = editor.info_panel.selected_object
            if panel_actor and panel_actor.id == editor.camera_processor._pedestrian_scaling_id:
                editor.info_panel._refresh_fields()
    elif (editor.camera_processor and editor.camera_processor.scaling_vehicle_trigger and buttons[0]):
        editor.camera_processor.update_vehicle_trigger_scaling(event.pos)
        if (editor.info_panel.visible and editor.info_panel.object_type == 'vehicle'):
            panel_actor = editor.info_panel.selected_object
            if panel_actor and panel_actor.id == editor.camera_processor._vehicle_scaling_id:
                editor.info_panel._refresh_fields()
    elif (editor.camera_processor and editor.camera_processor.scaling_trigger and buttons[0]):
        editor.camera_processor.update_trigger_scaling(event.pos)
    elif buttons[0] and getattr(editor, '_shift_press', None) and editor.camera_processor:
        # Shift+press on an object: past the drag threshold it promotes to a
        # free-move drag (Shift is held, so promotion picks no-snap); below
        # it the press stays a tap that toggles group membership on mouse-up.
        press = editor._shift_press
        drag_dx = event.pos[0] - press['pos'][0]
        drag_dy = event.pos[1] - press['pos'][1]
        if (drag_dx * drag_dx + drag_dy * drag_dy) ** 0.5 > editor.camera_processor.object_drag_threshold:
            editor._shift_press = None
            _promote_shift_press(editor, press, event.pos)
    elif buttons[0] and getattr(editor, '_marquee_press_pos', None) and editor.camera_processor:
        # Marquee (group selection): a left-drag that started on empty ground
        # becomes a rubber-band rectangle past the drag threshold.
        press = editor._marquee_press_pos
        drag_dx = event.pos[0] - press[0]
        drag_dy = event.pos[1] - press[1]
        if (editor.marquee_rect is not None or
                (drag_dx * drag_dx + drag_dy * drag_dy) ** 0.5 > editor.camera_processor.object_drag_threshold):
            editor.marquee_rect = (
                min(press[0], event.pos[0]),
                min(press[1], event.pos[1]),
                abs(drag_dx),
                abs(drag_dy),
            )
    elif editor.mouse_dragging and editor.camera_controller:
        dx = event.pos[0] - editor.last_mouse_pos[0]
        dy = event.pos[1] - editor.last_mouse_pos[1]

        # Right-drag pans, but a small wiggle stays a "tap" (release = one
        # Esc-style cancel step): hold panning until the press travels past
        # the drag threshold, then commit to the pan and kill the tap.
        allow_pan = True
        press_pos = getattr(editor, '_right_click_press_pos', None)
        if editor.mouse_drag_button == 3 and press_pos is not None:
            threshold = editor.camera_processor.object_drag_threshold if editor.camera_processor else 8
            total_dx = event.pos[0] - press_pos[0]
            total_dy = event.pos[1] - press_pos[1]
            if (total_dx * total_dx + total_dy * total_dy) ** 0.5 <= threshold:
                allow_pan = False
            else:
                editor._right_click_press_pos = None  # it's a pan, not a tap

        if allow_pan:
            if (dx != 0 or dy != 0) and editor.camera_processor:
                editor.camera_processor.suppress_vehicle_menu_for_camera_pan()
            cam = editor.camera_controller
            if (getattr(cam, "view_mode", "topdown") == "orbit"
                    and editor.mouse_drag_button == 3
                    and not is_shift_pressed(pygame.key.get_pressed())):
                # 3D view: right-drag orbits around the pivot. Shift+right keeps its
                # step-11 "plain pan" meaning, and middle-drag pans the pivot below.
                sensitivity = cam.ORBIT_DEG_PER_PIXEL
                cam.orbit(dx * sensitivity, dy * sensitivity)
            else:
                base_scale = 0.1
                navigation_height = cam.get_navigation_height()
                height_scale = navigation_height / 400.0
                scale = base_scale * height_scale
                world_dx = -dx * scale
                world_dy = -dy * scale
                cam.pan(world_dx, world_dy, 1.0)
            camera_moved = True
            editor.camera_panned_with_mouse = True

        editor.last_mouse_pos = event.pos
    return camera_moved

def _handle_mouse_wheel(editor, event, scenario_running):
    """Handle mouse wheel scrolling. Returns True if the camera moved."""
    active_menu = editor.get_active_selection_menu() if not scenario_running else None
    if active_menu and active_menu.handle_scroll(pygame.mouse.get_pos(), event.y):
        return False
    if editor.map_menu_visible and editor.map_menu and editor.map_menu.handle_scroll(
            pygame.mouse.get_pos(), event.y):
        return False
    if editor.info_panel.visible and editor.info_panel.handle_scroll(
            pygame.mouse.get_pos(), event.y):
        return False
    # Chase/cockpit lock the view to the ego (the camera is engine-attached). Zooming the free
    # camera_controller there fights the attach and breaks the view, so only allow zoom in top-down.
    if editor._is_locked_view_camera_active():
        return False
    if editor.camera_controller:
        if editor.camera_processor:
            editor.camera_processor.suppress_vehicle_menu_for_camera_pan()
        zoom_delta = -event.y
        editor.camera_controller.zoom(zoom_delta)
        if editor.camera_processor:
            editor.camera_processor.update_camera_position()
            editor.camera_processor.notify_manual_camera_adjustment()
            editor.camera_processor.restore_vehicle_menu_after_camera_pan()
        return True
    return False

def _handle_resize(editor, event):
    """Handle window resize events."""
    new_width = event.w
    new_height = event.h
    if (abs(editor.screen_width - new_width) > 10 or abs(editor.screen_height - new_height) > 10):
        editor.screen_width = new_width
        editor.screen_height = new_height
        if not editor.maximized:
            editor.windowed_width = new_width
            editor.windowed_height = new_height
        if editor.camera_processor:
            editor.camera_processor.update_screen_size(editor.screen_width, editor.screen_height)
        print(f"Window resized to: {editor.screen_width}x{editor.screen_height}")
        if editor.ui_manager:
            editor.ui_manager.set_window_resolution((editor.screen_width, editor.screen_height))
        if editor.tooltip_manager:
            editor.tooltip_manager.update_screen_size(editor.screen_width, editor.screen_height)

def calculate_acceleration_factor(editor, hold_time):
    """Calculate acceleration factor based on how long key has been held"""
    if hold_time <= 0:
        return 0.1
    
    normalized_time = min(hold_time / editor.max_acceleration_time, 1.0)
    acceleration_factor = pow(normalized_time, 1.0 / editor.acceleration_curve)
    return 0.1 + (0.9 * acceleration_factor)

def show_exit_confirmation(editor, *, window_title="Exit Confirmation",
                           action_long_desc=None) -> str:
    """Show a save-confirmation dialog. Returns ``"save"``/``"discard"``/``"cancel"``.

    If there are no unsaved changes the dialog is skipped and ``"discard"``
    (i.e. proceed) is returned immediately. Callers may override the title and
    body text (e.g. the crash-restart flow reuses this with restart wording).
    """
    if not editor._has_unsaved_changes():
        return "discard"

    if action_long_desc is None:
        action_long_desc = (
            "Do you want to save your changes before exiting?<br><br>"
            "Your unsaved changes will be lost if you don't save them."
        )

    editor._close_all_dropdowns(keep_info_panel=True)
    dialog_rect = editor._center_dialog_rect(460, 220)
    dialog = UISaveConfirmationDialog(
        rect=dialog_rect,
        action_long_desc=action_long_desc,
        manager=editor.ui_manager,
        window_title=window_title,
        blocking=True,
    )

    result = editor._run_modal_window(dialog, editor._unsaved_changes_handler(dialog))
    return result or "cancel"

def handle_events(editor, dt):
    """Handle pygame events for user interaction and camera control."""
    blocking_window_alive = bool(editor.result_window and editor.result_window.alive())

    if not editor.ready:
        for event in pygame.event.get():
            if editor.ui_manager:
                editor.ui_manager.process_events(event)

            if event.type == pygame.QUIT:
                if not editor._handle_quit_event():
                    return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if not editor._handle_quit_event():
                        return False
                elif event.key == pygame.K_F11:
                    editor.toggle_maximize()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if editor.start_screen_active:
                    editor._handle_start_screen_click(event.pos)
            elif event.type == pygame.VIDEORESIZE:
                editor._handle_resize(event)
        return True

    camera_moved = False
    scenario_running = bool(editor.scenario_running or (editor.scenario_process and editor.scenario_process.poll() is None))

    for event in pygame.event.get():
        if editor.ui_manager:
            editor.ui_manager.process_events(event)

        if event.type == pygame.QUIT:
            if not editor._handle_quit_event():
                return False
            continue

        if blocking_window_alive:
            if event.type == pygame.VIDEORESIZE:
                editor._handle_resize(event)
            # Swallow all other events while the blocking result window is open.
            continue

        if event.type == pygame.KEYDOWN:
            result = editor._handle_keydown(event, scenario_running)
            if result is False:
                return False
            if result:
                continue

        if event.type == pygame.KEYUP:
            editor._handle_keyup(event)
            continue

        if event.type == pygame.MOUSEBUTTONDOWN:
            if editor._handle_mouse_button_down(event, scenario_running):
                continue

        if event.type == pygame.MOUSEBUTTONUP:
            editor._handle_mouse_button_up(event)
            continue

        if event.type == pygame.MOUSEMOTION:
            if editor._handle_mouse_motion(event):
                camera_moved = True
            continue

        if event.type == pygame.MOUSEWHEEL:
            if editor._handle_mouse_wheel(event, scenario_running):
                camera_moved = True
            continue

        if event.type == pygame.VIDEORESIZE:
            editor._handle_resize(event)
            continue

    for key in editor.key_hold_times:
        editor.key_hold_times[key] += dt

    if editor.camera_controller and not editor._is_locked_view_camera_active():
        navigation_height = editor.camera_controller.get_navigation_height()
        height_scale = navigation_height / 400.0
        base_keyboard_speed = 30.0 * height_scale
        arrow_allowed = not (editor.camera_processor and editor.camera_processor.manual_control_armed)
        up_pressed = pygame.K_UP in editor.keys_pressed if arrow_allowed else False
        down_pressed = pygame.K_DOWN in editor.keys_pressed if arrow_allowed else False
        left_pressed = pygame.K_LEFT in editor.keys_pressed if arrow_allowed else False
        right_pressed = pygame.K_RIGHT in editor.keys_pressed if arrow_allowed else False

        if pygame.K_w in editor.keys_pressed or up_pressed:
            hold_time = editor.key_hold_times.get(pygame.K_w, 0.0)
            if up_pressed:
                hold_time = max(hold_time, editor.key_hold_times.get(pygame.K_UP, 0.0))
            acceleration_factor = editor.calculate_acceleration_factor(hold_time)
            speed = base_keyboard_speed * acceleration_factor
            if editor.camera_processor:
                editor.camera_processor.suppress_vehicle_menu_for_camera_pan()
                editor.camera_processor.begin_manual_camera_free_look("keyboard_pan")
            editor.camera_controller.pan(0, -speed, dt)
            camera_moved = True
        if pygame.K_s in editor.keys_pressed or down_pressed:
            keys = pygame.key.get_pressed()
            ctrl_pressed = is_ctrl_pressed(keys)
            if not ctrl_pressed:
                hold_time = editor.key_hold_times.get(pygame.K_s, 0.0)
                if down_pressed:
                    hold_time = max(hold_time, editor.key_hold_times.get(pygame.K_DOWN, 0.0))
                acceleration_factor = editor.calculate_acceleration_factor(hold_time)
                speed = base_keyboard_speed * acceleration_factor
                if editor.camera_processor:
                    editor.camera_processor.suppress_vehicle_menu_for_camera_pan()
                    editor.camera_processor.begin_manual_camera_free_look("keyboard_pan")
                editor.camera_controller.pan(0, speed, dt)
                camera_moved = True
        if pygame.K_a in editor.keys_pressed or left_pressed:
            hold_time = editor.key_hold_times.get(pygame.K_a, 0.0)
            if left_pressed:
                hold_time = max(hold_time, editor.key_hold_times.get(pygame.K_LEFT, 0.0))
            acceleration_factor = editor.calculate_acceleration_factor(hold_time)
            speed = base_keyboard_speed * acceleration_factor
            if editor.camera_processor:
                editor.camera_processor.suppress_vehicle_menu_for_camera_pan()
                editor.camera_processor.begin_manual_camera_free_look("keyboard_pan")
            editor.camera_controller.pan(-speed, 0, dt)
            camera_moved = True
        if pygame.K_d in editor.keys_pressed or right_pressed:
            hold_time = editor.key_hold_times.get(pygame.K_d, 0.0)
            if right_pressed:
                hold_time = max(hold_time, editor.key_hold_times.get(pygame.K_RIGHT, 0.0))
            acceleration_factor = editor.calculate_acceleration_factor(hold_time)
            speed = base_keyboard_speed * acceleration_factor
            if editor.camera_processor:
                editor.camera_processor.suppress_vehicle_menu_for_camera_pan()
                editor.camera_processor.begin_manual_camera_free_look("keyboard_pan")
            editor.camera_controller.pan(speed, 0, dt)
            camera_moved = True

    if camera_moved and editor.camera_processor:
        editor.camera_processor.update_camera_position()
        editor.camera_processor.notify_manual_camera_adjustment()

    return True
