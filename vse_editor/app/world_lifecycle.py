"""World lifecycle: world-reset detection/recovery, world-settings changes,
external-ego adoption/swap/monitoring and the manual tick pump (duck-typed
module functions, first arg `editor`). Moved verbatim from vse.py (step-37,
Phase 7).

Thread contract unchanged: bodies run on the main thread, the CARLA client
callback thread (_on_world_tick), the VSEWorldResetWait thread, the startup
thread and the restart thread exactly as before the move; flag-write ordering
(e.g. _world_reset_ready_world BEFORE _world_reset_finalize_pending) is
load-bearing. Locks stay on the editor instance (_world_reset_lock).
"""

import json
import math
import os
import threading
import time

from typing import Optional, Tuple

import carla

from vse_editor.carla_io.camera_stream import CameraImageProcessor
from vse_editor.rendering.overlays import OpenDriveOverlayRenderer
from vse_editor.ui.info_panel import InfoPanel


def _capture_world_settings_signature(editor, settings) -> Optional[Tuple[bool, bool, Optional[float], bool]]:
    """Return a tuple describing settings that matter to the editor for change detection."""
    if not settings:
        return None
    return (
        getattr(settings, "synchronous_mode", False),
        getattr(settings, "synchronous_mode_wait_for_vehicle_control_command", False),
        getattr(settings, "fixed_delta_seconds", None),
        getattr(settings, "no_rendering_mode", False),
    )

def _apply_rendering_mode(
    editor,
    desired_enabled: Optional[bool] = None,
    *,
    reason: Optional[str] = None,
    force: bool = False,
    world=None,
) -> bool:
    """
    Apply the preferred rendering mode to the active CARLA world.

    Args:
        desired_enabled: When provided, the user preference to enforce (True enables scene rendering).
        reason: Optional context string used in log output.
        force: When True, skip current-state short-circuit checks.
        world: Explicit world reference to update (defaults to self.world).

    Returns:
        True when the change was applied or already reflected in the world.
    """
    if desired_enabled is not None:
        editor._rendering_desired_enabled = bool(desired_enabled)
    desired_enabled = bool(editor._rendering_desired_enabled)

    target_world = world or editor.world
    if not target_world:
        return False

    try:
        settings = target_world.get_settings()
    except Exception as exc:
        print(f"[Rendering] Unable to read world settings: {exc}")
        return False

    desired_no_render = not desired_enabled
    current_no_render = getattr(settings, "no_rendering_mode", False)

    if not force and current_no_render == desired_no_render:
        editor.rendering_enabled = desired_enabled
        editor._world_settings_signature = editor._capture_world_settings_signature(settings)
        return True

    settings.no_rendering_mode = desired_no_render
    no_render_active = not desired_enabled
    state_label = "enabled" if no_render_active else "disabled"
    suffix = f" ({reason})" if reason else ""

    try:
        editor._rendering_apply_in_progress = True
        target_world.apply_settings(settings)
        print(f"[Rendering] No-rendering mode {state_label}{suffix}.")
    except Exception as exc:
        print(f"[Rendering] Failed to set no-rendering mode to {state_label}: {exc}")
        try:
            refreshed = target_world.get_settings()
            actual_no_render = getattr(refreshed, "no_rendering_mode", False)
            editor.rendering_enabled = not actual_no_render
            editor._rendering_desired_enabled = editor.rendering_enabled
            editor._world_settings_signature = editor._capture_world_settings_signature(refreshed)
        except Exception:
            pass
        return False
    finally:
        editor._rendering_apply_in_progress = False

    editor.rendering_enabled = desired_enabled
    editor._rendering_desired_enabled = desired_enabled
    editor._world_settings_signature = editor._capture_world_settings_signature(settings)
    return True

def _watchdog_client_dead(editor):
    """Remote-server liveness: version-ping on a DEDICATED carla client.

    Used only when there is no local server PID to check. A dedicated client
    (not editor.client, which the MiniRunner thread shares during playback)
    performs a real RPC handshake — safe for the server, unlike a raw TCP
    connect (see _server_confirmed_dead). Pings are throttled to 1/s and
    _server_dead_after_failures consecutive failures are required, so remote
    lag never one-shots a death verdict.
    """
    now = time.monotonic()
    if now - getattr(editor, '_watchdog_ping_last', 0.0) < 1.0:
        # No new information between pings: keep the current verdict.
        return editor._server_lost
    editor._watchdog_ping_last = now
    client = getattr(editor, '_watchdog_client', None)
    if client is None:
        try:
            profile = editor.connection_profile
            client = carla.Client(getattr(profile, 'host', '127.0.0.1') or '127.0.0.1',
                                  int(getattr(profile, 'port', 2000) or 2000))
            client.set_timeout(2.0)
            editor._watchdog_client = client
        except Exception:
            return False
    try:
        client.get_server_version()
        editor._watchdog_ping_failures = 0
        return False
    except Exception:
        failures = getattr(editor, '_watchdog_ping_failures', 0) + 1
        editor._watchdog_ping_failures = failures
        return failures >= editor._server_dead_after_failures


def _server_confirmed_dead(editor):
    """Definitive server-death check that CANNOT harm or misread the server.

    Local managed/adopted servers are checked by PROCESS liveness — zero
    interaction with CARLA:
      - a server we launched: Popen.poll() (also reaps the zombie, so a dead
        child can't read as alive);
      - a server we adopted by pid: os.kill(pid, 0) (ProcessLookupError = gone).
    A process check can't false-positive under load (the process exists no
    matter how busy it is) and detects a segfault within one frame.

    Raw TCP port probes are deliberately NOT used: CARLA's RPC plugin treats an
    accepted-then-aborted connection as an internal exception and
    carla::throw_exception KILLS the whole server (LowLevelFatalError,
    Carla.cpp:136) — the every-frame TCP probe of an earlier revision was
    itself crashing the server seconds after startup (proven from crash dumps
    2026-07-07). Same reason port-scanners crash CARLA simulators.

    Remote servers (no local pid) fall back to a throttled version-ping on a
    dedicated client (_watchdog_client_dead) — a real handshake, harmless.
    """
    manager = getattr(editor, 'server_manager', None)
    process = getattr(manager, 'process', None) if manager else None
    if process is not None:
        return process.poll() is not None
    pid = getattr(manager, 'known_server_pid', None) if manager else None
    if pid:
        try:
            os.kill(int(pid), 0)
            return False
        except ProcessLookupError:
            return True
        except OSError:
            return False  # e.g. EPERM: process exists but isn't ours
    return _watchdog_client_dead(editor)


def _check_world_replacement(editor):
    """Poll CARLA client to detect world reloads when tick callbacks are lost.

    Phase 8 (fix-05): the poll is throttled to _server_poll_interval_s (was
    every frame = ~120 RPC/s idle) and doubles as the server-liveness
    watchdog. While the server is lost the poll drops to
    _server_retry_interval_s and auto-clears the lost state when the server
    answers again (a fresh server on the same port then flows into the
    existing world-reset recovery via the world-id change below).

    Death is decided SOLELY by the TCP port being closed (confirmed across a
    few in-frame probes) — never by an RPC timeout. A live server keeps its
    listening port open regardless of how busy/laggy it is (the kernel
    completes the handshake even when CARLA's threads are saturated), so a
    slow-but-alive server is NEVER falsely declared dead; only a process that
    has actually exited (segfault/kill) closes the port. RPC failures on an
    open port are treated as transient lag and simply skip the poll.
    """
    if editor._world_reset_in_progress:
        return
    if not editor.client:
        return

    now = time.monotonic()

    # --- Liveness: probed EVERY frame (not throttled) ---------------------
    # During in-editor playback the MiniRunner thread shares editor.client; a
    # dead-server RPC there blocks the shared client for the full timeout, and
    # any main-thread per-frame RPC (camera update / overlay render) then
    # blocks behind it. If the liveness probe were throttled, the server could
    # die between two probes and the main thread would stall ~10 s per blocked
    # RPC before detection (the ~20 s the user saw). The probe is a cheap
    # localhost TCP connect (µs when the port is open — a single probe per
    # frame while healthy), so running it every frame flips _server_lost at the
    # TOP of the frame the server died, before those RPCs run and while the
    # not-_server_lost gates skip them. Suspended during map/world switches
    # (the port stays open through a load anyway).
    if not (editor.pending_map_switch or editor._world_reset_in_progress):
        if not editor._server_lost:
            if _server_confirmed_dead(editor):
                editor._server_lost = True
                print("[Server Watchdog] CARLA server unreachable")
                return
        elif now - editor._server_poll_last >= editor._server_retry_interval_s:
            # Already lost: retry on the slow cadence to detect recovery.
            editor._server_poll_last = now
            if _server_confirmed_dead(editor):
                return
            editor._server_lost = False
            print("[Server Watchdog] CARLA server reachable again")
        else:
            return  # still lost, not time to retry

    # --- World-reset detection RPCs: throttled ----------------------------
    if now - editor._server_rpc_last < editor._server_poll_interval_s:
        return
    editor._server_rpc_last = now

    # The port is open. get_world()/get_settings() below may still time out
    # under heavy lag — that is NOT death (the port proved the server alive),
    # so on any RPC failure just skip this poll and try again next tick.
    try:
        current_world = editor.client.get_world()
    except Exception:
        return

    if current_world is None:
        return

    # carla.Timestamp has no "episode" attribute (CARLA 0.9.15), so the old episode-based
    # detector never fired. world.id changes whenever the world is reloaded, so we use it
    # to detect externally-initiated reloads (e.g. by the ROS bridge / an external stack).
    try:
        world_id = current_world.id
    except Exception:
        world_id = None

    try:
        settings = current_world.get_settings()
        signature = editor._capture_world_settings_signature(settings)
    except Exception:
        # Lag on an open port — skip; do NOT feed None into the settings-change
        # handler (that would toggle manual-tick state off a transient stall).
        return

    if signature != editor._world_settings_signature:
        editor._world_settings_signature = signature
        editor._handle_world_settings_change(settings)

    trigger_reset = False

    with editor._world_reset_lock:
        if editor.world and current_world is not editor.world:
            editor._world_reset_candidate_world = current_world
        elif editor._world_reset_candidate_world is None and editor.world is None:
            editor._world_reset_candidate_world = current_world

        if world_id is not None:
            if editor._expected_world_id is None:
                # First observation: adopt as baseline (no reload to report yet).
                editor._expected_world_id = world_id
            elif (
                world_id != editor._expected_world_id
                and not editor.pending_map_switch
                and not editor._world_reset_in_progress
            ):
                # A reload we did not initiate (VSE-initiated swaps set pending_map_switch
                # and reboot the process, so they never reach here on a live instance).
                trigger_reset = True

    if trigger_reset and not editor._world_reset_in_progress:
        with editor._world_reset_lock:
            if not editor._world_reset_pending:
                editor._world_reset_pending = True
                editor._world_reset_detected_at = time.time()
                if editor._world_reset_candidate_world is None:
                    editor._world_reset_candidate_world = current_world

def _register_world_tick_handler(editor):
    """Subscribe to CARLA world ticks so we can detect world resets."""
    if not editor.world:
        return
    editor._unregister_world_tick_handler()
    editor._world_tick_times.clear()
    editor._world_tick_fps = 0.0
    # Baseline world.id for the world we are now attached to, so a later externally-initiated
    # reload (different world.id) is detected. Also reset the per-tick snapshot/actor-count
    # signals so the first ego that appears in this world counts as a fresh spawn.
    try:
        editor._expected_world_id = editor.world.id
    except Exception:
        editor._expected_world_id = None
    editor._latest_world_snapshot = None
    editor._last_seen_actor_count = None
    try:
        snapshot = editor.world.get_snapshot()
        editor._world_last_episode = getattr(snapshot.timestamp, "episode", None)
        editor._world_last_frame = snapshot.frame
    except Exception:
        editor._world_last_episode = None
        editor._world_last_frame = None
    try:
        editor._world_tick_subscription = editor.world.on_tick(editor._on_world_tick)
    except Exception as exc:
        print(f"[World Reset] Failed to register on_tick handler: {exc}")
        editor._world_tick_subscription = None

def _handle_world_settings_change(editor, settings):
    """React to runtime changes in CARLA world settings (e.g., synchronous mode)."""
    if not settings:
        return

    actual_rendering = not getattr(settings, "no_rendering_mode", False)
    if actual_rendering != editor.rendering_enabled:
        editor.rendering_enabled = actual_rendering
    if (
        not editor._rendering_apply_in_progress
        and actual_rendering != editor._rendering_desired_enabled
        and editor.world
    ):
        preferred = editor._rendering_desired_enabled
        mode_state = "enabled" if not preferred else "disabled"
        print(f"[Rendering] Detected external change; reapplying no-rendering mode {mode_state}.")
        editor._apply_rendering_mode(
            desired_enabled=preferred,
            reason="restoring preference after external change",
            force=True,
            world=editor.world,
        )

    synchronous = getattr(settings, "synchronous_mode", False)
    wait_for_control = getattr(settings, "synchronous_mode_wait_for_vehicle_control_command", False)
    fixed_delta = getattr(settings, "fixed_delta_seconds", None) or 0.05
    if synchronous:
        editor.manual_tick_required = True
        editor.manual_tick_interval = max(fixed_delta, 0.001)
        editor.manual_tick_recommendation = not editor.manual_tick_enabled
        editor._manual_tick_last_required_time = time.time()
        print(
            "[World Settings] Server entered synchronous mode"
            f"{' (waiting for external control commands)' if wait_for_control else ''}."
        )
        editor._map_refresh_disabled = True
        if editor.camera_processor:
            editor.camera_processor.restart_camera_sensor()

        # On large maps, save ego data from scenario for applying to external ego later
        # Note: The ROS bridge reloads the map, destroying VSE's ego actor - so we save from scenario data
        is_large_map = getattr(editor, 'large_map_active', False)
        should_bootstrap = (
            is_large_map
            and editor.external_ego_actor is None
            and not editor.scenario_running  # Don't bootstrap during playback
            and not getattr(editor, '_scenario_run_in_progress', False)  # Don't bootstrap during run setup
        )
        if should_bootstrap:
            cp = editor.camera_processor
            # Get ego data from the loaded scenario (survives map reload by ROS bridge)
            scenario_ego_data = None
            if cp and hasattr(cp, 'loaded_scenario_data') and cp.loaded_scenario_data:
                scenario_ego_data = cp.loaded_scenario_data.get('ego_vehicle')

            if scenario_ego_data:
                editor._pending_external_ego_data = scenario_ego_data
                loc = scenario_ego_data.get('location', {})
                print(f"[Bootstrap] Large map: saved ego position from scenario ({loc.get('x', 0):.1f}, {loc.get('y', 0):.1f}, {loc.get('z', 0):.1f})")

                # Move camera to awmini spawn point to ensure spawned ego is visible
                # On large maps, actors are only loaded near the spectator/camera position
                if editor.camera_processor and hasattr(editor.camera_processor, 'camera_sensor'):
                    # Save current camera transform for restoration later
                    current_transform = editor.camera_processor.camera_sensor.get_transform()
                    editor._bootstrap_original_camera_transform = current_transform

                    # Move to awmini spawn point (0, 0, 36) + some height for better visibility
                    spawn_location = carla.Location(x=0.0, y=0.0, z=50.0)
                    spawn_rotation = carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)
                    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
                    editor.camera_processor.camera_sensor.set_transform(spawn_transform)
                    print("[Bootstrap] Moved camera to spawn point (0, 0, 50) for external ego detection")
            else:
                print("[Bootstrap] Large map sync mode - no scenario ego data found")

            # Enable ticking immediately so awmini can spawn ego vehicle
            editor._set_large_map_bootstrap_ticking(
                True,
                log_message="[Bootstrap] Enabled Drive Clock for large map bootstrap",
            )

        editor._detect_external_ego_vehicle()
    else:
        if editor.manual_tick_required:
            print("[World Settings] Server returned to asynchronous mode.")
        editor.manual_tick_required = False
        editor.manual_tick_enabled = False
        editor.manual_tick_accumulator = 0.0
        editor.manual_tick_recommendation = False
        editor._map_refresh_disabled = False
        editor._external_ego_present_last_check = False
        editor._manual_tick_last_required_time = 0.0

def _switch_world_to_async_if_safe(editor, reason: Optional[str] = None, *, force: bool = False) -> bool:
    """Return the CARLA world to asynchronous mode when we can (or when forced)."""
    # Skip on large maps - apply_settings() crashes server when ROS bridge is connected
    is_large_map = getattr(editor, 'large_map_active', False)
    if is_large_map and not force:
        if not getattr(editor, '_large_map_async_skip_warned', False):
            print("[World Settings] Skipping async switch on large map - external bridge controls simulation")
            editor._large_map_async_skip_warned = True
        return False
    if is_large_map and force:
        # Still avoid apply_settings() while an external bridge/ego is present. This mirrors the
        # normal-map behavior ("restore local ego then return to async") but only after the
        # external ego has been removed and VSE is the sole simulation controller again.
        external_present = (
            getattr(editor, "external_ego_actor", None) is not None
            or getattr(editor, "external_ego_actor_id", None) is not None
            or getattr(editor, "_external_swap_active", False)
        )
        if external_present or getattr(editor, "_large_map_bootstrap_ticking", False):
            if not getattr(editor, '_large_map_async_skip_warned', False):
                print("[World Settings] Skipping async switch on large map - external bridge controls simulation")
                editor._large_map_async_skip_warned = True
            return False

    profile = getattr(editor, "connection_profile", None)
    if profile is not None and not profile.manage_server and not force:
        return False

    world = editor._get_current_world()
    if not world:
        return False

    try:
        settings = world.get_settings()
    except Exception as exc:
        print(f"[World Settings] Unable to read current settings for async switch: {exc}")
        return False

    synchronous = getattr(settings, "synchronous_mode", False)
    wait_for_control = getattr(settings, "synchronous_mode_wait_for_vehicle_control_command", False)
    if not synchronous and not wait_for_control:
        return False

    settings.synchronous_mode = False
    settings.synchronous_mode_wait_for_vehicle_control_command = False
    try:
        settings.fixed_delta_seconds = 0.0
    except Exception:
        pass

    try:
        try:
            world.apply_settings(settings, timeout=5.0)  # type: ignore[call-arg]
        except TypeError:
            try:
                world.apply_settings(settings, 5.0)  # type: ignore[misc]
            except TypeError:
                world.apply_settings(settings)
    except Exception as exc:
        print(f"[World Settings] Failed to switch to asynchronous mode: {exc}")
        return False

    message = "[World Settings] Switched to asynchronous mode"
    if reason:
        message += f" ({reason})."
    else:
        message += "."
    print(message)

    # If we had adopted an external ego and the bridge/frames stopped, treat this
    # as a disconnect signal and attempt to restore the local ego immediately.
    if editor._external_swap_active or editor.external_ego_actor_id is not None:
        try:
            editor._handle_external_ego_disconnect()
        except Exception as exc:
            print(f"[External Ego] Restore on async switch failed: {exc}")

    editor._handle_world_settings_change(settings)
    editor._world_settings_signature = editor._capture_world_settings_signature(settings)
    if editor.camera_processor:
        editor.camera_processor.restart_camera_sensor()
    return True

def _set_large_map_bootstrap_ticking(
    editor,
    enabled: bool,
    *,
    log_message: Optional[str] = None,
    disable_manual_tick: bool = False,
    clear_pending_external_ego: bool = False,
) -> None:
    """Toggle large-map bootstrap ticking without altering existing behavior."""
    if enabled:
        editor._large_map_bootstrap_ticking = True
        editor._bootstrap_tick_warned = False
        editor.manual_tick_enabled = True
        editor.manual_tick_accumulator = 0.0
    else:
        editor._large_map_bootstrap_ticking = False
        if disable_manual_tick:
            editor.manual_tick_enabled = False
        if clear_pending_external_ego:
            editor._pending_external_ego_data = None

    if log_message:
        print(log_message)

def _external_ego_can_prompt(editor) -> bool:
    """Return True if it is safe to automatically adopt an external ego."""
    if editor.scenario_running:
        return False
    cp = editor.camera_processor
    if cp and (getattr(cp, "manual_control_enabled", False) or getattr(cp, "manual_control_pending", False)):
        return False
    return True

def _show_external_swap_overlay(editor, message: Optional[str] = None) -> None:
    """Enable a non-blocking on-screen overlay during external ego adoption."""
    now = time.time()
    editor._external_swap_overlay = True
    editor._external_swap_overlay_message = message or "Adopting external ego..."
    editor._external_swap_overlay_started_at = now
    editor._external_swap_overlay_keep_until = now + 1.0  # keep visible briefly even if swap completes fast

def _clear_external_swap_overlay(editor) -> None:
    """Hide the external ego adoption overlay."""
    now = time.time()
    if editor._external_swap_overlay and now < editor._external_swap_overlay_keep_until:
        return
    editor._external_swap_overlay = False
    editor._external_swap_overlay_message = None
    editor._external_swap_overlay_started_at = 0.0
    editor._external_swap_overlay_keep_until = 0.0

def _perform_external_ego_swap(editor, actor: Optional[carla.Actor]) -> bool:
    """Record the external ego actor without replacing the editor placeholder.

    The editor placeholder ego is kept as the authoritative ego for selection and
    saving.  The external actor is only stored as a reference so it can be
    teleported to the placeholder position at scenario-play time.
    """
    cp = editor.camera_processor
    if not cp or not actor or not actor.is_alive:
        return False
    try:
        print(f"[External Ego] Detected external ego actor {actor.id} ({actor.type_id})")
    except Exception:
        pass

    # Store the external actor's current transform so we can return it later.
    try:
        editor._external_swap_last_transform = actor.get_transform()
    except Exception:
        editor._external_swap_last_transform = None
    try:
        editor._external_swap_last_blueprint = actor.type_id
    except Exception:
        editor._external_swap_last_blueprint = None
    try:
        editor._external_swap_last_color = actor.attributes.get('color')
    except Exception:
        editor._external_swap_last_color = None

    # Do NOT add the external actor to spawned_vehicles (keeps it non-clickable).
    # Do NOT destroy or replace the editor placeholder ego.
    # Do NOT register the external actor as the editor ego.

    editor._external_swap_active = True
    editor._external_swap_current_id = actor.id
    editor._external_ego_prompt_pending_id = None
    editor.external_ego_actor = actor
    editor.external_ego_actor_id = actor.id
    if getattr(editor, "large_map_active", False):
        now = time.time()
        editor._large_map_external_ego_connected_at = now
        editor._large_map_external_ego_handoff_until = now + 12.0
        editor._large_map_external_ego_had_frames = False
        editor._large_map_external_stall_warned = False

    # Rendering mode is left untouched on external ego connect: no-rendering
    # stays on by default and only the user can toggle it via the UI.

    print(f"[External Ego] External ego {actor.id} registered (placeholder preserved).")
    return True

def _update_external_swap_state(editor, actor: Optional[carla.Actor]) -> None:
    """Refresh cached state from the external ego while it is active."""
    if not (editor._external_swap_active and actor and actor.is_alive):
        return
    try:
        editor._external_swap_last_transform = actor.get_transform()
    except Exception:
        pass
    if editor._external_swap_last_blueprint is None:
        try:
            editor._external_swap_last_blueprint = actor.type_id
        except Exception:
            pass
    if editor._external_swap_last_color is None:
        try:
            editor._external_swap_last_color = actor.attributes.get('color')
        except Exception:
            pass

def _handle_external_ego_disconnect(editor) -> None:
    """Handle cleanup when an external ego disappears.

    The editor placeholder ego is never destroyed, so no restoration is needed.
    Just clear the external ego references.
    """
    try:
        print("[External Ego] Detected external ego disconnect; clearing references (placeholder preserved).")
    except Exception:
        pass
    # Rendering mode is left untouched on external ego disconnect: it stays
    # whatever the user last set via the UI (no-rendering on by default).
    editor._external_swap_active = False
    editor._external_swap_current_id = None
    editor.external_ego_actor = None
    editor.external_ego_actor_id = None
    editor._external_ego_prompt_pending_id = None
    editor._external_ego_present_last_check = False

def _maybe_prompt_external_ego_swap(editor, actor: Optional[carla.Actor]) -> None:
    """Auto-adopt the external ego when conditions allow."""
    if getattr(editor, "_suppress_external_ego_adoption", False):
        editor._external_ego_prompt_pending_id = None
        return
    if not actor or not actor.is_alive:
        return
    if editor._external_swap_active and editor._external_swap_current_id == actor.id:
        return
    if not editor._external_ego_can_prompt():
        editor._external_ego_prompt_pending_id = actor.id
        return

    editor._external_ego_prompt_pending_id = None
    try:
        print(f"[External Ego] Auto-adopting external ego actor {actor.id}")
    except Exception:
        pass
    # Lock agent mode to custom when external ego is connected
    editor.agent_mode = "custom"
    editor._remember_last_agent()
    editor._perform_external_ego_swap(actor)

def _monitor_external_ego_status(editor):
    """Detect when an external ego actor disappears and revert to async mode."""
    if not editor.ready:
        return
    if editor.scenario_running:
        return

    previously_present = editor._external_ego_present_last_check or (editor.external_ego_actor_id is not None)
    actor = editor._refresh_external_ego_actor_reference()
    currently_present = actor is not None
    now = time.time()

    # Authoritative liveness re-check for a believed-present external ego.
    # _refresh_external_ego_actor_reference()'s cheap fast path trusts the local
    # actor.is_alive flag whenever the tick snapshot is unavailable (None) or stale -- which
    # it can be while idle in async mode, or once an external bridge (e.g. awmini) stops
    # feeding ticks. In that state a removal is never noticed: currently_present stays True,
    # the disconnect branch below never runs (no log line, and Play stays enabled in custom
    # mode).
    #
    # Re-confirm with the SAME full get_actors() role scan that the Play path uses
    # (_prepare_external_ego_for_playback -> _resolve_external_ego_actor(force_scan=True) ->
    # _detect_external_ego_vehicle). A targeted world.get_actor(id) is NOT used here: the
    # editor's long-lived client adopted the ego earlier, so get_actor(id) can hand back a
    # client-side cached actor proxy that still reports is_alive=True for an id the server has
    # already destroyed -- which masks the removal. The full scan reflects server truth, which
    # is why pressing Play already detects "no external ego" while the toolbar stays stale.
    if (
        currently_present
        and editor.external_ego_actor_id is not None
        and (now - editor._last_external_ego_scan_time) >= 3.0
    ):
        rescanned = editor._detect_external_ego_vehicle(silent=True)
        actor = rescanned
        currently_present = rescanned is not None

    # Cheap, no-RPC spawn signal: the latest tick snapshot gives the actor count for free.
    # Only run the heavier get_actors() role scan when the population grew (something
    # spawned) or on a slow fallback. This works in async too (no dependency on sync mode /
    # manual_tick), so any stack that spawns a role_name ego is detected -- with or without
    # a world reload -- without continuous polling overhead.
    snap = editor._latest_world_snapshot
    try:
        actor_count = len(snap) if snap is not None else None
    except Exception:
        actor_count = None
    population_increased = (
        actor_count is not None
        and editor._last_seen_actor_count is not None
        and actor_count > editor._last_seen_actor_count
    )
    if actor_count is not None:
        editor._last_seen_actor_count = actor_count

    fallback_due = (now - editor._last_external_ego_scan_time) >= 3.0
    perform_scan = (
        not currently_present
        and not editor.scenario_running
        and (population_increased or fallback_due)
    )

    # Debug: log scan conditions during bootstrap
    if getattr(editor, '_large_map_bootstrap_ticking', False) and not getattr(editor, '_bootstrap_scan_debug_done', False):
        print(f"[Bootstrap DEBUG] Scan conditions: currently_present={currently_present}, scenario_running={editor.scenario_running}, "
              f"actor_count={actor_count}, population_increased={population_increased}, fallback_due={fallback_due}")
        if perform_scan:
            print("[Bootstrap DEBUG] Will perform external ego scan")

    if perform_scan:
        editor._bootstrap_scan_debug_done = True  # Only log once
        actor = editor._resolve_external_ego_actor(silent=True, allow_scan=True)
        currently_present = actor is not None
        if getattr(editor, '_large_map_bootstrap_ticking', False):
            print(f"[Bootstrap DEBUG] Scan result: actor={actor.id if actor else None}, currently_present={currently_present}")

    if actor and editor._external_ego_prompt_pending_id == actor.id and editor._external_ego_can_prompt():
        editor._maybe_prompt_external_ego_swap(actor)

    if actor and editor._external_swap_active and actor.id == editor._external_swap_current_id:
        editor._update_external_swap_state(actor)
    if actor:
        editor._maybe_prompt_external_ego_swap(actor)

    editor._external_ego_present_last_check = currently_present

    # Disable bootstrap ticking when external ego is detected on large map
    if currently_present and getattr(editor, '_large_map_bootstrap_ticking', False):
        if getattr(editor, "large_map_active", False):
            now = time.time()
            editor._large_map_external_ego_connected_at = float(
                getattr(editor, "_large_map_external_ego_connected_at", 0.0) or 0.0
            ) or now
            editor._large_map_external_ego_handoff_until = max(
                float(getattr(editor, "_large_map_external_ego_handoff_until", 0.0) or 0.0),
                now + 12.0,
            )
        editor._set_large_map_bootstrap_ticking(
            False,
            log_message="[Manual Tick] External ego detected - stopping large map bootstrap ticks",
            disable_manual_tick=True,
        )

        # Apply saved ego data to the external ego (teleport to scenario position)
        pending_data = getattr(editor, '_pending_external_ego_data', None)
        if pending_data and actor:
            try:
                loc = pending_data.get("location", {})
                rot = pending_data.get("rotation", {})
                new_location = carla.Location(
                    x=float(loc.get("x", 0)),
                    y=float(loc.get("y", 0)),
                    z=float(loc.get("z", 0))
                )
                new_rotation = carla.Rotation(
                    pitch=float(rot.get("pitch", 0)),
                    yaw=float(rot.get("yaw", 0)),
                    roll=float(rot.get("roll", 0))
                )
                new_transform = carla.Transform(new_location, new_rotation)
                applied = False
                if getattr(editor, "large_map_active", False):
                    cp = getattr(editor, "camera_processor", None)
                    if cp is not None and hasattr(cp, "_get_fast_world"):
                        try:
                            client_fast = getattr(cp, "_fast_rpc_client", None)
                            if client_fast is not None:
                                client_fast.apply_batch_sync(
                                    [carla.command.ApplyTransform(int(actor.id), new_transform)],
                                    False,
                                )
                                applied = True
                        except Exception:
                            applied = False
                if not applied:
                    actor.set_transform(new_transform)
                print(f"[Bootstrap] Applied saved ego position to external ego: ({new_location.x:.1f}, {new_location.y:.1f}, {new_location.z:.1f})")
                editor._pending_external_ego_data = None

                # Restore/move camera after teleporting ego
                if editor._bootstrap_original_camera_transform and editor.camera_processor and getattr(editor.camera_processor, "camera_sensor", None):
                    # Move camera to follow the external ego at its scenario position
                    ego_transform = new_transform
                    camera_location = new_location + carla.Location(z=50.0)
                    camera_rotation = carla.Rotation(
                        pitch=-15.0,
                        yaw=float(new_rotation.yaw),
                        roll=0.0,
                    )
                    camera_transform = carla.Transform(camera_location, camera_rotation)
                    if getattr(editor, "large_map_active", False) and hasattr(editor.camera_processor, "_queue_large_map_transform"):
                        try:
                            editor.camera_processor._queue_large_map_transform(camera_transform)
                        except Exception:
                            editor.camera_processor.camera_sensor.set_transform(camera_transform)
                    else:
                        editor.camera_processor.camera_sensor.set_transform(camera_transform)
                    print(f"[Bootstrap] Moved camera to follow external ego at ({new_location.x:.1f}, {new_location.y:.1f})")
                    editor._bootstrap_original_camera_transform = None
            except Exception as exc:
                print(f"[Bootstrap] Failed to apply saved ego data: {exc}")

    if currently_present and getattr(editor, "large_map_active", False):
        camera = getattr(editor, "camera_processor", None)
        if camera is not None and getattr(camera, "last_frame_time", None):
            try:
                connected_at = float(getattr(editor, "_large_map_external_ego_connected_at", 0.0) or 0.0)
            except Exception:
                connected_at = 0.0
            try:
                last_frame = float(camera.last_frame_time or 0.0)
            except Exception:
                last_frame = 0.0
            if connected_at and last_frame and last_frame >= connected_at:
                editor._large_map_external_ego_had_frames = True

    if previously_present and not currently_present:
        editor._handle_external_ego_disconnect()
        # On large maps, force reload the map to clear state (avoids timeouts)
        if getattr(editor, "large_map_active", False):
            editor._force_reload_current_map_for_external_ego()
            return
        # Non-large map: use existing async switch
        async_switched = editor._switch_world_to_async_if_safe(
            reason="external ego removed",
            force=True,
        )
        if (
            not async_switched
            and editor.manual_tick_required
            and not editor.manual_tick_enabled
        ):
            print("[External Ego] Enabling manual tick to keep sensors alive after external removal.")
            editor.manual_tick_enabled = True
            editor.manual_tick_accumulator = 0.0
        return

    camera = getattr(editor, "camera_processor", None)
    last_required = getattr(editor, "_manual_tick_last_required_time", 0.0)
    # Liveness heartbeat: treat the world as alive if EITHER a recent camera frame OR
    # a recent world tick was seen. World ticks (world.on_tick -> _on_world_tick) fire
    # at simulation rate regardless of how slowly the scene renders, so on large/tiled
    # maps -- where camera frames lag badly -- the tick keeps us from misreading slow
    # rendering as a frozen simulation and falsely swapping back from an external ego.
    # Small maps are unaffected (ticks fire there too).
    last_frame_time = getattr(camera, "last_frame_time", 0.0) if camera is not None else 0.0
    last_world_tick = editor._world_tick_times[-1] if editor._world_tick_times else 0.0
    last_alive = max(last_frame_time or 0.0, last_world_tick or 0.0)
    if (
        editor.manual_tick_required
        and not editor.manual_tick_enabled
        and last_alive
        and last_required
    ):
        frame_age = now - last_alive
        sync_age = now - last_required
        if frame_age >= 4.0 and sync_age >= 1.0:
            switched = editor._switch_world_to_async_if_safe(
                reason="no frames or world ticks in synchronous mode",
                force=True,
            )
            if not switched:
                is_large_map = getattr(editor, 'large_map_active', False)
                if not is_large_map:
                    print("[Manual Tick] Enabling Drive Clock to recover frames (no external ticks).")
                    editor.manual_tick_enabled = True
                    editor.manual_tick_accumulator = 0.0
                else:
                    # Large map: enable bootstrap ticking if no external ego yet
                    if editor.external_ego_actor is None and not editor._large_map_bootstrap_ticking:
                        editor._set_large_map_bootstrap_ticking(
                            True,
                            log_message="[Manual Tick] Large map bootstrap: enabling temporary ticks until external ego connects",
                        )
                    elif editor.external_ego_actor is not None:
                        # External ego exists - disable bootstrap ticking, let bridge control.
                        if editor._large_map_bootstrap_ticking:
                            editor._set_large_map_bootstrap_ticking(
                                False,
                                log_message="[Manual Tick] External ego connected on large map - disabling bootstrap ticks",
                                disable_manual_tick=True,
                            )
                            return

                        # If the external bridge was ticking before (we had frames) but now frames stopped,
                        # treat this as a bridge stall/exit - force map reload to clear state
                        if bool(getattr(editor, "_large_map_external_ego_had_frames", False)):
                            if not getattr(editor, "_large_map_external_stall_warned", False):
                                print(
                                    "[Manual Tick] Large map external ego stalled (no frames); "
                                    "reloading map to clear state."
                                )
                                editor._large_map_external_stall_warned = True
                                editor._force_reload_current_map_for_external_ego()
                            return

                        if not getattr(editor, '_large_map_tick_skip_warned', False):
                            print("[Manual Tick] On large map with external ego - bridge controls simulation")
                            editor._large_map_tick_skip_warned = True
                        return

def _unregister_world_tick_handler(editor):
    """Remove any active world tick subscription."""
    if editor._world_tick_subscription is None:
        return
    try:
        if editor.world:
            editor.world.remove_on_tick(editor._world_tick_subscription)
    except Exception:
        pass
    finally:
        editor._world_tick_subscription = None

def _on_world_tick(editor, snapshot):
    """Handle CARLA world tick callbacks (executed from CARLA thread)."""
    now = time.time()
    try:
        editor._world_tick_times.append(now)
        cutoff = now - editor._world_tick_window
        while editor._world_tick_times and editor._world_tick_times[0] < cutoff:
            editor._world_tick_times.popleft()
        if len(editor._world_tick_times) >= 2:
            span = editor._world_tick_times[-1] - editor._world_tick_times[0]
            if span > 0:
                editor._world_tick_fps = (len(editor._world_tick_times) - 1) / span
    except Exception:
        pass

    # Store the latest snapshot (O(1), no RPC). This is the cheap per-tick signal used by
    # external-ego spawn/leave detection in _monitor_external_ego_status (len()/has_actor()).
    # World-reset detection itself is handled by the polled world.id check in
    # _check_world_replacement; the old episode/frame trigger here was dead (carla.Timestamp
    # has no "episode" attribute in CARLA 0.9.15) and is intentionally removed.
    editor._latest_world_snapshot = snapshot

def _process_world_reset_events(editor):
    """Main-thread handler to process pending world reset workflow."""
    if not editor.ready and not editor._world_reset_in_progress and not editor._world_reset_pending:
        return

    editor._check_world_replacement()

    if editor._world_reset_pending:
        with editor._world_reset_lock:
            pending = editor._world_reset_pending
            if pending:
                editor._world_reset_pending = False
        if pending:
            editor._begin_world_reset_recovery()

    if editor._world_reset_finalize_pending:
        editor._finalize_world_reset()

    if getattr(editor, '_world_reset_failed', False):
        # Recovery deadline expired (fix-06): hand over to the server-lost
        # overlay and re-arm the reset machinery + watchdog.
        editor._world_reset_failed = False
        editor._world_reset_in_progress = False
        editor._world_reset_wait_thread = None
        editor.ready = True
        editor.loading_stage = ""
        editor._server_lost = True
        print("[Server Watchdog] CARLA server unreachable")

def _begin_world_reset_recovery(editor):
    """Start teardown and async wait for the world to recover after a reset."""
    if editor._world_reset_in_progress:
        return

    print("[World Reset] Detected CARLA world reload. Reinitializing editor resources...")
    editor._world_reset_in_progress = True
    editor.ready = False
    editor.loading_stage = "World reset detected. Waiting for CARLA..."
    editor._world_tick_times.clear()
    editor._world_tick_fps = 0.0

    if editor.camera_controller:
        # A 3D orbit view exits before the snapshot: the restore below (and the
        # spectator sync / fresh camera processor) only rewrites top-down fields.
        if getattr(editor.camera_controller, "view_mode", "topdown") == "orbit":
            editor.camera_controller.exit_orbit()
        editor._camera_restore_state = (
            float(getattr(editor.camera_controller, 'center_x', 0.0)),
            float(getattr(editor.camera_controller, 'center_y', 0.0)),
            float(getattr(editor.camera_controller, 'height', 200.0)),
        )
    else:
        editor._camera_restore_state = None

    if editor.scenario_running:
        print("[World Reset] Stopping active scenario due to world reset...")
        try:
            editor.stop_scenario()
        except Exception as exc:
            print(f"[World Reset] Failed to stop scenario cleanly: {exc}")

    editor._unregister_world_tick_handler()
    editor._teardown_world_resources_for_reset()

    if editor._world_reset_wait_thread and editor._world_reset_wait_thread.is_alive():
        return

    editor._world_reset_wait_thread = threading.Thread(
        target=editor._wait_for_world_recovery,
        name="VSEWorldResetWait",
        daemon=True,
    )
    editor._world_reset_wait_thread.start()

def _teardown_world_resources_for_reset(editor):
    """Release camera and overlay resources before reconnecting to a fresh world."""
    if editor.camera_processor:
        try:
            OpenDriveOverlayRenderer.disable_overlay(editor.camera_processor, silent=True)
        except Exception:
            pass
        try:
            editor.camera_processor.cleanup()
        except Exception as exc:
            print(f"[World Reset] Warning while cleaning camera processor: {exc}")
        editor.camera_processor = None
        if editor.vehicle_menu:
            editor.vehicle_menu.set_camera_processor(None)
        if editor.pedestrian_menu:
            editor.pedestrian_menu.set_camera_processor(None)
        if editor.ego_vehicle_menu:
            editor.ego_vehicle_menu.set_camera_processor(None)

    if editor.info_panel is not None:
        try:
            editor.info_panel.hide()
        except Exception:
            pass

    editor.external_ego_actor = None
    editor.external_ego_actor_id = None
    editor._external_swap_active = False
    editor._external_swap_current_id = None
    editor._external_swap_last_transform = None
    editor._external_swap_last_blueprint = None
    editor._external_swap_last_color = None
    editor._external_ego_prompt_pending_id = None
    # Clear "was present" / "already reloaded" state so the post-recovery scan starts clean:
    # avoids a spurious external-ego disconnect right after the world reload, and lets the
    # fresh ego (which appears after the reload) re-trigger scenario adoption.
    editor._external_ego_present_last_check = False
    editor._external_ego_scenario_reloaded = False

def _wait_for_world_recovery(editor):
    """Background thread: wait until CARLA world becomes available again.

    Phase 8 (fix-06): bounded by _world_recovery_deadline_s — if the server
    died during the reload this loop would otherwise spin forever behind an
    eternal "Waiting for CARLA..." screen. On expiry the main thread routes
    into the server-lost overlay (see _process_world_reset_events).
    """
    attempts = 0
    recovered_world = None
    candidate = editor._world_reset_candidate_world
    deadline = time.monotonic() + getattr(editor, '_world_recovery_deadline_s', 120.0)

    while time.monotonic() < deadline:
        if candidate is None and not editor.client:
            time.sleep(0.2)
            continue
        try:
            world = candidate or editor.client.get_world()
            candidate_snapshot = world.get_snapshot()
            if candidate_snapshot:
                recovered_world = world
                break
            candidate = None
        except Exception:
            pass

        attempts += 1
        if attempts % 25 == 0:
            print("[World Reset] Waiting for CARLA world to become available...")
        time.sleep(0.2)

    if recovered_world is None:
        # Deadline expired: signal failure; keep finalize_pending False.
        # Write order matters (main thread reads failed AFTER ready_world).
        editor._world_reset_ready_world = None
        editor._world_reset_failed = True
        print("[World Reset] Recovery timed out — treating server as lost")
        return

    editor._world_reset_ready_world = recovered_world
    editor._world_reset_finalize_pending = True

def _finalize_world_reset(editor):
    """Finish rebuilding editor state after the world has recovered."""
    if not editor._world_reset_ready_world:
        # Should not happen; nothing to finalize.
        editor._world_reset_finalize_pending = False
        editor._world_reset_in_progress = False
        return

    new_world = editor._world_reset_ready_world
    editor._world_reset_ready_world = None
    editor._world_reset_finalize_pending = False
    editor._world_reset_candidate_world = None

    try:
        editor.world = new_world
        new_map = None
        try:
            new_map = new_world.get_map()
        except Exception as exc:
            if editor.world_map is not None:
                print(f"[World Reset] Warning: failed to obtain new world map ({exc}). Reusing cached map.")
            else:
                raise

        if new_map is not None:
            editor.world_map = new_map
            editor.cached_map = new_map

        map_name = editor.world_map.name if editor.world_map else "<unknown>"
        print(f"[World Reset] Connected to CARLA world: {map_name}")
    except Exception as exc:
        print(f"[World Reset] Warning: failed to obtain world map: {exc}")
        if editor.world_map is None and editor.cached_map is None:
            print("[World Reset] No cached map available; lane snapping may be unavailable.")

    # Reapply preferred rendering mode after reconnecting to a fresh world.
    editor._apply_rendering_mode(
        desired_enabled=editor._rendering_desired_enabled,
        reason="world reset recovery",
        force=True,
        world=new_world,
    )
    # Reapply culling distance to the fresh world (or adopt its live value if unsafe).
    editor._apply_culling(world=new_world, force=True, reason="world reset recovery")

    restored_camera_state = getattr(editor, "_camera_restore_state", None)
    if editor.camera_controller:
        editor.camera_controller.world = new_world
        if restored_camera_state:
            cx, cy, height = restored_camera_state
            editor.camera_controller.center_x = cx
            editor.camera_controller.center_y = cy
            editor.camera_controller.height = height
    editor._camera_restore_state = None

    try:
        spectator = new_world.get_spectator()
        if editor.camera_controller:
            spectator.set_transform(editor.camera_controller.get_carla_transform())
    except Exception as exc:
        print(f"[World Reset] Unable to reposition spectator: {exc}")

    try:
        editor.camera_processor = CameraImageProcessor(
            new_world,
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
        editor.camera_processor._server_host = editor.connection_profile.host if editor.connection_profile else '127.0.0.1'
        editor.camera_processor._server_port = editor.connection_profile.port if editor.connection_profile else editor.local_port
        if editor.cached_map is not None:
            editor.camera_processor.coordinate_detector.world_map = editor.cached_map
        print("[World Reset] Camera processor reinitialized.")
        try:
            editor.camera_processor.update_camera_position()
        except Exception as exc:
            print(f"[World Reset] Warning: failed to update camera position after reset: {exc}")
    except Exception as exc:
        print(f"[World Reset] Failed to rebuild camera processor: {exc}")
        editor.loading_stage = f"Error rebuilding camera: {exc}"
        editor._world_reset_in_progress = False
        return

    editor.info_panel = InfoPanel(editor.camera_processor)

    try:
        if editor.vehicle_menu:
            editor.vehicle_menu.initialize_vehicles(new_world)
        if editor.pedestrian_menu:
            editor.pedestrian_menu.initialize_pedestrians(new_world)
        if editor.ego_vehicle_menu:
            editor.ego_vehicle_menu.initialize_ego_vehicle(new_world)
    except Exception as exc:
        print(f"[World Reset] Warning: failed to refresh actor menus: {exc}")

    try:
        OpenDriveOverlayRenderer.invalidate_cache(editor.camera_processor, drop_surfaces=True)
        editor.camera_processor.precompute_opendrive_lane_data()
    except Exception as exc:
        print(f"[World Reset] Warning: failed to precompute OpenDRIVE data: {exc}")

    if editor.camera_controller and editor.world:
        try:
            editor.camera_controller.update_min_height_from_terrain(editor.world, height_buffer=5.0)
        except Exception as exc:
            print(f"[World Reset] Warning: failed to update camera terrain height: {exc}")

    try:
        refreshed_weather = editor.world.get_weather()
    except Exception as exc:
        print(f"[Weather] Unable to refresh weather after world reset: {exc}")
    else:
        editor._update_weather_state(refreshed_weather)
        editor._capture_baseline_weather(refreshed_weather)
        if editor.weather_window and editor.weather_window.alive():
            editor.weather_window.apply_weather(refreshed_weather)

    editor._register_world_tick_handler()
    editor.loading_stage = "Ready!"
    editor.ready = True
    editor._world_reset_in_progress = False
    editor._world_reset_wait_thread = None
    try:
        snapshot = new_world.get_snapshot()
        with editor._world_reset_lock:
            editor._world_last_episode = getattr(snapshot.timestamp, "episode", None)
            editor._world_last_frame = snapshot.frame
    except Exception:
        with editor._world_reset_lock:
            editor._world_last_episode = None
            editor._world_last_frame = None
    print("[World Reset] Editor recovery complete.")

def _safe_get_world_map(editor, *, refresh=True):
    """Return cached CARLA map, optionally skipping refresh when unsafe."""
    if editor.cached_map is not None:
        return editor.cached_map

    if editor.world_map is not None:
        editor.cached_map = editor.world_map
        return editor.cached_map

    if not refresh or editor._map_refresh_disabled:
        return None

    if not editor.world:
        return None

    try:
        editor.cached_map = editor.world.get_map()
        editor.world_map = editor.cached_map
        return editor.cached_map
    except Exception as exc:
        print(f"[Map] Unable to refresh world map: {exc}")
        return None

def _expected_ego_roles(editor):
    """Return a set of role names that should be treated as ego vehicles."""
    roles = {"ego_vehicle", "hero"}
    try:
        if editor.current_scenario_path and os.path.isfile(editor.current_scenario_path):
            with open(editor.current_scenario_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            ego_record = data.get("ego_vehicle")
            # Some scenarios may store a list of ego vehicles
            if isinstance(ego_record, dict):
                role = ego_record.get("role")
                if role:
                    roles.add(str(role))
            elif isinstance(ego_record, list):
                for entry in ego_record:
                    if isinstance(entry, dict):
                        role = entry.get("role")
                        if role:
                            roles.add(str(role))
    except Exception as exc:
        print(f"Warning: Failed to resolve ego roles from scenario data: {exc}")
    return roles

def _scenario_ego_type(editor):
    """Return the ego vehicle blueprint/type from the current scenario JSON, or None.

    Used only as a non-gating confirmation that the expected ego vehicle spawned; role_name
    remains the sole adoption trigger, so a different vehicle is still adopted (with a warn).
    """
    try:
        if editor.current_scenario_path and os.path.isfile(editor.current_scenario_path):
            with open(editor.current_scenario_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            ego_record = data.get("ego_vehicle")
            if isinstance(ego_record, list):
                ego_record = ego_record[0] if ego_record else None
            if isinstance(ego_record, dict):
                ego_type = ego_record.get("type")
                if isinstance(ego_type, str) and ego_type.strip():
                    return ego_type.strip()
    except Exception as exc:
        print(f"Warning: Failed to resolve ego type from scenario data: {exc}")
    return None

def _get_current_world(editor):
    """Return the most up-to-date CARLA world reference available to the editor."""
    if editor.client:
        try:
            return editor.client.get_world()
        except Exception as exc:
            print(f"Warning: Failed to retrieve world from client: {exc}")
    if editor.camera_processor and hasattr(editor.camera_processor, "world"):
        return editor.camera_processor.world
    return None

def _refresh_external_ego_actor_reference(editor):
    """Ensure cached external ego actor reference is alive and update it if needed."""
    if editor.external_ego_actor_id is None:
        editor.external_ego_actor = None
        return None

    # Server-truth liveness via the latest tick snapshot (free, no RPC). actor.is_alive is a
    # local flag and does NOT notice an actor destroyed by another client (e.g. awmini
    # closing), so we consult the snapshot. has_actor() False -> skip the fast path and fall
    # through to the get_actor() RPC, which confirms and releases the reference. has_actor()
    # True/None -> keep the cheap is_alive fast path (no RPC in steady state).
    snap = editor._latest_world_snapshot
    snapshot_has_ego = None
    if snap is not None:
        try:
            snapshot_has_ego = snap.has_actor(editor.external_ego_actor_id)
        except Exception:
            snapshot_has_ego = None

    if snapshot_has_ego is not False and editor.external_ego_actor and editor.external_ego_actor.is_alive:
        return editor.external_ego_actor

    world = editor._get_current_world()
    if not world:
        editor.external_ego_actor = None
        return None

    try:
        actor = world.get_actor(editor.external_ego_actor_id)
    except Exception:
        actor = None

    if actor and actor.is_alive:
        try:
            role_name = actor.attributes.get('role_name', None)
        except Exception:
            role_name = None

        expected_roles = editor._expected_ego_roles()
        if role_name in expected_roles:
            editor.external_ego_actor = actor
            editor._external_ego_present_last_check = True
            return actor

        description = role_name if role_name is not None else "<unknown>"
        print(
            f"[Ego Detection] Cached external ego actor {actor.id} now has role '{description}'; "
            "releasing reference."
        )

    editor.external_ego_actor = None
    editor.external_ego_actor_id = None
    editor._external_ego_present_last_check = False
    return None

def _detect_external_ego_vehicle(editor, *, silent: bool = False):
    """Return externally spawned ego vehicle if one is already in the scene."""
    # Avoid misclassifying ScenarioRunner/MiniRunner ego actors as "external".
    #
    # During playback, MiniRunner spawns an internal ego with role_name "ego_vehicle",
    # which matches our external-ego roles and can incorrectly trigger overlays and
    # "preserve external ego" cleanup paths when restoring the editor preview.
    if editor.scenario_running or getattr(editor, "_restore_in_progress", False):
        return editor._refresh_external_ego_actor_reference()

    editor._last_external_ego_scan_time = time.time()
    world = editor._get_current_world()
    if world is None:
        editor._external_ego_present_last_check = False
        return None

    expected_roles = editor._expected_ego_roles()
    detected_roles = set()
    spawned_preview_ids = set()
    camera_proc = getattr(editor, "camera_processor", None)
    if camera_proc:
        spawned_preview_ids = {
            actor.id for actor in getattr(camera_proc, "spawned_vehicles", []) if actor and actor.is_alive
        }

    try:
        vehicles = world.get_actors().filter('vehicle.*')
    except Exception as exc:
        print(f"Warning: Failed to query vehicles when searching for external ego: {exc}")
        editor._external_ego_present_last_check = False
        return None

    previous_id = editor.external_ego_actor_id
    suppress_actions = bool(getattr(editor, "_suppress_external_ego_adoption", False))

    for actor in vehicles:
        try:
            role_name = actor.attributes.get('role_name')
            if role_name:
                detected_roles.add(role_name)
            if role_name in expected_roles:
                if actor.id in spawned_preview_ids:
                    # Allow the cached external or swapped ego to be recognized even if tracked as a preview.
                    if actor.id != previous_id and actor.id != editor._external_swap_current_id:
                        continue
                if actor.is_alive:
                    try:
                        loc = actor.get_location()
                        if (not silent) or (previous_id != actor.id):
                            print(
                                f"Detected candidate external ego vehicle {actor.id} "
                                f"({actor.type_id}) with role '{role_name}' at "
                                f"({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})"
                            )
                    except Exception:
                        if (not silent) or (previous_id != actor.id):
                            print(
                                f"Detected candidate external ego vehicle {actor.id} "
                                f"({actor.type_id}) with role '{role_name}'"
                            )

                    # Non-gating "name" confirmation: role_name above is the sole adoption
                    # trigger; here we only log whether the spawned vehicle matches the
                    # scenario's expected ego type. A mismatch warns but never blocks
                    # adoption (another editor-driving stack may use a different vehicle).
                    if (not silent) or (previous_id != actor.id):
                        expected_type = editor._scenario_ego_type()
                        if expected_type:
                            if actor.type_id == expected_type:
                                print(
                                    f"[Ego Detection] Confirmed external ego type "
                                    f"'{actor.type_id}' matches scenario ego type."
                                )
                            else:
                                print(
                                    f"[Ego Detection] Note: external ego type "
                                    f"'{actor.type_id}' differs from scenario ego type "
                                    f"'{expected_type}'; adopting by role '{role_name}' anyway."
                                )

                    # Handle automatic scenario reload for new external ego connection
                    if (
                        not suppress_actions
                        and editor.external_ego_actor is None
                        and not editor._external_ego_scenario_reloaded
                    ):

                        editor._external_ego_scenario_reloaded = True  # Set immediately to prevent re-entry

                        if (not editor.scenario_running
                            and editor.current_scenario_path
                            and editor.camera_processor):

                            try:
                                # Surface an early overlay so users know we're adopting an external ego.
                                try:
                                    editor._show_external_swap_overlay("Adopting external ego...")
                                    editor._render_external_swap_overlay_immediate()
                                except Exception:
                                    pass
                                print(
                                    f"[External Ego] Auto-reloading scenario '{editor.current_scenario_path}' "
                                    f"for new external ego connection (actor {actor.id})"
                                )
                                editor.camera_processor.load_waypoint_data_from_file(
                                    editor.current_scenario_path,
                                    preserve_camera=True,
                                    skip_ego_spawn=False,
                                )
                                try:
                                    scenario_data = getattr(editor.camera_processor, "loaded_scenario_data", None)
                                    if not scenario_data and os.path.isfile(editor.current_scenario_path):
                                        with open(editor.current_scenario_path, "r", encoding="utf-8") as fh:
                                            scenario_data = json.load(fh)
                                    if scenario_data:
                                        editor._apply_weather_from_json_data(scenario_data)
                                except Exception as exc:
                                    print(f"[External Ego] Failed to re-apply weather: {exc}")
                            except Exception as exc:
                                print(
                                    f"[External Ego] Failed to reload scenario '{editor.current_scenario_path}': {exc}"
                                )

                    editor.external_ego_actor = actor
                    editor.external_ego_actor_id = actor.id
                    editor._external_ego_present_last_check = True
                    if suppress_actions:
                        editor._external_ego_prompt_pending_id = None
                        return actor
                    editor._maybe_prompt_external_ego_swap(actor)
                    return actor
                else:
                    print(
                        f"Found actor {actor.id} with role '{role_name}' but actor.is_alive == False; "
                        "skipping as external ego."
                    )
        except Exception:
            continue

    if expected_roles:
        if not silent:
            print(
                "No external ego vehicle found. "
                f"Expected roles: {sorted(expected_roles)}, detected scene roles: {sorted(detected_roles)}"
            )
    if previous_id is not None and previous_id in spawned_preview_ids:
        # Keep the previous reference alive when it is one of our tracked actors.
        try:
            actor = world.get_actor(previous_id)
            if actor and actor.is_alive:
                editor.external_ego_actor = actor
                editor.external_ego_actor_id = previous_id
                editor._external_ego_present_last_check = True
                return actor
        except Exception:
            pass
    editor.external_ego_actor = None
    editor.external_ego_actor_id = None
    editor._external_ego_present_last_check = False
    editor._external_ego_prompt_pending_id = None
    editor._external_ego_scenario_reloaded = False  # Reset for next connection
    return None

def _resolve_external_ego_actor(
    editor,
    *,
    silent: bool,
    force_scan: bool = False,
    allow_scan: bool = False,
) -> Optional[carla.Actor]:
    """Return the external ego actor by refreshing cached state and optionally scanning."""
    if force_scan:
        editor._detect_external_ego_vehicle(silent=silent)
        return editor._refresh_external_ego_actor_reference()
    actor = editor._refresh_external_ego_actor_reference()
    if actor is None and allow_scan:
        actor = editor._detect_external_ego_vehicle(silent=silent)
    return actor

def _focus_camera_on_ego_vehicle(editor) -> bool:
    """Snap the editor camera to the active ego vehicle if one exists."""
    cp = editor.camera_processor
    if not cp:
        return False
    editor._camera_debug("[CameraDebug] _focus_camera_on_ego_vehicle invoked")
    editor._debug_camera_pose("focus-ego-before")

    def _valid_location(loc: Optional[carla.Location]) -> bool:
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

    location: Optional[carla.Location] = None

    # 1) Editor-managed ego actor (highest priority)
    try:
        editor_ego = cp.get_editor_ego_actor()
    except Exception:
        editor_ego = None
    if editor_ego:
        try:
            loc = editor_ego.get_location()
            if _valid_location(loc):
                location = loc
        except Exception:
            pass

    # 2) Cached editor ego transform
    if location is None:
        transform = getattr(cp, "ego_vehicle_transform", None)
        if transform:
            loc = transform.location
            loc = carla.Location(loc.x, loc.y, loc.z)
            if _valid_location(loc):
                location = loc

    # 3) Ego pose from loaded scenario JSON
    if location is None:
        scenario_data = getattr(cp, "loaded_scenario_data", None)
        if isinstance(scenario_data, dict):
            ego_meta = scenario_data.get("ego_vehicle") or {}
            loc_meta = ego_meta.get("location") or {}
            try:
                loc = carla.Location(
                    float(loc_meta.get("x", 0.0)),
                    float(loc_meta.get("y", 0.0)),
                    float(loc_meta.get("z", 0.0)),
                )
                if _valid_location(loc):
                    location = loc
            except Exception:
                pass

    # 4) External/any ego-role actor (last resort, only if valid)
    if location is None:
        try:
            ego_actor = cp.get_ego_vehicle_actor()
        except Exception:
            ego_actor = None
        if ego_actor and ego_actor.is_alive:
            try:
                loc = ego_actor.get_location()
                if _valid_location(loc):
                    location = loc
            except Exception:
                pass

    if location:
        cp.focus_camera_on_location(location)
        try:
            editor._camera_debug(
                f"[CameraDebug] Focused on ego at ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"
            )
            editor._debug_camera_pose("focus-ego-after")
        except Exception:
            editor._camera_debug("[CameraDebug] Focused on ego at <unknown location>")
        return True
    editor._camera_debug("[CameraDebug] No ego location available to focus")
    return False

def _manual_world_tick(editor, dt):
    """Advance the CARLA simulation when operating in synchronous remote mode."""
    if not (editor.manual_tick_required and editor.manual_tick_enabled):
        return

    # During scenario playback, tick ownership belongs to ScenarioRunner/MiniRunner
    # (or an external bridge). Avoid double-ticking from the UI loop.
    if getattr(editor, "scenario_running", False):
        return

    if not editor.world:
        return

    interval = max(editor.manual_tick_interval, 0.001)
    is_large_map = bool(getattr(editor, "large_map_active", False))
    is_bootstrap = bool(is_large_map and getattr(editor, "_large_map_bootstrap_ticking", False))
    if is_bootstrap:
        # During large-map bootstrap we poll slowly (the external bridge is expected to tick).
        interval = max(interval, 0.25)
    editor.manual_tick_accumulator += dt

    ticks_executed = 0
    max_ticks_per_frame = 1 if is_large_map else 5

    if is_large_map:
        now = time.time()
        next_attempt = float(getattr(editor, "_large_map_manual_tick_next_attempt", 0.0) or 0.0)
        if now < next_attempt:
            editor.manual_tick_accumulator = 0.0
            return

        # If an external ego is present and frames are flowing, let the bridge control ticks.
        cp = getattr(editor, "camera_processor", None)
        if (
            getattr(editor, "external_ego_actor", None) is not None
            and not getattr(editor, "_large_map_bootstrap_ticking", False)
            and cp is not None
            and getattr(cp, "last_frame_time", None)
        ):
            frame_age = now - float(cp.last_frame_time or 0.0)
            if 0.0 <= frame_age < 0.25:
                # Keep manual ticks running unless an external tick source is clearly active.
                try:
                    camera_fps = float(getattr(cp, "_camera_fps_display", 0.0) or 0.0)
                except Exception:
                    camera_fps = 0.0
                try:
                    world_fps = float(getattr(editor, "_world_tick_fps", 0.0) or 0.0)
                except Exception:
                    world_fps = 0.0
                if max(camera_fps, world_fps) >= 8.0:
                    editor.manual_tick_accumulator = 0.0
                    return

    while editor.manual_tick_accumulator >= interval and ticks_executed < max_ticks_per_frame:
        try:
            is_bootstrap = getattr(editor, '_large_map_bootstrap_ticking', False)
            # During large-map bootstrap, avoid driving ticks ourselves; wait for the external
            # tick source (e.g., bridge) so we don't fight for tick ownership.
            if is_large_map and is_bootstrap and hasattr(editor.world, "wait_for_tick"):
                editor.world.wait_for_tick(0.5)
            elif hasattr(editor.world, "tick"):
                tick_world = editor.world
                fast_client = None
                if is_large_map:
                    cp = getattr(editor, "camera_processor", None)
                    if cp is not None and hasattr(cp, "_get_fast_world"):
                        try:
                            fast_world = cp._get_fast_world()
                        except Exception:
                            fast_world = None
                        if fast_world is not None:
                            tick_world = fast_world
                            fast_client = getattr(cp, "_fast_rpc_client", None)

                if is_large_map and fast_client is not None:
                    try:
                        prev_timeout = float(getattr(cp, "_large_map_rpc_timeout_s", 2.0))
                    except Exception:
                        prev_timeout = 2.0
                    try:
                        fast_client.set_timeout(min(prev_timeout, 0.25))
                    except Exception:
                        pass
                    try:
                        tick_world.tick()
                    finally:
                        try:
                            fast_client.set_timeout(prev_timeout)
                        except Exception:
                            pass
                else:
                    tick_world.tick()
            elif hasattr(editor.world, "wait_for_tick"):
                # Avoid long UI stalls: poll briefly instead of blocking for seconds.
                editor.world.wait_for_tick(0.05 if is_large_map else 1.0)
            else:
                raise RuntimeError("Current CARLA world does not expose tick/wait_for_tick")
        except Exception as exc:
            # During bootstrap on large maps, suppress repeated warnings
            if is_bootstrap:
                if not getattr(editor, '_bootstrap_tick_warned', False):
                    print(f"[Tick] Bootstrap waiting for external tick source...")
                    editor._bootstrap_tick_warned = True
            else:
                if not is_large_map:
                    print(f"[Tick] Warning: world tick failed ({exc})")
                else:
                    failures = int(getattr(editor, "_large_map_manual_tick_failures", 0) or 0) + 1
                    editor._large_map_manual_tick_failures = failures
                    backoff = 0.25 * (2.0 ** min(4, failures))
                    editor._large_map_manual_tick_next_attempt = time.time() + min(2.0, backoff)
            editor.manual_tick_accumulator = 0.0
            break
        else:
            if is_large_map:
                editor._large_map_manual_tick_failures = 0
                editor._large_map_manual_tick_next_attempt = 0.0
            editor.manual_tick_accumulator -= interval
            ticks_executed += 1

    if ticks_executed == max_ticks_per_frame and editor.manual_tick_accumulator >= interval:
        editor.manual_tick_accumulator = 0.0
