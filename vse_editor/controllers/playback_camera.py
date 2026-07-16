"""Playback camera controller (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-26).

Top-Down / Chase / Cockpit playback camera follow state machine, including the
CARLA world-tick callback. All state lives as plain attributes on the processor
(initialized in CameraImageProcessor.__init__); the processor keeps one-line
delegates for every function here, so registration sites, editor callers and
getattr-string reads are unchanged.
"""

import math
import time
from typing import Optional

import carla


def request_playback_camera_follow(processor) -> None:
    """Begin watching for an ego vehicle to follow during playback (no keyboard control)."""
    processor.playback_camera_follow_pending = True
    # The starting camera mode is assigned by the editor from its remembered Play Cam
    # preference right after this call; do not force it here.
    processor.playback_camera_follow_enabled = False
    processor.playback_camera_follow_actor = None
    processor._playback_camera_follow_search_timer = 0.0
    processor._playback_camera_follow_local_offset = (0.0, 0.0, 0.0)
    processor._playback_camera_target_transform = None

def toggle_playback_camera_mode(processor) -> str:
    """Cycle the playback camera 'topdown' -> 'chase' -> 'cockpit' -> 'topdown'; return new mode.

    topdown : free overhead (editor default).
    chase   : behind the vehicle. External ego rigid-attaches to the CARLA spectator (zero-lag,
              server-window-identical); local ego computes its own chase pose (sync -> smooth).
    cockpit : driver-seat view, rigid-attached to the ego itself so it turns with the car. Works
              for both external (bridge-driven) and local (sync) ego.
    """
    order = ("topdown", "chase", "cockpit")
    idx = order.index(processor.playback_camera_mode) if processor.playback_camera_mode in order else 0
    return processor.set_playback_camera_mode(order[(idx + 1) % len(order)])

def set_playback_camera_mode(processor, mode: str) -> str:
    """Set the playback camera mode directly and apply the matching attach; return the mode."""
    if mode in ("topdown", "chase", "cockpit"):
        processor.playback_camera_mode = mode
    processor._apply_playback_camera_attach()
    return processor.playback_camera_mode

def _apply_playback_camera_attach(processor) -> None:
    """Attach/detach VSE's camera sensor to match the current playback_camera_mode.

    Leaves the free top-down camera in place for 'topdown'. Detaches any attach that doesn't match
    the new mode, then enters the new mode's attach (spectator for external-ego chase; ego for
    cockpit). Local-ego chase is compute-own, so it needs no attach here.
    """
    mode = processor.playback_camera_mode
    external = processor.playback_camera_follow_enabled
    # Leave any attach that no longer matches the mode (each detach re-spawns the free camera).
    if processor._camera_attached_to_spectator and mode != "chase":
        processor._detach_camera_from_spectator()
    if processor._camera_attached_to_ego and mode != "cockpit":
        processor._detach_camera_from_ego()
    # Enter the new mode's attach.
    if mode == "chase" and external and not processor._camera_attached_to_spectator:
        processor._attach_camera_to_spectator()
    elif mode == "cockpit" and not processor._camera_attached_to_ego:
        actor = processor.playback_camera_follow_actor if external else processor.manual_control_actor
        processor._attach_camera_to_ego_cockpit(actor)

def _attach_camera_to_spectator(processor) -> None:
    """Rigid-attach VSE's camera to the CARLA spectator (awmini-driven) -> zero-lag server view.
    Falls back to the compute-own free-camera chase if no spectator is available."""
    spectator = None
    try:
        spectator = processor.world.get_spectator() if processor.world else None
    except Exception:
        spectator = None
    if spectator is None:
        processor._camera_attached_to_spectator = False  # fall back to compute-own chase (free cam)
        return
    processor._camera_attached_to_spectator = True
    try:
        processor.restart_camera_sensor(
            attach_to=spectator,
            transform=carla.Transform(),  # identity offset -> exactly the spectator's view
            attachment_type=carla.AttachmentType.Rigid,
        )
    except Exception as exc:
        print(f"[Camera] Failed to attach to spectator: {exc}")
        processor._camera_attached_to_spectator = False
        try:
            processor.restart_camera_sensor(attach_to=None, transform=processor.camera_controller.get_carla_transform())
        except Exception:
            pass

def _detach_camera_from_spectator(processor) -> None:
    """Re-spawn the free top-down camera and resume the normal follow."""
    if not processor._camera_attached_to_spectator:
        return
    processor._camera_attached_to_spectator = False
    try:
        processor.restart_camera_sensor(attach_to=None, transform=processor.camera_controller.get_carla_transform())
    except Exception as exc:
        print(f"[Camera] Failed to detach from spectator: {exc}")

def _attach_camera_to_ego_cockpit(processor, actor: Optional[carla.Actor]) -> None:
    """Rigid-attach VSE's camera to the ego at the driver seat -> smooth cockpit view.

    The offset is in the ego's own frame, so the view turns/pitches/rolls with the car. The ego is
    engine-driven (bridge teleport in VIL, sync locally), so the attached camera is smooth with no
    per-frame VSE positioning. Falls back to the free top-down camera if no live actor.
    """
    try:
        alive = bool(actor and actor.is_alive)
    except Exception:
        alive = False
    if not alive:
        processor._camera_attached_to_ego = False
        return
    processor._camera_attached_to_ego = True
    try:
        processor.restart_camera_sensor(
            attach_to=actor,
            transform=carla.Transform(
                carla.Location(x=processor.cockpit_x, y=processor.cockpit_y, z=processor.cockpit_z),
                carla.Rotation(pitch=processor.cockpit_pitch),
            ),
            attachment_type=carla.AttachmentType.Rigid,
        )
    except Exception as exc:
        print(f"[Camera] Failed to attach cockpit camera to ego: {exc}")
        processor._camera_attached_to_ego = False
        try:
            processor.restart_camera_sensor(attach_to=None, transform=processor.camera_controller.get_carla_transform())
        except Exception:
            pass

def _detach_camera_from_ego(processor) -> None:
    """Re-spawn the free top-down camera after a cockpit attach."""
    if not processor._camera_attached_to_ego:
        return
    processor._camera_attached_to_ego = False
    try:
        processor.restart_camera_sensor(attach_to=None, transform=processor.camera_controller.get_carla_transform())
    except Exception as exc:
        print(f"[Camera] Failed to detach cockpit camera: {exc}")

def _is_camera_engine_attached(processor) -> bool:
    """True while the camera is rigid-attached to a CARLA actor (spectator chase or ego cockpit).
    The engine drives it, so the follow tick/step/apply must not also call set_transform."""
    return processor._camera_attached_to_spectator or processor._camera_attached_to_ego

def _stop_playback_camera_follow(processor) -> None:
    """Stop playback camera following and clear tick-based updates."""
    processor._detach_camera_from_spectator()  # no-op unless spectator-attached (chase)
    processor._detach_camera_from_ego()        # no-op unless ego-attached (cockpit)
    processor.playback_camera_mode = "topdown"
    processor._playback_camera_unregister_tick_callback()
    processor.playback_camera_follow_enabled = False
    processor.playback_camera_follow_pending = False
    processor.playback_camera_follow_actor = None
    processor._playback_camera_follow_local_offset = (0.0, 0.0, 0.0)
    processor._playback_camera_target_transform = None
    processor._manual_camera_free_look_sources.clear()
    processor._pending_free_look_restores.clear()

def _start_playback_camera_follow(processor, actor: carla.Actor) -> None:
    """Begin camera following for the given actor during playback."""
    # Reset predictive-follow state so a new follow doesn't inherit stale velocity.
    processor._playback_camera_follow_vel = (0.0, 0.0)
    processor._playback_camera_follow_prev_target = None
    processor._playback_camera_follow_prev_t = 0.0
    controller = processor.camera_controller
    if not actor or not controller:
        processor._playback_camera_follow_local_offset = (0.0, 0.0, 0.0)
        processor._playback_camera_target_transform = None
        return
    try:
        if not actor.is_alive:
            processor._playback_camera_follow_local_offset = (0.0, 0.0, 0.0)
            processor._playback_camera_target_transform = None
            return
        actor_transform = actor.get_transform()
    except Exception:
        processor._playback_camera_follow_local_offset = (0.0, 0.0, 0.0)
        processor._playback_camera_target_transform = None
        return

    # Playback follow always sits directly above the ego (top-down); the camera was
    # focused on the editor ego, not this follow target, so a relative horizontal
    # offset would be stale. Zero horizontal offset keeps the camera over the ego.
    local_offset = (0.0, 0.0, controller.height - actor_transform.location.z)
    processor._playback_camera_follow_local_offset = local_offset
    processor._playback_camera_target_transform = processor._build_follow_target(actor_transform, local_offset)
    processor._playback_camera_apply_target_transform()
    processor._playback_camera_register_tick_callback()

def _playback_camera_apply_target_transform(processor, *, force: bool = False) -> None:
    """Apply the computed camera transform to the camera controller and spectator."""
    if processor._is_camera_engine_attached():
        return  # camera is engine-attached to the spectator; don't fight it with set_transform
    transform = processor._playback_camera_target_transform
    if not transform:
        return
    controller = processor.camera_controller
    if not controller:
        return
    if processor.manual_camera_free_look_active and not force:
        return
    controller.center_x = transform.location.x
    controller.center_y = transform.location.y
    controller.height = transform.location.z
    if hasattr(controller, "stop_moving"):
        controller.stop_moving()
    try:
        if processor.camera_sensor:
            processor.camera_sensor.set_transform(transform)
        # Spectator intentionally NOT moved: it's unhooked from the editor
        # camera. During manual control / playback an ego ("ego_vehicle"/"hero")
        # exists and is CARLA's large-map streaming anchor, so the spectator is
        # dropped by the LargeMapManager and moving it would have no effect.
    except Exception:
        pass
    processor.camera_is_moving = False
    processor.camera_movement_timer = time.time()

def _playback_camera_register_tick_callback(processor) -> None:
    """Subscribe to CARLA world ticks to keep the camera in sync during playback."""
    if not processor.world or processor._playback_camera_tick_subscription is not None:
        return
    try:
        processor._playback_camera_tick_subscription = processor.world.on_tick(processor._playback_camera_on_world_tick)
    except Exception:
        processor._playback_camera_tick_subscription = None

def _playback_camera_unregister_tick_callback(processor) -> None:
    """Remove any active playback camera tick subscription."""
    if processor._playback_camera_tick_subscription is None:
        return
    try:
        if processor.world:
            processor.world.remove_on_tick(processor._playback_camera_tick_subscription)
    except Exception:
        pass
    finally:
        processor._playback_camera_tick_subscription = None

def _playback_camera_on_world_tick(processor, snapshot) -> None:
    """Update the follow target each simulator tick (playback mode).

    Reads the ego pose from the tick's world snapshot (the exact pose for the frame being
    rendered) rather than a fresh get_transform() RPC, so the target matches the rendered
    frame. The camera is eased toward this target per render frame in
    _playback_camera_follow_step(); it is intentionally NOT snapped here, which is what keeps
    the top-down view smooth when the ego is teleported asynchronously (e.g. VIL).
    """
    if not processor.playback_camera_follow_enabled:
        processor._playback_camera_unregister_tick_callback()
        return
    if processor._is_camera_engine_attached():
        return  # engine drives the camera while attached to the spectator
    actor = processor.playback_camera_follow_actor
    if not actor:
        return
    try:
        if not actor.is_alive:
            return
        # B: same-frame pose from the snapshot; fall back to an RPC if not present yet.
        actor_snapshot = snapshot.find(actor.id) if snapshot is not None else None
        actor_transform = actor_snapshot.get_transform() if actor_snapshot is not None else actor.get_transform()
    except Exception:
        return

    local_offset = processor._playback_camera_follow_local_offset
    processor._playback_camera_target_transform = processor._build_follow_target(actor_transform, local_offset)

def _playback_camera_follow_update(processor, dt: float) -> None:
    """Update playback camera follow state - find ego and start following."""
    # Already following? Keep the camera eased onto the ego (and stop if it died).
    if processor.playback_camera_follow_enabled:
        actor = processor.playback_camera_follow_actor
        if not actor:
            processor._stop_playback_camera_follow()
            return
        try:
            if not actor.is_alive:
                processor._stop_playback_camera_follow()
                return
        except Exception:
            processor._stop_playback_camera_follow()
            return
        processor._playback_camera_follow_step(dt)
        return

    # Not pending? Nothing to do
    if not processor.playback_camera_follow_pending:
        return

    # Search for ego vehicle periodically
    processor._playback_camera_follow_search_timer -= dt
    if processor._playback_camera_follow_search_timer > 0:
        return
    processor._playback_camera_follow_search_timer = 0.25  # Search every 250ms

    # Reuse the manual control actor finder logic
    actor = processor._find_manual_control_actor()
    if actor is None:
        return

    # Found the ego - start following
    processor.playback_camera_follow_actor = actor
    processor.playback_camera_follow_enabled = True
    processor.playback_camera_follow_pending = False
    processor._start_playback_camera_follow(actor)
    # Attach for the starting camera mode (chase/cockpit) now that the ego exists.
    processor._apply_playback_camera_attach()

def _playback_camera_follow_step(processor, dt: float) -> None:
    """Ease the top-down camera toward the follow target (set per world tick).

    Two tunable class attrs (edit then restart VSE):
      _playback_camera_follow_tau  - exponential smoothing time constant (s); 0 = snap, larger = floatier.
      _playback_camera_follow_lead - predictive lead (s): aim ahead by the ego's estimated velocity
                                     to cancel the camera's one-tick set_transform lag. 0 = off.
    Easing removes the async-teleport shake; the optional lead tightens the lock without the snap
    wobble. Velocity is estimated from observed target motion (works with CARLA physics off on the
    ego). Skipped during free-look so the user can still detach and pan.
    """
    if processor._is_camera_engine_attached():
        return  # engine drives the camera while attached to the spectator
    target = processor._playback_camera_target_transform
    controller = processor.camera_controller
    if target is None or not controller:
        return
    if processor.manual_camera_free_look_active:
        return
    tx = float(target.location.x)
    ty = float(target.location.y)
    tz = float(target.location.z)

    # Estimate ego velocity from how the target moves (only when it actually changes), then
    # lead the aim point so the camera arrives where the ego will be, not where it was.
    now = time.time()
    prev = processor._playback_camera_follow_prev_target
    moved = prev is None or abs(tx - prev[0]) > 1e-4 or abs(ty - prev[1]) > 1e-4
    if prev is not None and moved:
        dtt = max(1e-3, now - processor._playback_camera_follow_prev_t)
        raw_vx = (tx - prev[0]) / dtt
        raw_vy = (ty - prev[1]) / dtt
        vbeta = 0.5  # low-pass the velocity estimate across target updates
        vx, vy = processor._playback_camera_follow_vel
        processor._playback_camera_follow_vel = (vx + (raw_vx - vx) * vbeta,
                                            vy + (raw_vy - vy) * vbeta)
    if moved:
        processor._playback_camera_follow_prev_target = (tx, ty)
        processor._playback_camera_follow_prev_t = now
    lead = max(0.0, float(processor._playback_camera_follow_lead))
    tx = tx + processor._playback_camera_follow_vel[0] * lead
    ty = ty + processor._playback_camera_follow_vel[1] * lead

    cx = float(getattr(controller, "center_x", tx))
    cy = float(getattr(controller, "center_y", ty))
    cz = float(getattr(controller, "height", tz))
    tau = max(1e-3, float(processor._playback_camera_follow_tau))
    alpha = 1.0 - math.exp(-max(0.0, float(dt)) / tau)
    # Snap the residual when essentially on target to avoid endless micro-stepping.
    nx = tx if abs(tx - cx) < 0.01 else cx + (tx - cx) * alpha
    ny = ty if abs(ty - cy) < 0.01 else cy + (ty - cy) * alpha
    nz = tz if abs(tz - cz) < 0.01 else cz + (tz - cz) * alpha
    controller.center_x = nx
    controller.center_y = ny
    controller.height = nz
    if hasattr(controller, "stop_moving"):
        controller.stop_moving()
    try:
        if processor.camera_sensor:
            processor.camera_sensor.set_transform(
                carla.Transform(carla.Location(x=nx, y=ny, z=nz), target.rotation)
            )
    except Exception:
        pass
    # Mirror _playback_camera_apply_target_transform: follow is not treated as a user pan.
    processor.camera_is_moving = False
    processor.camera_movement_timer = time.time()
