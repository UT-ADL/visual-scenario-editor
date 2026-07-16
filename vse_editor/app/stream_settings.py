"""Camera-stream settings, no-camera pose cache, camera debug helpers
(duck-typed module functions, first arg `editor`). Moved verbatim from vse.py
(step-40, Phase 7).
"""

import time

from typing import List, Optional, Tuple

import carla

from vse_editor.constants import NO_CAMERA_STREAM_RESOLUTIONS


def _camera_debug(editor, message: str) -> None:
    """Emit camera-related debug logs when debug mode is enabled."""
    if editor.camera_debug_enabled:
        print(message)

def _debug_camera_pose(editor, label: str):
    """Print current camera center/height for debugging jumps."""
    if not editor.camera_debug_enabled:
        return
    ctrl = getattr(editor, "camera_controller", None)
    if not ctrl:
        editor._camera_debug(f"[CameraDebug] {label}: camera_controller unavailable")
        return
    try:
        pose = f"({ctrl.center_x:.2f}, {ctrl.center_y:.2f}) h={getattr(ctrl, 'height', 0.0):.2f}"
        editor._camera_debug(f"[CameraDebug] {label}: center={pose}")
        editor._debug_last_pose_label = label
    except Exception as exc:
        editor._camera_debug(f"[CameraDebug] {label}: unable to read camera pose ({exc})")

def _auto_camera_allowed(editor) -> bool:
    """Return True if automatic camera moves are allowed."""
    return bool(getattr(editor, "camera_stream_enabled", False))

def _process_pending_camera_focus(editor) -> None:
    """Apply any scheduled camera focus requests (main-thread)."""
    target = getattr(editor, "_pending_camera_focus_target", None)
    if not target:
        return
    now = time.time()
    if now < getattr(editor, "_pending_camera_focus_not_before", 0.0):
        return
    if now > getattr(editor, "_pending_camera_focus_until", 0.0):
        editor._pending_camera_focus_target = None
        editor._pending_camera_focus_reason = None
        editor._pending_camera_focus_debug_logged = False
        return
    last_attempt = getattr(editor, "_pending_camera_focus_last_attempt", 0.0)
    if now - last_attempt < 0.12:
        return
    editor._pending_camera_focus_last_attempt = now
    cp = editor.camera_processor
    if not cp:
        return
    try:
        x, y, z = target
        focus_loc = carla.Location(x=float(x), y=float(y), z=float(z))
    except Exception:
        editor._pending_camera_focus_target = None
        editor._pending_camera_focus_reason = None
        editor._pending_camera_focus_debug_logged = False
        return
    if editor.camera_debug_enabled and not editor._pending_camera_focus_debug_logged:
        try:
            editor._camera_debug(
                f"[CameraDebug] Pending focus scheduled ({editor._pending_camera_focus_reason or 'unspecified'}) "
                f"→ ({focus_loc.x:.2f}, {focus_loc.y:.2f}, {focus_loc.z:.2f})"
            )
            editor._debug_camera_pose("pending-focus-before")
        except Exception:
            pass
        editor._pending_camera_focus_debug_logged = True
    try:
        cp.focus_camera_on_location(focus_loc)
    except Exception:
        return
    if editor.camera_debug_enabled and editor._pending_camera_focus_debug_logged:
        try:
            editor._debug_camera_pose("pending-focus-after")
        except Exception:
            pass

def _using_remote_server(editor) -> bool:
    profile = editor.connection_profile
    return bool(profile and profile.is_remote)

def _apply_stream_settings(editor):
    if not editor.camera_processor:
        return
    target_fps = editor.stream_fps if editor._using_remote_server() else 0
    if not editor.camera_stream_enabled and getattr(editor, "large_map_active", False):
        editor.camera_processor.pause_camera_stream()
        return
    editor.camera_processor.enable_camera_sensor()

    if editor.camera_stream_enabled:
        editor.camera_processor.apply_stream_settings(
            resolution=editor.stream_resolution,
            fps=target_fps,
        )
        return

    candidates: List[Tuple[int, int]] = []
    if editor._no_camera_resolution_override:
        candidates.append(editor._no_camera_resolution_override)
    for option in NO_CAMERA_STREAM_RESOLUTIONS:
        if option not in candidates:
            candidates.append(option)

    last_error: Optional[Exception] = None
    for option in candidates:
        try:
            editor.camera_processor.apply_stream_settings(
                resolution=option,
                fps=target_fps,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[Stream] Failed to apply placeholder camera resolution "
                f"{option[0]}x{option[1]}: {exc}"
            )
            continue

        editor._no_camera_resolution_override = option
        if option != NO_CAMERA_STREAM_RESOLUTIONS[0]:
            print(
                f"[Stream] Using fallback placeholder camera resolution "
                f"{option[0]}x{option[1]}"
            )
        return

    print("[Stream] Unable to maintain placeholder camera feed; disabling sensor.")
    if last_error:
        print(f"[Stream] Last placeholder error: {last_error}")
    editor.camera_processor.disable_camera_sensor()
    editor._no_camera_resolution_override = None

def _cache_camera_pose_for_no_camera(editor) -> None:
    """Remember the current camera pose so we can restore it after 'No Camera'."""
    if editor._no_camera_saved_pose is not None:
        return
    ctrl = getattr(editor, "camera_controller", None)
    if not ctrl:
        return
    editor._no_camera_saved_pose = (
        float(getattr(ctrl, "center_x", 0.0)),
        float(getattr(ctrl, "center_y", 0.0)),
        float(getattr(ctrl, "height", 0.0)),
    )
    editor._camera_debug(
        f"[CameraDebug] Cached pose before disabling stream: "
        f"({editor._no_camera_saved_pose[0]:.1f}, {editor._no_camera_saved_pose[1]:.1f}) "
        f"h={editor._no_camera_saved_pose[2]:.1f}"
    )

def _apply_camera_pose(editor, pose: Tuple[float, float, float]) -> None:
    """Apply a raw camera pose and sync the CARLA spectator/sensor."""
    ctrl = getattr(editor, "camera_controller", None)
    if not ctrl:
        return
    ctrl.center_x, ctrl.center_y, ctrl.height = pose
    if editor.camera_processor:
        editor.camera_processor.update_camera_position()
        editor.camera_processor.notify_manual_camera_adjustment()
        editor.camera_processor.restore_vehicle_menu_after_camera_pan()
    if hasattr(ctrl, "stop_moving"):
        ctrl.stop_moving()

def _jump_camera_to_no_camera_pose(editor) -> None:
    """Move the camera out of view when the stream is disabled."""
    if getattr(editor, "large_map_active", False):
        editor._camera_debug("[CameraDebug] Large map: keeping camera pose for 'No Camera'.")
        return
    placeholder_pose = (-1000.0, -1000.0, 0.0)
    editor._camera_debug(
        f"[CameraDebug] Moving camera to placeholder pose: "
        f"({placeholder_pose[0]:.1f}, {placeholder_pose[1]:.1f}) h={placeholder_pose[2]:.1f}"
    )
    editor._apply_camera_pose(placeholder_pose)

def _restore_camera_pose_after_no_camera(editor) -> None:
    """Restore the cached pose after re-enabling a camera stream."""
    if not editor._no_camera_saved_pose:
        return
    pose = editor._no_camera_saved_pose
    editor._no_camera_saved_pose = None
    editor._camera_debug(
        f"[CameraDebug] Restoring camera pose after re-enabling stream: "
        f"({pose[0]:.1f}, {pose[1]:.1f}) h={pose[2]:.1f}"
    )
    editor._apply_camera_pose(pose)

def _set_stream_resolution(editor, resolution: Optional[Tuple[int, int]]):
    label = "[Remote]" if editor._using_remote_server() else "[Stream]"
    if resolution is None:
        if editor.camera_stream_enabled:
            editor._cache_camera_pose_for_no_camera()
        editor.camera_stream_enabled = False
        if None in editor.remote_resolution_options:
            editor.remote_resolution_index = editor.remote_resolution_options.index(None)
        else:
            editor.remote_resolution_index = 0
        print(f"{label} Camera feed disabled (No Camera).")
        editor._jump_camera_to_no_camera_pose()
        editor._apply_stream_settings()
        return

    was_disabled = not editor.camera_stream_enabled
    editor.camera_stream_enabled = True
    width, height = int(resolution[0]), int(resolution[1])
    editor.stream_resolution = (width, height)
    if resolution in editor.remote_resolution_options:
        editor.remote_resolution_index = editor.remote_resolution_options.index(resolution)
    else:
        editor.remote_resolution_index = 0
    if was_disabled:
        editor._restore_camera_pose_after_no_camera()
    print(f"{label} Stream resolution set to {width}x{height}")
    editor._apply_stream_settings()

def _set_stream_fps(editor, fps: int):
    fps = int(fps)
    if fps < editor.remote_fps_min:
        fps = editor.remote_fps_min
    elif fps > editor.remote_fps_max:
        fps = editor.remote_fps_max
    if fps == editor.stream_fps:
        return
    editor.stream_fps = fps
    print(f"[Remote] Stream FPS set to {editor.stream_fps}")
    editor._apply_stream_settings()
