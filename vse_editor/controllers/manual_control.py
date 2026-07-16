"""Manual ego control + free-look controller (moved verbatim from
CameraImageProcessor, self -> processor rename only — step-27).

The manual-driving state machine (arrow keys, autopilot toggle, camera follow)
and the shared free-look family. All state lives as plain attributes on the
processor; the processor keeps one-line delegates (plus the
manual_camera_free_look_active @property and two staticmethod aliases), so the
world.on_tick registration, editor reach-ins and getattr-string reads are
unchanged. The two transform builders were @staticmethods (no self) and are
plain module functions here.
"""

import math
import time
from typing import Optional, Tuple

import carla
import pygame


def request_manual_control(processor) -> None:
    """Begin watching for an ego vehicle to attach for manual control."""
    processor.manual_control_armed = True
    processor.manual_control_pending = True
    # The starting camera mode is assigned by the editor from its remembered Play Cam
    # preference right after this call; do not force it here.
    processor.manual_control_enabled = False
    processor.manual_control_actor = None
    processor._manual_control_search_timer = 0.0
    processor._manual_control_debug_printed = False
    processor.manual_control_state.update({
        'control': None,
        'steer_cache': 0.0,
        'reverse': False,
        'auto_reverse_engaged': False,
        'hand_brake_pressed': False,
        'autopilot_enabled': False,
        'autopilot_active': False,
        'camera_follow_local_offset': (0.0, 0.0, 0.0),
    })
    processor._manual_control_unregister_tick_callback()
    processor._manual_control_target_transform = None
    processor._manual_camera_free_look_sources.clear()
    processor._pending_free_look_restores.clear()

def disable_manual_control(processor) -> None:
    """Detach manual control from the ego vehicle."""
    processor._manual_control_stop_camera_follow()
    processor._stop_playback_camera_follow()
    final_transform = None
    if processor.manual_control_enabled and processor.manual_control_actor and processor.manual_control_actor.is_alive:
        try:
            if processor.manual_control_state.get('autopilot_active'):
                processor.manual_control_actor.set_autopilot(False)
        except Exception:
            pass
        try:
            processor.manual_control_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass
        try:
            final_transform = processor.manual_control_actor.get_transform()
        except Exception:
            final_transform = None
        should_destroy = processor.manual_control_actor.id != processor.ego_vehicle_id
        if should_destroy:
            try:
                processor.manual_control_actor.destroy()
            except Exception:
                pass
    if final_transform is None and processor.ego_vehicle_transform is not None:
        final_transform = processor.ego_vehicle_transform

    if final_transform:
        processor.update_editor_ego_transform(final_transform)
        if processor.session.saved_scene_vehicles:
            for actor_data in processor.session.saved_scene_vehicles:
                actor = actor_data.get('actor')
                if actor and actor.id == processor.ego_vehicle_id:
                    actor_data['transform'] = carla.Transform(final_transform.location, final_transform.rotation)
                    break
    processor.manual_control_armed = False
    processor.manual_control_enabled = False
    processor.manual_control_pending = False
    processor.manual_control_actor = None
    processor.manual_control_state.update({
        'control': None,
        'steer_cache': 0.0,
        'reverse': False,
        'auto_reverse_engaged': False,
        'hand_brake_pressed': False,
        'autopilot_enabled': False,
        'autopilot_active': False,
        'camera_follow_local_offset': (0.0, 0.0, 0.0),
    })
    processor._manual_control_target_transform = None
    processor._manual_control_debug_printed = False
    processor._manual_camera_free_look_sources.clear()
    processor._pending_free_look_restores.clear()

def handle_manual_control_key(processor, key: int, is_pressed: bool) -> bool:
    """Route arrow-key input to manual control when active. Returns True if the key was consumed."""
    if not processor.manual_control_armed:
        return False

    state = processor.manual_control_state
    arrow_keys = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)
    if key in arrow_keys:
        return True
    if key == pygame.K_SPACE:
        state['hand_brake_pressed'] = is_pressed
        return True
    if key == pygame.K_q:
        if is_pressed:
            state['reverse'] = not state.get('reverse', False)
            state['auto_reverse_engaged'] = False
        return True
    if key == pygame.K_p:
        if is_pressed:
            state['autopilot_enabled'] = not state.get('autopilot_enabled', False)
        return True
    return False

def _manual_control_update(processor, dt: float) -> None:
    """Attach manual control when ready and send controls to the ego vehicle."""
    if processor.manual_control_enabled:
        actor = processor.manual_control_actor
        if not actor or not actor.is_alive:
            # The "not alive" read can be a transient false positive (stale proxy). Re-pend
            # the search instead of tearing down, so manual control re-attaches when the ego
            # reappears and arrows stay armed (never leaking to the camera mid-run).
            processor.request_manual_control()
            return
        processor._apply_manual_control(dt)
        return

    if not processor.manual_control_pending:
        return

    processor._manual_control_search_timer += dt
    if processor._manual_control_search_timer < 0.25:
        return
    processor._manual_control_search_timer = 0.0

    actor = processor._find_manual_control_actor()
    if actor is None:
        return

    # Attach to the ego actor
    try:
        actor.set_simulate_physics(True)
    except Exception:
        pass
    processor.manual_control_actor = actor
    processor.manual_control_enabled = True
    processor.manual_control_pending = False
    processor.manual_control_state.update({
        'control': carla.VehicleControl(),
        'steer_cache': 0.0,
        'reverse': False,
        'auto_reverse_engaged': False,
        'autopilot_active': False,
        'camera_follow_local_offset': (0.0, 0.0, 0.0),
    })
    processor._manual_control_start_camera_follow(actor)
    processor._manual_control_register_tick_callback()
    # Attach for the starting camera mode (chase/cockpit) now that the ego exists.
    processor._apply_playback_camera_attach()
    print("Manual ego control enabled. Arrow keys now steer the ego vehicle.")

def _find_manual_control_actor(processor) -> Optional[carla.Actor]:
    """Search the world for an ego vehicle spawned by ScenarioRunner."""
    world = processor.world
    if world is None and hasattr(processor, 'get_world'):
        try:
            world = processor.get_world()
        except Exception:
            world = None
    if world is None:
        editor = getattr(processor, 'editor', None)
        if editor and hasattr(editor, '_get_current_world'):
            try:
                world = editor._get_current_world()
            except Exception:
                world = None
    if world is None:
        return None

    try:
        if processor.ego_vehicle_id:
            actor = world.get_actor(processor.ego_vehicle_id)
            if actor and actor.is_alive and actor.get_location().z > -100.0:
                return actor
    except Exception:
        pass

    try:
        actors = list(world.get_actors().filter('vehicle.*'))
    except RuntimeError:
        return None

    preferred_roles = {'hero', 'ego_vehicle', 'ego', 'player'}
    for actor in actors:
        if not actor.is_alive:
            continue
        if actor.get_location().z < -100.0:
            continue
        role_name = actor.attributes.get('role_name', '').lower()
        if role_name in preferred_roles:
            return actor

    blueprint_id = getattr(processor, 'ego_vehicle_blueprint', None)
    transform_ref = getattr(processor, 'ego_vehicle_transform', None)
    expected_blueprints = []
    if blueprint_id:
        expected_blueprints.append(blueprint_id)

    scenario_transform = None
    scenario_blueprint = None
    scenario_data = getattr(processor, 'loaded_scenario_data', None)
    if isinstance(scenario_data, dict):
        ego_meta = scenario_data.get('ego_vehicle')
        if isinstance(ego_meta, dict):
            scenario_blueprint = ego_meta.get('type')
            if scenario_blueprint and scenario_blueprint not in expected_blueprints:
                expected_blueprints.append(scenario_blueprint)
            if transform_ref is None:
                loc_meta = ego_meta.get('location') or {}
                rot_meta = ego_meta.get('rotation') or {}
                scenario_transform = carla.Transform(
                    carla.Location(
                        loc_meta.get('x', 0.0),
                        loc_meta.get('y', 0.0),
                        loc_meta.get('z', 0.0),
                    ),
                    carla.Rotation(
                        rot_meta.get('pitch', 0.0),
                        rot_meta.get('yaw', 0.0),
                        rot_meta.get('roll', 0.0),
                    ),
                )

    if transform_ref is None and scenario_transform is not None:
        transform_ref = scenario_transform

    if not expected_blueprints and scenario_blueprint:
        expected_blueprints.append(scenario_blueprint)
    if not expected_blueprints:
        # Fallback to the default Lexus blueprint used by the editor when no info is cached.
        expected_blueprints.append('vehicle.lexus.utlexus')

    fallback_candidates = []
    for actor in actors:
        if not actor:
            continue
        if not actor.is_alive:
            refreshed = None
            try:
                refreshed = world.get_actor(actor.id)
            except Exception:
                refreshed = None
            if refreshed and refreshed.is_alive:
                actor = refreshed
            else:
                continue
        try:
            location = actor.get_location()
        except Exception:
            continue
        if location.z < -100.0:
            continue
        blueprint_match = actor.type_id in expected_blueprints
        if not blueprint_match and actor.type_id == 'vehicle.lexus.utlexus':
            blueprint_match = True
        if not blueprint_match:
            continue
        fallback_candidates.append(actor)

    if not fallback_candidates:
        return None

    if transform_ref:
        best_actor = None
        best_distance = float('inf')
        for actor in fallback_candidates:
            try:
                distance = actor.get_location().distance(transform_ref.location)
            except Exception:
                continue
            if distance < best_distance:
                best_distance = distance
                best_actor = actor
        if best_actor:
            return best_actor

    actor = fallback_candidates[0] if fallback_candidates else None
    return actor

def _apply_manual_control(processor, dt: float) -> None:
    """Gradually apply user steering/throttle inputs to the ego vehicle."""
    actor = processor.manual_control_actor
    if not actor or not actor.is_alive:
        processor.disable_manual_control()
        return

    processor._manual_control_apply_target_transform()

    # In autopilot mode, skip keyboard input — only camera follow is active
    agent_mode = processor.session.agent_mode
    if agent_mode == "autopilot" and processor.session.scenario_running:
        return

    state = processor.manual_control_state
    control = state.get('control')
    if control is None:
        control = carla.VehicleControl()
        state['control'] = control

    requested_autopilot = state.get('autopilot_enabled', False)
    active_autopilot = state.get('autopilot_active', False)
    if requested_autopilot != active_autopilot:
        try:
            actor.set_autopilot(requested_autopilot)
            state['autopilot_active'] = requested_autopilot
        except Exception:
            pass
        active_autopilot = state.get('autopilot_active', False)

    if active_autopilot:
        return

    keys = pygame.key.get_pressed()
    steer_cache = state.get('steer_cache', 0.0)
    steer_increment = 0.5 * dt
    left_pressed = keys[pygame.K_LEFT]
    right_pressed = keys[pygame.K_RIGHT]
    up_pressed = keys[pygame.K_UP]
    down_pressed = keys[pygame.K_DOWN]
    if left_pressed and not right_pressed:
        steer_cache -= steer_increment
    elif right_pressed and not left_pressed:
        steer_cache += steer_increment
    else:
        steer_cache = 0.0
    steer_cache = max(-0.7, min(0.7, steer_cache))
    state['steer_cache'] = steer_cache

    auto_reverse_engaged = state.get('auto_reverse_engaged', False)
    reverse_enabled = state.get('reverse', False)
    auto_reverse_stop_speed = 0.35
    try:
        velocity = actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    except Exception:
        speed = 0.0
    is_stopped = speed < auto_reverse_stop_speed

    if not auto_reverse_engaged and not reverse_enabled:
        if down_pressed and not up_pressed and is_stopped:
            reverse_enabled = True
            auto_reverse_engaged = True
    elif auto_reverse_engaged and up_pressed and is_stopped:
        reverse_enabled = False
        auto_reverse_engaged = False

    state['reverse'] = reverse_enabled
    state['auto_reverse_engaged'] = auto_reverse_engaged

    if auto_reverse_engaged:
        control.throttle = 1.0 if down_pressed else 0.0
        control.brake = 1.0 if up_pressed else 0.0
    else:
        control.throttle = 1.0 if up_pressed else 0.0
        control.brake = 1.0 if down_pressed else 0.0
    control.steer = round(steer_cache, 1)
    control.hand_brake = state.get('hand_brake_pressed', False)
    control.reverse = reverse_enabled

    try:
        actor.apply_control(control)
        processor.update_editor_ego_transform(actor.get_transform())
    except Exception:
        # Transient RPC failure (stale proxy / glitch): re-pend and re-attach rather than
        # tearing down, keeping arrows armed for the ego instead of releasing them to the camera.
        processor.request_manual_control()

def _manual_control_compute_local_offset(
    processor,
    actor_transform: carla.Transform,
    controller: Optional["TopDownCamera"],
) -> Tuple[float, float, float]:
    """Return the camera offset expressed in the actor's local frame."""
    if controller is None:
        return (0.0, 0.0, 0.0)
    actor_loc = actor_transform.location
    actor_rot = actor_transform.rotation

    dx = controller.center_x - actor_loc.x
    dy = controller.center_y - actor_loc.y
    dz = controller.height - actor_loc.z

    yaw_rad = math.radians(actor_rot.yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    return (local_x, local_y, dz)

def _manual_control_build_world_transform(
    actor_transform: carla.Transform,
    local_offset: Tuple[float, float, float],
) -> carla.Transform:
    """Convert a local offset into a world-space transform with fixed orientation."""
    local_x, local_y, local_z = local_offset
    actor_loc = actor_transform.location
    actor_rot = actor_transform.rotation

    yaw_rad = math.radians(actor_rot.yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    world_x = actor_loc.x + cos_yaw * local_x - sin_yaw * local_y
    world_y = actor_loc.y + sin_yaw * local_x + cos_yaw * local_y
    world_z = actor_loc.z + local_z

    world_location = carla.Location(x=world_x, y=world_y, z=world_z)
    world_rotation = carla.Rotation(pitch=-90.0, yaw=270.0, roll=0.0)
    return carla.Transform(world_location, world_rotation)

def _build_chase_world_transform(
    actor_transform: carla.Transform,
    distance_behind: float,
    height_above: float,
    pitch: float,
) -> carla.Transform:
    """Behind-the-vehicle chase pose (CARLA frame), matching awmini's spectator recipe:
    sit `distance_behind` m behind the ego along its heading, `height_above` m up, tilted
    down `pitch` degrees, facing the ego's heading."""
    loc = actor_transform.location
    yaw = actor_transform.rotation.yaw
    yaw_rad = math.radians(yaw)
    world_location = carla.Location(
        x=loc.x - distance_behind * math.cos(yaw_rad),
        y=loc.y - distance_behind * math.sin(yaw_rad),
        z=loc.z + height_above,
    )
    world_rotation = carla.Rotation(pitch=-float(pitch), yaw=yaw, roll=0.0)
    return carla.Transform(world_location, world_rotation)

def _build_follow_target(processor, actor_transform, local_offset):
    """Build the follow-camera target for the current playback_camera_mode.

    'topdown' -> overhead (editor default); 'chase' -> behind the vehicle. Every follow path
    routes through here, so toggling the mode (C) is reflected wherever the target is built.
    local_offset only applies to topdown; chase uses its own distance/height/pitch.
    """
    if processor.playback_camera_mode == "chase":
        return processor._build_chase_world_transform(
            actor_transform,
            processor.chase_distance_behind,
            processor.chase_height_above,
            processor.chase_pitch,
        )
    return processor._manual_control_build_world_transform(actor_transform, local_offset)

def _manual_control_apply_target_transform(processor, *, force: bool = False) -> None:
    """Apply the cached target transform to the controller and sensor."""
    if processor._is_camera_engine_attached():
        return  # engine drives the camera while attached to the ego (cockpit); don't fight it
    transform = processor._manual_control_target_transform
    controller = processor.camera_controller
    if transform is None or not controller:
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
        # Move the spectator to track the local ego when culling is on, so the CARLA server window
        # follows the ego and camera-relative culling renders correctly. The ego (not the spectator)
        # stays the large-map streaming anchor, so this only repositions the viewport. Gated on the
        # combined "Cull" option via editor.culling_enabled (off => spectator left alone, as before).
        if processor.world is not None and processor.session.culling_enabled:
            processor.world.get_spectator().set_transform(transform)
    except Exception:
        pass
    processor.camera_is_moving = False
    processor.camera_movement_timer = time.time()

def _manual_control_start_camera_follow(processor, actor: Optional[carla.Actor]) -> None:
    """Begin free-camera following for manual control."""
    controller = processor.camera_controller
    if not actor or not controller:
        processor.manual_control_state['camera_follow_local_offset'] = (0.0, 0.0, 0.0)
        processor._manual_control_target_transform = None
        return
    try:
        if not actor.is_alive:
            processor.manual_control_state['camera_follow_local_offset'] = (0.0, 0.0, 0.0)
            processor._manual_control_target_transform = None
            return
        actor_transform = actor.get_transform()
    except Exception:
        processor.manual_control_state['camera_follow_local_offset'] = (0.0, 0.0, 0.0)
        processor._manual_control_target_transform = None
        return

    local_offset = processor._manual_control_compute_local_offset(actor_transform, controller)
    processor.manual_control_state['camera_follow_local_offset'] = local_offset
    processor._manual_control_target_transform = processor._build_follow_target(actor_transform, local_offset)
    processor._manual_control_apply_target_transform()

def _manual_control_stop_camera_follow(processor) -> None:
    """Stop camera following and clear tick-based updates."""
    processor._manual_control_unregister_tick_callback()
    processor._manual_control_target_transform = None
    processor.manual_control_state['camera_follow_local_offset'] = (0.0, 0.0, 0.0)

def manual_camera_free_look_active(processor) -> bool:
    """Return True while user temporarily detaches the camera from the ego follow."""
    return bool(processor._manual_camera_free_look_sources)

def begin_manual_camera_free_look(processor, source: str) -> None:
    """Pause camera follow updates while ``source`` (mouse/keyboard) holds free-look."""
    if not processor.manual_control_enabled and not processor.playback_camera_follow_enabled:
        return
    key = source or "unknown"
    processor._manual_camera_free_look_sources.add(key)
    processor._pending_free_look_restores.pop(key, None)

def end_manual_camera_free_look(processor, source: str) -> None:
    """Resume follow once all free-look sources have released."""
    key = source or "unknown"
    if key not in processor._manual_camera_free_look_sources:
        return
    if len(processor._manual_camera_free_look_sources) > 1:
        processor._manual_camera_free_look_sources.remove(key)
        processor._pending_free_look_restores.pop(key, None)
        return
    if processor._restore_camera_follow_after_free_look():
        processor._manual_camera_free_look_sources.remove(key)
        processor._pending_free_look_restores.pop(key, None)
        return
    processor._pending_free_look_restores[key] = time.time() + processor._manual_camera_restore_retry_window

def _restore_camera_follow_after_free_look(processor) -> bool:
    """Snap the camera back onto the ego vehicle using the cached offset."""
    if processor.manual_control_enabled:
        actor = processor.manual_control_actor
        controller = processor.camera_controller
        if not actor or not controller:
            return False
        try:
            if not actor.is_alive:
                return False
            actor_transform = actor.get_transform()
        except Exception:
            return False
        local_offset = processor._manual_control_compute_local_offset(actor_transform, controller)
        processor.manual_control_state['camera_follow_local_offset'] = local_offset
        processor._manual_control_target_transform = processor._build_follow_target(
            actor_transform, local_offset
        )
        processor._manual_control_apply_target_transform(force=True)
        return True
    if processor.playback_camera_follow_enabled:
        actor = processor.playback_camera_follow_actor
        controller = processor.camera_controller
        if not actor or not controller:
            return False
        try:
            if not actor.is_alive:
                return False
            actor_transform = actor.get_transform()
        except Exception:
            return False
        local_offset = processor._manual_control_compute_local_offset(actor_transform, controller)
        processor._playback_camera_follow_local_offset = local_offset
        processor._playback_camera_target_transform = processor._build_follow_target(
            actor_transform, local_offset
        )
        processor._playback_camera_apply_target_transform(force=True)
        return True
    return False

def notify_manual_camera_adjustment(processor) -> None:
    """Recalculate camera offset when the user manually moves the view."""
    if processor.manual_camera_free_look_active:
        return
    if processor.manual_control_enabled:
        actor = processor.manual_control_actor
        controller = processor.camera_controller
        if not actor or not controller:
            return
        try:
            if not actor.is_alive:
                return
            actor_transform = actor.get_transform()
        except Exception:
            return
        local_offset = processor._manual_control_compute_local_offset(actor_transform, controller)
        processor.manual_control_state['camera_follow_local_offset'] = local_offset
        processor._manual_control_target_transform = processor._build_follow_target(actor_transform, local_offset)
        processor._manual_control_apply_target_transform()
    elif processor.playback_camera_follow_enabled:
        actor = processor.playback_camera_follow_actor
        controller = processor.camera_controller
        if not actor or not controller:
            return
        try:
            if not actor.is_alive:
                return
            actor_transform = actor.get_transform()
        except Exception:
            return
        local_offset = processor._manual_control_compute_local_offset(actor_transform, controller)
        processor._playback_camera_follow_local_offset = local_offset
        processor._playback_camera_target_transform = processor._build_follow_target(actor_transform, local_offset)
        processor._playback_camera_apply_target_transform()

def _process_pending_free_look_restores(processor) -> None:
    """Retry deferred camera snapping once the ego transform is available."""
    if not processor._pending_free_look_restores:
        return
    now = time.time()
    pending_items = list(processor._pending_free_look_restores.items())
    for key, deadline in pending_items:
        if key not in processor._manual_camera_free_look_sources:
            processor._pending_free_look_restores.pop(key, None)
            continue
        if not processor.manual_control_enabled and not processor.playback_camera_follow_enabled:
            processor._manual_camera_free_look_sources.discard(key)
            processor._pending_free_look_restores.pop(key, None)
            continue
        if processor._restore_camera_follow_after_free_look():
            processor._manual_camera_free_look_sources.discard(key)
            processor._pending_free_look_restores.pop(key, None)
            continue
        if now >= deadline:
            processor._manual_camera_free_look_sources.discard(key)
            processor._pending_free_look_restores.pop(key, None)

def _manual_control_register_tick_callback(processor) -> None:
    """Subscribe to CARLA world ticks to keep the camera in sync."""
    if not processor.world or processor._manual_control_tick_subscription is not None:
        return
    try:
        processor._manual_control_tick_subscription = processor.world.on_tick(processor._manual_control_on_world_tick)
    except Exception:
        processor._manual_control_tick_subscription = None

def _manual_control_unregister_tick_callback(processor) -> None:
    """Remove any active manual-control tick subscription."""
    if processor._manual_control_tick_subscription is None:
        return
    try:
        if processor.world:
            processor.world.remove_on_tick(processor._manual_control_tick_subscription)
    except Exception:
        pass
    finally:
        processor._manual_control_tick_subscription = None

def _manual_control_on_world_tick(processor, _snapshot) -> None:
    """Snap the camera to the ego offset each simulator tick."""
    if not processor.manual_control_enabled:
        processor._manual_control_unregister_tick_callback()
        return
    actor = processor.manual_control_actor
    if not actor:
        return
    try:
        if not actor.is_alive:
            return
        actor_transform = actor.get_transform()
    except Exception:
        return

    local_offset = processor.manual_control_state.get('camera_follow_local_offset', (0.0, 0.0, 0.0))
    processor._manual_control_target_transform = processor._build_follow_target(actor_transform, local_offset)
    processor._manual_control_apply_target_transform()


def is_manual_control_actor(processor, actor_id: Optional[int]) -> bool:
    return (
        actor_id is not None
        and processor.manual_control_actor is not None
        and processor.manual_control_actor.id == actor_id
    )
