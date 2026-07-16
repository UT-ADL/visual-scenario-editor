"""CARLA world utilities shared by playback modules (moved verbatim from
vse_play.py): walker grounding, temporary client timeouts, batched actor
destruction and CarlaDataProvider initialization / stale-entry purge.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import List, Optional

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from vse_common.geometry import get_ground_height

logger = logging.getLogger(__name__)

# Skeleton hits of walkers carry the Pedestrians label and can be filtered;
# their collision capsules report NONE and cannot (NONE is also legitimate
# terrain on custom maps).
_WALKER_GROUND_IGNORE_LABELS = frozenset({carla.CityObjectLabel.Pedestrians})


def _cdp_location_or_live(actor):
    """Per-tick cached location for CDP-registered actors (opt-09).

    Falls back to a live get_location RPC when the actor is not in the CDP
    map -- an adopted external/VIL ego may never be registered. Never
    swallows errors: a dead actor raises from the live call exactly as the
    direct RPC did before.
    """
    location = CarlaDataProvider.get_location(actor)
    if location is not None:
        return location
    return actor.get_location()


def _walker_ground_height(world, walker, sample_location, cached_map=None):
    """Ground height for walker grounding (snap/idle/destination) operations.

    When the walker is standing at/near the sample XY, its own standing height
    (actor center minus bounding-box half-height) is the ground truth — a
    raycast there would hit the walker itself: client commands are queued until
    the next world tick while cast_ray queries run immediately, so nothing can
    move the walker out of the ray within one tick, and its capsule hits are
    NONE-labeled (unfilterable by label). Only when the walker is away from the
    sample point (e.g. a teleport snap that has not been applied yet) is the
    raycast used, where a self-hit is impossible.
    """
    try:
        bbox_extent_z = float(getattr(walker.bounding_box.extent, "z", 1.0))
    except Exception:
        bbox_extent_z = 1.0
    try:
        loc = walker.get_location()
        if math.hypot(loc.x - sample_location.x, loc.y - sample_location.y) <= 0.7:
            return loc.z - bbox_extent_z
    except Exception:
        pass
    return get_ground_height(
        world,
        sample_location,
        debug=False,
        cached_map=cached_map,
        probe_on_miss=True,
        ignore_labels=_WALKER_GROUND_IGNORE_LABELS,
    )


# =============================================================================
# UTILITY FUNCTIONS
# Map detection, timeouts, actor cleanup, CARLA data provider initialization
# =============================================================================


def _temporary_client_timeout(client: Optional[carla.Client], timeout_s: float):
    """Context manager that temporarily sets the CARLA client timeout."""
    _state = getattr(_temporary_client_timeout, "_state", None)
    if _state is None:
        _state = threading.local()
        _temporary_client_timeout._state = _state  # type: ignore[attr-defined]

    class _TimeoutCtx:
        def __enter__(self_nonlocal):
            if client:
                depth = getattr(_state, "depth", 0)
                if depth <= 0:
                    try:
                        client.set_timeout(float(timeout_s))
                    except Exception:
                        pass
                _state.depth = depth + 1
            return self_nonlocal

        def __exit__(self_nonlocal, exc_type, exc, tb):
            if client:
                depth = getattr(_state, "depth", 1)
                depth = max(0, depth - 1)
                _state.depth = depth
                if depth == 0:
                    try:
                        client.set_timeout(10.0)
                    except Exception:
                        pass
            return False

    return _TimeoutCtx()


def _destroy_actor_ids(
    client: Optional[carla.Client],
    actor_ids: List[int],
    *,
    do_tick: bool,
    log_fn=None,
) -> None:
    """Destroy actors by id using a single batched command."""
    if not client:
        return
    cleaned: List[int] = []
    seen = set()
    for raw in actor_ids:
        try:
            actor_id = int(raw)
        except Exception:
            continue
        if actor_id <= 0 or actor_id in seen:
            continue
        seen.add(actor_id)
        cleaned.append(actor_id)
    if not cleaned:
        return
    try:
        commands = [carla.command.DestroyActor(actor_id) for actor_id in cleaned]
        client.apply_batch_sync(commands, do_tick=bool(do_tick))
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Failed to destroy actors %s: %s", cleaned, exc)
        if log_fn:
            try:
                log_fn(f"Failed to destroy actors {cleaned}: {exc}")
            except Exception:
                pass
        else:
            print(f"[Cleanup] Failed to destroy actors {cleaned}: {exc}")


def _cdp_purge_actor_id(actor_id: int) -> None:
    """Remove any CarlaDataProvider actor-map entries for ``actor_id`` before re-registering it.

    CarlaDataProvider is a process-global singleton that is never cleaned between runs (no
    cleanup()/reset() anywhere). Its actor maps are keyed by the proxy *object*, but
    ``get_velocity``/``get_location`` match by id and return the *first* id-match. Across runs the
    reused ego (the persistent placeholder, re-fetched as a new proxy each run) leaves a stale prior
    entry that shadows the fresh one; after an external-ego world switch that stale proxy stops being
    refreshed by on_carla_tick, so ``get_velocity`` returns its frozen ~0 speed and CollisionTest
    skips every real collision (speed < EPSILON => "not the actor's fault"). Dropping same-id entries
    right before register_actor leaves only the fresh, alive proxy — without disturbing other actors.
    """
    for _m in (CarlaDataProvider._actor_velocity_map,
               CarlaDataProvider._actor_location_map,
               CarlaDataProvider._actor_transform_map):
        for _k in [a for a in list(_m.keys()) if getattr(a, "id", None) == actor_id]:
            try:
                del _m[_k]
            except Exception:
                pass
    CarlaDataProvider._all_actors = None


def _init_carla_data_provider(world: carla.World, client: Optional[carla.Client] = None) -> bool:
    """
    Initialize ScenarioRunner's CarlaDataProvider without blocking on large maps.

    Returns True when the "fast init" path was used (no GlobalRoutePlanner build).
    """
    if not world:
        return False
    carla_map = None
    try:
        carla_map = world.get_map()
    except Exception:
        carla_map = None

    if client:
        try:
            CarlaDataProvider.set_client(client)
        except Exception:
            pass

    # Fast init: mirror CarlaDataProvider.set_world() minus the GRP build.
    #
    # We avoid constructing CARLA's GlobalRoutePlanner here because it can be slow and
    # (depending on CARLA build/driver/threading) has been observed to crash the Python
    # process during Play/Stop races. VSE drives agents via set_global_plan() and uses
    # explicit routes for criteria/weather, so CDP does not require a pre-built GRP.
    CarlaDataProvider._world = world  # type: ignore[attr-defined]
    try:
        CarlaDataProvider._sync_flag = bool(world.get_settings().synchronous_mode)  # type: ignore[attr-defined]
    except Exception:
        CarlaDataProvider._sync_flag = False  # type: ignore[attr-defined]
    CarlaDataProvider._map = carla_map  # type: ignore[attr-defined]
    try:
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()  # type: ignore[attr-defined]
    except Exception:
        CarlaDataProvider._blueprint_library = None  # type: ignore[attr-defined]
    CarlaDataProvider._grp = None  # type: ignore[attr-defined]
    try:
        CarlaDataProvider.generate_spawn_points()
    except Exception:
        pass
    try:
        CarlaDataProvider.prepare_map()
    except Exception:
        pass
    return True
