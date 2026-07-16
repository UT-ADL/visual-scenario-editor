"""Client-side cache for CARLA actor bounding boxes.

CARLA 0.9.16 ("Synchronized actor BoundingBox between server and client", PR #8334) turned
every ``actor.bounding_box`` access into a blocking ``get_actor_bounding_box`` RPC — measured
~11 ms/call against a local 0.9.16 server, vs a free client-cached attribute on 0.9.15. VSE
reads bounding boxes in per-mouse-event, per-frame and per-tick paths, so those paths must go
through this cache; a spawned actor's bounding box is static for its lifetime, which is exactly
the 0.9.15 semantics this restores.

Call ``clear_bounding_box_cache()`` whenever the world is (re)loaded: actor ids restart per
episode, so a stale entry could hand a recycled id another blueprint's box.
"""

from __future__ import annotations

from typing import Dict

import carla

_BBOX_CACHE: Dict[int, carla.BoundingBox] = {}


def cached_bounding_box(actor: carla.Actor) -> carla.BoundingBox:
    """Return the actor's bounding box, reading it from the actor at most once per actor id."""
    bbox = _BBOX_CACHE.get(actor.id)
    if bbox is None:
        bbox = actor.bounding_box  # blocking RPC on CARLA >= 0.9.16
        _BBOX_CACHE[actor.id] = bbox
    return bbox


def clear_bounding_box_cache() -> None:
    """Drop all cached boxes; call on world (re)load, where actor ids restart."""
    _BBOX_CACHE.clear()
