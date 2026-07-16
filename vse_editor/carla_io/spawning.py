"""Grounded actor spawning for the editor (moved verbatim from vse.py)."""

from __future__ import annotations

import carla

# Ground clearance for editor placeholders. Vehicle transforms sit near the
# chassis bottom; a walker transform is body-center (~bbox half-height above its
# feet), so walkers add their bounding box. Physics settles actors onto the
# ground when playback starts; the clearance only keeps them from clipping
# uneven terrain in the editor — and, crucially, keeps the SAVED Z high enough
# that the post-Play restore (which respawns via try_spawn_actor) doesn't get
# rejected for overlapping curb/sidewalk geometry.
VEHICLE_PLACEMENT_CLEARANCE_M = 0.20
PEDESTRIAN_PLACEMENT_CLEARANCE_M = 0.10


def spawn_vehicle_grounded(world, blueprint, x, y, rotation, ground_z,
                           *, start_buffer=VEHICLE_PLACEMENT_CLEARANCE_M,
                           step=0.05, max_buffer=2.0):
    """try_spawn_actor a (non-walker) actor at the lowest collision-free height.

    Escalates Z from ground+start_buffer upward until try_spawn_actor accepts it,
    so the actor is placed as low as the geometry allows while guaranteeing the
    saved Z is one CARLA will accept again on restore. Returns (actor, z) or
    (None, None) if no height up to ground+max_buffer works.
    """
    buf = start_buffer
    while buf <= max_buffer + 1e-6:
        z = ground_z + buf
        actor = world.try_spawn_actor(
            blueprint, carla.Transform(carla.Location(x, y, z), rotation)
        )
        if actor is not None:
            return actor, z
        buf = round(buf + step, 3)
    return None, None


def lowest_spawnable_z(world, blueprint, x, y, rotation, ground_z,
                       *, start_buffer=VEHICLE_PLACEMENT_CLEARANCE_M,
                       step=0.05, max_buffer=2.0):
    """Lowest Z (>= ground+start_buffer) where try_spawn_actor would succeed.

    Probes with a throwaway actor and returns the validated Z, for repositioning
    an existing actor (drag-release) to a height that survives restore. Falls
    back to ground+start_buffer if nothing up to ground+max_buffer works.
    """
    actor, z = spawn_vehicle_grounded(
        world, blueprint, x, y, rotation, ground_z,
        start_buffer=start_buffer, step=step, max_buffer=max_buffer,
    )
    if actor is not None:
        try:
            actor.destroy()
        except Exception:
            pass
        return z
    return ground_z + start_buffer
