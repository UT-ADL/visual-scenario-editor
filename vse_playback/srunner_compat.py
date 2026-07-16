"""Compatibility shim so VSE playback runs against stock CARLA ScenarioRunner,
not only the UT-ADL fork.

Stock ScenarioRunner's ``RunningRedLightTest.get_traffic_light_waypoints`` does
``wpx.next(0.5)[0]`` — it indexes the successor list *before* checking it is
non-empty, so it raises ``IndexError: list index out of range`` on any
traffic-light lane whose ``waypoint.next(0.5)`` returns ``[]`` (a lane end or map
edge). ``RunningRedLightTest.__init__`` walks *every* traffic light in the map, so
one such lane aborts the whole scenario during construction. The UT-ADL fork
guards that access; stock 0.9.15 and 0.9.16 do not.

VSE builds ``RunningRedLightTest`` for the ego on every run, so we subclass it and
override the one method with the fork's guarded body. The guarded logic is
identical to the fork's, so playback output stays byte-identical when running
against the fork.

CARLA renamed the waypoint property ``is_intersection`` (0.9.15) to ``is_junction``
(0.9.16); both are aliases for the same value. The name is resolved once against
the active ``carla.Waypoint`` (preferring the fork's ``is_intersection`` when
present) so the override works on both client versions.

Body adapted from scenario_runner
``srunner/scenariomanager/scenarioatomics/atomic_criteria.py``
(``RunningRedLightTest.get_traffic_light_waypoints``); the sole functional change
is guarding the empty-``next()`` list access.
"""

from __future__ import annotations

import numpy as np

import carla

from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    RunningRedLightTest as _SrunnerRunningRedLightTest,
)

# carla.Waypoint.is_intersection (0.9.15) was renamed is_junction (0.9.16); same
# value. Resolve once: prefer the fork's original name for byte-identical parity,
# fall back to the new name on clients where the old one is gone.
_JUNCTION_ATTR = "is_intersection" if hasattr(carla.Waypoint, "is_intersection") else "is_junction"


def _wp_is_junction(waypoint) -> bool:
    """Return whether a waypoint lies in a junction, across CARLA versions."""
    return bool(getattr(waypoint, _JUNCTION_ATTR))


class GuardedRunningRedLightTest(_SrunnerRunningRedLightTest):
    """RunningRedLightTest that tolerates traffic-light lanes with no successor.

    Overrides only ``get_traffic_light_waypoints`` to guard the empty-``next()``
    list access that raises ``IndexError`` in stock ScenarioRunner. Everything else
    (``__init__``, ``update``, scoring) is inherited unchanged.
    """

    def get_traffic_light_waypoints(self, traffic_light):
        """Return the trigger-area location and stop waypoints for a traffic light."""
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not _wp_is_junction(wpx):
                next_wp = wpx.next(0.5)  # guard: empty at lane ends -> stock srunner's [0] raises
                if next_wp and not _wp_is_junction(next_wp[0]):
                    wpx = next_wp[0]
                else:
                    break
            wps.append(wpx)

        return area_loc, wps
