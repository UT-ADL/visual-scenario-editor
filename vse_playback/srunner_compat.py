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

fix-28: ``update`` is likewise overridden with a copy of the fork's body
(``RunningRedLightTest.update``, atomic_criteria.py lines 1659-1749); the sole
change is reading the ego's bounding-box extent through
``vse_common.actor_cache.cached_bounding_box`` — on CARLA 0.9.16 the inherited
body's raw ``self.actor.bounding_box`` read is a blocking ~11 ms game-thread RPC
issued once per behavior-tree tick for the entire armed run. The fork's and stock
master's ``update`` bodies are functionally identical (line wrapping aside), so
playback parity holds against both, as with the waypoints override above.
"""

from __future__ import annotations

import numpy as np
import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    RunningRedLightTest as _SrunnerRunningRedLightTest,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType

from vse_common.actor_cache import cached_bounding_box

# carla.Waypoint.is_intersection (0.9.15) was renamed is_junction (0.9.16); same
# value. Resolve once: prefer the fork's original name for byte-identical parity,
# fall back to the new name on clients where the old one is gone.
_JUNCTION_ATTR = "is_intersection" if hasattr(carla.Waypoint, "is_intersection") else "is_junction"


def _wp_is_junction(waypoint) -> bool:
    """Return whether a waypoint lies in a junction, across CARLA versions."""
    return bool(getattr(waypoint, _JUNCTION_ATTR))


class GuardedRunningRedLightTest(_SrunnerRunningRedLightTest):
    """RunningRedLightTest that tolerates traffic-light lanes with no successor.

    Overrides ``get_traffic_light_waypoints`` to guard the empty-``next()`` list
    access that raises ``IndexError`` in stock ScenarioRunner, and ``update`` to
    read the ego bounding box through the client cache (fix-28; a raw read is a
    blocking per-tick RPC on CARLA >= 0.9.16). Everything else (``__init__``,
    scoring) is inherited unchanged, and both overrides are behavior-identical
    copies of the inherited bodies.
    """

    # fix-28 provenance: copied verbatim from scenario_runner
    # srunner/scenariomanager/scenarioatomics/atomic_criteria.py
    # (RunningRedLightTest.update, lines 1659-1749; fork and stock bodies are functionally
    # identical). Sole change: the per-tick raw ``self.actor.bounding_box`` read (a blocking
    # ~11 ms RPC on CARLA >= 0.9.16) goes through cached_bounding_box.
    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        transform = CarlaDataProvider.get_transform(self.actor)
        location = transform.location
        if location is None:
            return new_status

        veh_extent = cached_bounding_box(self.actor).extent.x  # fix-28: cached (was self.actor.bounding_box)

        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0, 0), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0, 0), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:

            if self.debug:
                z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(155, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 155, 0)
                else:
                    color = carla.Color(155, 155, 0)
                self._world.debug.draw_point(center + carla.Location(z=z), size=0.2, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(
                        wp.transform.location + carla.Location(x=1, z=z), text, color=color, life_time=0.01)
                    self._world.debug.draw_point(
                        wp.transform.location + carla.Location(z=z), size=0.1, color=color, life_time=0.01)

            center_loc = carla.Location(center)

            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:

                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = CarlaDataProvider.get_transform(self.actor).get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and ve_dir.dot(wp_dir) > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):

                        self.test_status = "FAILURE"
                        self.actual_value += 1
                        location = traffic_light.get_transform().location
                        red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION, frame=GameTime.get_frame())
                        red_light_event.set_message(
                            "Agent ran a red light {} at (x={}, y={}, z={})".format(
                                traffic_light.id,
                                round(location.x, 3),
                                round(location.y, 3),
                                round(location.z, 3)))
                        red_light_event.set_dict({'id': traffic_light.id, 'location': location})

                        self.events.append(red_light_event)
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

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
