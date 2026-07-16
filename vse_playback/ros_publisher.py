"""Forked ROS plan-publisher process (moved verbatim from vse_play.py).

_ros_plan_publisher_process is a multiprocessing target and must remain a
module-level function. AGENT_DEBUG gates its diagnostics.
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import carla

from agents.navigation.local_planner import RoadOption
from srunner.tools.route_manipulation import downsample_route, interpolate_trajectory

# Debug toggle for the forked CARLA Minimal Agent publisher process. When True,
# _ros_plan_publisher_process emits step markers and surfaces the agent's own
# rospy.loginfo to stderr. Set False to silence.
AGENT_DEBUG = True


def _ros_plan_publisher_process(
    raw_waypoints: List[Tuple[float, float, float, float]],
    downsample_interval: float,
    ros_publish_delay: float,
    agent_path: str,
    skip_interpolation: bool,
    cancel_now=None,
    cancel_done=None,
) -> None:
    """Child process entry: publish a plan via a ScenarioRunner agent with a fresh ROS node.

    cancel_now/cancel_done are optional multiprocessing.Event handles used for a graceful
    route cancel: the parent sets cancel_now and waits for cancel_done so the route is
    cancelled while this agent is still healthy (clean /planning/cancel_route reply) rather
    than during SIGTERM teardown (which yields a "returned no response" error).
    """
    parent_pid = os.getppid()

    # Graceful shutdown: convert SIGTERM into a cooperative exit so the
    # finally block can call cancel_route before the process dies.
    _shutdown = threading.Event()
    def _sigterm_handler(signum, frame):
        _shutdown.set()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Forked-agent diagnostics (gated by the module-level AGENT_DEBUG constant).
    # stderr + flush so messages reach the VSE terminal even though this runs in
    # a child process; the agent itself uses rospy.loginfo (surfaced below).
    def _dbg(msg):
        if AGENT_DEBUG:
            print("[agent-dbg] %s" % msg, file=sys.stderr, flush=True)

    _dbg("entry  ROS_MASTER_URI=%s  ROS_IP=%s  ROS_HOSTNAME=%s"
         % (os.environ.get("ROS_MASTER_URI"), os.environ.get("ROS_IP"), os.environ.get("ROS_HOSTNAME")))
    _dbg("agent_path=%s  raw_waypoints=%d  skip_interpolation=%s"
         % (agent_path, len(raw_waypoints or []), skip_interpolation))

    agent_file = None
    try:
        agent_file = str(Path(agent_path).expanduser().resolve())
        agent_dir = str(Path(agent_file).parent)
        module_name = Path(agent_file).stem
        if agent_dir and agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        module_agent = importlib.import_module(module_name)
        if hasattr(module_agent, "get_entry_point"):
            agent_class_name = str(module_agent.get_entry_point())
        else:
            agent_class_name = module_agent.__name__.title().replace("_", "")
        agent_class = getattr(module_agent, agent_class_name)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[MiniRunner] Failed to load agent from {agent_file or agent_path}: {exc}")
        return

    _dbg("agent class resolved: %s.%s" % (module_name, agent_class_name))

    if not raw_waypoints or len(raw_waypoints) < 2:
        print("[MiniRunner] No waypoints available for agent; skipping publish")
        return

    try:
        keypoints: List[carla.Location] = []
        for x, y, z, _yaw in raw_waypoints:
            keypoints.append(carla.Location(x=float(x), y=float(y), z=float(z)))

        route: List[Tuple] = []
        gps_route: List[Tuple] = []
        if len(keypoints) >= 2 and not skip_interpolation:
            try:
                gps_route, route = interpolate_trajectory(keypoints, hop_resolution=1.0)
            except Exception as exc:
                print(f"[MiniRunner] Route interpolation failed for minimal agent: {exc}")
                route = []
                gps_route = []

        if not route:
            for x, y, z, yaw in raw_waypoints:
                loc = carla.Location(x=float(x), y=float(y), z=float(z))
                rot = carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
                tf = carla.Transform(loc, rot)
                route.append((tf, RoadOption.LANEFOLLOW))
                gps_route.append((loc, RoadOption.LANEFOLLOW))
        else:
            try:
                sampled_ids = downsample_route(route, downsample_interval)
                route = [route[i] for i in sampled_ids if 0 <= i < len(route)]
                gps_route = [gps_route[i] for i in sampled_ids if 0 <= i < len(gps_route)]
            except Exception as exc:
                print(f"[MiniRunner] Route downsample failed; using full route: {exc}")
            if not route:
                for x, y, z, yaw in raw_waypoints:
                    loc = carla.Location(x=float(x), y=float(y), z=float(z))
                    rot = carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
                    tf = carla.Transform(loc, rot)
                    route.append((tf, RoadOption.LANEFOLLOW))
                    gps_route.append((loc, RoadOption.LANEFOLLOW))

        def _wait_for_ros_time(timeout=5.0) -> bool:
            """Wait until ROS time becomes non-zero.

            ROS time comes from the CARLA ROS bridge's /clock (sync mode). It
            must be live before we publish, because the agent stamps its Path
            and goal messages with rospy.Time.now(). This uses the public
            rospy clock directly (rospy is initialised by the agent's setup()
            -> rospy.init_node); no agent internals are touched.
            """
            import rospy
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    if not rospy.Time.now().is_zero():
                        return True
                except Exception:
                    pass
                time.sleep(0.1)
            return False

        if AGENT_DEBUG:
            # Surface the agent's own rospy.loginfo (e.g. "Published /initialpose",
            # "Publishing plan..") on stderr. Also catches rospy/tf startup INFO.
            _h = logging.StreamHandler(sys.stderr)
            _h.setFormatter(logging.Formatter("[agent-ros] %(levelname)s %(message)s"))
            logging.getLogger().addHandler(_h)
            logging.getLogger().setLevel(logging.INFO)

        _dbg("instantiating agent (runs setup(): init_node, params, TF lookup)...")
        try:
            agent = agent_class("")
        except Exception as exc:
            _dbg("AGENT setup() FAILED: %r" % exc)
            raise
        _dbg("agent instantiated OK")

        # Hand the route to the agent and let IT publish, through its own
        # ScenarioRunner interface -- no reaching into private members.
        # run_step(input_data, timestamp) publishes the plan + /initialpose +
        # goals and flips its own global_plan_published guard when
        # timestamp > init_goal_delay (carla_minimal_agent.run_step). input_data
        # is ignored by run_step, so passing {} avoids the sensor-queue block a
        # full agent __call__() would hit (the agent declares no sensors).
        #
        # No warm-up mini-route and no publish burst: the agent's publishers are
        # latch=True, so a subscriber that connects after this still receives the
        # last Path/goal/initialpose. The old warm-up published a second
        # /initialpose + /move_base_simple/goal, which placed/launched the ego
        # twice at the start (observed in sync mode) -- dropping it fixes that.
        _dbg("set_global_plan: %d route pts" % len(route))
        agent.set_global_plan(gps_route, route)
        if not _wait_for_ros_time(timeout=3.0):
            print("[MiniRunner] ROS time still zero after 3s; publishing anyway "
                  "(messages may carry a zero stamp)")
        try:
            gate_timestamp = float(getattr(agent, "init_goal_delay", 0.0)) + 1.0
            agent.run_step({}, gate_timestamp)
            _dbg("run_step published plan (published=%s)"
                 % getattr(agent, "global_plan_published", "?"))
            print("[MiniRunner] Route published via agent.run_step()")
        except Exception as exc:
            print(f"[MiniRunner] Failed to publish plan via run_step: {exc}")

        cancelled = False
        try:
            # Keep the ROS publishers alive so late subscribers can still receive the
            # latched Path message. This addresses first-run cases where an external
            # stack connects after the publish completes (otherwise only the final goal
            # may be observed).
            while not _shutdown.is_set():
                # Graceful cancel: when the parent requests a stop it sets cancel_now while
                # this process is still healthy, so the /planning/cancel_route round-trip
                # completes cleanly (no "returned no response"). Cancel exactly once here,
                # then signal cancel_done and exit; the finally below will not re-cancel.
                if cancel_now is not None and cancel_now.is_set():
                    try:
                        agent.set_global_plan([], [])
                        print("[MiniRunner] Route cancelled via agent.set_global_plan([], [])")
                    except Exception as exc:
                        print(f"[MiniRunner] Could not cancel route: {exc}")
                    cancelled = True
                    if cancel_done is not None:
                        try:
                            cancel_done.set()
                        except Exception:
                            pass
                    break
                try:
                    current_ppid = os.getppid()
                    # Exit if parent died (ppid changed or became 1/init)
                    if current_ppid != parent_pid or current_ppid == 1:
                        break
                except (OSError, AttributeError):
                    # Parent process no longer exists
                    break
                # Use Event.wait for instant wakeup on SIGTERM
                _shutdown.wait(timeout=0.1)
        except Exception:
            pass
        finally:
            # Fallback cancel only if the graceful in-loop cancel did not run (e.g. hard
            # SIGTERM or parent died). This may log a "returned no response" because it
            # happens during teardown, but the planner still cancels the route.
            if not cancelled:
                try:
                    agent.set_global_plan([], [])
                    print("[MiniRunner] Route cancelled via agent.set_global_plan([], [])")
                except Exception as exc:
                    print(f"[MiniRunner] Could not cancel route: {exc}")
                if cancel_done is not None:
                    try:
                        cancel_done.set()
                    except Exception:
                        pass
            try:
                agent.destroy()
            except Exception:
                pass
    finally:
        pass
