"""Standalone CLI for VSE scenario playback (moved verbatim from
vse_play.py): argument parsing, CARLA connection, external-ego / tick-mode
resolution and main().
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import carla

from vse_common.env import env_float, env_int
from vse_playback.runner import (
    MiniRunner,
    _expected_ego_roles,
    _find_external_ego,
    _scenario_ego_blueprint_id,
    _scenario_has_ego,
)

logger = logging.getLogger(__name__)

def _resolve_json_arg(scenario_arg: str) -> Path:
    """Resolve a scenario argument to an existing JSON file path."""
    candidates: List[Path] = []

    expanded = Path(os.path.expanduser(scenario_arg)).resolve()
    candidates.append(expanded)
    if expanded.suffix != ".json":
        candidates.append(expanded.with_suffix(".json"))

    cwd = Path.cwd()
    rel = cwd / scenario_arg
    candidates.append(rel)
    if rel.suffix != ".json":
        candidates.append(rel.with_suffix(".json"))

    script_dir = Path(__file__).resolve().parent
    script_rel = script_dir / scenario_arg
    candidates.append(script_rel)
    if script_rel.suffix != ".json":
        candidates.append(script_rel.with_suffix(".json"))

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not find scenario JSON for '{scenario_arg}'. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a VSE scenario JSON via the built-in MiniRunner (no ScenarioRunner, no XML).",
    )
    parser.add_argument(
        "scenario",
        help="Scenario JSON path or name (e.g., my_scenario.json or /path/to/my_scenario.json).",
    )
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=env_int("CARLA_PORT", 2000))
    parser.add_argument("--client-timeout", type=float, default=env_float("CARLA_TIMEOUT", 10.0))
    parser.add_argument(
        "--tick-mode",
        choices=("auto", "own", "ros"),
        default="auto",
        help="World tick source: auto (match VSE), own (call world.tick), ros (wait_for_tick).",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=None,
        help="Fixed delta seconds when forcing sync mode (default: world setting or 0.05).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=env_float("VSE_SCENARIO_RUNNER_TIMEOUT", 18000.0),
        help="Scenario timeout seconds (default: $VSE_SCENARIO_RUNNER_TIMEOUT or 18000).",
    )
    parser.add_argument(
        "--wait-for-ego",
        action="store_true",
        help="Attach to an existing ego vehicle in the world (like VSE external-ego mode).",
    )
    parser.add_argument(
        "--external-ego-actor-id",
        type=int,
        default=None,
        help="Explicit external ego actor id to attach/warp (implies --wait-for-ego).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to a ScenarioRunner-compatible agent Python file (required when using an external ego / tick-mode=ros).",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def _configure_logging(debug_enabled: bool) -> None:
    # Configure logging based on --debug flag
    if debug_enabled:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("Debug logging enabled via --debug flag")
    else:
        # Only show warnings and errors by default
        logging.basicConfig(
            level=logging.WARNING,
            format='%(name)s - %(levelname)s - %(message)s'
        )


def _load_scenario_json(parser: argparse.ArgumentParser, scenario_arg: str) -> Tuple[Path, dict]:
    try:
        json_path = _resolve_json_arg(scenario_arg)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            scenario_data = json.load(fh)
    except Exception as exc:
        parser.error(f"Failed to read scenario JSON: {exc}")
    return json_path, scenario_data


def _connect_carla(
    parser: argparse.ArgumentParser,
    *,
    host: str,
    port: int,
    client_timeout: float,
) -> Tuple[carla.Client, carla.World]:
    try:
        client = carla.Client(host, int(port))
        client.set_timeout(float(client_timeout))
        world = client.get_world()
    except Exception as exc:
        parser.error(f"Failed to connect to CARLA at {host}:{port}: {exc}")
    return client, world


def _resolve_external_ego(
    parser: argparse.ArgumentParser,
    *,
    world: carla.World,
    args: argparse.Namespace,
    scenario_data: dict,
) -> Tuple[bool, Optional[int], Optional[carla.Actor]]:
    scenario_has_ego = _scenario_has_ego(scenario_data)
    expected_roles = _expected_ego_roles(scenario_data)
    ego_blueprint_id = _scenario_ego_blueprint_id(scenario_data)

    external_ego_id = None
    external_ego_actor = None
    if args.external_ego_actor_id is not None:
        try:
            actor = world.get_actor(int(args.external_ego_actor_id))
        except Exception:
            actor = None
        if not actor or not getattr(actor, "is_alive", False):
            parser.error(f"External ego actor id {args.external_ego_actor_id} not found/alive in world.")
        external_ego_id = int(args.external_ego_actor_id)
        external_ego_actor = actor
    elif args.tick_mode in ("auto", "ros") or args.wait_for_ego:
        # Auto-detect an already-spawned ego and use it as the ego vehicle source + tick authority.
        external_actor = _find_external_ego(world, expected_roles, blueprint_id=ego_blueprint_id)

        # Optional grace period to allow external stacks to spawn/register the ego
        # before we decide to run in standalone mode.
        #
        # Default is 1s (helps avoid a race where the ego exists but hasn't shown
        # up in the actor list yet). Override via $VSE_EXTERNAL_EGO_DETECT_TIMEOUT_S.
        detect_timeout_s = env_float("VSE_EXTERNAL_EGO_DETECT_TIMEOUT_S", 1.0)
        if args.wait_for_ego:
            detect_timeout_s = max(detect_timeout_s, 10.0)
        detect_timeout_s = max(0.0, detect_timeout_s)

        if external_actor is None and detect_timeout_s > 0.0:
            deadline = time.time() + detect_timeout_s
            while external_actor is None and time.time() < deadline:
                try:
                    world.wait_for_tick(0.25)
                except Exception:
                    time.sleep(0.25)
                external_actor = _find_external_ego(world, expected_roles, blueprint_id=ego_blueprint_id)

        if external_actor is not None:
            external_ego_id = int(external_actor.id)
            external_ego_actor = external_actor
        elif args.wait_for_ego:
            parser.error("External ego requested but not found in world actor list.")

    return scenario_has_ego, external_ego_id, external_ego_actor


def _resolve_tick_mode_and_agent(
    parser: argparse.ArgumentParser,
    *,
    args: argparse.Namespace,
    scenario_has_ego: bool,
    external_ego_id: Optional[int],
) -> Tuple[str, Optional[str], bool]:
    tick_mode = args.tick_mode
    if tick_mode == "auto":
        tick_mode = "ros" if external_ego_id is not None else "own"

    if tick_mode == "ros" and external_ego_id is None:
        parser.error("Tick mode 'ros' requires an external ego vehicle, but none was detected in the world.")

    if tick_mode == "ros":
        if not args.agent:
            parser.error("External ego detected; --agent /path/to/agent.py is required to publish the route.")
    else:
        if args.agent:
            parser.error("--agent was provided but no external ego is in use (tick-mode is not 'ros').")

    agent_path: Optional[str] = None
    if args.agent:
        try:
            agent_candidate = Path(str(args.agent)).expanduser().resolve()
        except Exception:
            agent_candidate = None
        if not agent_candidate or not agent_candidate.exists() or not agent_candidate.is_file():
            parser.error(f"Agent file not found: {args.agent}")
        agent_path = str(agent_candidate)

    wait_for_ego = bool(args.wait_for_ego or args.external_ego_actor_id is not None)
    if not wait_for_ego:
        wait_for_ego = bool(tick_mode == "ros" and scenario_has_ego and external_ego_id is not None)

    return tick_mode, agent_path, wait_for_ego


def _resolve_fixed_delta(world: carla.World, fixed_delta_arg: Optional[float]) -> float:
    fixed_delta = fixed_delta_arg
    if fixed_delta is None:
        fixed_delta = 0.05
        try:
            settings = world.get_settings()
            if settings and getattr(settings, "fixed_delta_seconds", None):
                fixed_delta = float(settings.fixed_delta_seconds)
        except Exception:
            fixed_delta = 0.05
    return float(fixed_delta)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(bool(args.debug))

    json_path, scenario_data = _load_scenario_json(parser, args.scenario)
    client, world = _connect_carla(
        parser,
        host=str(args.host),
        port=int(args.port),
        client_timeout=float(args.client_timeout),
    )
    scenario_has_ego, external_ego_id, external_ego_actor = _resolve_external_ego(
        parser,
        world=world,
        args=args,
        scenario_data=scenario_data,
    )
    tick_mode, agent_path, wait_for_ego = _resolve_tick_mode_and_agent(
        parser,
        args=args,
        scenario_has_ego=scenario_has_ego,
        external_ego_id=external_ego_id,
    )

    if external_ego_actor is not None and getattr(external_ego_actor, "is_alive", False):
        try:
            role_name = external_ego_actor.attributes.get("role_name", "")
        except Exception:
            role_name = ""
        try:
            type_id = external_ego_actor.type_id
        except Exception:
            type_id = ""
        print(f"[MiniRunner] Using external ego actor {external_ego_id} ({type_id}) role='{role_name}'")

    fixed_delta = _resolve_fixed_delta(world, args.fixed_delta)

    timeout_s = max(30.0, float(args.timeout_s or 18000.0))

    finish: dict = {"reason": None}

    def _on_finish(reason: str):
        finish["reason"] = reason
        print(f"[MiniRunner] Scenario finished ({reason})")

    runner = MiniRunner(
        client=client,
        world=world,
        json_path=str(json_path),
        tick_mode=str(tick_mode),
        fixed_delta=float(fixed_delta),
        wait_for_ego=bool(wait_for_ego),
        ego_role_name="ego_vehicle",
        timeout_s=float(timeout_s),
        external_ego_actor_id=external_ego_id,
        log_fn=lambda msg: print(f"[MiniRunner] {msg}"),
        debug=bool(args.debug),
        on_finish=_on_finish,
        agent_path=agent_path,
    )

    print(f"[MiniRunner] Running {json_path} (mode={tick_mode}, timeout={timeout_s}s)")
    runner.start()

    thread = getattr(runner, "_thread", None)
    if not thread:
        return 1

    try:
        while thread.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("[MiniRunner] Stop requested by user.")
        try:
            runner.request_stop()
        except Exception:
            pass

        # Give the worker thread a chance to clean up, but do not crash on a
        # second Ctrl-C; instead force cleanup and exit.
        deadline = time.time() + 5.0
        while thread.is_alive() and time.time() < deadline:
            try:
                thread.join(timeout=0.25)
            except KeyboardInterrupt:
                break

        try:
            runner._cleanup("stopped by user")
        except KeyboardInterrupt:
            pass
        except Exception:
            pass

        try:
            thread.join(timeout=1.0)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        return 130

    reason = str(finish.get("reason") or "")
    reason_l = reason.lower()
    # Abnormal terminations exit nonzero: a scenario failure/timeout, or a lost
    # CARLA server (the run couldn't complete — fix-08).
    if (reason_l.startswith("failed") or reason_l.startswith("timeout")
            or "server lost" in reason_l):
        return 1
    return 0
