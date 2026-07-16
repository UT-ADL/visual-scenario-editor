"""Entrypoint main() for the Visual Scenario Editor (moved verbatim from
vse.py, step-43, Phase 7). The `if __name__ == "__main__"` block stays in the
root vse.py shim (self-relaunch replays sys.argv, so vse.py remains the
launched script)."""

import argparse
import logging
import os
import traceback

from vse_editor.app.editor import VisualScenarioEditor

logger = logging.getLogger(__name__)


def main():
    """Entry point for Visual Scenario Editor."""
    parser = argparse.ArgumentParser(description="Visual Scenario Editor", add_help=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug logs.")
    parser.add_argument("--port", type=int, help="CARLA port (default 2000).")
    parser.add_argument("--carla-path", dest="carla_path", help="Path to CarlaUE4.sh.")
    parser.add_argument("--no-culling", action="store_true",
                        help="Start with mesh culling off (max_culling_distance=0).")
    parser.add_argument("--cull-distance", dest="cull_distance", type=float, default=None,
                        help="Initial cull distance in metres (0 = off; default: remembers your last choice, otherwise Off).")
    # Legacy positional args (carla_path, port) preserved for compatibility.
    parser.add_argument("legacy_carla_path", nargs="?", default=None)
    parser.add_argument("legacy_port", nargs="?", type=int, default=None)

    args = parser.parse_args()

    # Configure logging based on --debug flag
    if args.debug:
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

    carla_path = args.carla_path or args.legacy_carla_path
    port = args.port if args.port is not None else (args.legacy_port if args.legacy_port is not None else 2000)

    # CLI culling overrides -> env vars read by VisualScenarioEditor._load_culling (CLI wins over a
    # pre-set env var; --no-culling beats --cull-distance). If neither flag is given, leave any
    # existing VSE_CULLING / VSE_CULL_DISTANCE untouched.
    if args.no_culling:
        os.environ['VSE_CULLING'] = '0'
    if args.cull_distance is not None:
        os.environ['VSE_CULL_DISTANCE'] = str(args.cull_distance)

    print("Visual Scenario Editor for CARLA")
    print("================================")
    print("Shift+Right Click to move camera to position")
    print("ESC exits during loading, cancels operations after loading")
    print("Alt+F4 or close button (X) to exit with confirmation")

    try:
        editor = VisualScenarioEditor(carla_path, port, debug=args.debug)
        editor.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

    return 0
