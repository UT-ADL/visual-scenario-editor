#!/usr/bin/env python3
"""Visual Scenario Editor -- entrypoint shim.

The implementation lives in the vse_editor/ package. This file stays at the
repo root under this name because self-relaunch replays
``[sys.executable] + sys.argv`` and callers import the module as ``vse``.
"""

import sys
import warnings

# Suppress noisy pygame_gui label size warnings while retaining existing layout.
warnings.filterwarnings(
    "ignore",
    message="Label Rect is too small for text:",
)
warnings.filterwarnings(
    "ignore",
    message="Pygame GUI event types can now be used directly as event.type",
    category=DeprecationWarning,
)


# Import CARLA
try:
    import carla
    from carla import ColorConverter as cc
except ImportError:
    print("CARLA Python API not found. Please add CARLA to your PYTHONPATH.")
    print("Example: export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
    sys.exit(1)

# Re-exports: same objects as their vse_editor/vse_common homes (tests assert
# identity -- never wrap or redefine these).
from vse_common.geometry import (
    cast_ray_with_tile_offset_compensation,
    get_ground_height,
    is_large_map as is_large_map_name,
)
from vse_common.traffic_lights import (
    TRAFFIC_LIGHT_FINGERPRINT_SCALE,
    compute_traffic_light_fingerprint,
    normalize_traffic_light_fingerprint,
)
from vse_editor.carla_io.camera_stream import CameraImageProcessor
from vse_editor.carla_io.profiles import ConnectionProfile
from vse_editor.app.editor import VisualScenarioEditor
from vse_editor.app.main import main

# Entry point for script execution
if __name__ == "__main__":
    sys.exit(main())
