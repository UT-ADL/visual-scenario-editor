"""VSE playback package (MiniRunner engine).

This package owns the ScenarioRunner/py_trees side of VSE. Importing it
guarantees SCENARIO_RUNNER_ROOT is on sys.path BEFORE any submodule imports
srunner.* (Python runs a package's __init__ before its submodules).

Layering: no module in this package may import pygame/pygame_gui or the
editor (vse_editor, vse). See tools/checks/import_smoke.py.
"""

import os
import sys
from pathlib import Path

_SCENARIO_RUNNER_ROOT = os.environ.get("SCENARIO_RUNNER_ROOT")
if _SCENARIO_RUNNER_ROOT:
    _scenario_runner_path = Path(_SCENARIO_RUNNER_ROOT).resolve()
    _scenario_runner_str = str(_scenario_runner_path)
    if _scenario_runner_str not in sys.path:
        sys.path.insert(0, _scenario_runner_str)
