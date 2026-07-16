#!/usr/bin/env python

"""vse_play.py — thin entrypoint for VSE scenario playback.

The playback engine lives in the vse_playback/ package (runner.py holds
MiniRunner, scenario.py the vse_play scenario class, cli.py the argument
parsing and main()). This file preserves the two historical contracts:

- `python vse_play.py <scenario.json> [flags]` — standalone playback
  (no editor needed), exactly as before.
- `from vse_play import MiniRunner` — the editor's in-process import
  (vse.py run_scenario), which also installs the atexit/SIGTERM/SIGINT
  cleanup handlers unless VSE_PLAY_INSTALL_HANDLERS disables them.
"""

import sys

from vse_playback.cli import main
from vse_playback.runner import MiniRunner
from vse_playback.scenario import install_handlers_from_env, vse_play

# Historical import-time handler gate: plain `import vse_play` installs the
# cleanup handlers unless VSE_PLAY_INSTALL_HANDLERS disables it; running as a
# script defaults to NOT installing (main() handles Ctrl-C itself).
install_handlers_from_env(default="0" if __name__ == "__main__" else "1")

if __name__ == "__main__":
    sys.exit(main())
