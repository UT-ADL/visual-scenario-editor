"""SessionState: a read-only live view of the editor's session flags (step-23).

The processor (and the scenario_io functions it hosts) historically read
editor session state through getattr-with-default probes. SessionState keeps
exactly those semantics — every property is a live `getattr(editor, name,
default)` against the editor object, with the per-site default the monolith
used — while giving the reads one typed, greppable seam.

Deliberately NOT storage: the editor's ~40 existing write sites stay the
single source of truth, so no synchronization (and no drift) is possible.
With `editor=None` (headless harnesses) every property returns its default,
matching the old `self.editor and self.editor.X` falsiness guards.

The one exception is `scene_preview_destroyed`, which the scenario-load path
WRITES (setter forwards to the editor). Per the refactor plan, SessionState
plus the CommandHistory on_change callback are the entire mechanism budget —
no events, no polling.
"""


class SessionState:
    def __init__(self, editor):
        self._editor = editor

    def _get(self, name, default):
        if self._editor is None:
            return default
        return getattr(self._editor, name, default)

    # -- playback / lifecycle ------------------------------------------------

    @property
    def scenario_running(self):
        return self._get("scenario_running", False)

    @property
    def scene_preview_destroyed(self):
        return self._get("scene_preview_destroyed", False)

    @scene_preview_destroyed.setter
    def scene_preview_destroyed(self, value):
        if self._editor is not None:
            self._editor.scene_preview_destroyed = value

    @property
    def saved_scene_vehicles(self):
        return self._get("saved_scene_vehicles", [])

    # -- external ego (VIL) ---------------------------------------------------

    @property
    def external_ego_actor(self):
        return self._get("external_ego_actor", None)

    @property
    def external_ego_actor_id(self):
        return self._get("external_ego_actor_id", None)

    @property
    def _external_swap_current_id(self):
        return self._get("_external_swap_current_id", None)

    # -- connection / world ---------------------------------------------------

    @property
    def client(self):
        return self._get("client", None)

    @property
    def cached_map(self):
        return self._get("cached_map", None)

    @property
    def _map_refresh_disabled(self):
        return self._get("_map_refresh_disabled", False)

    # -- editor modes ----------------------------------------------------------

    @property
    def agent_mode(self):
        return self._get("agent_mode", "autopilot")

    @property
    def camera_stream_enabled(self):
        return self._get("camera_stream_enabled", True)

    @property
    def culling_enabled(self):
        # editor-side this is a computed @property (culling_distance_m > 0)
        return self._get("culling_enabled", False)
