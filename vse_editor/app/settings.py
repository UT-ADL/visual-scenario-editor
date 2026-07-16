"""Settings persistence (~/.cache/vse) + culling / play-camera appliers
(duck-typed module functions, first arg `editor`). Moved verbatim from vse.py
(step-39, Phase 7). Class attrs (_LEGACY_SETTINGS_FILES, PLAY_CAMERA_MODES,
CULLING_PRESETS, CULLING_DEFAULT_M) stay on VisualScenarioEditor; culling_enabled
stays a real @property on the class delegating here.
"""

import json
import os
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


def _get_vse_cache_dir(editor) -> Path:
    """Return the base cache directory for VSE."""
    cache_base = os.environ.get('XDG_CACHE_HOME')
    if not cache_base:
        cache_base = os.path.join(Path.home(), '.cache')
    return Path(cache_base) / 'vse'

def _get_settings_path(editor) -> Path:
    """Return the unified VSE settings file (replaces the per-feature last_*.json caches)."""
    return editor._get_vse_cache_dir() / 'settings.json'

def _load_settings(editor) -> Dict[str, Any]:
    """Read the unified settings file as a dict (empty on missing/corrupt; unlink if corrupt)."""
    settings_path = editor._get_settings_path()
    if not settings_path.is_file():
        return {}
    try:
        data = json.loads(settings_path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[Settings] Failed to load {settings_path.name}: {exc}")
        try:
            settings_path.unlink()
        except Exception:
            pass
        return {}

def _read_settings_section(editor, key: str) -> Dict[str, Any]:
    """Return a copy of one dict section of the unified settings file ({} if absent/non-dict)."""
    section = editor._load_settings().get(key)
    return dict(section) if isinstance(section, dict) else {}

def _write_settings_section(editor, key: str, payload: Any) -> None:
    """Merge one section into the unified settings file (atomic temp-file + os.replace)."""
    settings_path = editor._get_settings_path()
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = editor._load_settings()
        data[key] = payload
        data['updated'] = time.time()
        tmp_path = settings_path.with_name(settings_path.name + '.tmp')
        tmp_path.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp_path), str(settings_path))
    except Exception as exc:
        print(f"[Settings] Failed to write section '{key}': {exc}")

def _migrate_legacy_settings(editor) -> None:
    """One-time fold of the legacy per-feature cache files into settings.json, then remove them.

    Runs before any _load_*_cache() in __init__. Copies each legacy file's raw JSON under its
    section key (the section loaders already normalize the shapes), writes settings.json, and
    deletes the legacy files. No-op once settings.json exists.
    """
    settings_path = editor._get_settings_path()
    if settings_path.is_file():
        return  # already migrated
    cache_dir = editor._get_vse_cache_dir()
    migrated: Dict[str, Any] = {}
    for key, filename in editor._LEGACY_SETTINGS_FILES.items():
        legacy_path = cache_dir / filename
        if not legacy_path.is_file():
            continue
        try:
            migrated[key] = json.loads(legacy_path.read_text())
        except Exception:
            continue
    if migrated:
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            migrated['updated'] = time.time()
            settings_path.write_text(json.dumps(migrated, indent=2))
        except Exception as exc:
            print(f"[Settings] Migration write failed: {exc}")
            return
    for filename in editor._LEGACY_SETTINGS_FILES.values():
        try:
            (cache_dir / filename).unlink()
        except Exception:
            pass

def _load_last_remote_cache(editor) -> None:
    """Load cached remote host/port from the unified settings file."""
    editor.last_remote_host: Optional[str] = None
    editor.last_remote_port: Optional[str] = None
    data = editor._read_settings_section('remote')
    host = data.get('host')
    port = data.get('port')
    if host:
        editor.last_remote_host = str(host)
    if port is not None and str(port).strip():
        editor.last_remote_port = str(port)

def _remember_last_remote(editor, host: str, port: Union[str, int, None]) -> None:
    """Persist the most recent remote host/port selection."""
    payload = {
        'host': host or '',
        'port': str(port) if port is not None else '',
    }
    editor._write_settings_section('remote', payload)
    editor.last_remote_host = payload['host']
    editor.last_remote_port = payload['port']

def _load_last_agent_cache(editor) -> None:
    """Load cached playback agent path, mode, and behavior from the unified settings file."""
    editor.agent_path: Optional[str] = None
    editor.agent_mode: str = "autopilot"       # "autopilot" | "human" | "custom"
    editor.agent_behavior: str = "normal"       # "cautious" | "normal" | "aggressive"
    try:
        from agents.navigation import behavior_agent as _ba_mod
        editor._behavior_agent_path: str = _ba_mod.__file__
    except Exception:
        editor._behavior_agent_path = "agents/navigation/behavior_agent.py"
    data = editor._read_settings_section('agent')
    path = data.get('path')
    mode = data.get('mode')
    behavior = data.get('behavior')
    if mode in ("autopilot", "human", "custom"):
        editor.agent_mode = mode
    if behavior in ("cautious", "normal", "aggressive"):
        editor.agent_behavior = behavior
    if path and os.path.isfile(path):
        editor.agent_path = os.path.abspath(path)
        return
    if editor.agent_mode == "custom":
        # Custom agent path no longer valid; fall back to autopilot
        editor.agent_mode = "autopilot"
    editor._clear_last_agent_cache(remove_file=True)

def _remember_last_agent(editor, agent_path: Optional[str] = None) -> None:
    """Persist the most recently used playback agent path, mode, and behavior."""
    if agent_path and os.path.isfile(agent_path):
        editor.agent_path = os.path.abspath(agent_path)
    payload: Dict[str, Any] = {
        'mode': editor.agent_mode,
        'behavior': editor.agent_behavior,
        'updated': time.time(),
    }
    if editor.agent_path:
        payload['path'] = editor.agent_path
        payload['directory'] = os.path.dirname(editor.agent_path)
    else:
        # Preserve last-used directory for the file picker
        existing_dir = editor._get_last_agent_directory()
        if existing_dir:
            payload['directory'] = existing_dir
    editor._write_settings_section('agent', payload)

def _clear_last_agent_cache(editor, *, remove_file: bool = False) -> None:
    """Clear cached playback agent path but preserve directory, mode, and behavior."""
    editor.agent_path = None
    if remove_file:
        existing = editor._read_settings_section('agent')
        existing_dir = existing.get('directory') or (
            os.path.dirname(existing.get('path', '')) if existing.get('path') else None
        )
        payload: Dict[str, Any] = {
            'mode': editor.agent_mode,
            'behavior': editor.agent_behavior,
        }
        if existing_dir and os.path.isdir(existing_dir):
            payload['directory'] = existing_dir
        editor._write_settings_section('agent', payload)

def _get_last_agent_directory(editor) -> Optional[str]:
    """Return the last used agent directory from the unified settings file, if available."""
    data = editor._read_settings_section('agent')
    directory = data.get('directory')
    if not directory and data.get('path'):
        directory = os.path.dirname(data['path'])
    if directory and os.path.isdir(directory):
        return directory
    return None

def _load_last_scenario_cache(editor) -> None:
    """Load cached recent scenario reference from the unified settings file if available."""
    data = editor._load_settings().get('scenario')
    if data is None:
        return
    try:
        entries_raw = []
        if isinstance(data, dict) and 'recent' in data:
            entries_raw = data.get('recent', [])
        elif isinstance(data, dict) and 'path' in data:
            entries_raw = [data]
        elif isinstance(data, list):
            entries_raw = data

        recent: List[Dict[str, str]] = []
        seen_paths: Set[str] = set()
        for raw in entries_raw:
            path = raw.get('path') if isinstance(raw, dict) else None
            name = raw.get('name') if isinstance(raw, dict) else None
            map_name = raw.get('map_name') if isinstance(raw, dict) else None
            if not path:
                continue
            norm_path = os.path.abspath(path)
            if norm_path in seen_paths:
                continue
            seen_paths.add(norm_path)
            exists = os.path.isfile(norm_path)
            if not name:
                name = os.path.basename(norm_path)
            recent.append({
                'path': norm_path,
                'name': name,
                'map_name': map_name or 'Unknown',
                'exists': exists,
            })
            if len(recent) >= 3:
                break

        editor.recent_scenarios = recent
        first = recent[0] if recent else None
        editor.last_scenario_path = first['path'] if first else None
        editor.last_scenario_name = first['name'] if first else None
        editor.last_scenario_map = first.get('map_name') if first else None
    except Exception as exc:
        print(f"[Recent Scenario] Failed to load cache: {exc}")
        editor._clear_last_scenario_cache(remove_file=True)

def _remember_last_scenario(editor, file_path: str, scenario_name: Optional[str], scenario_map: Optional[str]) -> None:
    """Update in-memory and the unified settings file with the most recent scenarios (max 3)."""
    if not file_path or not os.path.isfile(file_path):
        return
    name = scenario_name or os.path.basename(file_path)
    norm_path = os.path.abspath(file_path)
    map_name = scenario_map or 'Unknown'
    existing = [entry for entry in editor.recent_scenarios if entry.get('path') != norm_path]
    recent = [{
        'path': norm_path,
        'name': name,
        'map_name': map_name,
        'exists': True,
        'updated': time.time(),
    }] + existing
    recent = recent[:3]
    editor._write_settings_section('scenario', {'recent': recent})
    editor.recent_scenarios = recent
    first = recent[0]
    editor.last_scenario_path = first['path']
    editor.last_scenario_name = first['name']
    editor.last_scenario_map = first.get('map_name')

def _clear_last_scenario_cache(editor, *, remove_file: bool = False) -> None:
    """Clear recent scenario references."""
    editor.recent_scenarios = []
    editor.last_scenario_path = None
    editor.last_scenario_name = None
    editor.last_scenario_map = None
    if remove_file:
        editor._write_settings_section('scenario', {'recent': []})

def _load_last_play_camera_cache(editor) -> None:
    """Load the remembered play camera mode; default to 'chase' if missing/invalid."""
    editor.play_camera_mode: str = "chase"
    data = editor._read_settings_section('play_camera')
    mode = data.get('mode')
    if mode in editor.PLAY_CAMERA_MODES:
        editor.play_camera_mode = mode

def _remember_last_play_camera(editor) -> None:
    """Persist the current play camera mode selection."""
    editor._write_settings_section('play_camera', {
        'mode': editor.play_camera_mode,
        'updated': time.time(),
    })

def _load_culling(editor) -> None:
    """Load the remembered culling distance (metres; 0 = Off) into self.culling_distance_m.

    Precedence: --no-culling/VSE_CULLING=0 -> Off; explicit --cull-distance/VSE_CULL_DISTANCE;
    else the remembered value; else CULLING_DEFAULT_M (Off when no `culling` section exists yet).
    """
    if os.environ.get('VSE_CULLING') == '0':
        editor.culling_distance_m = 0.0
        return
    env_dist = os.environ.get('VSE_CULL_DISTANCE')
    if env_dist is not None:
        try:
            editor.culling_distance_m = max(0.0, float(env_dist))
            return
        except ValueError:
            pass
    dist = editor._read_settings_section('culling').get('distance_m')
    try:
        editor.culling_distance_m = max(0.0, float(dist)) if dist is not None else editor.CULLING_DEFAULT_M
    except (TypeError, ValueError):
        editor.culling_distance_m = editor.CULLING_DEFAULT_M

def _remember_last_culling(editor) -> None:
    """Persist the current culling distance selection."""
    editor._write_settings_section('culling', {
        'distance_m': editor.culling_distance_m,
        'updated': time.time(),
    })

def culling_enabled(editor) -> bool:
    """True when culling is on (distance > 0). Gates the spectator-follow re-add for local ego."""
    return editor.culling_distance_m > 0

def _culling_apply_safe(editor) -> bool:
    """False when changing world settings is unsafe: on large maps or while an external
    bridge/ego owns the sim, apply_settings() can crash the server (see
    _switch_world_to_async_if_safe). Culling is only editable while VSE solely controls the world.
    """
    if getattr(editor, "large_map_active", False):
        return False
    if (getattr(editor, "external_ego_actor", None) is not None
            or getattr(editor, "external_ego_actor_id", None) is not None
            or getattr(editor, "_external_swap_active", False)):
        return False
    return True

def _apply_culling(editor, *, world=None, force: bool = False, reason: str = "") -> bool:
    """Push self.culling_distance_m to WorldSettings.max_culling_distance (metres -> cm; 0 = off).

    When applying is unsafe (large map / external bridge), do NOT touch settings - instead adopt
    the live value into culling_distance_m so the dimmed dropdown reflects reality. Returns True
    when the world matches the desired value afterwards.
    """
    target_world = world or editor.world
    if not target_world:
        return False
    if not editor._culling_apply_safe():
        try:
            live = getattr(target_world.get_settings(), "max_culling_distance", 0.0)
            editor.culling_distance_m = max(0.0, float(live) / 100.0)
        except Exception:
            pass
        return False
    try:
        settings = target_world.get_settings()
    except Exception as exc:
        print(f"[Culling] Unable to read world settings: {exc}")
        return False
    target_cm = editor.culling_distance_m * 100.0
    current_cm = float(getattr(settings, "max_culling_distance", 0.0) or 0.0)
    if not force and abs(current_cm - target_cm) < 1e-6:
        return True
    try:
        settings.max_culling_distance = target_cm
        target_world.apply_settings(settings)
        label = "Off" if editor.culling_distance_m <= 0 else f"{editor.culling_distance_m:.0f} m"
        suffix = f" ({reason})" if reason else ""
        print(f"[Culling] Max draw distance set to {label}{suffix}.")
    except Exception as exc:
        print(f"[Culling] Failed to set max_culling_distance: {exc}")
        return False
    return True

def _set_culling_distance(editor, meters: float) -> None:
    """Set the culling distance (metres; 0 = Off), apply it live, remember it, close dropdowns."""
    editor.culling_distance_m = max(0.0, float(meters))
    editor._apply_culling(force=True, reason="user")
    editor._remember_last_culling()
    editor._close_all_dropdowns()

def _set_play_camera_mode(editor, mode: str, *, persist: bool = True, apply_live: bool = True) -> None:
    """Set the play camera preference, optionally persist it, and (when a run is active)
    apply the live camera + UI-visibility rule.

    UI visibility only changes while a playback run is active (camera follow or manual
    control): Top-Down always shows the overlays; Chase/Cockpit auto-hide them for a clean
    view. The backtick/`~` key still toggles `hide_all_ui` manually afterwards.
    """
    if mode not in editor.PLAY_CAMERA_MODES:
        return
    editor.play_camera_mode = mode
    if persist:
        editor._remember_last_play_camera()
    run_active = bool(editor.camera_processor and (
        editor.camera_processor.playback_camera_follow_enabled
        or editor.camera_processor.manual_control_enabled
    ))
    if run_active:
        editor.hide_all_ui = (mode != "topdown")
        if apply_live:
            editor.camera_processor.set_playback_camera_mode(mode)
