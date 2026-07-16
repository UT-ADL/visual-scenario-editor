"""Weather controller (moved verbatim from VisualScenarioEditor,
self -> editor rename only — step-32).

The editor-side weather window plumbing: baseline capture/reset, keyframe
sanitize/ensure/add/delete/navigate, slider preview/commit (undo via
WeatherEditCommand), window metadata refresh, JSON apply, and the
_restore_start_weather_from_presets hook that MiniRunner's _on_finish closure
calls from the runner thread (no UI calls may be added here). Storage is the
scene-backed weather forwarders (_weather_state/_weather_keyframes/
_active_weather_index/_pending_weather_pct — single storage on SceneModel).
The editor keeps one-line delegates for every function: WeatherControlWindow
stores eight of them as callbacks at construction, WeatherEditCommand calls
four by attribute name during undo/redo, and getattr-with-default reads
(_scenario_restore_weather exists only after first play; _baseline_weather_state
lazily) move verbatim. The weather button rect/render and click routing stay
in the editor (toolbar surface, not weather logic).
"""

import copy
import math
from typing import Dict, Iterable, List, Optional

import carla
import pygame

from vse_editor.commands import WeatherEditCommand
from vse_editor.constants import WEATHER_PARAMETER_SPECS
from vse_editor.ui.weather_window import WeatherControlWindow


def _restore_start_weather_from_presets(editor):
    """Restore pre-play weather (fallback: first weather keyframe)."""
    if not editor.world:
        return

    restore_weather = getattr(editor, "_scenario_restore_weather", None)
    if restore_weather is not None:
        try:
            editor.world.set_weather(restore_weather)
            return
        except Exception as exc:
            print(f"[Weather] Failed to restore pre-play weather: {exc}")
        finally:
            try:
                editor._scenario_restore_weather = None
            except Exception:
                pass

    editor._ensure_weather_keyframes()
    frame = editor._weather_keyframes[0] if editor._weather_keyframes else None
    if not isinstance(frame, dict):
        return
    weather = editor._weather_params_from_dict(frame)
    try:
        editor.world.set_weather(weather)
    except Exception as exc:
        print(f"[Weather] Failed to restore start weather: {exc}")

def _weather_dict_from_params(editor, weather: "carla.WeatherParameters") -> Dict[str, float]:
    """Convert WeatherParameters to a plain dict for the UI/state."""
    payload: Dict[str, float] = {}
    for spec in WEATHER_PARAMETER_SPECS:
        try:
            payload[spec.name] = float(getattr(weather, spec.name))
        except Exception:
            payload[spec.name] = spec.min_value
    return payload

def _capture_baseline_weather(editor, weather: Optional["carla.WeatherParameters"]) -> None:
    """Store the current world weather as the baseline for 'New Scenario' resets."""
    if not weather:
        return
    try:
        editor._baseline_weather_state = editor._weather_dict_from_params(weather)
    except Exception:
        editor._baseline_weather_state = None

def _reset_weather_to_baseline(editor) -> None:
    """Reset world weather + editor weather UI state back to the session baseline."""
    if not editor.world:
        return
    baseline = getattr(editor, "_baseline_weather_state", None)
    if not isinstance(baseline, dict) or not baseline:
        try:
            baseline_weather = editor.world.get_weather()
        except Exception:
            baseline_weather = None
        editor._capture_baseline_weather(baseline_weather)
        baseline = getattr(editor, "_baseline_weather_state", None)

    if not isinstance(baseline, dict) or not baseline:
        return

    try:
        weather = editor._weather_params_from_dict(baseline)
    except Exception:
        return

    try:
        editor.world.set_weather(weather)
    except Exception as exc:
        print(f"[Weather] Failed to reset weather: {exc}")
        return

    base_frame = {"route_percentage": 0.0, **dict(baseline)}
    editor._weather_keyframes = [
        base_frame,
        {**base_frame, "route_percentage": 100.0},
    ]
    editor._active_weather_index = 0
    editor._pending_weather_pct = 0.0
    editor._update_weather_state(weather, keyframe_index=0)
    if editor.weather_window and editor.weather_window.alive():
        try:
            editor.weather_window.apply_weather(weather)
        except Exception:
            pass
    editor._refresh_weather_window_metadata()

def _weather_params_from_dict(editor, values: Dict[str, float]) -> "carla.WeatherParameters":
    """Convert a payload dict back into WeatherParameters."""
    weather = carla.WeatherParameters()
    for spec in WEATHER_PARAMETER_SPECS:
        if spec.name in values:
            try:
                setattr(weather, spec.name, float(values[spec.name]))
            except Exception:
                pass
    return weather

def _sanitize_weather_keyframes_payload(editor, keyframes: Optional[Iterable[Dict[str, float]]]) -> List[Dict[str, float]]:
    """Normalize a list of keyframes (dicts) to a sorted, deduped payload."""
    defaults: Dict[str, float] = {}
    try:
        defaults = editor._weather_dict_from_params(editor.world.get_weather()) if editor.world else {}
    except Exception:
        defaults = dict(editor._weather_state)
    cleaned: List[Dict[str, float]] = []

    def _append(frame: Dict[str, float]) -> None:
        try:
            pct = float(frame.get("route_percentage", frame.get("route_percent", frame.get("pct", 0.0))))
        except Exception:
            return
        normalized: Dict[str, float] = {"route_percentage": max(0.0, min(100.0, pct))}
        for spec in WEATHER_PARAMETER_SPECS:
            val = frame.get(spec.name)
            if val is None and spec.name.lower() in frame:
                val = frame.get(spec.name.lower())
            if val is None:
                continue
            try:
                normalized[spec.name] = float(val)
            except Exception:
                normalized[spec.name] = spec.min_value
        cleaned.append(normalized)

    for frame in keyframes or []:
        if isinstance(frame, dict):
            _append(frame)

    if not cleaned:
        base = defaults or dict(editor._weather_state)
        default_frame = {"route_percentage": 0.0, **base}
        cleaned = [
            default_frame,
            {**default_frame, "route_percentage": 100.0},
        ]

    cleaned = sorted(cleaned, key=lambda kf: float(kf.get("route_percentage", 0.0)))
    deduped: List[Dict[str, float]] = []
    for frame in cleaned:
        pct = float(frame.get("route_percentage", 0.0))
        if deduped and abs(deduped[-1].get("route_percentage", 0.0) - pct) < 1e-3:
            deduped[-1].update({k: v for k, v in frame.items() if k != "route_percentage"})
        else:
            deduped.append(frame)

    if deduped[0].get("route_percentage", 0.0) > 0.0:
        deduped.insert(0, {"route_percentage": 0.0, **deduped[0]})
    if deduped[-1].get("route_percentage", 0.0) < 100.0:
        deduped.append({"route_percentage": 100.0, **deduped[-1]})

    for frame in deduped:
        for spec in WEATHER_PARAMETER_SPECS:
            if spec.name not in frame:
                frame[spec.name] = float(defaults.get(spec.name, editor._weather_state.get(spec.name, spec.min_value)))
    return deduped

def _ensure_weather_keyframes(editor) -> None:
    """Ensure we have at least 0% and 100% keyframes with full parameter coverage."""
    editor._weather_keyframes = editor._sanitize_weather_keyframes_payload(editor._weather_keyframes)
    editor._active_weather_index = min(editor._active_weather_index, len(editor._weather_keyframes) - 1)

def _update_weather_state(editor, weather: "carla.WeatherParameters", *, keyframe_index: Optional[int] = None) -> None:
    """Cache latest weather values and sync active keyframe if provided."""
    for spec in WEATHER_PARAMETER_SPECS:
        try:
            value = getattr(weather, spec.name)
        except AttributeError:
            value = spec.min_value
        editor._weather_state[spec.name] = float(value)

    if keyframe_index is None:
        keyframe_index = editor._active_weather_index
    if editor._weather_keyframes and 0 <= keyframe_index < len(editor._weather_keyframes):
        for spec in WEATHER_PARAMETER_SPECS:
            editor._weather_keyframes[keyframe_index][spec.name] = editor._weather_state[spec.name]
    else:
        # Seed if empty
        editor._ensure_weather_keyframes()
        if editor._weather_keyframes:
            for spec in WEATHER_PARAMETER_SPECS:
                editor._weather_keyframes[0][spec.name] = editor._weather_state[spec.name]

def _on_weather_window_closed(editor) -> None:
    """Callback when the weather window is dismissed."""
    editor.weather_window = None

def _on_weather_slider_preview(editor, parameter: str, value: float) -> None:
    """Apply live slider changes without creating undo entries."""
    spec = editor._weather_spec_lookup.get(parameter)
    if spec:
        value = float(max(spec.min_value, min(spec.max_value, value)))
    if not editor.world:
        return
    editor._ensure_weather_keyframes()
    if not editor._weather_keyframes:
        return
    active_index = max(0, min(editor._active_weather_index, len(editor._weather_keyframes) - 1))

    if editor._weather_drag_snapshot is None:
        editor._weather_drag_snapshot = {
            "keyframes": copy.deepcopy(editor._weather_keyframes),
            "index": active_index,
            "pending": editor._pending_weather_pct,
        }

    try:
        editor._weather_keyframes[active_index][parameter] = value
    except Exception:
        return

    try:
        frame = editor._weather_keyframes[active_index]
        weather = editor._weather_params_from_dict(frame)
        if editor.world:
            editor.world.set_weather(weather)
        editor._update_weather_state(weather, keyframe_index=active_index)
    except Exception as exc:
        print(f"[Weather] Unable to apply live slider change for {parameter}: {exc}")
        return

    editor._refresh_weather_window_metadata()

def _on_weather_slider_commit(editor, parameter: str, value: float) -> None:
    """Commit a weather slider change to undo/redo after drag ends."""
    if not editor._weather_keyframes:
        return
    active_index = max(0, min(editor._active_weather_index, len(editor._weather_keyframes) - 1))

    start = editor._weather_drag_snapshot
    editor._weather_drag_snapshot = None
    if start is None:
        return  # No drag was recorded; nothing to commit.

    old_keyframes = start.get("keyframes", [])
    old_index = int(start.get("index", active_index))
    old_pending = float(start.get("pending", editor._pending_weather_pct))

    # Ensure final value is stored in the active frame
    spec = editor._weather_spec_lookup.get(parameter)
    if spec:
        value = float(max(spec.min_value, min(spec.max_value, value)))
    try:
        editor._weather_keyframes[active_index][parameter] = value
    except Exception:
        return

    new_keyframes = copy.deepcopy(editor._weather_keyframes)
    command = WeatherEditCommand(
        editor,
        old_keyframes,
        new_keyframes,
        old_index,
        active_index,
        old_pending,
        editor._pending_weather_pct,
        description=f"Set weather {parameter}",
    )
    editor.execute_command(command)

def _refresh_weather_window_metadata(editor) -> None:
    """Sync keyframe metadata UI (label, percent slider, buttons)."""
    if not (editor.weather_window and editor.weather_window.alive()):
        return
    editor._ensure_weather_keyframes()
    if not editor._weather_keyframes:
        return
    count = len(editor._weather_keyframes)
    index = max(0, min(editor._active_weather_index, count - 1))
    # Use pending slider position for display to keep user-chosen insert point.
    pct = float(max(0.0, min(100.0, editor._pending_weather_pct)))
    can_delete = count > 2 and index not in (0, count - 1)
    # Disable add when the pending percentage matches an existing keyframe.
    matches_existing = any(abs(float(frame.get("route_percentage", 0.0)) - pct) < 1e-3 for frame in editor._weather_keyframes)
    try:
        editor.weather_window.update_keyframe_metadata(
            count=count,
            index=index,
            percentage=pct,
            percent_editable=True,
            can_delete=can_delete,
            can_add=not matches_existing,
        )
    except Exception:
        pass

def _set_active_weather_index(editor, index: int) -> None:
    """Select a different weather keyframe and apply it to the world."""
    editor._ensure_weather_keyframes()
    if not editor._weather_keyframes or not editor.world:
        return
    index = max(0, min(index, len(editor._weather_keyframes) - 1))
    editor._active_weather_index = index
    frame = editor._weather_keyframes[index]
    try:
        editor._pending_weather_pct = float(frame.get("route_percentage", 0.0))
    except Exception:
        editor._pending_weather_pct = 0.0
    try:
        weather = editor._weather_params_from_dict(frame)
    except Exception:
        return
    try:
        editor.world.set_weather(weather)
    except Exception as exc:
        print(f"[Weather] Failed to apply keyframe {index}: {exc}")
    editor._update_weather_state(weather, keyframe_index=index)
    if editor.weather_window and editor.weather_window.alive():
        try:
            editor.weather_window.apply_weather(weather)
        except Exception:
            pass
    editor._refresh_weather_window_metadata()

def _set_active_weather_percentage(editor, percentage: float) -> None:
    """Track the desired insert percentage from the slider."""
    try:
        editor._pending_weather_pct = float(max(0.0, min(100.0, percentage)))
    except Exception:
        editor._pending_weather_pct = 0.0
    editor._refresh_weather_window_metadata()

def _add_weather_keyframe_at(editor, percentage: float) -> None:
    """Insert a new keyframe at the requested route percentage."""
    editor._ensure_weather_keyframes()
    if not editor._weather_keyframes:
        return
    pct = max(0.0, min(100.0, float(percentage)))
    for idx, frame in enumerate(editor._weather_keyframes):
        if abs(float(frame.get("route_percentage", 0.0)) - pct) < 1e-3:
            # Selecting an existing keyframe is navigation, not an edit.
            editor._set_active_weather_index(idx)
            editor._pending_weather_pct = pct
            editor._refresh_weather_window_metadata()
            return
    new_frame: Dict[str, float] = {"route_percentage": pct}
    for spec in WEATHER_PARAMETER_SPECS:
        new_frame[spec.name] = float(editor._weather_state.get(spec.name, spec.min_value))
    old_keyframes = copy.deepcopy(editor._weather_keyframes)
    new_keyframes = copy.deepcopy(editor._weather_keyframes)
    new_keyframes.append(new_frame)
    new_keyframes = sorted(new_keyframes, key=lambda kf: float(kf.get("route_percentage", 0.0)))
    new_index = new_keyframes.index(new_frame)
    command = WeatherEditCommand(
        editor,
        old_keyframes,
        new_keyframes,
        editor._active_weather_index,
        new_index,
        editor._pending_weather_pct,
        pct,
        description="Add weather keyframe",
    )
    editor.execute_command(command)

def _delete_active_weather_keyframe(editor) -> None:
    """Remove the current keyframe when allowed (not first/last)."""
    editor._ensure_weather_keyframes()
    if len(editor._weather_keyframes) <= 2:
        return
    if editor._active_weather_index in (0, len(editor._weather_keyframes) - 1):
        return
    old_keyframes = copy.deepcopy(editor._weather_keyframes)
    new_keyframes = copy.deepcopy(editor._weather_keyframes)
    new_keyframes.pop(editor._active_weather_index)
    new_index = max(0, min(editor._active_weather_index, len(new_keyframes) - 1))
    command = WeatherEditCommand(
        editor,
        old_keyframes,
        new_keyframes,
        editor._active_weather_index,
        new_index,
        editor._pending_weather_pct,
        float(new_keyframes[new_index].get("route_percentage", editor._pending_weather_pct)) if new_keyframes else 0.0,
        description="Delete weather keyframe",
    )
    editor.execute_command(command)

def _on_weather_keyframe_percentage_change(editor, value: float) -> None:
    editor._set_active_weather_percentage(value)

def _on_weather_add_keyframe(editor) -> None:
    pct = float(max(0.0, min(100.0, getattr(editor, "_pending_weather_pct", 0.0))))
    editor._add_weather_keyframe_at(pct)

def _on_weather_delete_keyframe(editor) -> None:
    editor._delete_active_weather_keyframe()

def _on_weather_prev_keyframe(editor) -> None:
    target = editor._active_weather_index - 1
    if target < 0:
        return
    editor._set_active_weather_index(target)

def _on_weather_next_keyframe(editor) -> None:
    target = editor._active_weather_index + 1
    if target >= len(editor._weather_keyframes):
        return
    editor._set_active_weather_index(target)

def _apply_weather_from_json_data(editor, scenario_data: Optional[dict]) -> bool:
    """Apply weather keyframes from a JSON payload if present."""
    if not editor.world or not scenario_data:
        return False

    raw_keyframes = scenario_data.get("weather_keyframes")
    if not isinstance(raw_keyframes, list):
        return False

    cleaned = editor._sanitize_weather_keyframes_payload(raw_keyframes)
    if not cleaned:
        return False

    editor._weather_keyframes = cleaned
    editor._active_weather_index = 0
    try:
        editor._pending_weather_pct = float(cleaned[0].get("route_percentage", 0.0))
    except Exception:
        editor._pending_weather_pct = 0.0

    try:
        start_weather = editor._weather_params_from_dict(cleaned[0])
        editor.world.set_weather(start_weather)
        editor._update_weather_state(start_weather, keyframe_index=0)
        if editor.weather_window and editor.weather_window.alive():
            editor.weather_window.apply_weather(start_weather)
            try:
                editor.weather_window.update_keyframe_metadata(
                    count=len(cleaned),
                    index=editor._active_weather_index,
                    percentage=float(cleaned[0].get("route_percentage", 0.0)),
                    percent_editable=len(cleaned) > 1 and editor._active_weather_index not in (0, len(cleaned) - 1),
                    can_delete=len(cleaned) > 2 and editor._active_weather_index not in (0, len(cleaned) - 1),
                )
            except Exception:
                pass
        return True
    except Exception as exc:
        print(f"[Weather] Failed to apply start weather from JSON keyframes: {exc}")
        return False

def toggle_weather_window(editor) -> None:
    """Show or hide the weather control window."""
    if not editor.ready or not editor.world:
        print("Weather controls are available once the CARLA world finishes loading.")
        return

    if editor.weather_window and editor.weather_window.alive():
        window = editor.weather_window
        editor.weather_window = None
        window.kill()
        return

    available_width = editor.screen_width - 20
    width_candidate = max(280, min(520, available_width))
    width = min(width_candidate, available_width) if available_width > 0 else 360

    # Estimate required height based on layout metrics from WeatherControlWindow.
    columns = 2 if width >= 420 and len(WEATHER_PARAMETER_SPECS) > 6 else 1
    items_per_column = math.ceil(len(WEATHER_PARAMETER_SPECS) / columns)
    row_height = 58
    keyframe_controls_height = 120
    bottom_padding = 24
    window_chrome = 52  # title bar + margins inside UIWindow
    desired_content_height = keyframe_controls_height + (items_per_column * row_height) + bottom_padding
    desired_height = desired_content_height + window_chrome

    available_height = editor.screen_height - (editor.top_ui_height + editor.mode_button_height + 40)
    height_candidate = max(280, min(580, available_height))
    height = min(height_candidate, max(320, desired_height)) if available_height > 0 else 420
    bottom_margin = 12
    top_margin = editor.top_ui_height + editor.mode_button_height + 12
    top = max(top_margin, editor.screen_height - height - bottom_margin)
    rect = pygame.Rect(12, top, width, height)

    editor.weather_window = WeatherControlWindow(
        rect=rect,
        manager=editor.ui_manager,
        specs=WEATHER_PARAMETER_SPECS,
        on_change_live=editor._on_weather_slider_preview,
        on_change_commit=editor._on_weather_slider_commit,
        keyframe_count=len(editor._weather_keyframes) if editor._weather_keyframes else 1,
        active_index=editor._active_weather_index,
        active_percentage=float(max(0.0, min(100.0, getattr(editor, "_pending_weather_pct", 0.0)))),
        percent_editable=True,
        can_delete=bool(len(editor._weather_keyframes) > 2 and editor._active_weather_index not in (0, len(editor._weather_keyframes) - 1)),
        on_percentage_change=editor._on_weather_keyframe_percentage_change,
        on_add_keyframe=editor._on_weather_add_keyframe,
        on_delete_keyframe=editor._on_weather_delete_keyframe,
        on_prev_keyframe=editor._on_weather_prev_keyframe,
        on_next_keyframe=editor._on_weather_next_keyframe,
        on_close=editor._on_weather_window_closed,
    )

    try:
        editor._ensure_weather_keyframes()
        weather = None
        if editor._weather_keyframes:
            weather = editor._weather_params_from_dict(editor._weather_keyframes[editor._active_weather_index])
        if weather is None:
            weather = editor.world.get_weather()
    except Exception as exc:
        print(f"[Weather] Unable to read current weather: {exc}")
        weather = None

    if weather:
        editor._update_weather_state(weather)
        editor.weather_window.apply_weather(weather)
        editor._refresh_weather_window_metadata()
    elif editor._weather_state:
        try:
            fallback = carla.WeatherParameters(**editor._weather_state)
        except Exception:
            fallback = None
        if fallback:
            editor.weather_window.apply_weather(fallback)
            editor._refresh_weather_window_metadata()
