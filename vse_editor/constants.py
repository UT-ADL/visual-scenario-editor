"""Editor constants shared across vse_editor modules and vse.py
(moved verbatim from vse.py).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

OVERLAY_ICON_TOOLTIPS: Dict[str, str] = {
    'delete': "Delete (Del)",
    'rotate': "Rotate",
    'waypoint': "Add/Edit Waypoints",
    'autoroute': "Auto-route to Destination",
    'ego_destination': "Set Ego Destination",
    'scale': "Adjust Trigger Radius",
    'add_trigger': "Add Trigger",
    'remove_trigger': "Remove Trigger",
}

# Floating-menu icon rows shared by the hit-test and render sides — one source
# of truth: OverlayMenuRenderer lays icons out purely by index, so a hit-test/
# render list mismatch silently misroutes clicks.
TRIGGER_MENU_ICONS: Tuple[str, ...] = ('delete', 'scale')
PERSONAL_TRIGGER_MENU_ICONS: Tuple[str, ...] = ('delete', 'scale')
GROUP_MENU_ICONS: Tuple[str, ...] = ('delete',)

DEFAULT_PERSONAL_TRIGGER_RADIUS = 2.0
MIN_PERSONAL_TRIGGER_RADIUS = DEFAULT_PERSONAL_TRIGGER_RADIUS

# Height (m) above the stored waypoint Z used for BOTH drawing the marker and
# hit-testing/hovering it. Must stay one constant: with the perspective camera,
# points at different heights project to different pixels away from the screen
# center, so a draw/hit-test height mismatch makes markers unclickable near the
# screen edges.
WAYPOINT_MARKER_Z_OFFSET = 0.5


INFO_PANEL_BUTTON_TOOLTIPS: Dict[str, str] = {
    'add_traffic_light_step': "Add a new sequence step",
    'clear_traffic_light_steps': "Remove all sequence steps",
}

TOP_UI_BUTTON_TOOLTIPS: Dict[str, str] = {
    'map': "Change Map",
    'play_camera': "Playback camera: Top-Down, Chase, or Cockpit (also C key during play)",
    'connection': "Switch Connection Mode",
    'view3d': "3D orbit view (Tab). Right-drag orbit, middle-drag pan, wheel zoom; editing works in top-down",
    'resolution': "Change Resolution",
    'culling': "Cull distance: hide distant meshes beyond N m",
    'fps': "Adjust Frame Rate",
    'manual_tick': "Toggle Drive Clock",
    'scenario': "Scenario Menu",
    'play': "Start Playback",
    'stop': "Stop Playback",
    'agent': "Ego Agent Mode",
    'npc_mode': "NPC driving mode: Simulated or Scripted",
    'weather': "Weather Controls",
}

MODE_BUTTON_TOOLTIPS: Dict[str, str] = {
    'VEHICLE': "Spawn NPC Vehicles",
    'PEDESTRIAN': "Spawn Pedestrians",
    'EGO': "Spawn Ego Vehicle",
    'TRIGGER': "Place/Edit Triggers",
}


@dataclass(frozen=True)
class WeatherParameterSpec:
    """Describe a single adjustable weather parameter."""

    name: str
    display_name: str
    min_value: float
    max_value: float
    step: float
    decimals: int = 1


WEATHER_PARAMETER_SPECS: Tuple[WeatherParameterSpec, ...] = (
    WeatherParameterSpec("cloudiness", "Cloudiness", 0.0, 100.0, 1.0),
    WeatherParameterSpec("precipitation", "Precipitation", 0.0, 100.0, 1.0),
    WeatherParameterSpec("precipitation_deposits", "Precipitation Deposits", 0.0, 100.0, 1.0),
    WeatherParameterSpec("wind_intensity", "Wind Intensity", 0.0, 100.0, 1.0),
    WeatherParameterSpec("sun_azimuth_angle", "Sun Azimuth (°)", 0.0, 360.0, 1.0, 0),
    WeatherParameterSpec("sun_altitude_angle", "Sun Altitude (°)", -90.0, 90.0, 1.0, 1),
    WeatherParameterSpec("fog_density", "Fog Density", 0.0, 100.0, 1.0),
    WeatherParameterSpec("fog_distance", "Fog Distance (m)", 0.0, 500.0, 5.0, 0),
    WeatherParameterSpec("fog_falloff", "Fog Falloff", 0.0, 5.0, 0.05, 2),
    WeatherParameterSpec("wetness", "Wetness", 0.0, 100.0, 1.0),
    WeatherParameterSpec("scattering_intensity", "Scattering Intensity", 0.0, 100.0, 1.0),
    WeatherParameterSpec("mie_scattering_scale", "Mie Scattering Scale", 0.0, 1.0, 0.01, 2),
    WeatherParameterSpec("rayleigh_scattering_scale", "Rayleigh Scattering Scale", 0.0, 1.0, 0.001, 3),
    WeatherParameterSpec("dust_storm", "Dust Storm", 0.0, 100.0, 1.0),
)

REMOTE_STREAM_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (320, 180),
    (480, 270),
    (640, 360),
    (800, 450),
    (960, 540),
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
)
NO_CAMERA_STREAM_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 1),   # Try a 1x1 placeholder feed first to keep integrations alive
    (2, 2),   # Fallback if CARLA rejects the tiny resolution
)
