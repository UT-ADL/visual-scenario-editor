# VSE Quick Tutorial: Create and Play a Scenario

This is a short, beginner-friendly tutorial for `vse.py` (Visual Scenario Editor).

## What You Can Create

In VSE you can build a scenario JSON that includes:
- A single **ego vehicle** (always one in this tutorial)
- Optional **NPC vehicles** and **pedestrians**
- **Routes** (waypoints or autoroute destinations)
- **Triggers** (a global trigger, plus optional personal triggers and traffic-light triggers)
- Optional **weather** keyframes

## Start VSE

1) Start the editor:
   - `python vse.py`
2) On the start screen:
   - Click **Open Map** and pick a Town (e.g., `Town03`).

## Create Your First Scenario (Step-by-step)

Follow these steps in order. Each step is written as “do X, expect Y”.

### 1) Spawn the ego vehicle

1) Click **Ego Vehicle** mode (left side).
2) Choose an ego blueprint from the dropdown.
3) Hold `Ctrl` and **Left Click** on the map to spawn.

Expect:
- The ego appears at your click location.
- By default, ego placement **snaps to the nearest lane direction**.

Tips:
- Hold `Ctrl+Shift` while clicking to place without lane snap (free placement).
- To move/rotate later: click the ego to open the small floating icon menu.

### 2) Give the ego a route/destination

1) Click the ego to select it.
2) In the floating menu, click the **Ego destination** (pin) icon.
3) Click one or more points in the world to define the ego route/destination.
4) Press `ESC` (or **Right Click**) to finish route placement.

Expect:
- A visible waypoint/path for the ego.
- During playback, VSE will drive the ego automatically when it has a route.

### 3) Add the global trigger (required for this tutorial)

1) Click **Trigger** mode.
2) Click in the world to place the trigger circle.
3) Click the trigger to select it, then use the floating menu:
   - **Move**: drag the center
   - **Scale**: drag to change radius
   - **Delete**: remove it

Expect:
- A single trigger zone exists (only one global trigger per scenario).

### 4) Add at least one NPC vehicle with a route

1) Click **NPC Vehicles** mode.
2) Choose a vehicle blueprint.
3) Hold `Ctrl` and **Left Click** to spawn it.
4) Click the spawned NPC vehicle to select it.

Now pick one routing method:

**A) Manual waypoints**
1) Click the **Waypoints** icon in the floating menu.
2) Left-click to place waypoints.
3) Press `ESC` (or **Right Click**) to finish.

**B) Autoroute to a destination**
1) Click the **Autoroute** (pin) icon in the floating menu.
2) Click a destination point.

Expect:
- The NPC now has a path, and it becomes “playable” during playback.

### 5) (Optional) Add a pedestrian

1) Click **Pedestrians** mode.
2) Choose a pedestrian blueprint.
3) Hold `Ctrl` and **Left Click** to spawn.
4) Add waypoints via the floating menu (same idea as vehicles).

### 6) (Optional) Add personal triggers (per NPC / pedestrian)

Personal triggers are per-actor trigger circles (not supported for the ego).

1) Select an NPC vehicle or pedestrian.
2) In the floating menu, click **Add Trigger**.
3) Click to place the trigger center.
4) Select the trigger circle to move/scale it (or edit radius in the Info panel).

### 7) (Optional) Traffic-light triggers

1) Press `T` to enable the traffic-light stop line overlay.
2) Click a red stop-line marker to select a traffic-light *group*.
3) Click **Add Trigger** for that group.
4) In the Info panel, set:
   - Trigger radius
   - Optional sequence steps (color + duration)

### 8) (Optional) Weather

1) Click **Weather** to open the weather window.
2) Adjust sliders (live preview).
3) Add keyframes (0–100% route percentage) if you want weather changes over time.

## Save and Play

### Save
- Press `Ctrl+S` to save the scenario JSON (or use the **Scenario** menu → **Save**).

Expect:
- A `.json` scenario file is written, including map name, actors, routes, triggers, and optional weather keyframes.

### Play / Stop
1) Click **Play** (top-right).
2) If prompted about unsaved changes, choose whether to discard or cancel.
3) Click **Play** again to stop.

Expect:
- While playing, editing is disabled.
- When finished/stopped, VSE returns to edit mode and shows a results window.
- Results are written next to your scenario JSON as `<scenario>.txt`.

## Shortcuts You’ll Use Most

Camera:
- Pan: `W/A/S/D` or arrow keys or holding down the right mouse button
- Zoom: mouse wheel
- Jump to cursor: `Shift + Right Click`

Editing:
- Save: `Ctrl+S`
- Load: `Ctrl+L`
- Undo / Redo: `Ctrl+Z` / `Ctrl+Y` (or `Ctrl+Shift+Z`)
- Delete selected thing: `Delete`
- Toggle overlays: `O` (OpenDRIVE lanes), `T` (traffic lights)
- Cancel placement / clear selection: `ESC` or right mouse click

## Common Options (How Things Behave)

- Lane snap vs free placement:
  - Spawn: `Ctrl+Click` snaps (vehicles/ego), `Ctrl+Shift+Click` is free placement.
  - Move/waypoints: snap by default; hold `Shift` for free placement while dragging/placing.
- Selection:
  - Click an actor to select; a floating icon menu appears near it.
  - The Info panel (right side) shows editable fields like speed, idle time, trigger radius, etc.

## Troubleshooting

- **“CARLA Python API not found”**: your `PYTHONPATH` is missing CARLA’s `.egg`.
- **Map list is empty**: set `CARLA_ROOT` so VSE can find `.umap` files.
- **“Cannot change map while a scenario is running”**: press **Stop** first.
- **Play fails but editing works**: playback uses `vse_play.py` dependencies; check console logs.
- **Low FPS with an external ego (agent-controlled)**: set **Stream resolution** (top bar) to **No Camera** to disable the camera stream (often improves FPS).
- **No camera frames / editor looks frozen**: try enabling **Drive Clock** in the top bar if the world is in synchronous mode.
