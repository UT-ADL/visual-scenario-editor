# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-01-23

### Added
- VSE now launches to a main screen where users can choose between tasks
- Support for large CARLA maps (e.g., tartu_large)
- Route scenario support in vse_play.py
- List of traffic light groups with a trigger are displayed under the "Triggers" button
- --agent parameter to add external ego vehicle controller
- Dynamic weather system for scenarios
- Scenario results are also shown in the editor window and saved to disk
- Deviation support for actor speed values in waypoints
- Tooltips when hovering over menu icons

### Changed
- Single ego vehicle in VSE when awmini connects
- Ego vehicle can now be controlled by a Carla agent 
- Scenarios now run in synchronous mode
- vse_play.py now uses carla_minimal_agent to publish goals
- vse_play.py can again be used as a standalone script (with or without awmini)
- Remote server connection redesigned
- Manual control update rate set to 20 Hz
- “No camera” mode now moves the camera out of view 
- Ignore-traffic-lights checkbox now works for the ego vehicle in manual mode and with awmini
- Scenario ending criteria updated (no longer waits for untriggered traffic lights)
- Traffic light groups UI is disabled during scenario playback
- Undo/Redo system updated

### Fixed
- vse_play.py spawning pedestrians at incorrect elevation
- UI layering issues (UI elements no longer render on top of menus)

## [0.3.0] - 2025-11-26

### Added
- Confirmation dialog when opening scenario files with unsaved changes
- Recent files history (last 3 opened files)
- Visual circles around pedestrians in editor mode for easier visibility
- FPS counters for UI, CARLA camera, and CARLA wall time

### Changed
- vse_play.py now takes path to JSON file as argument (no longer copied to scenario folder)
- VSE command line log shows vse_play.py output in real-time
- 2D UI elements now hide when panning and zooming camera view
- Right Control and right Shift keys now work as modifiers
- Middle mouse button now pans camera (in addition to right mouse button)

### Fixed
- Scene cleanup logic when running scenario with unsaved changes
- Pedestrian yaw now updates when first waypoint is modified

## [0.2.0] - 2025-11-22

### Added
- Actor list in vehicle, pedestrian, and ego menus (double-click to select and focus camera)
- Waypoint splitting with Ctrl+left click
- Reverse driving for ego vehicle while holding down arrow key
- Camera auto-moves to ego vehicle when Play button is used

### Changed
- Drive clock button is now always visible
- Selected actors remain selected after scenario finishes and camera moves to them
- Drop-down menus no longer overlap
- Traffic lights UI hides when panning or zooming
- OpenDrive overlay hides when panning or zooming
- Play button no longer auto-saves; shows warning for unsaved changes instead
- Traffic light ID assignment now based on location
- Added "No camera" mode option in resolution selector
- Undo/Redo stack increased to 25 operations

### Fixed
- Camera panning while using manual control
- Traffic light triggers now require selecting parent stop line first
- Shift+right click no longer removes all waypoints when adding waypoints
- Personal trigger height updates when moving triggers
- Traffic light group trigger display when group has no trigger
- Creating new scenario now clears traffic light group triggers
- Pedestrian elevation check when moving actors
- Destination waypoint no longer snaps to cursor when re-entering waypoint mode

## [0.1.0] - 2025-11-12

### Added
- Create new scenario option in scenario menu
- Dead zone for mouse movement when right-click canceling in waypoint mode
- Personal triggers now use custom location

### Changed
- Default pedestrian turning time set to 0
- Play button now opens save dialog for unsaved scenarios
- Actor placement and movement height check logic improved
- Selection tolerance increased for actors, triggers, and waypoints
- Info panel stays visible when canceling waypoint addition after selecting destination

### Fixed
- Camera panning during manual control
- Waypoints no longer move randomly when selecting through UI
