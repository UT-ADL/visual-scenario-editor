# Visual Scenario Editor

A graphical tool for creating and editing driving scenarios for the CARLA Simulator.

[![Watch the video](https://img.youtube.com/vi/jaBjUwShg9Q/maxresdefault.jpg)](https://www.youtube.com/watch?v=jaBjUwShg9Q)

## Features

- Top-down map viewer with panning and zooming
- Vehicle and pedestrian spawning and management
- Waypoint creation and editing for scenario paths
- Traffic light configuration and triggers
- Weather control
- Undo/redo system
- Scenario saving/loading to JSON files
- Integration with CARLA ScenarioRunner

## Installation

### Prerequisites

- Python 3.8+
- CARLA Simulator 0.9.15 (or compatible version)
- CARLA ScenarioRunner

### 1. Install CARLA Simulator

Download CARLA 0.9.15 from the [official releases](https://github.com/carla-simulator/carla/releases):

```bash
# Extract to your preferred location
tar -xzf ~/Downloads/CARLA_0.9.15.tar.gz -C ~

# Set CARLA_ROOT environment variable
export CARLA_ROOT=$HOME/CARLA_0.9.15
```

### 2. Install CARLA ScenarioRunner

Clone the ScenarioRunner repository:

```bash
git clone https://github.com/UT-ADL/scenario_runner.git

# Install ScenarioRunner dependencies
pip install -r scenario_runner/requirements.txt

# Set SCENARIO_RUNNER_ROOT environment variable
export SCENARIO_RUNNER_ROOT=$HOME/scenario_runner
```

### 3. Configure PYTHONPATH Environment Variable

Set up the PYTHONPATH environment variable:

```bash
# Add CARLA Python API to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

### 4. Install VSE Dependencies

```bash
pip install -r requirements.txt
```

### Persistent Configuration

To avoid setting environment variables each session, add them to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# CARLA configuration
export CARLA_ROOT=$HOME/CARLA_0.9.15
export SCENARIO_RUNNER_ROOT=$HOME/scenario_runner
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

## Usage

Launch the editor:
```bash
python vse.py
```

Run a saved scenario on existing Carla server:
```bash
python vse_play.py path/to/scenario.json
```

Run a saved scenario on existing Carla server with external agent:
```bash
python vse_play.py scenario.json --agent /path/to/my_agent.py
```

For futher instructions check the [tutorial](tutorial.md)


### Launching VSE with Autoware Mini

Set up [Autoware Mini](https://github.com/UT-ADL/autoware_mini)


Launch the editor:
```bash
python vse.py
```
Launch Autoware Mini
```
roslaunch autoware_mini start_carla.launch use_scenario_runner:=true map_name:="name_of_a_map"
```

Use the Play button in VSE


## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
