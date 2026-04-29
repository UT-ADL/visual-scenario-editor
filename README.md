# Visual Scenario Editor

A graphical tool for creating and editing driving scenarios for the CARLA Simulator.

[![Watch the video](https://i.imgur.com/JQU4g6Z.jpeg)](https://www.youtube.com/watch?v=jaBjUwShg9Q)

## Features

- Top-down map viewer with panning and zooming
- Vehicle and pedestrian spawning and management
- Waypoint creation and editing for scenario paths
- Traffic light configuration and triggers
- Weather control
- Undo/redo system
- Scenario saving/loading to JSON files
- Integration with CARLA ScenarioRunner
- Export to OpenSCENARIO (.xosc)

## Installation

### Prerequisites

- Python 3.8+
- CARLA Simulator 0.9.15 (or compatible version)
- CARLA ScenarioRunner
- CARLA UT Lexus


#### 1. Create a directory into which to install carla and export it as `CARLA_ROOT`. Also update `PYTHONPATH` to make Carla agents importable in Python.
   ```
   mkdir ~/CARLA_0.9.15
   export CARLA_ROOT=$HOME/CARLA_0.9.15
   export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
   ```
   **Note:** Putting the above exports in `~/.bashrc` will reduce the hassle of exporting them every time you open a terminal.
   ```
   echo "export CARLA_ROOT=$HOME/CARLA_0.9.15" >> ~/.bashrc
   echo "export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla" >> ~/.bashrc
   ```
#### 2. Change into the directory and download [Carla 0.9.15](https://tiny.carla.org/carla-0-9-15-linux).
   ```
   cd $CARLA_ROOT
   wget https://tiny.carla.org/carla-0-9-15-linux -O CARLA_0.9.15.tar.gz
   ```
#### 3. Extract the file
   ```
   tar xzvf CARLA_0.9.15.tar.gz
   ```
#### 4. Delete the downloaded carla archive file.
   ```
   rm CARLA_0.9.15.tar.gz
   ```
#### 5. Go to the `Import` directory
   ```
   cd Import
   ```
#### 6. Download [utlexus.tar.gz](https://github.com/UT-ADL/carla_lexus/releases/download/v0.9.15/utlexus.tar.gz).
   ```
   wget https://github.com/UT-ADL/carla_lexus/releases/download/v0.9.15/utlexus.tar.gz
   ```
#### 7. Move to the parent directory
   ```
   cd ..
   ```
#### 8. Import the UT Lexus vehicle model.
   ```
   ./ImportAssets.sh
   ```
#### 9. Delete the `utlexus.tar.gz` file from the `Import` directory
  ```
  rm Import/utlexus.tar.gz
  ```
#### 10. Now, install the Carla Python module
  ```
  pip install carla==0.9.15
  ```
#### 11. Install CARLA dependencies:
  ```
  sudo apt install libomp5
  ```
#### 12. Clone [Scenario Runner](https://scenario-runner.readthedocs.io/en/latest/) to a directory of your choice
   ```
   git clone https://github.com/UT-ADL/scenario_runner.git
   ```
#### 13. Install Scenario Runner requirements
   ```
   pip install -r scenario_runner/requirements.txt
   ```
#### 14. Set SCENARIO_RUNNER_ROOT environment variable
   ```
   echo "export SCENARIO_RUNNER_ROOT=<path_to>/scenario_runner" >> ~/.bashrc
   ```
#### 15. Clone VSE to a directory of your choice
   ```
   git clone https://github.com/UT-ADL/visual-scenario-editor.git
   ```
#### 16. Install VSE requirements
   ```
   pip install -r visual-scenario-editor/requirements.txt
   ```

## Usage

[![Watch the video](https://i.imgur.com/FaeVRuc.png)](https://www.youtube.com/watch?v=kjzwnp27A2o)

Launch the editor:
```bash
python vse.py
```

Run a saved scenario using VSE:
```bash
Open scenario.json file in VSE and press Play button
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
