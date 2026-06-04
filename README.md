# Gravity Assistance Simulation

A Python module for simulating a 2D gravitational assist trajectory in a Cartesian coordinate system. The project integrates the motion of a probe under the gravitational field of a planet using custom Runge–Kutta methods and SciPy's adaptive solver, saves the trajectory to CSV, and can optionally generate and open an animated GIF of the motion.

## Features
- Configurable simulation through a `.ini` file
- Numerical integrators:
  - RK2 (Ralston)
  - RK4 (3/8 rule)
  - SciPy `DOP853`
  
---

## Project structure

```text
gravitatis-assist/
│
├── assisted_probe/
│   ├── __init__.py
│   ├── assisted_probe.py
│   ├── config.ini              # .ini file with system parameters
│   └── outputfolder/           # Created after running the module
│       ├── *.csv
│       └── *.gif
│
├── pyproject.toml
├── README.md
├── tests/
│   └── test_mission_titan.py
│
└── analysis/
    └── mission_titan.ipynb
```

---

## Installation
From the root of the project run,
```bash
pip install -e .
```
This also allows the module to be run inside a Jupyter notebook.

---

## Configuration file
The system parameters are stored in the **config.ini** file. It has default values and to override a parameter simply
add the new definition under the **[simulation]** section.

## Usage
- To run it from the terminal without generating a GIF:
```bash 
python assisted_probe.py --config config.ini
```
- To run it from the terminal and generate and animation:
```bash
python assisted_probe.py --config config.ini --gif
```
The GIF will open automatically.

### In Python

To run it on scripts or Jupyter notebooks:

```python
from assisted_probe import GravityAssistance, RunAssistance, AnimateAssistance

simulation = GravityAssistance("config.ini")
runner = RunAssistance(simulation)
time, state = runner.integrator()
```
---
 
## Output files
The module saves a *.csv* file by default per simulation. If desired it also saves a *.gif* file.

---

## Running tests
This project includes a suite of unit tests using pytest to verify:

- Correct input mapping to the initial state vector
- Proper handling of invalid integration methods
- Conservation of specific orbital energy

Before running please install **pytest**.
To test, run from the root of the project:
```bash
pytest
```

---

## Troubleshooting
- If the GIF is not generated, make sure **pillow** is installed:
```bash 
pip install pillow
```

- If the *science* style is not found, make sure **scienceplots** is installed:
```bash
pip install scienceplots
```

---

## Example terminal session
```bash
$ python assisted_probe.py --config config.ini --gif

╔══════════════════════════════════════════════════════╗
║            GRAVITY ASSISTANCE SIMULATION             ║
╠══════════════════════════════════════════════════════╣
║ Planet:                Saturn                        ║
║ Method:                RK4                           ║
║ Simulation time (yr):  0.4                           ║
║ Steps:                 2000                          ║
╚══════════════════════════════════════════════════════╝

Running simulation...
Generating GIF...
GIF saved to: ./outputfolder/traj_at_Saturn_rk4.gif
```

---

## License
This project is licensed under the MIT License.
