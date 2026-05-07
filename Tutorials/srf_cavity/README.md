# SRF Cavity — QUAM State & Experiment Tutorials

Tutorials for bring-up and calibration of a fixed-frequency transmon coupled to
SRF storage cavities (Alice and Bob) using QUA and the QUAM hardware abstraction layer.

---

## Contents

| File | Purpose |
|------|---------|
| `create_quam_state_opx1000.ipynb` | Create + populate a QUAM state for **OPX1000** (MW-FEM) |
| `create_quam_state_opx_plus_octave.ipynb` | Create + populate a QUAM state for **OPX+ / Octave** |
| `qubit_spectroscopy_standalone.ipynb` | Standalone qubit spectroscopy experiment using the QUAM interface |
| `srf_qubit_calibration.ipynb` | Full calibration notebook (full Qualibrate node sequence) |
| `quam_config/` | Local QUAM class definition (copy of the project `quam_config`) |
| `srf_quam.py` | Alternate standalone QUAM class definition |

---

## Prerequisites

Install the required packages into your Python environment:

```bash
pip install qm-qua qualang-tools quam quam-builder qualibrate
```

The notebooks assume:
- A running OPX1000 or OPX+ cluster reachable at the IP set in the "Create QUAM state" cell.
- The `QUAM_CONFIG_PATH` environment variable pointing to your project's `state/` folder,
  **or** you run Jupyter/VS Code with the `state/` folder's parent as the working directory.

---

## Concept: what is a QUAM state?

**QUAM** (Quantum Abstract Machine) is a hardware abstraction layer that sits between
your experiment code and the raw QUA configuration dictionary. Instead of hand-editing
a large JSON config, you:

1. Describe your hardware topology once (`state.json` + `wiring.json`).
2. Access qubits, resonators, and cavity modes through Python objects.
3. Let QUAM generate the QUA config automatically via `machine.generate_config()`.

A **QUAM state** is the pair of files `state/state.json` and `state/wiring.json`.
They live in your project folder and are loaded at the start of every session with
`machine = Quam.load()`.

---

## Step 0 — Select a Qualibrate project

Qualibrate organises data into **projects**. Each project has its own storage folder
for calibration results and, most importantly, its own `state/` directory containing
`state.json` and `wiring.json`.

The preamble cell in every notebook switches to the correct project:

```python
from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path
from qualibrate_config.core.project.switch import switch_project

config_path = get_qualibrate_config_path()
config      = get_qualibrate_config(config_path)

desired_project = "my_project_name"          # <-- edit this
if config.project != desired_project:
    switch_project(config_path, desired_project)
    config = get_qualibrate_config(config_path)

print(f"Project  : {config.project}")
print(f"State dir: {config.storage.location}")
```

**Why this matters for QUAM:** `Quam.load()` (and `machine.save()`) resolve the
`state/` path from `QUAM_CONFIG_PATH`, which Qualibrate sets automatically when you
switch projects. If you are running experiments outside Qualibrate (e.g. directly
in a Jupyter notebook), make sure the project is switched — or pass the path explicitly:

```python
machine = Quam.load("/absolute/path/to/my_project/state/state.json")
```

---

## Step 1 — Create the QUAM state (run once)

Open the appropriate notebook for your hardware:

- **OPX1000 (MW-FEM):** `create_quam_state_opx1000.ipynb`
- **OPX+ / Octave:**    `create_quam_state_opx_plus_octave.ipynb`

### What the "Create" cell does

It uses `qualang-tools` to:

1. **Define instruments** — which OPX controller and which FEM slots / Octave units
   are present in your setup.
2. **Assign channel addresses** — which physical port carries the readout, XY drive, etc.
3. **Build wiring** — `build_quam_wiring()` generates `state/wiring.json` with the
   port-to-element mapping.
4. **Build QUAM** — `build_quam()` populates `state/state.json` with the initial
   component tree (qubits, resonators, ports).

```python
# Minimal example — OPX1000
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1, 2])

connectivity = Connectivity()
connectivity.add_resonator_line(qubits=1, constraints=q1_res_ch)
connectivity.add_qubit_drive_lines(qubits=1, constraints=q1_drive_ch)
allocate_wiring(connectivity, instruments)

machine = Quam()
build_quam_wiring(connectivity, host_ip, cluster_name, machine)
machine = Quam.load()
build_quam(machine)
```

> **Run this cell only once per project.** Re-running it overwrites `wiring.json` and
> resets the component tree — you will lose any calibration data stored in `state.json`.

---

## Step 2 — Populate the QUAM state with initial values

After creating the wiring, the QUAM state contains the correct topology but all
frequencies and amplitudes are at their defaults. The "Populate" cell sets every
hardware parameter to a known starting point.

### What the "Populate" cell does

- Sets **LO frequencies** (upconverter_frequency for MW-FEM, or Octave LO).
- Sets **IF amplitudes / full-scale powers** derived from a desired output power in dBm.
- Sets **readout parameters**: resonator frequency, time-of-flight, depletion time.
- Creates the standard **pulse library**: saturation, x180/x90 DRAG Gaussian, EF pulses,
  selective pulse, displacement pulse.
- Creates **cavity objects** (`Cavity`, `CavityMode`) with their drive channels.
- Creates **CavityTransmonPair** objects with the f0g1 sideband drive.
- Initialises the **temporary calibration state** used by adaptive nodes.

Edit the `USER PARAMETERS` block at the top of that cell and re-run it whenever
you want to reset the state to a clean starting point. Individual calibration nodes
will overwrite specific fields (e.g. the qubit frequency after spectroscopy), but
re-running the populate cell resets everything back.

```python
# Key parameters to edit
rr_freq   = 7.500e9   # Hz  — readout resonator frequency
xy_freq   = 4.600e9   # Hz  — qubit ge transition frequency
xy_LO     = 4.400e9   # Hz  — upconverter / Octave LO
T1        = 200e-6    # s   — initial T1 estimate (used for wait times)
```

### Power helper functions

Both notebooks include a helper that converts a desired output power (dBm) into the
hardware-specific knob:

| Hardware | Helper | Output |
|----------|--------|--------|
| OPX1000 MW-FEM | `get_full_scale_power_dBm_and_amplitude(power_dBm)` | `(full_scale_power_dbm, waveform_amplitude)` |
| OPX+ / Octave  | `get_octave_gain_and_amplitude(power_dBm)` | `(octave_gain_dB, IF_amplitude)` |

---

## Step 3 — Verify the state

Run the "Verify" cell (last cell in each create/populate notebook) to confirm that
the QUAM loaded correctly and that all expected objects are present:

```python
machine = Quam.load()
for q_name, qubit in machine.qubits.items():
    print(q_name, qubit.f_01 / 1e9, "GHz")
print("Cavities:", list(machine.cavities.keys()))
```

---

## Step 4 — Run a standalone experiment

`qubit_spectroscopy_standalone.ipynb` shows the full workflow for a single experiment
**without** the Qualibrate node structure. It is the recommended starting point for
writing custom experiments.

### Loading the machine

```python
from quam_config import Quam   # local package in this folder
machine = Quam.load()
qubit   = machine.qubits["q1"]
```

### Writing a QUA program using QUAM

QUAM provides channel methods that hide the QUA element names and IF bookkeeping:

```python
from qm.qua import *

with program() as prog:
    I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
    df = declare(int)

    machine.initialize_qpu(target=qubit)
    align()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
            qubit.xy.play("saturation", duration=saturation_length // 4)
            align()
            qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
            qubit.resonator.wait(machine.depletion_time * u.ns)
            save(I[0], I_st[0])
            save(Q[0], Q_st[0])

    with stream_processing():
        n_st.save("n")
        I_st[0].buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dfs)).average().save("Q1")
```

Key QUAM methods:

| Method | What it does |
|--------|-------------|
| `machine.connect()` | Opens `QuantumMachinesManager` using IP from `state.json` |
| `machine.generate_config()` | Returns the full QUA config dict |
| `machine.declare_qua_variables()` | Declares I/Q variables and streams for all qubits |
| `machine.initialize_qpu(target=qubit)` | Sets flux bias etc. (no-op for fixed-frequency) |
| `machine.depletion_time` | Max resonator depletion time across active qubits (ns) |
| `qubit.xy.update_frequency(new_if)` | Updates the XY channel intermediate frequency in QUA |
| `qubit.xy.play(op, duration=...)` | Plays a pulse operation; duration in QUA clock cycles (ns/4) |
| `qubit.resonator.measure(op, qua_vars=...)` | Measures and demodulates into I/Q variables |
| `qubit.resonator.wait(t)` | Waits `t` QUA clock cycles on the resonator element |

### Executing and fetching data

```python
qmm    = machine.connect()
config = machine.generate_config()
qm     = qmm.open_qm(config)
job    = qm.execute(prog)
print("Job ID:", job.id)          # job.id is a string property in QM API v2

from qualang_tools.results import fetching_tool
results = fetching_tool(job, ["n", "I1", "Q1"], mode="live")
while results.is_processing():
    n_val, I, Q = results.fetch_all()
    # update live plot here if desired
n_val, I, Q = results.fetch_all()
```

> **Note:** `job.id` is a plain string property in QM API v2 — do **not** call it as
> `job.id()`.

---

## The local `quam_config` package

The `quam_config/` folder in this directory is a self-contained copy of the Quam class.
It exists so these notebooks work without requiring the full `qua-libs` project tree on
`sys.path`.

```
quam_config/
├── __init__.py    # re-exports Quam, TemporaryCalibrationData, CavityTransmonPair
└── my_quam.py     # Quam class definition
```

`Quam` extends `FixedFrequencyQuam` (from `quam_builder`) with three extra fields:

| Field | Type | Purpose |
|-------|------|---------|
| `cavities` | `Dict[str, Cavity]` | SRF storage cavities (Alice, Bob) |
| `cavity_transmon_pairs` | `Dict[str, CavityTransmonPair]` | Qubit–cavity coupling, f0g1 drive |
| `temp_calibration` | `Dict[str, TemporaryCalibrationData]` | Transient adaptive calibration state |

If you do not need the SRF cavity fields, you can use `FixedFrequencyQuam` directly
without any local package:

```python
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam as Quam
machine = Quam.load()
```

The notebooks ensure the local package is found by inserting the notebook directory
at the front of `sys.path`:

```python
import sys, os
_here = os.path.abspath('')
if _here not in sys.path:
    sys.path.insert(0, _here)
from quam_config import Quam
```

If you copy a notebook to a different folder, copy the `quam_config/` directory
alongside it, or point `sys.path` at this folder explicitly.

---

## Quick reference — typical session

```
1. Open the relevant create/populate notebook for your hardware.
2. Run the preamble cell  →  switches to the correct Qualibrate project.
3. Run the populate cell  →  resets all hardware parameters (edit USER PARAMETERS first).
4. Run the verify cell    →  confirms the state loaded correctly.
5. Open qubit_spectroscopy_standalone.ipynb (or your own experiment).
6. Run imports + load     →  machine = Quam.load()
7. Edit sweep parameters.
8. Run QUA program cell.
9. Run execute cell       →  job = qm.execute(prog)
10. Run fetch cell        →  live data collection into xarray Dataset
11. Run analyse + plot    →  Lorentzian fit, extract qubit frequency
12. Optionally update state → machine.save()
```
