# BO Bootstrap Qubit Calibration — Context Document

Self-contained reference for continuing this implementation in a new Claude Code session.

---

## Background

**Author:** Leonardo (Fermilab SQMS / Northwestern QuantA lab, CS PhD starting Sept 2026)
**Repo:** `https://github.com/NQI-SQMS/qua-libs`, branch `1q_calibration_sqms`
**Path:** `qualibration_graphs/superconducting/`

### The Problem

The existing graph 92 (`92_calibration_graph_bringup_fixed_frequency_transmon_adaptive.py`)
discovers the qubit via a four-level nested retry FSM:

```
outer loop: should_restart_qubit_calibration
  spec_vs_power [inner loop: should_repeat_spec_vs_power]
  → qubit_spec
  → power_rabi  [inner loop: should_repeat_rabi_amplitude]
```

This is fragile: fails when the qubit is far from the QUAM prior, or when a spurious
mode exists nearby.

### The Solution

Replace the entire qubit discovery sequence with a single measurement (time-domain Rabi)
and a Gaussian-Process Bayesian Optimisation loop over drive frequency and amplitude.

Inspired by: **Wolff et al. APS 2026** ("Autonomous closed-loop initial tuneup of
superconducting qubit drive and readout" — Pfafflab / UIUC).

---

## Files Created

All paths relative to `qualibration_graphs/superconducting/`:

```
calibration_utils/qubit_bo_bootstrap/
    __init__.py             Export symbols (Parameters, BOOptimizer, fit_rabi_trace, compute_cost)
    parameters.py           NodeSpecificParameters + Parameters class
    bo_optimizer.py         BOOptimizer — sklearn GP + EI/UCB + LHS sampling
    analysis.py             BoFitResult, fit_rabi_trace(), compute_cost() (Wolff formula)
    plotting.py             4 figure types: trajectory, convergence, best Rabi, cost landscape
    README_bo_bootstrap.md  ← this file

calibrations/1Q_calibrations/
    03e_qubit_bo_bootstrap.py       QualibrationNode (main BO node)
    93_calibration_graph_bringup_bo.py  QualibrationGraph (new clean graph)
```

---

## Algorithm

### Three-phase loop

```
Phase 1a — Broad LHS:
    n_initial_lhs=12 quasi-random Latin Hypercube points
    over (ω_d ± freq_search_radius_mhz, V_d in [amp_min, amp_max])
    → seeds GP with global cost landscape

Phase 1b — Zoom LHS:
    n_zoom_lhs=10 points within zoom_radius_mhz=50 MHz of Phase-1a best
    → gives GP resolution to distinguish true minimum from nearby modes

Phase 2 — BO acquisition:
    for i in range(n_bo_iterations=40):
        x_next = optimizer.suggest()   # maximize EI over zoomed search space
        cost = run_time_rabi(x_next)   # one QUA execution + fit
        optimizer.register(x_next, cost)
        x_opt = optimizer.predict_optimum()
        if |x_opt - x_opt_prev| < convergence_tolerance_mhz: break

Write best (ω_q, V_π) to QUAM → hand off to x180_fine_calibration
```

### Wolff cost function

```
C = w_rabi × |Ω_R − Ω_T| / Ω_T  −  w_amp × log(A)  −  w_snr × log(SNR)
```

- `Ω_R` = fit Rabi frequency (MHz) from time-domain trace
- `Ω_T` = target Rabi frequency (default 20 MHz)
- `A` = normalised oscillation amplitude ∈ (0,1] — peaks at resonance
- `SNR` = fit_amplitude / residual_RMS

Defaults: `w_rabi=10, w_amp=3, w_snr=1`

### Why BO instead of gradient descent

1. **Global optimizer:** EI exploration term forces evaluation of uncertain regions,
   preventing convergence to local minima (spurious modes, aliased Rabi)
2. **Sample efficient:** GP surrogate models the landscape in software; fewer hardware evals
3. **LHS seeding:** 12 quasi-random points give the GP a global picture before it guides evaluations

---

## Hardware Architecture

### What runs on vs off the FPGA

| Task | Where |
|------|-------|
| Time-Rabi pulse sequence (play x180 at variable duration, measure) | On FPGA (QUA) |
| num_shots averaging + stream_processing | On FPGA |
| Frequency/amplitude update between BO iterations | Off FPGA (Python) |
| GP fitting, EI maximization, cost function | Off FPGA (Python/sklearn) |
| Convergence decision, QUAM state update | Off FPGA (Python) |

Each BO iteration = one `qm.execute()` call with a freshly regenerated config.
Overhead: ~200 ms config regen × 50 total evaluations ≈ 10 s overhead.

### Dual-amplitude hardware (Octave + OPX+)

The drive signal power is controlled by TWO gains:
- **Octave gain:** sets the RF output power at the fridge input port (dBm)
- **OPX+ IF amplitude:** the waveform amplitude in Volts (max 0.5 V)

For the BO search:
- If candidate V_d ≤ `max_amplitude_opx=0.45 V`: set OPX amplitude directly
- If candidate V_d > `max_amplitude_opx`: call `set_output_power(power_dbm, max_amplitude=0.45)`
  which distributes the power between Octave gain and OPX amplitude optimally

The QUAM state is restored after each BO evaluation (try/finally in `_evaluate()`).
The best result is committed in `update_state` only after the full BO loop completes.

### IF frequency constraint

The OPX+ has ±250 MHz IF bandwidth around the LO frequency.
The BO bounds are automatically clamped:
```python
f_min_hz = max(prior_freq - radius, LO_freq - 250e6)
f_max_hz = min(prior_freq + radius, LO_freq + 250e6)
```

---

## Prerequisites

1. **Resonator bringup complete** — readout must be calibrated before the BO can run.
   The BO cost function uses the SNR of the readout signal; a miscalibrated resonator
   produces noisy Rabi traces that confuse the GP.
2. **Mixer/Octave calibration done** (node 01a).
3. **QUAM prior for qubit frequency** — does NOT need to be accurate. The BO searches
   ±200 MHz around it by default.

---

## How to Run

### Standalone node (simulation mode, no hardware)

```python
from calibrations.1Q_calibrations import 03e_qubit_bo_bootstrap as node_module
# node.parameters.simulate = True is the default for standalone runs
```

Or from CLI:
```bash
cd qualibration_graphs/superconducting
python calibrations/1Q_calibrations/03e_qubit_bo_bootstrap.py
```

### Full bring-up graph (hardware)

```bash
python calibrations/1Q_calibrations/93_calibration_graph_bringup_bo.py
```

Or via QualibrationLibrary:
```python
library.graphs["93_calibration_graph_bringup_bo"].run(parameters={
    "qubits": ["q1"],
    "bo_freq_search_radius_mhz": 200.0,
    "bo_n_initial_lhs": 12,
})
```

### 4D joint readout optimization (experimental)

Set `bo_optimize_readout_jointly=True` in the graph parameters.
This also sweeps readout frequency (±3 MHz) and amplitude (±6 dB) simultaneously.
Requires more LHS points: set `bo_n_initial_lhs ≥ 30, bo_n_zoom_lhs ≥ 30`.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bo_num_shots` | 100 | Shots per Rabi trace |
| `bo_min/max_duration_ns` | 16 / 448 | Time-Rabi sweep range (ns) |
| `bo_freq_search_radius_mhz` | 200 | Half-width of frequency search |
| `bo_amp_search_min/max_db` | −20 / +20 | Amplitude search range (dB rel. to prior) |
| `bo_n_initial_lhs` | 12 | Phase 1a points |
| `bo_n_zoom_lhs` | 10 | Phase 1b points |
| `bo_zoom_radius_mhz` | 50 | Phase 1b zoom half-width |
| `bo_n_bo_iterations` | 40 | Max Phase 2 iterations |
| `bo_convergence_tolerance_mhz` | 1.0 | Convergence threshold (MHz) |
| `bo_acq` | "EI" | Acquisition function (EI or UCB) |
| `bo_target_rabi_freq_mhz` | 20 | Target Ω_T in MHz |
| `bo_w_rabi / bo_w_amp / bo_w_snr` | 10 / 3 / 1 | Wolff cost weights |

---

## Open TODOs

1. **Square pulse operation:** The BO works best with a flat-top (constant envelope)
   drive pulse so that "amplitude = drive Rabi frequency" is linear. Check whether
   QUAM has a `"const"` or `"square_drive"` operation. If not, add one:
   ```python
   # In QUAM state:
   qubit.xy.operations["const_drive"] = SquarePulse(amplitude=0.1, length=16)
   ```
   Then set `bo_operation = "const_drive"` and scale via `qubit.xy.operations["const_drive"].amplitude`.

2. **Post-convergence Ramsey check:** After the BO converges, run a short Ramsey
   sequence to verify the qubit frequency. If Ramsey T2* < 500 ns, flag as spurious
   mode and add the frequency to a blacklist. This can be added as an additional
   `run_action` in `03e_qubit_bo_bootstrap.py`.

3. **Multi-qubit parallelism:** Currently sequential (one qubit at a time). For
   multiplexed setups, the BO for each qubit could run in parallel Python threads
   while the QUA program runs all qubits simultaneously. Requires the QUA program
   to include all qubits and the BO to track per-qubit data streams.

4. **Aliasing guard:** When Ω_R >> Ω_T (far from resonance, high detuning),
   the time-Rabi trace aliases. Add a soft penalty: `+ w_alias × max(0, Ω_R - 2Ω_T) / Ω_T`.

5. **BoTorch upgrade path:** If sklearn GP becomes slow (N > 200 observations),
   swap `BOOptimizer.gp` for `botorch.models.SingleTaskGP`. The `register()` /
   `suggest()` / `predict_optimum()` interface is unchanged.

---

## Simulator Reference

The offline simulator (`automatic_calibration_idea/`) validates the algorithm:
- `simulator/transmon_sim.py` — analytical/QuTiP transmon model with noise
- `simulator/run_bo_simulation.py` — 3-phase BO loop on the simulator
- `bo_optimizer.py` — same BOOptimizer class (ported directly)

The simulator tested: 10/10 clean, 3/5 noisy (with binomial projection noise,
thermal population, readout assignment errors, and 1/f drift).

---

## Continuing in a New Claude Code Session

Tell Claude Code:

> "Read `calibration_utils/qubit_bo_bootstrap/README_bo_bootstrap.md` for context.
> We have implemented the BO bootstrap node (03e) and graph (93) on the
> `1q_calibration_sqms` branch of `https://github.com/NQI-SQMS/qua-libs`.
> The next TODO is: [paste one of the TODOs above]."

Key files to read first:
- This README (you're reading it)
- `parameters.py` — all tunable knobs
- `03e_qubit_bo_bootstrap.py` — node structure and BO loop implementation
- `04c_time_rabi.py` — reference template this was modeled after
