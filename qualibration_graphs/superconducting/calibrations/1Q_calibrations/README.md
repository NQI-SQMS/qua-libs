# 1Q Calibrations

This folder contains the standard single-qubit calibration library for superconducting transmon qubits on the QUA platform, originally developed by Quantum Machines (QM). NQI-SQMS extended and corrected these nodes to support:

- SRF transmon-cavity systems (dispersive coupling, parity measurement, Wigner tomography)
- Adaptive calibration with error-code-driven recovery and per-qubit blacklisting
- Extended readout optimization (depletion time, readout length, 2D frequency-amplitude, GEF)
- EF (second excited state) calibration
- Long-run monitoring (T₁ monitor, thermal monitor)

See [`cavity_calibrations/`](../cavity_calibrations/README.md) for the cavity-specific nodes.

---

## Nodes added by NQI-SQMS

### Resonator characterization
| Node | File | Purpose |
|------|------|---------|
| 02d | `02d_broad_resonator_spectroscopy.py` | Wide-span sweep to locate a resonator from scratch; used in adaptive discovery |
| 02e | `02e_resonator_punch_out.py` | Power-dependent punch-out to identify qubit hybridization |
| 02f | `02f_resonator_bringup_graph.py` | Orchestrated graph: spectroscopy → punch-out → frequency optimization |

### Qubit spectroscopy
| Node | File | Purpose |
|------|------|---------|
| 03c | `03c_qubit_spectroscopy_vs_power.py` | Adaptive power sweep; feeds error codes to the adaptive calibration flow |

### Pulse calibration & EF
| Node | File | Purpose |
|------|------|---------|
| 04d | `04d_time_rabi_ef.py` | Time-domain Rabi for EF gate duration calibration |
| 05 | `05_x180_fine_calibration_graph.py` | Fine x180 calibration graph (power Rabi error amplification) |
| 05b | `05b_T1_ef.py` | T₁ of the \|f⟩ state |
| 06b | `06b_ramsey_ef.py` | Ramsey for \|f⟩ state frequency and T₂\* |

### Readout optimization
| Node | File | Purpose |
|------|------|---------|
| 07a | `07a_dispersive_shift.py` | Measure dispersive shift χ (qubit GE transition) |
| 07b | `07b_dispersive_shift_gef.py` | Measure χ for GEF levels |
| 08c | `08c_readout_depletion.py` | Optimize resonator depletion time |
| 08d | `08d_readout_length_optimization.py` | Optimize readout pulse length for SNR |
| 08e | `08e_readout_frequency_amplitude_optimization.py` | 2D frequency × amplitude readout optimization |
| 14b | `14b_readout_gef_power_optimization.py` | GEF readout power optimization |
| 14c | `14c_readout_gef_length_optimization.py` | GEF readout length optimization |
| 14d | `14d_readout_gef_freq_amp_optimization.py` | GEF 2D frequency × amplitude optimization |

### Long-run monitoring
| Node | File | Purpose |
|------|------|---------|
| 29 | `29_T1_monitor.py` | Long-run T₁ monitoring (repeated measurements over time) |
| 32 | `32_T1_thermal_monitor.py` | T₁ monitoring normalized to thermal state as reference |

### Calibration graphs (orchestration)
| Node | File | Purpose |
|------|------|---------|
| 91 | `91_ge_readout_optimization_graph.py` | Readout optimization chain (GE) |
| 91b | `91b_gef_readout_optimization_graph.py` | Readout optimization chain (GEF) |
| 92 | `92_ge_bringup_graph.py` | GE bring-up from scratch |
| 92a | `92a_ge_discovery_graph.py` | Adaptive GE discovery (spectroscopy + power sweep + Rabi loop) |
| 93 | `93_ef_bringup_graph.py` | EF bring-up from a calibrated GE state |
| 93a | `93a_ef_discovery_graph.py` | Adaptive EF discovery graph |
| 96 | `96_ge_retuning_graph.py` | GE retuning from a known state |
| 97 | `97_ef_retuning_graph.py` | EF retuning graph |

---

## Changes to QM nodes (changelog)

The following QM nodes were modified. Changes are grouped by theme.

### Timing fix: `depletion_time * u.ns` → `depletion_time // 4`

**Affected:** `01a_time_of_flight.py`, `07_iq_blobs.py`, `02a_resonator_spectroscopy.py`, and all nodes that wait for resonator depletion.

**What changed:** Wait calls used `rr.wait(depletion_time * u.ns)`. Since `u.ns = 1` in `qualang_tools.units`, this passed nanoseconds directly to a function that expects QUA clock cycles (1 cycle = 4 ns). The fix uses `depletion_time // 4`.

**Why:** The original code produced waits that were 4× too short, causing readout contamination in high-depletion-time setups (e.g., SRF cavities).

### Adaptive calibration support (error codes + blacklisting)

**Affected:** `02a_resonator_spectroscopy.py`, `03a_qubit_spectroscopy.py`, `03b_qubit_spectroscopy_vs_flux.py`, `04b_power_rabi.py`.

**What changed:** Added `ErrorCode` / `CorrectiveAction` enum tracking to each node's `update_state` action. On specific failure modes (e.g., `NO_DIP_FOUND`, `NO_OSCILLATION`), failed frequencies or amplitudes are added to `machine.temp_calibration[qubit_name].blacklisted_*` so that upstream discovery nodes avoid retrying the same bad points.

**Why:** Enables the adaptive calibration graphs (`92a_ge_discovery_graph.py`, `93a_ef_discovery_graph.py`) to automatically re-route failed qubits through the discovery flow without human intervention. This introduced the concept of adaptive node chaining using `TemporaryCalibrationData`.

### Multiplexed readout separation

**Affected:** `07_iq_blobs.py`, `04b_power_rabi.py`, and several other nodes.

**What changed:** Separated the reset+drive loop from the readout loop into distinct `for i, qubit in ...` blocks with an `align()` between them.

**Why:** The original code drove and read each qubit sequentially. The fix enables proper simultaneous readout of all qubits.

### Readout power override via `tracked_updates`

**Affected:** `02a_resonator_spectroscopy.py`, `08a_readout_frequency_optimization.py`, `08b_readout_power_optimization.py`.

**What changed:** Added optional `readout_power_dbm` parameter. When set, the node temporarily adjusts `resonator.set_locked_output_power()` for the duration of the experiment, and reverts it in `update_state` via `tracked_resonator.revert_changes()`.

**Why:** SRF/3D cavity systems require low readout power during sensitive calibration steps to avoid drive-induced dephasing, without permanently altering the QUAM state.

### GE pi-pulse parameter

**Affected:** `07_iq_blobs.py`.

**What changed:** Excited-state preparation now uses `qubit.xy.play(node.parameters.ge_pi_pulse)` instead of hardcoded `"x180"`.

**Why:** Allows using a selective or differently-named π-pulse (e.g., with AC Stark shift compensation).

### GEF readout improvements

**Affected:** `14_gef_readout_frequency_optimization.py`, `15_iq_blobs_gef.py`.

**What changed:** Added GEF classifier support (LDA-based 3-state discrimination) and fixed crash for missing GEF classifier fields. Added missing optimization axes (power, length, 2D freq-amp) as new nodes (14b–14d).

**Why:** GEF readout is essential for EF-gate calibration and parity measurement in cavity QED systems.

### Bringup/retuning graph updates

**Affected:** `80_calibration_graph_bringup_flux_tunable_transmon.py`, `81_calibration_graph_retuning_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`, `91_calibration_graph_retuning_fixed_frequency_transmon.py`.

**What changed:** Updated to reference new readout optimization nodes (08c, 08d, 08e) and extended with EF calibration steps.

### Additional modifications

**Affected:** `06a_ramsey.py`, `06b_echo.py`, `09a_ramsey_vs_flux_calibration.py`, `10b_drag_calibration_180_minus_180.py`, `11a_single_qubit_randomized_benchmarking.py`, `11b_single_qubit_randomized_benchmarking_interleaved.py`, `12_Qubit_Spectroscopy_E_to_F.py`, `13_power_rabi_ef.py`, `16a_xyz_delay.py`, `16b_xy_coupler_z_delay.py`, `17_pi_vs_flux_long_distortions.py`, `18_cryoscope.py`, `19_zz_off_jazz.py`.

**What changed:** Various bug fixes, parameter additions, analysis improvements, and hardware compatibility fixes across these nodes. See git diff vs. upstream/main for per-file details.

---

## calibration_utils — modified modules

| Module | Change |
|--------|--------|
| `resonator_spectroscopy/` | Added circle-fit analysis, error codes (`ResonatorSpectroscopyErrorCode`), `readout_power_dbm` parameter |
| `broad_resonator_spectroscopy/` | New — wide-span analysis with blacklist filtering for adaptive discovery |
| `power_rabi/` | Added `PowerRabiErrorCode`, `NO_OSCILLATION` blacklisting, `operation_length_in_ns` override |
| `qubit_spectroscopy/` | Added error codes, adaptive power sweep support |
| `iq_blobs/` | Added `ge_pi_pulse` parameter, multiplexed readout fix, histogram improvements |
| `iq_blobs_ef/` | GEF LDA classifier, crash fix for missing fields |
| `ramsey/` | Improved decay fit robustness, additional plotting modes |
| `ramsey_versus_flux_calibration/` | Extended for 2D flux-vs-frequency fitting |
| `drag_calibration_180_minus180/` | Added convergence diagnostics |
| `single_qubit_randomized_benchmarking/` | Added 1Q-Clifford diagnostic mode, raw-IQ plot support |
| `T1/analysis.py` | Improved exponential fit initial guess |
| `T2echo/analysis.py` | Improved echo decay robustness |
| `T1_ef/` | New — T₁ analysis for the \|f⟩ state |
| `T1_monitor/` | New — repeated T₁ measurement analysis |
| `T1_thermal_monitor/` | New — T₁ normalized to thermal reference |
| `dispersive_shift/` | New — χ extraction from qubit spectroscopy |
| `dispersive_shift_gef/` | New — χ extraction for GEF levels |
| `readout_depletion/` | New — depletion time optimization |
| `readout_length_optimization/` | New — readout length optimization |
| `readout_frequency_amplitude_optimization/` | New — 2D readout optimization |
| `readout_gef_*/` | New — GEF readout optimization variants (power, length, freq-amp) |
| `bringup_graphs.py` | New — shared utilities for bringup/retuning graphs (`_ensure_temp_calibration`, `should_keep_retuning`) |

All cavity-specific modules (`cavity_*`, `fNgN1_*`, `displacement_*`, `wigner_*`, `parity_*`) are documented in [`cavity_calibrations/README.md`](../cavity_calibrations/README.md).

---

## QUAM infrastructure additions

| File | Purpose |
|------|---------|
| `quam_config/recursive_calibration_scan.py` | Monkeypatches `QualibrationLibrary` to scan `1Q_calibrations/`, `CZ_calibrations/`, and `cavity_calibrations/` in a single call |
| `quam_config/data_fetch_retry_patch.py` | Retry logic for QM data fetching on transient failures |
| `quam_config/hardware_batching_patch.py` | Hardware slot batching for OPX1000 with many elements |
| `quam_config/wiring_examples/` | Wiring configs for OPX+/Octave+cavity and OPX1000/MW-FEM+cavity |
