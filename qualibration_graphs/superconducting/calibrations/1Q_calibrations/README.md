# 1Q Calibrations

This folder contains the standard single-qubit calibration library for superconducting transmon qubits on the QUA platform, originally developed by Quantum Machines (QM) and extended by NQI-SQMS at Northwestern/Fermilab to support:

- SRF transmon-cavity systems (dispersive regime, Wigner tomography)
- Adaptive calibration graphs with error-code-driven recovery
- Extended readout optimization (depletion, length, 2D frequency-amplitude)
- EF (second excited state) calibration nodes
- Flux-vs-time coherence characterization
- Long-run monitoring (T₁ monitor, thermal monitor)

See [`cavity_calibrations/`](../cavity_calibrations/README.md) for the cavity-specific nodes.

---

## Nodes added by NQI-SQMS

### Resonator characterization
| Node | File | Purpose |
|------|------|---------|
| 02d | `02d_broad_resonator_spectroscopy.py` | Wide-span sweep to locate resonator from scratch; used in adaptive discovery |
| 02e | `02e_resonator_punch_out.py` | Power-dependent punch-out measurement to identify qubit hybridization |
| 02f | `02f_resonator_bringup_graph.py` | Orchestrated bringup graph: spectroscopy → punch-out → optimization |
| 02g | `02g_resonator_spectroscopy_wide_pyloop.py` | Pure-Python loop version for very wide frequency sweeps |
| 02h | `02h_resonator_spectroscopy_single.py` | Single-shot resonator spectroscopy (no averaging loop) |
| 02i | `02i_resonator_spectroscopy_vs_power_iq.py` | IQ-resolved power sweep for resonator characterization |
| 02j | `02j_resonator_spectroscopy_vs_coupler_flux_new.py` | Resonator spectroscopy vs. coupler flux bias |

### Qubit spectroscopy
| Node | File | Purpose |
|------|------|---------|
| 03c | `03c_qubit_spectroscopy_vs_power.py` | Adaptive power sweep; feeds error codes to adaptive graph |
| 03d | `03d_qubit_spectroscopy_vs_amplitude.py` | Amplitude-swept spectroscopy (alternative to power in dBm) |
| 03e | `03e_qubit_spectroscopy_vs_flux_b.py` | Qubit spectroscopy vs. auxiliary (B-coil) flux |
| 04 | `04_twpa_calibration.py` | TWPA pump frequency and power optimization |

### Pulse calibration
| Node | File | Purpose |
|------|------|---------|
| 04c | `04c_time_rabi.py` | Time-domain Rabi for GE gate duration calibration |
| 04d | `04d_time_rabi_ef.py` | Time-domain Rabi for EF gate duration calibration |
| 05 | `05_x180_fine_calibration_graph.py` | Fine x180 calibration graph (power Rabi error amplification) |
| 10a | `10a_stark_detuning.py` | AC Stark shift measurement for detuning calibration |
| 10c | `10c_allxy.py` | AllXY gate quality diagnostic (streamlined version) |
| 22b | `22b_all_xy.py` | AllXY gate quality diagnostic (alternative implementation) |

### Coherence characterization
| Node | File | Purpose |
|------|------|---------|
| 05b | `05b_T1_ef.py` | T₁ of the |f⟩ state |
| 05c | `05c_T1_vs_flux.py` | T₁ as a function of flux bias |
| 06b | `06b_ramsey_ef.py` | Ramsey for |f⟩ state frequency and T₂* |
| 06c | `06c_echo_vs_flux.py` | T₂ echo as a function of flux bias |
| 09c | `09c_T2star_vs_flux.py` | T₂* (Ramsey) as a function of flux bias |
| 29 | `29_T1_monitor.py` | Long-run T₁ monitoring (repeated measurements) |
| 32 | `32_T1_thermal_monitor.py` | T₁ monitoring normalized to thermal state as reference |

### Readout optimization
| Node | File | Purpose |
|------|------|---------|
| 07a | `07a_dispersive_shift.py` | Measure dispersive shift χ (qubit GE transition) |
| 07b | `07b_dispersive_shift_gef.py` | Measure χ for GEF levels |
| 08c | `08c_readout_depletion.py` | Optimize resonator depletion time |
| 08d | `08d_readout_length_optimization.py` | Optimize readout pulse length for SNR |
| 08e | `08e_readout_frequency_amplitude_optimization.py` | 2D frequency × amplitude readout optimization |
| 08f | `08f_fullscale_dbm_adjustment.py` | Adjust DAC full-scale to target output power in dBm |
| 14b | `14b_readout_gef_power_optimization.py` | GEF readout power optimization |
| 14c | `14c_readout_gef_length_optimization.py` | GEF readout length optimization |
| 14d | `14d_readout_gef_freq_amp_optimization.py` | GEF 2D frequency × amplitude optimization |

### Flux distortion calibration
| Node | File | Purpose |
|------|------|---------|
| 17b | `17b_coupler_flux_long_distortion_qubitspec.py` | Long-timescale flux distortion on coupler (qubit spec method) |
| 17c | `17c_coupler_flux_long_distortion_ramsey.py` | Long-timescale flux distortion on coupler (Ramsey method) |
| 17d | `17d_qubit_flux_long_distortion_ramsey.py` | Long-timescale flux distortion on qubit (Ramsey method) |
| 17e | `17e_qubit_flux_long_distortion_qubitspec.py` | Long-timescale flux distortion on qubit (qubit spec method) |
| 18b | `18b_coupler_flux_short_distortion.py` | Short-timescale flux distortion on coupler (cryoscope) |
| 18c | `18c_qubit_flux_short_distortion.py` | Short-timescale flux distortion on qubit (cryoscope) |
| 18d | `18d_coupler_zero_point_coarse.py` | Coarse coupler flux zero-point calibration |

### Miscellaneous
| Node | File | Purpose |
|------|------|---------|
| 20 | `20_qubit_rpm.py` | Qubit RPM (resonance population measurement) |
| 20b | `20b_ef_rabi_rpm.py` | EF Rabi RPM |
| 99 | `99_filter_plot.py` | Utility: visualize flux distortion filter impulse response |

### Calibration graphs (orchestration)
| Node | File | Purpose |
|------|------|---------|
| 91 | `91_ge_readout_optimization_graph.py` | Readout optimization graph (GE) |
| 91b | `91b_gef_readout_optimization_graph.py` | Readout optimization graph (GEF) |
| 92 | `92_ge_bringup_graph.py` | GE bring-up from scratch |
| 92a | `92a_ge_discovery_graph.py` | Adaptive GE discovery (spectroscopy + Rabi loop) |
| 93 | `93_ef_bringup_graph.py` | EF bring-up graph (from GE calibrated state) |
| 93a | `93a_ef_discovery_graph.py` | Adaptive EF discovery graph |
| 96 | `96_ge_retuning_graph.py` | GE retuning from a known state |
| 97 | `97_ef_retuning_graph.py` | EF retuning graph |
| 994 | `994_rb_success_exit.py` | Exit node for RB-based adaptive loop (success condition) |
| 995 | `995_readout_chain_graph.py` | Readout calibration chain (depletion → length → freq → amp) |
| 996 | `996_single_qubit_gate_graph.py` | Single-qubit gate calibration chain |
| 997 | `997_single_qubit_rb_graph.py` | RB-gated gate calibration (runs RB to verify) |
| 999 | `999_adaptive_graph.py` | Full adaptive calibration graph with RB-driven retuning loop |

---

## Changes to QM nodes (changelog)

The following nodes from the original QM library were modified. Changes are grouped by theme.

### Timing fix: `depletion_time * u.ns` → `depletion_time // 4`

**Affected:** `01a_time_of_flight.py`, `07_iq_blobs.py`, `02a_resonator_spectroscopy.py`, and all other nodes that wait for resonator depletion.

**What changed:** The wait call was `rr.wait(depletion_time * u.ns)`. Since `u.ns = 1` in `qualang_tools.units`, this multiplied nanoseconds by 1 and passed nanoseconds to a function that expects QUA clock cycles (1 cycle = 4 ns). The fix uses integer division by 4 to convert correctly.

**Why:** The original code produced waits that were 4× too short, causing readout contamination when depletion time was large relative to the repetition rate.

### Adaptive calibration support

**Affected:** `02a_resonator_spectroscopy.py`, `04b_power_rabi.py`, `03a_qubit_spectroscopy.py`, `03b_qubit_spectroscopy_vs_flux.py`.

**What changed:** Added `ErrorCode` / `CorrectiveAction` enum tracking to each node's `update_state` action. On failure, failed frequencies or amplitudes are added to `machine.temp_calibration[qubit_name].blacklisted_*` so that upstream discovery nodes (02d broad spectroscopy, 03c vs. power) avoid retrying the same bad points.

**Why:** Enables the adaptive calibration graph (`999_adaptive_graph.py`) to automatically re-route failed qubits through the discovery flow without human intervention.

### Multiplexed readout separation

**Affected:** `07_iq_blobs.py`, `04b_power_rabi.py`, and several other nodes.

**What changed:** Separated the reset+drive loop from the readout loop into distinct `for i, qubit in ...` blocks with an `align()` between them.

**Why:** The original code drove and read each qubit sequentially, which breaks multiplexed operation on multi-qubit setups. The fix allows simultaneous readout of all qubits after the drive phase.

### Readout power override via `tracked_updates`

**Affected:** `02a_resonator_spectroscopy.py`, `08a_readout_frequency_optimization.py`, `08b_readout_power_optimization.py`.

**What changed:** Added optional `readout_power_dbm` parameter. When set, the node temporarily adjusts `resonator.set_locked_output_power()` for the duration of the experiment and reverts it in `update_state` via `tracked_resonator.revert_changes()`.

**Why:** SRF/3D cavity systems require low readout power during qubit spectroscopy to avoid drive-induced dephasing. The override allows using a low-power readout for sensitive steps without permanently changing the QUAM state.

### GE pi-pulse parameter

**Affected:** `07_iq_blobs.py`.

**What changed:** The excited-state preparation now uses `qubit.xy.play(node.parameters.ge_pi_pulse)` instead of hardcoded `"x180"`, where `ge_pi_pulse` defaults to `"x180"`.

**Why:** Some experiments need to use a selective x180 pulse or a different operation name (e.g., for SRF with ac Stark shift).

### GEF readout improvements

**Affected:** `14_gef_readout_frequency_optimization.py`, `15_iq_blobs_gef.py`.

**What changed:** Added GEF classifier support (LDA-based 3-state discrimination), crash fixes for missing GEF classifier fields, and additional optimization axes (power, length, 2D freq-amp).

**Why:** GEF readout is essential for EF-gate calibration and parity measurement. The original code only had frequency optimization for GEF.

### Bringup/retuning graph updates

**Affected:** `80_calibration_graph_bringup_flux_tunable_transmon.py`, `81_calibration_graph_retuning_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`, `91_calibration_graph_retuning_fixed_frequency_transmon.py`.

**What changed:** Updated to reference the new user-added readout optimization nodes (08c, 08d, 08e, 08f) and to use the recursive calibration scanner (`recursive_calibration_scan.py`) that finds nodes across all subfolders.

**Why:** The original graphs assumed a flat 1Q_calibrations folder and didn't include the extended readout chain.

### calibration_utils changes

| Module | Change |
|--------|--------|
| `resonator_spectroscopy/` | Added circle-fit analysis, error codes, `readout_power_dbm` parameter |
| `power_rabi/` | Added `PowerRabiErrorCode`, `NO_OSCILLATION` path for blacklisting, `operation_length_in_ns` override |
| `qubit_spectroscopy/` | Added error codes, adaptive power sweep support |
| `iq_blobs/` | Added `ge_pi_pulse` parameter, multiplexed readout fix, histogram improvements |
| `iq_blobs_ef/` | GEF LDA classifier, crash fix for missing fields |
| `ramsey/` | Improved decay fit robustness, added plotting modes |
| `ramsey_versus_flux_calibration/` | Extended to support flux-vs-qubit-frequency 2D fitting |
| `drag_calibration_180_minus180/` | Added convergence diagnostics |
| `single_qubit_randomized_benchmarking/` | Added 1Q-Clifford diagnostic mode, raw-IQ plot |
| `T1/analysis.py` | Improved exponential fit initial guess |
| `T2echo/analysis.py` | Added echo decay robustness |
| `cryoscope/` | Extended for coupler flux |
| `chevron_cz/`, `cz_conditional_phase/` | CZ calibration improvements (see CZ_calibrations/README.md) |

---

## New calibration_utils modules (supporting added nodes)

| Module | Purpose |
|--------|---------|
| `bringup_graphs.py` | Shared utilities for all bringup/retuning graphs (`_ensure_temp_calibration`, `should_keep_retuning`) |
| `broad_resonator_spectroscopy/` | Wide-span spectroscopy analysis + blacklist filtering |
| `T1_ef/`, `T1_vs_flux/`, `T1_monitor/`, `T1_thermal_monitor/` | T₁ variants |
| `T2echo_vs_flux/`, `T2star_vs_flux/` | T₂ vs. flux bias |
| `all_xy/`, `allxy/` | AllXY gate diagnostic |
| `stark_detuning_calibration/` | AC Stark shift fitting |
| `readout_frequency_amplitude_optimization/` | 2D readout optimization |
| `readout_depletion/`, `readout_length_optimization/` | Readout depletion and length optimization |
| `readout_gef_*/` | GEF readout optimization variants |
| `dispersive_shift/`, `dispersive_shift_gef/` | χ measurement analysis |
| `qubit_gef_thresholds/` | GEF LDA threshold calibration |
| `ef_rabi_rpm/`, `qubit_rpm/` | RPM measurement modules |

All cavity-specific modules (`cavity_*`, `fNgN1_*`, `displacement_*`, `wigner_*`, `parity_*`) are documented in [`cavity_calibrations/README.md`](../cavity_calibrations/README.md).

---

## Adaptive calibration graph (999)

`999_adaptive_graph.py` implements a fully automated calibration loop:

1. Run initial RB → identify qubits below `fidelity_threshold`
2. Failed qubits enter a retune subgraph:
   - Re-find idle flux point + qubit frequency (`09a_ramsey_vs_flux_calibration`)
   - Refine qubit frequency (`06a_ramsey`)
   - Refine x180 amplitude (`04b_power_rabi` error amplification)
   - Refine x90 amplitude
   - Verify via RB (`997_single_qubit_rb_graph`)
3. Loop continues until all qubits pass or `max_retune_iterations` is reached

State is tracked in `machine.temp_calibration` (`TemporaryCalibrationData`). A tutorial notebook is provided at `998_adaptive_graph_tutorial.ipynb`.

---

## QUAM infrastructure additions

The following files in `quam_config/` support multi-subfolder calibration scanning and hardware patches:

| File | Purpose |
|------|---------|
| `recursive_calibration_scan.py` | Monkeypatches `QualibrationLibrary` to scan `1Q_calibrations/`, `CZ_calibrations/`, and `cavity_calibrations/` in one call |
| `data_fetch_retry_patch.py` | Retry logic for QM data fetching on transient failures |
| `hardware_batching_patch.py` | Hardware slot batching for OPX1000 with many elements |
| `wiring_examples/` | Wiring configs for OPX+/Octave+cavity and OPX1000/MW-FEM+cavity |
