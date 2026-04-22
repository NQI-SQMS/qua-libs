# %%
"""
93_calibration_graph_bringup_bo.py
═══════════════════════════════════════════════════════════════════════════════
Fixed-Frequency Transmon Bring-Up Graph — BO Bootstrap version.

Replaces the nested retry FSM (92_..._adaptive.py) with a clean 3-stage pipeline:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. mixer_calibration                                                   │
  │  2. resonator_bringup (subgraph — unchanged from adaptive graph):       │
  │       broad_resonator_spectroscopy                                      │
  │       ──► resonator_spectroscopy_high_power  [retry on no dip]         │
  │       ──► resonator_punch_out                [retry on failure]         │
  │       ──► resonator_spectroscopy_low_power                              │
  │  3. qubit_bo_bootstrap  ← NEW: replaces the entire qubit FSM           │
  │       Phase 1: LHS seeding (n_initial_lhs quasi-random evaluations)    │
  │       Phase 2: GP-BO (EI acquisition, converges when Δω_q < tol)       │
  │       [Optional] Ramsey sanity check for spurious mode detection        │
  │  4. x180_fine_calibration (subgraph — unchanged):                       │
  │       rabi_ramsey [loop until freq converges]                           │
  │         power_rabi ──► ramsey                                           │
  │  5. T1                                                                  │
  │  6. readout_frequency_optimization                                      │
  │  7. readout_power_optimization                                          │
  └─────────────────────────────────────────────────────────────────────────┘

Key differences from 92_calibration_graph_bringup_fixed_frequency_transmon_adaptive.py:
  - No condition functions for qubit discovery (should_repeat_*, should_restart_*)
  - No error code FSM (PowerRabiErrorCode, QubitSpectroscopyErrorCode, etc.)
  - No spec_vs_power, qubit_spec, power_rabi nodes in the bring-up phase
  - The BO bootstrap node is a single node that handles all qubit parameter discovery

The resonator bringup subgraph is preserved unchanged — its condition functions
are appropriate (hardware-level failures, not parameter-estimation failures).
"""

from typing import List, Optional

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import (
    build_resonator_bringup,
    build_x180_fine_calibration,
)

library = QualibrationLibrary.get_active_library()
test_qubits = ["q0"]


# ── Graph parameters ───────────────────────────────────────────────────────────

class BringUpBoParameters(GraphParameters):
    """
    Parameters for the BO-based fixed-frequency transmon bring-up graph.

    Sections:
      - Qubit list and multiplexing
      - Mixer calibration
      - Resonator bringup (forwarded to build_resonator_bringup unchanged)
      - BO bootstrap (forwarded to 03e_qubit_bo_bootstrap node)
      - X180 fine calibration (forwarded to build_x180_fine_calibration unchanged)
      - T1 and readout optimization
    """

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # ── Mixer ──────────────────────────────────────────────────────────────────
    mixer_calibrate_resonator: bool = True
    mixer_calibrate_drive: bool = True

    # ── Resonator bringup — forwarded unchanged ────────────────────────────────
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5
    broad_frequency_span_mhz: float = 200.0
    broad_frequency_step_mhz: float = 0.1
    broad_num_shots: int = 50
    broad_peak_prominence: float = 2.0
    broad_peak_width: tuple = (1, 10.0)
    broad_readout_power_dbm: Optional[float] = 0.0
    broad_max_amp: float = 0.1
    blacklist_exclusion_radius_mhz: float = 10.0
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0.0
    high_power_max_amp: float = 0.1
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    use_adaptive_span: bool = True
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1
    save_resonator_amplitudes: bool = True
    min_dip_contrast: float = 0.02
    lo_leakage_exclusion_mhz: float = 10.0

    # ── BO bootstrap ───────────────────────────────────────────────────────────
    bo_target_rabi_freq_mhz: float = 20.0
    """Target Rabi frequency Ω_T (MHz). Sets cost function and time axis.
    Must be << 1/t_rise and >> 1/T2*. 20 MHz works for most SQMS transmons."""

    bo_freq_search_radius_mhz: float = 200.0
    """±radius (MHz) around QUAM prior frequency for drive frequency search."""

    bo_amp_search_min_db: float = -20.0
    """Lower amplitude bound (dB, relative to QUAM x180 amplitude)."""

    bo_amp_search_max_db: float = 20.0
    """Upper amplitude bound (dB, relative to QUAM x180 amplitude)."""

    bo_n_initial_lhs: int = 12
    """Latin Hypercube Sampling points to seed the GP before BO acquisition."""

    bo_n_bo_iterations: int = 40
    """Maximum BO iterations after LHS phase. Total evaluations ≤ 12+40 = 52."""

    bo_acq_function: str = "EI"
    """Acquisition function: 'EI' (Expected Improvement) or 'UCB'."""

    bo_convergence_tolerance_mhz: float = 1.0
    """Stop BO early if GP-predicted optimum moves < this between iterations."""

    bo_num_shots: int = 100
    """Averages per time-Rabi measurement."""

    bo_run_ramsey_sanity_check: bool = True
    """Run a quick Ramsey after convergence to verify T2* (spurious mode detection)."""

    bo_min_t2star_sanity_ns: float = 500.0
    """Minimum T2* (ns) to accept the converged frequency as real qubit."""

    bo_optimize_readout_jointly: bool = False
    """Expand BO to 4D (also optimize ω_r, V_r). Default False — use resonator
    bringup result as-is. Only enable if resonator bringup is unreliable."""

    # ── X180 fine calibration — forwarded unchanged ────────────────────────────
    x180_rabi_min_amp_factor: float = 0.001
    x180_rabi_max_amp_factor: float = 1.99
    x180_rabi_amp_factor_step: float = 0.005
    x180_rabi_num_shots: int = 50
    x180_rabi_max_number_pulses_per_sweep: int = 1
    x180_ramsey_num_shots: int = 100
    x180_ramsey_frequency_detuning_in_mhz: float = 1.0
    x180_ramsey_max_wait_time_in_ns: int = 10_000
    x180_ramsey_wait_time_num_points: int = 200
    x180_ramsey_log_or_linear_sweep: str = "linear"
    x180_freq_threshold_hz: float = 50_000.0
    x180_max_iterations: int = 10

    # ── T1 ─────────────────────────────────────────────────────────────────────
    t1_num_shots: int = 1000
    t1_min_wait_time_ns: int = 16
    t1_max_wait_time_ns: int = 200_000
    t1_wait_time_num_points: int = 100
    t1_log_or_linear_sweep: str = "log"

    # ── Readout optimization ───────────────────────────────────────────────────
    readout_freq_frequency_span_mhz: float = 2.0
    readout_freq_frequency_step_mhz: float = 0.01
    readout_freq_num_shots: int = 100
    readout_power_num_shots: int = 2000
    readout_power_start_amp: float = 0.5
    readout_power_end_amp: float = 1.5
    readout_power_num_amps: int = 10


# ── Graph construction ─────────────────────────────────────────────────────────

with QualibrationGraph.build(
    "transmon_bringup_bo",
    parameters=BringUpBoParameters(),
) as graph:

    # ── 1. Mixer calibration ──────────────────────────────────────────────────
    mixer_calibration = library.nodes["01a_mixer_calibration"].copy(
        name="mixer_calibration",
        calibrate_resonator=graph.parameters.mixer_calibrate_resonator,
        calibrate_drive=graph.parameters.mixer_calibrate_drive,
    )
    graph.add_node(mixer_calibration)

    # ── 2. Resonator bringup (unchanged subgraph) ─────────────────────────────
    # This subgraph's condition functions are appropriate — they handle
    # hardware-level failures (no dip found, punch-out failed) which are
    # qualitatively different from parameter estimation failures.
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

    # ── 3. BO qubit bootstrap ─────────────────────────────────────────────────
    # Single node — no condition functions, no nested retry logic.
    # All intelligence is inside the BOOptimizer + cost function.
    qubit_bo_bootstrap = library.nodes["03e_qubit_bo_bootstrap"].copy(
        name="qubit_bo_bootstrap",
        target_rabi_freq_mhz       = graph.parameters.bo_target_rabi_freq_mhz,
        freq_search_radius_mhz     = graph.parameters.bo_freq_search_radius_mhz,
        amp_search_min_db          = graph.parameters.bo_amp_search_min_db,
        amp_search_max_db          = graph.parameters.bo_amp_search_max_db,
        n_initial_lhs              = graph.parameters.bo_n_initial_lhs,
        n_bo_iterations            = graph.parameters.bo_n_bo_iterations,
        acq_function               = graph.parameters.bo_acq_function,
        convergence_tolerance_mhz  = graph.parameters.bo_convergence_tolerance_mhz,
        num_shots                  = graph.parameters.bo_num_shots,
        run_ramsey_sanity_check    = graph.parameters.bo_run_ramsey_sanity_check,
        min_t2star_sanity_ns       = graph.parameters.bo_min_t2star_sanity_ns,
        optimize_readout_jointly   = graph.parameters.bo_optimize_readout_jointly,
    )
    graph.add_node(qubit_bo_bootstrap)

    # ── 4. X180 fine calibration (unchanged subgraph) ─────────────────────────
    # After BO bootstrap gives a rough ω_q and V_π, this subgraph refines them
    # with Ramsey (for ω_q) and error-amplified power_rabi (for V_π).
    x180_fine_calibration = build_x180_fine_calibration(graph, library)
    graph.add_node(x180_fine_calibration)

    # ── 5. T1 ─────────────────────────────────────────────────────────────────
    t1 = library.nodes["05_T1"].copy(
        name="T1",
        num_shots=graph.parameters.t1_num_shots,
        min_wait_time_ns=graph.parameters.t1_min_wait_time_ns,
        max_wait_time_ns=graph.parameters.t1_max_wait_time_ns,
        wait_time_num_points=graph.parameters.t1_wait_time_num_points,
        log_or_linear_sweep=graph.parameters.t1_log_or_linear_sweep,
    )
    graph.add_node(t1)

    # ── 6. Readout frequency optimization ─────────────────────────────────────
    readout_freq_opt = library.nodes["08a_readout_frequency_optimization"].copy(
        name="readout_frequency_optimization",
        frequency_span_mhz=graph.parameters.readout_freq_frequency_span_mhz,
        frequency_step_mhz=graph.parameters.readout_freq_frequency_step_mhz,
        num_shots=graph.parameters.readout_freq_num_shots,
    )
    graph.add_node(readout_freq_opt)

    # ── 7. Readout power optimization ─────────────────────────────────────────
    readout_power_opt = library.nodes["08b_readout_power_optimization"].copy(
        name="readout_power_optimization",
        num_shots=graph.parameters.readout_power_num_shots,
        start_amp=graph.parameters.readout_power_start_amp,
        end_amp=graph.parameters.readout_power_end_amp,
        num_amps=graph.parameters.readout_power_num_amps,
    )
    graph.add_node(readout_power_opt)
