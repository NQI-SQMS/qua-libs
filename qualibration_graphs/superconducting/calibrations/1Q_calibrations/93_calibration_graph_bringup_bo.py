# %%
"""
93_calibration_graph_bringup_bo.py — Fixed-Frequency Transmon Bring-Up Graph (BO)

Replaces the nested FSM qubit-discovery sequence in graph 92 with a single
Gaussian-Process Bayesian Optimisation node (03e_qubit_bo_bootstrap).

Architecture
────────────

  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. mixer_calibration                                               │
  │  2. resonator_bringup (subgraph — unchanged from graph 92):         │
  │       broad_resonator_spectroscopy                                  │
  │       ──► resonator_spectroscopy_high_power  [loop: retry on fail]  │
  │       ──► resonator_punch_out                [loop: retry on fail]  │
  │       ──► resonator_spectroscopy_low_power                          │
  │  3. qubit_bo_bootstrap  (NEW — replaces spec_vs_power + FSM)        │
  │       BO discovers (ω_q, V_π) in ≤ 60 time-Rabi evaluations        │
  │       No loop needed: convergence is internal to the BO node        │
  │  4. x180_fine_calibration (subgraph — unchanged from graph 92):     │
  │       power_rabi → ramsey  [loop: until frequency converges]        │
  │  5. T1                                                              │
  │  6. readout_frequency_optimization                                  │
  │  7. readout_power_optimization                                      │
  └─────────────────────────────────────────────────────────────────────┘

Why this is simpler than graph 92
──────────────────────────────────
Graph 92 has a four-level nested retry FSM for qubit discovery:
  outer loop: should_restart_qubit_calibration (blacklist + restart)
    inner subgraph: spec_vs_power [loop: should_repeat_spec_vs_power]
      → qubit_spec
      → power_rabi [loop: should_repeat_rabi_amplitude]

The BO bootstrap replaces ALL of this with a single node.  There are no
condition functions needed for qubit discovery.  The only fallback is a
single manual retry if the BO outcome is 'failed' (rare; indicates the
search space needs to be widened or the resonator readout is unreliable).

Usage
─────
    python 93_calibration_graph_bringup_bo.py
or via QualibrationLibrary:
    library.graphs["93_calibration_graph_bringup_bo"].run(parameters={...})
"""

from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import (
    build_resonator_bringup,
    build_x180_fine_calibration,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q2"]


# ─── Top-level graph parameters ───────────────────────────────────────────────

class TransmonBringUpBOParameters(GraphParameters):
    """
    Parameters for the BO-based fixed-frequency transmon bring-up graph.

    Resonator bring-up and X180 fine calibration parameters are forwarded to
    the corresponding subgraph builders (same as graph 92).

    BO-specific parameters are forwarded directly to the
    03e_qubit_bo_bootstrap node.
    """

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # ── Iteration limits ──────────────────────────────────────────────────────
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5

    # ── Mixer calibration ──────────────────────────────────────────────────────
    mixer_calibrate_resonator: bool = True
    mixer_calibrate_drive: bool = True

    # ── Resonator – broad spectroscopy ────────────────────────────────────────
    broad_frequency_span_mhz: float = 200.0
    broad_frequency_step_mhz: float = 0.1
    broad_num_shots: int = 50
    broad_peak_prominence: float = 2.0
    broad_peak_width: tuple = (1, 10.0)
    broad_readout_power_dbm: Optional[float] = 0.0
    broad_max_amp: float = 0.1
    blacklist_exclusion_radius_mhz: float = 10.0

    # ── Resonator – high-power confirmation ───────────────────────────────────
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0.0
    high_power_max_amp: float = 0.1

    # ── Resonator – punch-out ─────────────────────────────────────────────────
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    use_adaptive_span: bool = True

    # ── Resonator – low-power fine spectroscopy ───────────────────────────────
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1

    # ── BO bootstrap ──────────────────────────────────────────────────────────
    # All parameters forwarded to 03e_qubit_bo_bootstrap.
    # See calibration_utils/qubit_bo_bootstrap/parameters.py for full docs.

    bo_num_shots: int = 100
    """Shots per time-Rabi trace.  100 shots ≈ 1 MHz projection-noise floor."""

    bo_min_duration_ns: int = 16
    """Minimum drive pulse duration in ns (must be multiple of 4)."""
    bo_max_duration_ns: int = 448
    """Maximum drive pulse duration in ns. 448 ns ≈ 9 periods at 20 MHz target."""
    bo_duration_step_ns: int = 4
    """Step size for duration sweep in ns."""
    bo_operation: str = "x180"
    """QUAM operation name to use for the time-Rabi pulse."""

    bo_target_rabi_freq_mhz: float = 20.0
    """Target Rabi frequency Ω_T in MHz for the Wolff cost function."""
    bo_w_rabi: float = 10.0
    """Weight of the Rabi-frequency cost term."""
    bo_w_amp: float = 3.0
    """Weight of the oscillation-amplitude cost term."""
    bo_w_snr: float = 1.0
    """Weight of the SNR cost term."""

    bo_freq_search_radius_mhz: float = 200.0
    """Half-width of the drive frequency search window (MHz)."""
    bo_amp_search_min_db: float = -20.0
    """Lower bound for drive amplitude search (dB relative to QUAM prior)."""
    bo_amp_search_max_db: float = 20.0
    """Upper bound for drive amplitude search (dB relative to QUAM prior)."""
    bo_max_amplitude_opx: float = 0.45
    """Hard cap on OPX IF waveform amplitude (V). Leave 0.05 V margin below 0.5 V max."""

    bo_n_initial_lhs: int = 12
    """Number of Phase 1a broad LHS points."""
    bo_n_zoom_lhs: int = 10
    """Number of Phase 1b zoom LHS points."""
    bo_zoom_radius_mhz: float = 50.0
    """Half-width of Phase 1b zoom window (MHz)."""
    bo_n_bo_iterations: int = 40
    """Maximum number of Phase 2 BO acquisition iterations."""
    bo_convergence_tolerance_mhz: float = 1.0
    """GP-predicted optimum shift threshold for convergence (MHz)."""
    bo_acq: str = "EI"
    """Acquisition function: 'EI' (Expected Improvement) or 'UCB'."""
    bo_seed: Optional[int] = None
    """Random seed for LHS sampler and GP initialisation."""

    bo_optimize_readout_jointly: bool = False
    """If True, also optimize readout frequency and amplitude (4D search)."""
    bo_readout_freq_radius_mhz: float = 3.0
    """Half-width of readout frequency search window (4D only, MHz)."""
    bo_readout_amp_search_min_db: float = -6.0
    """Lower bound for readout amplitude search (4D only, dB)."""
    bo_readout_amp_search_max_db: float = 6.0
    """Upper bound for readout amplitude search (4D only, dB)."""

    # ── X180 fine calibration ─────────────────────────────────────────────────
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

    # ── T1 ────────────────────────────────────────────────────────────────────
    t1_num_shots: int = 1000
    t1_min_wait_time_ns: int = 16
    t1_max_wait_time_ns: int = 200_000
    t1_wait_time_num_points: int = 100
    t1_log_or_linear_sweep: str = "log"

    # ── Readout frequency optimization ────────────────────────────────────────
    readout_freq_frequency_span_mhz: float = 2.0
    readout_freq_frequency_step_mhz: float = 0.01
    readout_freq_num_shots: int = 100

    # ── Readout power optimization ────────────────────────────────────────────
    readout_power_num_shots: int = 2000
    readout_power_start_amp: float = 0.5
    readout_power_end_amp: float = 1.5
    readout_power_num_amps: int = 10


# ─── Graph construction ────────────────────────────────────────────────────────

with QualibrationGraph.build(
    "93_calibration_graph_bringup_bo",
    parameters=TransmonBringUpBOParameters(),
) as graph:

    p = graph.parameters

    # ── 1. Mixer calibration ──────────────────────────────────────────────────
    mixer_calibration = library.nodes["01a_mixer_calibration"].copy(
        name="mixer_calibration",
        calibrate_resonator=p.mixer_calibrate_resonator,
        calibrate_drive=p.mixer_calibrate_drive,
    )
    graph.add_node(mixer_calibration)

    # ── 2. Resonator bring-up (unchanged from graph 92) ───────────────────────
    # The resonator must be calibrated before the qubit BO can run, because the
    # BO cost function relies on the readout SNR term — a miscalibrated readout
    # produces noisy Rabi traces that confuse the GP.
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

    # ── 3. BO qubit bootstrap (replaces spec_vs_power + qubit_spec + power_rabi FSM)
    # The BO node handles convergence internally. No condition function is needed.
    # If the BO outcome is 'failed' for all qubits, the graph will stop here and
    # report the failure — the operator should widen bo_freq_search_radius_mhz or
    # check the resonator readout before retrying.
    qubit_bo_bootstrap = library.nodes["03e_qubit_bo_bootstrap"].copy(
        name="qubit_bo_bootstrap",
        multiplexed=p.multiplexed,
        num_shots=p.bo_num_shots,
        min_duration_ns=p.bo_min_duration_ns,
        max_duration_ns=p.bo_max_duration_ns,
        duration_step_ns=p.bo_duration_step_ns,
        operation=p.bo_operation,
        target_rabi_freq_mhz=p.bo_target_rabi_freq_mhz,
        w_rabi=p.bo_w_rabi,
        w_amp=p.bo_w_amp,
        w_snr=p.bo_w_snr,
        freq_search_radius_mhz=p.bo_freq_search_radius_mhz,
        amp_search_min_db=p.bo_amp_search_min_db,
        amp_search_max_db=p.bo_amp_search_max_db,
        max_amplitude_opx=p.bo_max_amplitude_opx,
        n_initial_lhs=p.bo_n_initial_lhs,
        n_zoom_lhs=p.bo_n_zoom_lhs,
        zoom_radius_mhz=p.bo_zoom_radius_mhz,
        n_bo_iterations=p.bo_n_bo_iterations,
        convergence_tolerance_mhz=p.bo_convergence_tolerance_mhz,
        acq=p.bo_acq,
        bo_seed=p.bo_seed,
        optimize_readout_jointly=p.bo_optimize_readout_jointly,
        readout_freq_radius_mhz=p.bo_readout_freq_radius_mhz,
        readout_amp_search_min_db=p.bo_readout_amp_search_min_db,
        readout_amp_search_max_db=p.bo_readout_amp_search_max_db,
    )
    graph.add_node(qubit_bo_bootstrap)

    # ── 4. X180 fine calibration (unchanged from graph 92) ────────────────────
    # The BO leaves the qubit state with a rough (ω_q, V_π) — good enough to
    # see coherent oscillations.  The x180 fine calibration refines this to
    # sub-MHz frequency accuracy and sub-percent amplitude accuracy using
    # power_rabi + Ramsey with ~50 kHz convergence threshold.
    x180_fine_calibration = build_x180_fine_calibration(graph, library)
    graph.add_node(x180_fine_calibration)

    # ── 5. T1 ─────────────────────────────────────────────────────────────────
    t1 = library.nodes["05_T1"].copy(
        name="T1",
        num_shots=p.t1_num_shots,
        min_wait_time_in_ns=p.t1_min_wait_time_ns,
        max_wait_time_in_ns=p.t1_max_wait_time_ns,
        wait_time_num_points=p.t1_wait_time_num_points,
        log_or_linear_sweep=p.t1_log_or_linear_sweep,
    )
    graph.add_node(t1)

    # ── 6. Readout frequency optimization ─────────────────────────────────────
    readout_freq_opt = library.nodes["08a_readout_frequency_optimization"].copy(
        name="readout_frequency_optimization",
        multiplexed=p.multiplexed,
        num_shots=p.readout_freq_num_shots,
        frequency_span_in_mhz=p.readout_freq_frequency_span_mhz,
        frequency_step_in_mhz=p.readout_freq_frequency_step_mhz,
    )
    graph.add_node(readout_freq_opt)

    # ── 7. Readout power optimization ─────────────────────────────────────────
    readout_power_opt = library.nodes["08b_readout_power_optimization"].copy(
        name="readout_power_optimization",
        num_shots=p.readout_power_num_shots,
        start_amp=p.readout_power_start_amp,
        end_amp=p.readout_power_end_amp,
        num_amps=p.readout_power_num_amps,
    )
    graph.add_node(readout_power_opt)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(mixer_calibration, resonator_bringup)
    graph.connect(resonator_bringup, qubit_bo_bootstrap)
    graph.connect(qubit_bo_bootstrap, x180_fine_calibration)
    graph.connect(x180_fine_calibration, t1)
    graph.connect(t1, readout_freq_opt)
    graph.connect(readout_freq_opt, readout_power_opt)


graph.run()
