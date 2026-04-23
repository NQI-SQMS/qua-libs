# %%
"""
Fixed-Frequency Transmon Bring-Up Graph (Adaptive FSM)

Full automated bring-up sequence for a fixed-frequency transmon qubit.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1.  mixer_calibration                                                  │
  │  2.  resonator_bringup (subgraph):                                      │
  │        resonator_discovery [loop: retry on no dip]:                     │
  │          broad_resonator_spectroscopy                                   │
  │          ──► resonator_spectroscopy_high_power                          │
  │        ──► resonator_punch_out        [loop: retry on failure]          │
  │        ──► resonator_spectroscopy_low_power                             │
  │  3.  qubit_calibration (subgraph, nested loops):                        │
  │        qubit_spectroscopy_vs_power    [inner loop: span expansion]      │
  │          (power-broadening fit → sets saturation & x180 amplitude)      │
  │        ──► time_rabi (saturation pulse)                                 │
  │        [outer loop: restart on NO_OSCILLATION → new freq search]        │
  │  4.  x180_fine_calibration (subgraph):                                  │
  │        rabi_ramsey [loop: repeat until freq converges]                  │
  │          power_rabi ──► ramsey                                          │
  │  5.  T1                                                                 │
  │  6.  readout_frequency_optimization                                     │
  │  7.  readout_length_optimization                                        │
  │  8.  readout_power_optimization                                         │
  │  9.  [ef_bringup] (if run_ef_calibration=True):                        │
  │        ef_spectroscopy [loop: retry on no peak]                         │
  │        ──► ef_power_rabi                                                │
  │  10. [cavity_bringup] (if run_cavity_calibration=True):                 │
  │        cavity_mode_spectroscopy                                         │
  │        ──► displacement_calibration                                     │
  │        ──► cavity_T1                                                    │
  │        ──► parity_time_measurement                                      │
  └─────────────────────────────────────────────────────────────────────────┘

Opt-out flags (evaluated at graph load time — set before importing the graph):
  - run_ef_calibration=False   → omits the EF-transition bringup subgraph
  - run_cavity_calibration=True → appends the cavity mode bringup subgraph
"""

from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import (
    build_resonator_bringup,
    build_qubit_calibration,
    build_x180_fine_calibration,
    build_ef_bringup,
    build_cavity_bringup,
    should_restart_qubit_calibration,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q1"]


# ─── Top-level parameters ─────────────────────────────────────────────────────

class TransmonBringUpParameters(GraphParameters):
    """Parameters for the full fixed-frequency transmon bring-up graph (adaptive FSM)."""

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # ── Iteration limits ──────────────────────────────────────────────────────
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5
    max_spec_vs_power_iterations: int = 5
    max_qubit_calibration_iterations: int = 3

    # ── Mixer calibration ──────────────────────────────────────────────────────
    mixer_calibrate_resonator: bool = True
    mixer_calibrate_drive: bool = True
    mixer_calibrate_cavity_drive: bool = True
    mixer_calibrate_sideband_drive: bool = True

    # ── Resonator – broad spectroscopy ────────────────────────────────────────
    broad_frequency_span_mhz: float = 200.0
    broad_frequency_step_mhz: float = 0.1
    broad_num_shots: int = 50
    broad_peak_prominence: float = 2.0
    broad_peak_width: List[float] = [1.0, 10.0]
    broad_peak_height: Optional[float] = None
    broad_peak_threshold: Optional[float] = None
    broad_readout_power_dbm: Optional[float] = 0.0
    broad_max_amp: float = 0.1
    blacklist_exclusion_radius_mhz: float = 10.0

    # ── Resonator – high-power confirmation ───────────────────────────────────
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0.0
    high_power_max_amp: float = 0.1
    high_power_save_readout_amplitude: bool = True

    # ── Resonator – punch-out ─────────────────────────────────────────────────
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    punch_out_sweep_left_offset_mhz: float = 4.0
    """MHz to extend the punch-out sweep to the LEFT of the bare resonator frequency,
    so that the dispersive-shifted low-power resonance is within the swept window."""
    use_adaptive_span: bool = True

    # ── Resonator – low-power fine spectroscopy ───────────────────────────────
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1
    low_power_save_readout_amplitude: bool = True

    # ── Qubit spectroscopy vs power ───────────────────────────────────────────
    spec_vs_power_frequency_span_mhz: float = 200.0
    spec_vs_power_frequency_step_mhz: float = 2.0
    spec_vs_power_num_power_points: int = 10
    spec_vs_power_num_shots: int = 100
    spec_vs_power_min_power_dbm: int = -80
    spec_vs_power_max_power_dbm: int = 0
    spec_vs_power_operation: str = "saturation"
    spec_vs_power_operation_len_ns: int = 200_000
    spec_vs_power_linewidth_threshold_hz: float = 10e6
    """Linewidth threshold (Hz) for qubit spectroscopy vs power operating-point selection."""
    spec_vs_power_max_amplitude_opx: float = 0.24
    spec_vs_power_min_amplitude_opx: float = 0.01
    spec_vs_power_power_buffer_db: float = 3.0
    spec_vs_power_signal_source: str = "I_rot"
    spec_vs_power_peak_persistence_lookahead: int = 0
    spec_vs_power_peak_persistence_freq_tolerance_hz: float = 5e6
    spec_vs_power_use_adaptive_span: bool = True
    """Enable adaptive frequency span and power expansion in qubit spectroscopy vs power."""
    spec_vs_power_rabi_target_periods: int = 3
    """Target Rabi periods within the time Rabi sweep for amplitude estimation."""
    spec_vs_power_rabi_sweep_max_duration_ns: float = 300.0
    """Upper bound of the time Rabi sweep [ns], used with rabi_target_periods to set T_pi."""
    # ── Time Rabi ─────────────────────────────────────────────────────────────
    time_rabi_min_duration_ns: int = 16
    time_rabi_max_duration_ns: int = 300
    time_rabi_duration_step_ns: int = 4
    time_rabi_num_shots: int = 200
    time_rabi_operation: str = "saturation"
    """Operation used for the time Rabi sweep. 'saturation' is used here because
    spec_vs_power sets its amplitude to match the desired π-pulse time."""
    time_rabi_operation_amplitude_factor: float = 1.0
    time_rabi_drive_power_dbm: Optional[float] = None
    time_rabi_max_amplitude_opx: float = 0.1

    # ── X180 fine calibration ─────────────────────────────────────────────────
    x180_rabi_min_amp_factor: float = 0.001
    x180_rabi_max_amp_factor: float = 1.99
    x180_rabi_amp_factor_step: float = 0.005
    x180_rabi_num_shots: int = 50
    x180_rabi_operation: str = "x180"
    x180_rabi_operation_length_in_ns: Optional[int] = None
    x180_rabi_max_number_pulses_per_sweep: int = 1
    x180_rabi_use_adaptive: bool = True
    """Enable adaptive amplitude/gain/duration escalation in power_rabi."""
    x180_rabi_max_amplitude_iterations: int = 5
    """Max retries of power_rabi to converge the period count before running Ramsey."""
    x180_rabi_octave_gain_step_db: float = 3.0
    """Max Octave gain step (dB) per adaptive iteration when base amplitude is maxed."""
    x180_rabi_update_x90: bool = True
    x180_ramsey_num_shots: int = 100
    x180_ramsey_frequency_detuning_in_mhz: float = 1.0
    x180_ramsey_min_wait_time_in_ns: int = 16
    x180_ramsey_max_wait_time_in_ns: int = 10_000
    x180_ramsey_wait_time_num_points: int = 200
    x180_ramsey_log_or_linear_sweep: str = "linear"
    x180_ramsey_x180_operation: str = "x180"
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

    # ── Readout length optimization ───────────────────────────────────────────
    readout_length_max_ns: int = 8000
    """Maximum readout pulse length to sweep during fidelity optimization [ns]."""
    readout_length_division_ns: int = 16
    """Accumulated demodulation chunk size [ns]. Must be a multiple of 4."""
    readout_length_num_shots: int = 10000
    """Single-shot averages for readout length optimization."""
    readout_length_readout_operation: str = "readout"
    readout_length_cos_weight_name: str = "iw1"
    readout_length_sin_weight_name: str = "iw2"
    readout_length_minus_sin_weight_name: str = "iw3"

    # ── Readout power optimization ────────────────────────────────────────────
    readout_power_num_shots: int = 2000
    readout_power_start_amp: float = 0.5
    readout_power_end_amp: float = 1.5
    readout_power_num_amps: int = 10
    readout_power_outliers_threshold: float = 0.98
    readout_power_plot_raw: bool = False

    # ── EF-transition calibration (opt-out) ───────────────────────────────────
    run_ef_calibration: bool = True
    """Set False before loading the graph to omit the EF bringup subgraph entirely.
    This flag is evaluated at graph-load time: the subgraph is only added to the
    graph when run_ef_calibration=True."""
    ef_spec_frequency_span_mhz: float = 100.0
    """Frequency span for EF spectroscopy around the anharmonicity-derived prior [MHz]."""
    ef_spec_frequency_step_mhz: float = 0.25
    """Frequency step for EF spectroscopy [MHz]."""
    ef_spec_operation: str = "saturation"
    """Drive operation used during EF spectroscopy."""
    ef_spec_operation_len_in_ns: Optional[int] = None
    """Duration override for the EF spectroscopy drive pulse [ns]. None = use pulse default."""
    ef_spec_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the saturation pulse during EF spectroscopy."""
    ef_spec_num_shots: int = 100
    """Averages per frequency point during EF spectroscopy."""
    ef_spec_target_peak_width: float = 3e6
    """Target FWHM used to scale saturation/x180 amplitudes in EF spectroscopy [Hz]."""
    ef_spec_update_pulses_amplitude: bool = False
    """Whether to update EF pulse amplitudes based on the fitted peak width."""
    ef_spec_find_dip: bool = False
    """Set True if the EF transition appears as a dip in I_rot (e.g. reflection readout)."""
    max_ef_spec_iterations: int = 3
    """Maximum retries for EF spectroscopy before giving up."""
    ef_rabi_min_amp_factor: float = 0.001
    """Minimum amplitude factor for EF power Rabi sweep."""
    ef_rabi_max_amp_factor: float = 1.99
    """Maximum amplitude factor for EF power Rabi sweep."""
    ef_rabi_amp_factor_step: float = 0.005
    """Amplitude step for EF power Rabi sweep."""
    ef_rabi_num_shots: int = 50
    """Averages per amplitude point during EF power Rabi."""

    # ── Cavity mode calibration (opt-in) ──────────────────────────────────────
    run_cavity_calibration: bool = False
    """Set True before loading the graph to append the cavity mode bringup subgraph.
    This flag is evaluated at graph-load time: the subgraph is only added when True."""
    cavity_mode_name: str = "alice"
    """Name of the cavity mode to calibrate. Must match an attribute on the Cavity
    component in QUAM (e.g. 'alice', 'bob')."""
    cavity_spec_frequency_span_mhz: float = 400.0
    """Frequency span for cavity mode spectroscopy [MHz]."""
    cavity_spec_frequency_step_mhz: float = 1.0
    """Frequency step for cavity mode spectroscopy [MHz]."""
    cavity_spec_operation: str = "saturation"
    """Drive operation for cavity mode spectroscopy."""
    cavity_spec_operation_len_in_ns: Optional[int] = None
    """Duration override for cavity spectroscopy drive pulse [ns]. None = use default."""
    cavity_spec_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the saturation pulse during cavity spectroscopy."""
    cavity_spec_num_shots: int = 100
    """Averages per frequency point during cavity spectroscopy."""
    cavity_spec_qubit_probe_operation: str = "selective_x180"
    """Qubit probe operation used to conditional-map cavity state during spectroscopy."""
    cavity_spec_use_state_discrimination: bool = True
    """Use state discrimination (True) or raw IQ threshold (False) in cavity spectroscopy."""
    cavity_spec_min_dip_fraction: float = 0.05
    """Minimum dip depth as a fraction of the baseline for cavity spectroscopy peak detection."""
    cavity_disp_amp_min: float = 0.0
    """Minimum displacement amplitude_scale for vacuum calibration."""
    cavity_disp_amp_max: float = 2.0
    """Maximum displacement amplitude_scale for vacuum calibration."""
    cavity_disp_amp_points: int = 51
    """Number of amplitude points for displacement calibration."""
    cavity_disp_num_shots: int = 1000
    """Averages per amplitude point for displacement calibration."""
    cavity_disp_qubit_pulse: str = "selective_x180"
    """Qubit probe pulse for displacement calibration."""
    cavity_disp_cavity_reset_type: str = "thermal"
    """Cavity reset method for displacement calibration: 'thermal' or 'active_sideband'."""
    cavity_disp_active_reset: bool = True
    """Enable active qubit reset before each displacement calibration shot."""
    cavity_disp_use_state_discrimination: bool = True
    """Use state discrimination in displacement calibration."""
    cavity_t1_min_wait_ns: int = 16
    """Minimum wait time for coherent cavity T1 measurement [ns]."""
    cavity_t1_max_wait_ns: int = 5_000_000
    """Maximum wait time for coherent cavity T1 measurement [ns]."""
    cavity_t1_num_points: int = 51
    """Number of time points for coherent cavity T1 measurement."""
    cavity_t1_num_shots: int = 1000
    """Averages per time point for coherent cavity T1 measurement."""
    cavity_t1_log_or_linear_sweep: str = "log"
    """Sweep spacing for cavity T1 time axis: 'log' or 'linear'."""
    cavity_t1_displacement_scale: float = 1.0
    """Scale factor applied to the displacement pulse amplitude for cavity T1."""
    cavity_t1_use_state_discrimination: bool = True
    """Use state discrimination in cavity T1 measurement."""
    cavity_t1_cavity_reset_type: str = "thermal"
    """Cavity reset method for T1 measurement: 'thermal' or 'active_sideband'."""
    parity_min_delay_ns: int = 16
    """Minimum Ramsey wait time for parity time measurement [ns]."""
    parity_max_delay_ns: int = 4000
    """Maximum Ramsey wait time for parity time measurement [ns]."""
    parity_delay_step_ns: int = 16
    """Delay step for parity time measurement [ns]."""
    parity_num_shots: int = 1000
    """Averages per delay point for parity time measurement."""
    parity_displacement_scale: float = 0.5
    """Displacement amplitude scale for parity time measurement."""
    parity_use_state_discrimination: bool = True
    """Use state discrimination in parity time measurement."""
    parity_cavity_reset_type: str = "thermal"
    """Cavity reset method for parity measurement: 'thermal' or 'active_sideband'."""


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "transmon_bringup_adaptive",
    parameters=TransmonBringUpParameters(),
) as graph:

    # ── 1. Mixer calibration ──────────────────────────────────────────────────
    mixer_calibration = library.nodes["01a_mixer_calibration"].copy(
        name="mixer_calibration",
        calibrate_resonator=graph.parameters.mixer_calibrate_resonator,
        calibrate_drive=graph.parameters.mixer_calibrate_drive,
        calibrate_cavity_drive=graph.parameters.mixer_calibrate_cavity_drive,
        calibrate_sideband_drive=graph.parameters.mixer_calibrate_sideband_drive,
    )
    graph.add_node(mixer_calibration)

    # ── 2. Resonator bring-up ─────────────────────────────────────────────────
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

    # ── 3. Qubit calibration (FSM: spec-vs-power → spec → power Rabi) ─────────
    qubit_calibration = build_qubit_calibration(graph, library)
    graph.add_node(qubit_calibration)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=graph.parameters.max_qubit_calibration_iterations,
    )

    # ── 4. X180 fine calibration (Ramsey → power Rabi loop) ───────────────────
    x180_fine_calibration = build_x180_fine_calibration(graph, library)
    graph.add_node(x180_fine_calibration)

    # ── 5. T1 ─────────────────────────────────────────────────────────────────
    t1 = library.nodes["05_T1"].copy(
        name="T1",
        num_shots=graph.parameters.t1_num_shots,
        min_wait_time_in_ns=graph.parameters.t1_min_wait_time_ns,
        max_wait_time_in_ns=graph.parameters.t1_max_wait_time_ns,
        wait_time_num_points=graph.parameters.t1_wait_time_num_points,
        log_or_linear_sweep=graph.parameters.t1_log_or_linear_sweep,
    )
    graph.add_node(t1)

    # ── 6. Readout frequency optimization ─────────────────────────────────────
    readout_freq_opt = library.nodes["08a_readout_frequency_optimization"].copy(
        name="readout_frequency_optimization",
        multiplexed=graph.parameters.multiplexed,
        num_shots=graph.parameters.readout_freq_num_shots,
        frequency_span_in_mhz=graph.parameters.readout_freq_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.readout_freq_frequency_step_mhz,
    )
    graph.add_node(readout_freq_opt)

    # ── 7. Readout length optimization ────────────────────────────────────────
    readout_length_opt = library.nodes["08d_readout_length_optimization"].copy(
        name="readout_length_optimization",
        max_readout_length_in_ns=graph.parameters.readout_length_max_ns,
        division_length_in_ns=graph.parameters.readout_length_division_ns,
        num_shots=graph.parameters.readout_length_num_shots,
        readout_operation=graph.parameters.readout_length_readout_operation,
        cos_weight_name=graph.parameters.readout_length_cos_weight_name,
        sin_weight_name=graph.parameters.readout_length_sin_weight_name,
        minus_sin_weight_name=graph.parameters.readout_length_minus_sin_weight_name,
    )
    graph.add_node(readout_length_opt)

    # ── 8. Readout power optimization ─────────────────────────────────────────
    readout_power_opt = library.nodes["08b_readout_power_optimization"].copy(
        name="readout_power_optimization",
        num_shots=graph.parameters.readout_power_num_shots,
        start_amp=graph.parameters.readout_power_start_amp,
        end_amp=graph.parameters.readout_power_end_amp,
        num_amps=graph.parameters.readout_power_num_amps,
        outliers_threshold=graph.parameters.readout_power_outliers_threshold,
        plot_raw=graph.parameters.readout_power_plot_raw,
    )
    graph.add_node(readout_power_opt)

    # ── 9. EF-transition bringup (optional, default on) ───────────────────────
    # This block is evaluated at graph-load time.  Set run_ef_calibration=False
    # in TransmonBringUpParameters (or override after import) before the graph
    # is scanned to omit these nodes.
    if graph.parameters.run_ef_calibration:
        ef_bringup = build_ef_bringup(graph, library)
        graph.add_node(ef_bringup)

    # ── 10. Cavity mode bringup (optional, default off) ───────────────────────
    # Set run_cavity_calibration=True and cavity_mode_name="alice" (or "bob")
    # before the graph is scanned to include the cavity bringup subgraph.
    if graph.parameters.run_cavity_calibration:
        cavity_bringup = build_cavity_bringup(graph, library)
        graph.add_node(cavity_bringup)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(mixer_calibration, resonator_bringup)
    graph.connect(resonator_bringup, qubit_calibration)
    graph.connect(qubit_calibration, x180_fine_calibration)
    graph.connect(x180_fine_calibration, t1)
    graph.connect(t1, readout_freq_opt)
    graph.connect(readout_freq_opt, readout_length_opt)
    graph.connect(readout_length_opt, readout_power_opt)

    # Determine the last mandatory node, then chain optional subgraphs after it.
    _last_node = readout_power_opt
    if graph.parameters.run_ef_calibration:
        graph.connect(_last_node, ef_bringup)
        _last_node = ef_bringup
    if graph.parameters.run_cavity_calibration:
        graph.connect(_last_node, cavity_bringup)


graph.run()
