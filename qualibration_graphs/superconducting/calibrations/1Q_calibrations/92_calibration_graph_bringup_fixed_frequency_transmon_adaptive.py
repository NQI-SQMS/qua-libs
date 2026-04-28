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
  │  9.  [ef_bringup] (if run_ef_calibration=True)                         │
  │  10. [cavity_bringup] (if run_cavity_calibration=True)                  │
  └─────────────────────────────────────────────────────────────────────────┘

Graph-level parameters control graph FLOW (iteration limits, convergence
thresholds, opt-in flags).  Node-specific measurement parameters are baked
into the node copies in bringup_graphs.py and can be edited there or through
the Qualibrate GUI at the individual node level.
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
    """Graph-flow parameters for the fixed-frequency transmon bring-up graph.

    Only parameters that control graph STRUCTURE or LOOP BEHAVIOUR live here.
    Node-specific measurement parameters (frequency spans, shot counts, etc.)
    are set directly in the node copies inside bringup_graphs.py, which
    eliminates the duplicated-parameter view in the Qualibrate GUI.
    """

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # ── Iteration limits ──────────────────────────────────────────────────────
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5
    max_spec_vs_power_iterations: int = 5
    max_qubit_calibration_iterations: int = 3
    max_ef_discovery_iterations: int = 3
    """Max retries of ef_spectroscopy + ef_tentative_rabi when no EF oscillation is found."""
    max_ef_spec_iterations: int = 3
    ef_max_iterations: int = 5
    """Max iterations of the ef_power_rabi → ef_ramsey convergence loop."""
    ef_rabi_max_amplitude_iterations: int = 5
    """Max inner retries of ef_power_rabi to converge period count before EF Ramsey."""
    x180_max_iterations: int = 10
    x180_rabi_max_amplitude_iterations: int = 5
    """Max retries of power_rabi to converge the period count before running Ramsey."""

    # ── Mixer calibration ──────────────────────────────────────────────────────
    mixer_calibrate_resonator: bool = True
    mixer_calibrate_drive: bool = True
    mixer_calibrate_cavity_drive: bool = True
    mixer_calibrate_sideband_drive: bool = True

    # ── Adaptive behaviour ─────────────────────────────────────────────────────
    use_adaptive_span: bool = True
    """Enable adaptive span/power expansion in the resonator punch-out."""
    spec_vs_power_use_adaptive_span: bool = True
    """Enable adaptive frequency span and power expansion in qubit spectroscopy vs power."""
    x180_rabi_use_adaptive: bool = True
    """Enable adaptive amplitude/gain/duration escalation in power_rabi."""
    x180_rabi_octave_gain_step_db: float = 10.0
    """Max Octave gain step (dB) per adaptive iteration when base amplitude is maxed."""

    # ── Convergence thresholds ─────────────────────────────────────────────────
    x180_freq_threshold_hz: float = 50_000.0
    """Stop the power_rabi → ramsey loop when |GE detuning| < this value [Hz]."""
    ef_freq_threshold_hz: float = 50_000.0
    """Stop the ef_power_rabi → ef_ramsey loop when |EF detuning| < this value [Hz]."""

    # ── Optional subgraphs (evaluated at graph scan time) ─────────────────────
    run_ef_calibration: bool = False
    """Set False before loading the graph to omit the EF bringup subgraph entirely."""
    run_cavity_calibration: bool = False
    """Set True before loading the graph to append the cavity mode bringup subgraph."""
    cavity_mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""


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

    # ── 3. Qubit calibration (FSM: spec-vs-power → time Rabi) ─────────────────
    qubit_calibration = build_qubit_calibration(graph, library)
    graph.add_node(qubit_calibration)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=graph.parameters.max_qubit_calibration_iterations,
    )

    # ── 4. X180 fine calibration (power_rabi → ramsey loop) ───────────────────
    x180_fine_calibration = build_x180_fine_calibration(graph, library)
    graph.add_node(x180_fine_calibration)

    # ── 5. T1 ─────────────────────────────────────────────────────────────────
    t1 = library.nodes["05_T1"].copy(
        name="T1",
        num_shots=1000,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=200_000,
        wait_time_num_points=100,
        log_or_linear_sweep="linear",
    )
    graph.add_node(t1)

    # ── 6. Readout frequency optimization ─────────────────────────────────────
    readout_freq_opt = library.nodes["08a_readout_frequency_optimization"].copy(
        name="readout_frequency_optimization",
        multiplexed=graph.parameters.multiplexed,
        num_shots=100,
        frequency_span_in_mhz=20.0,
        frequency_step_in_mhz=0.1,
    )
    graph.add_node(readout_freq_opt)

    # ── 7. Readout length optimization ────────────────────────────────────────
    readout_length_opt = library.nodes["08d_readout_length_optimization"].copy(
        name="readout_length_optimization",
        max_readout_length_in_ns=12000,
        division_length_in_ns=160,
        num_shots=2000,
        readout_operation="readout",
        cos_weight_name="iw1",
        sin_weight_name="iw2",
        minus_sin_weight_name="iw3",
    )
    graph.add_node(readout_length_opt)

    # ── 8. Readout power optimization ─────────────────────────────────────────
    readout_power_opt = library.nodes["08b_readout_power_optimization"].copy(
        name="readout_power_optimization",
        num_shots=2000,
        start_amp=0.5,
        end_amp=1.5,
        num_amps=10,
        outliers_threshold=0.98,
        plot_raw=False,
    )
    graph.add_node(readout_power_opt)

    # ── 9. EF-transition bringup (optional, default on) ───────────────────────
    if graph.parameters.run_ef_calibration:
        ef_bringup = build_ef_bringup(graph, library)
        graph.add_node(ef_bringup)

    # ── 10. Cavity mode bringup (optional, default off) ───────────────────────
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

    _last_node = readout_power_opt
    if graph.parameters.run_ef_calibration:
        graph.connect(_last_node, ef_bringup)
        _last_node = ef_bringup
    if graph.parameters.run_cavity_calibration:
        graph.connect(_last_node, cavity_bringup)


graph.run()
