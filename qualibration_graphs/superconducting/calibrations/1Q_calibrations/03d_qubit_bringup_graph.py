# %%
"""
Qubit Bringup Graph — Sequential

A simple sequential graph that runs calibration nodes in order:
  spec_vs_power  →  power_rabi

Each node is a plain standalone node with no retry loops or FSM logic.
Retry orchestration (if needed) is handled externally by CausalOrchestrator
via bringup_causal.py.
"""

from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q1"]


class QubitBringupParameters(GraphParameters):
    """Parameters for the sequential qubit bringup graph."""

    qubits: List[str] = test_qubits

    # Qubit spectroscopy vs power
    spec_vs_power_frequency_span_mhz: float = 200
    spec_vs_power_frequency_step_mhz: float = 2
    spec_vs_power_num_power_points: int = 10
    spec_vs_power_num_shots: int = 100
    spec_vs_power_min_power_dbm: int = -80
    spec_vs_power_max_power_dbm: int = 0
    spec_vs_power_operation: str = "saturation"
    spec_vs_power_operation_len_ns: int = 200_000
    spec_vs_power_linewidth_threshold_hz: float = 10e6
    spec_vs_power_max_amplitude_opx: float = 0.24
    spec_vs_power_min_amplitude_opx: float = 0.01
    spec_vs_power_power_buffer_db: float = 3.0
    spec_vs_power_signal_source: str = "I_rot"
    spec_vs_power_peak_persistence_lookahead: int = 0
    spec_vs_power_peak_persistence_freq_tolerance_hz: float = 5e6

    # Power Rabi
    rabi_min_amp_factor: float = 0.8
    rabi_max_amp_factor: float = 1.2
    rabi_amp_factor_step: float = 0.01
    rabi_num_shots: int = 200
    rabi_operation: str = "x180"
    rabi_max_number_pulses_per_sweep: int = 1


with QualibrationGraph.build(
    "qubit_bringup",
    parameters=QubitBringupParameters(),
) as graph:
    spec_vs_power = library.nodes["03c_qubit_spectroscopy_vs_power"].copy(
        name="spec_vs_power",
        qubits=graph.parameters.qubits,
        frequency_span_in_mhz=graph.parameters.spec_vs_power_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.spec_vs_power_frequency_step_mhz,
        num_power_points=graph.parameters.spec_vs_power_num_power_points,
        num_shots=graph.parameters.spec_vs_power_num_shots,
        min_power_dbm=graph.parameters.spec_vs_power_min_power_dbm,
        max_power_dbm=graph.parameters.spec_vs_power_max_power_dbm,
        operation=graph.parameters.spec_vs_power_operation,
        operation_len_in_ns=graph.parameters.spec_vs_power_operation_len_ns,
        linewidth_threshold_hz=graph.parameters.spec_vs_power_linewidth_threshold_hz,
        max_amplitude_opx=graph.parameters.spec_vs_power_max_amplitude_opx,
        min_amplitude_opx=graph.parameters.spec_vs_power_min_amplitude_opx,
        power_buffer_db=graph.parameters.spec_vs_power_power_buffer_db,
        signal_source=graph.parameters.spec_vs_power_signal_source,
        peak_persistence_lookahead=graph.parameters.spec_vs_power_peak_persistence_lookahead,
        peak_persistence_freq_tolerance_hz=graph.parameters.spec_vs_power_peak_persistence_freq_tolerance_hz,
    )

    power_rabi = library.nodes["04b_power_rabi"].copy(
        name="power_rabi",
        qubits=graph.parameters.qubits,
        min_amp_factor=graph.parameters.rabi_min_amp_factor,
        max_amp_factor=graph.parameters.rabi_max_amp_factor,
        amp_factor_step=graph.parameters.rabi_amp_factor_step,
        num_shots=graph.parameters.rabi_num_shots,
        operation=graph.parameters.rabi_operation,
        max_number_pulses_per_sweep=graph.parameters.rabi_max_number_pulses_per_sweep,
    )

    graph.add_node(spec_vs_power)
    graph.add_node(power_rabi)

    graph.connect(spec_vs_power, power_rabi)

graph.run()
