# %% {Resonator Bring-up}
"""
Resonator Bring-Up Graph — Sequential

A plain sequential graph that runs calibration nodes in order:
  broad_spectroscopy  →  high_power_spectroscopy  →  punch_out  →  low_power_spectroscopy

Each step is a standalone node with no retry loops or FSM logic.
"""
from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q5"]


class ResonatorBringUpParameters(GraphParameters):
    """Parameters for the sequential resonator bringup graph."""

    qubits: List[str] = test_qubits

    # Broad spectroscopy (02d)
    broad_frequency_span_mhz: float = 200.0
    broad_frequency_step_mhz: float = 0.1
    broad_num_shots: int = 50
    broad_peak_prominence: float = 2
    broad_peak_width: tuple = (1, 10.0)
    broad_peak_height: Optional[float] = None
    broad_peak_threshold: Optional[float] = None
    blacklist_exclusion_radius_mhz: float = 10.0
    broad_readout_power_dbm: Optional[float] = 0
    broad_max_amp: float = 0.1

    # High-power spectroscopy (02a)
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0
    high_power_max_amp: float = 0.1
    high_power_save_readout_amplitude: bool = False

    # Punch-out (02e)
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    punch_out_sweep_left_offset_mhz: float = 4.0
    use_adaptive_span: bool = True

    # Low-power spectroscopy (02a)
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1
    low_power_save_readout_amplitude: bool = True


with QualibrationGraph.build(
    "resonator_optimization",
    parameters=ResonatorBringUpParameters(),
) as graph:
    broad_spectroscopy = library.nodes["02d_broad_resonator_spectroscopy"].copy(
        name="broad_spectroscopy",
        qubits=graph.parameters.qubits,
        frequency_span_in_mhz=graph.parameters.broad_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.broad_frequency_step_mhz,
        num_shots=graph.parameters.broad_num_shots,
        peak_prominence=graph.parameters.broad_peak_prominence,
        peak_width=graph.parameters.broad_peak_width,
        peak_height=graph.parameters.broad_peak_height,
        peak_threshold=graph.parameters.broad_peak_threshold,
        blacklist_exclusion_radius_mhz=graph.parameters.blacklist_exclusion_radius_mhz,
        readout_power_dbm=graph.parameters.broad_readout_power_dbm,
        max_amp=graph.parameters.broad_max_amp,
    )

    high_power_spectroscopy = library.nodes["02a_resonator_spectroscopy"].copy(
        name="high_power_spectroscopy",
        qubits=graph.parameters.qubits,
        frequency_span_in_mhz=graph.parameters.high_power_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.high_power_frequency_step_mhz,
        num_shots=graph.parameters.high_power_num_shots,
        readout_power_dbm=graph.parameters.high_power_readout_power_dbm,
        max_amp=graph.parameters.high_power_max_amp,
        save_readout_amplitude=graph.parameters.high_power_save_readout_amplitude,
    )

    punch_out = library.nodes["02e_resonator_punch_out"].copy(
        name="punch_out",
        qubits=graph.parameters.qubits,
        frequency_span_in_mhz=graph.parameters.punch_out_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.punch_out_frequency_step_mhz,
        min_power_dbm=graph.parameters.punch_out_min_power_dbm,
        max_power_dbm=graph.parameters.punch_out_max_power_dbm,
        num_power_points=graph.parameters.punch_out_num_power_points,
        max_amp=graph.parameters.punch_out_max_amp,
        num_shots=graph.parameters.punch_out_num_shots,
        frequency_shift_threshold_in_hz=graph.parameters.punch_out_frequency_shift_threshold_hz,
        sweep_left_offset_mhz=graph.parameters.punch_out_sweep_left_offset_mhz,
        use_adaptive_span=graph.parameters.use_adaptive_span,
    )

    low_power_spectroscopy = library.nodes["02a_resonator_spectroscopy"].copy(
        name="low_power_spectroscopy",
        qubits=graph.parameters.qubits,
        frequency_span_in_mhz=graph.parameters.low_power_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.low_power_frequency_step_mhz,
        num_shots=graph.parameters.low_power_num_shots,
        readout_power_dbm=graph.parameters.low_power_readout_power_dbm,
        max_amp=graph.parameters.low_power_max_amp,
        save_readout_amplitude=graph.parameters.low_power_save_readout_amplitude,
    )

    graph.add_node(broad_spectroscopy)
    graph.add_node(high_power_spectroscopy)
    graph.add_node(punch_out)
    graph.add_node(low_power_spectroscopy)

    graph.connect(broad_spectroscopy, high_power_spectroscopy)
    graph.connect(high_power_spectroscopy, punch_out)
    graph.connect(punch_out, low_power_spectroscopy)

graph.run()
