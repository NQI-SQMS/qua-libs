# %% {Resonator Bring-up}
"""
Resonator Bring-Up Graph

This graph finds the ideal resonator parameters (frequency and readout amplitude) through:
1. Broad resonator spectroscopy  → finds rough resonator frequency over wide span
2. Resonator spectroscopy (high power) → confirms frequency with strong signal
   → If no dip found: blacklist that frequency, loop back to step 1
3. Resonator punch-out → measures Kerr shift to find optimal readout power
4. Resonator spectroscopy (low power) → precise frequency at optimal power
"""
from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import build_resonator_bringup

library = QualibrationLibrary.get_active_library()


class ResonatorBringUpParameters(GraphParameters):
    """Parameters for the resonator optimization graph."""
    qubits: List[str] = ["q5"]

    # General
    multiplexed: bool = False

    # Broad spectroscopy
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

    # High-power spectroscopy
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0
    high_power_max_amp: float = 0.1
    high_power_save_readout_amplitude: bool = False

    # Punch-out
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2  # Must be 2 for shift-based analysis
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    punch_out_sweep_left_offset_mhz: float = 4.0
    """MHz to extend the punch-out sweep to the LEFT of the bare resonator frequency,
    so that the dispersive-shifted low-power resonance is within the swept window."""
    use_adaptive_span: bool = True

    # Low-power spectroscopy
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1
    low_power_save_readout_amplitude: bool = True

    # Iteration limits
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5

    # Misc


with QualibrationGraph.build(
    "resonator_optimization",
    parameters=ResonatorBringUpParameters(),
) as graph:
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

graph.run()
