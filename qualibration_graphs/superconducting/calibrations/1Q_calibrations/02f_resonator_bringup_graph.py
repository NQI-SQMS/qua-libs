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
from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import build_resonator_bringup

library = QualibrationLibrary.get_active_library()


class ResonatorBringUpParameters(GraphParameters):
    """Graph-flow parameters for the resonator bring-up graph."""
    qubits: List[str] = ["q5"]
    multiplexed: bool = False
    use_adaptive_span: bool = True
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5


with QualibrationGraph.build(
    "resonator_optimization",
    parameters=ResonatorBringUpParameters(),
) as graph:
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

graph.run()
