# %%
"""
EF-Transition Bring-Up Graph

Calibrates the EF (|e>→|f>) transition of a fixed-frequency transmon:

  ef_discovery [loop: retry if no EF oscillation]:
    ef_spectroscopy  [loop: retry on no peak]
    → ef_tentative_rabi       (checks oscillation; blacklists EF freq on failure)
  → ef_rabi_ramsey [loop: until |EF detuning| converges]:
      ef_power_rabi [inner loop: amplitude convergence]
      → ef_ramsey
  → ef_T1
  → gef_readout_frequency_optimization
  → gef_iq_blobs
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import build_ef_bringup

library = QualibrationLibrary.get_active_library()


class EFBringUpParameters(GraphParameters):
    """Graph-flow parameters for the EF-transition bring-up graph."""

    qubits: List[str] = ["q1"]
    # Iteration limits
    max_ef_discovery_iterations: int = 3
    max_ef_spec_iterations: int = 3
    ef_max_iterations: int = 5
    ef_rabi_max_amplitude_iterations: int = 5
    # Convergence threshold
    ef_freq_threshold_hz: float = 50_000.0
    """Stop the ef_power_rabi → ef_ramsey loop when |EF detuning| < this value [Hz]."""


with QualibrationGraph.build(
    "ef_bringup_graph",
    parameters=EFBringUpParameters(),
) as graph:
    ef_bringup = build_ef_bringup(graph, library)
    graph.add_node(ef_bringup)


graph.run()
