# %%
"""
Cavity Mode Bring-Up Graph

Calibrates a single cavity mode (alice or bob):

  cavity_mode_spectroscopy
  → displacement_calibration     (vacuum state; finds 1-photon amplitude)
  → cavity_T1                    (coherent T1)
  → parity_time_measurement      (optimal parity mapping time)
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import build_cavity_bringup

library = QualibrationLibrary.get_active_library()


class CavityBringUpParameters(GraphParameters):
    """Graph-flow parameters for the cavity mode bring-up graph."""

    qubits: List[str] = ["q1"]
    cavity_mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""


with QualibrationGraph.build(
    "cavity_bringup_graph",
    parameters=CavityBringUpParameters(),
) as graph:
    cavity_bringup = build_cavity_bringup(graph, library)
    graph.add_node(cavity_bringup)


graph.run()
