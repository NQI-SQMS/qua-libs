# %%
"""
Sideband Bring-Up Graph

Calibrates a single f|k>g|k+1> sideband transition selected by ``sideband_level``.
Setting ``sideband_level = N`` calibrates the f|N-1>g|N> transition (1-based):

  sideband_level = 1  →  f0g1
  sideband_level = 2  →  f1g2
  sideband_level = 3  →  f2g3
  …

The calibration chain for the selected level is:

  f{k}g{k+1}_spectroscopy
  -> f{k}g{k+1}_time_rabi
  -> f{k}g{k+1}_ramsey          (sideband frequency fine-tuning)
  -> f{k}g{k+1}_ge_spectroscopy (qubit GE frequency shift in Fock |k>)
  -> f{k}g{k+1}_ge_ramsey       (precise qubit GE frequency at Fock |k>)
  -> f{k}g{k+1}_ef_spectroscopy (qubit EF frequency shift in Fock |k>)
  -> f{k}g{k+1}_ef_ramsey       (Kerr-corrected EF frequency)

For levels k > 0, lower-level transitions (f0g1 … f{k-1}g{k}) must already
be calibrated in the machine state so that the Fock-state preparation works.

Note: ``sideband_level`` controls the graph structure at build time.
Changing it after the library is loaded has no effect until the library
is reloaded (i.e., this file is re-executed).
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import build_sideband_bringup

library = QualibrationLibrary.get_active_library()


class SidebandBringUpParameters(GraphParameters):
    """Graph-flow parameters for the sideband bring-up graph."""

    qubits: List[str] = ["q1"]
    mode_name: str = "alice"
    """Cavity mode to calibrate: 'alice' or 'bob'."""
    sideband_level: int = 1
    """Sideband transition to calibrate (1-based): 1 → f0g1, 2 → f1g2, …
    Changing this requires reloading the library."""


with QualibrationGraph.build(
    "sideband_bringup_graph",
    parameters=SidebandBringUpParameters(),
) as graph:
    sideband_bringup = build_sideband_bringup(graph, library)
    graph.add_node(sideband_bringup)


graph.run()
