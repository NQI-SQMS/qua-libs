# %%
"""
Sideband Re-Tuning Graph

Quick re-tuning of a single f|k>g|k+1> sideband transition selected by
``sideband_level``.  Setting ``sideband_level = N`` re-tunes the f|N-1>g|N>
transition (1-based):

  sideband_level = 1  →  f0g1
  sideband_level = 2  →  f1g2
  sideband_level = 3  →  f2g3
  …

The calibration chain for the selected level is:

  fNgN1_spectroscopy
  -> fNgN1_time_rabi
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class SidebandRetuningParameters(GraphParameters):
    """Graph-flow parameters for the sideband re-tuning graph."""

    qubits: List[str] = ["q1"]
    mode_name: str = "alice"
    """Cavity mode to re-tune: 'alice' or 'bob'."""
    sideband_level: int = 1
    """Sideband transition to re-tune (1-based): 1 → f0g1, 2 → f1g2, …
    Changing this requires reloading the library."""


with QualibrationGraph.build(
    "sideband_retuning_graph",
    parameters=SidebandRetuningParameters(),
) as graph:

    k = graph.parameters.sideband_level - 1
    qubits = graph.parameters.qubits
    mode = graph.parameters.mode_name

    n_spec = library.nodes["26_fNgN1_spectroscopy"].copy(
        name="fNgN1_spectroscopy",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        frequency_span_in_mhz=10.0,
        frequency_step_in_mhz=0.05,
        operation_amplitude_factor=1.0,
        operation_len_in_ns=20_000,
        num_shots=500,
    )
    n_rabi = library.nodes["26b_fNgN1_time_rabi"].copy(
        name="fNgN1_time_rabi",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        min_duration_ns=16,
        max_duration_ns=20_000,
        duration_step_ns=4,
        num_shots=100,
        cavity_thermalization_time_ns=200_000,
    )
    for node in [n_spec, n_rabi]:
        graph.add_node(node)

    graph.connect(n_spec, n_rabi)


graph.run()
