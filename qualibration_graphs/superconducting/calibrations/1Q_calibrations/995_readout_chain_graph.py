"""
Simple readout-chain tune-up graph (flux-tunable transmons).

Scope: bring the *readout* up to a target single-shot discrimination fidelity — nothing
about the qubit gates. It demonstrates two graph constructs on a deliberately small
workflow: a reusable **sub-graph** and a **conditional loop**.

Workflow:
    resonator_spectroscopy            (find the readout resonator)
      -> resonator_spectroscopy_vs_flux   (track it vs flux -> idle point)
      -> [readout_optimization]            <-- sub-graph, looped:
             readout_power_optimization
               -> readout_frequency_optimization
               -> iq_blobs                 (single-shot fidelity + thresholds)
The `readout_optimization` sub-graph is repeated (conditional loop) until the IQ-blob
`readout_fidelity` reaches `readout_fidelity_threshold` for each qubit, or
`max_optimization_iterations` is hit.

No failure routing / success sink here — those are shown in 997_single_qubit_rb_graph.py.
See 998_adaptive_graph_tutorial.ipynb for the building blocks and 999_adaptive_graph.py
for the full adaptive RB+retune graph.
"""

# %%
from typing import List, Optional

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary, QualibrationNode
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator

library = QualibrationLibrary.get_active_library()


# %% {Graph parameters}
class Parameters(GraphParameters):
    qubits: Optional[List[str]] = None
    """Qubits to operate on. None = all active qubits."""

    readout_fidelity_threshold: float = 90.0
    """Target IQ-blob single-shot assignment fidelity, in percent. The readout
    optimization sub-graph is repeated until each qubit reaches this."""

    max_optimization_iterations: int = 2
    """Hard cap on how many times the readout-optimization sub-graph repeats."""


parameters = Parameters()


class SubgraphParameters(GraphParameters):
    """Inner sub-graph parameters; `qubits` is filled in by the orchestrator."""

    qubits: Optional[List[str]] = None


# %% {Loop condition: is this qubit's readout fidelity still below target?}
def readout_below_threshold(subgraph: QualibrationGraph, target: str) -> bool:
    """Per-qubit loop predicate: inspect the sub-graph's terminal `iq_blobs` node and
    return True (keep looping) while `readout_fidelity` is below the target."""
    iq_node = subgraph._elements["iq_blobs"]
    fit = iq_node.results.get("fit_results", {}).get(target, {})
    fidelity = fit.get("readout_fidelity") if fit else None
    if fidelity is None:
        return True  # no/failed result -> retry
    return float(fidelity) < parameters.readout_fidelity_threshold


# %% {Build the graph}
with QualibrationGraph.build("readout_chain_tuneup", parameters=parameters) as graph:
    # ---- 1. Find the readout resonator and its flux dependence ----
    resonator_spectroscopy = library.nodes["03_resonator_spectroscopy_single"].copy(
        name="resonator_spectroscopy"
    )
    graph.add_node(resonator_spectroscopy)

    resonator_spectroscopy_vs_flux = library.nodes["06_resonator_spectroscopy_vs_flux"].copy(
        name="resonator_spectroscopy_vs_flux"
    )
    graph.add_node(resonator_spectroscopy_vs_flux)
    graph.connect(resonator_spectroscopy, resonator_spectroscopy_vs_flux)

    # ---- 2. Readout-optimization sub-graph (power -> frequency -> IQ blobs) ----
    with QualibrationGraph.build(
        "readout_optimization",
        parameters=SubgraphParameters(),
        orchestrator=BasicOrchestrator(skip_failed=False),
    ) as readout_optimization:
        readout_power_optimization = library.nodes["15b_readout_power_optimization"].copy(
            name="readout_power_optimization"
        )
        readout_frequency_optimization = library.nodes["15a_readout_frequency_optimization"].copy(
            name="readout_frequency_optimization"
        )
        # `iq_blobs` MUST be the terminal node: its `readout_fidelity` drives the loop.
        iq_blobs = library.nodes["16_iq_blobs"].copy(name="iq_blobs")

        readout_optimization.add_node(readout_power_optimization)
        readout_optimization.add_node(readout_frequency_optimization)
        readout_optimization.add_node(iq_blobs)
        readout_optimization.connect(readout_power_optimization, readout_frequency_optimization)
        readout_optimization.connect(readout_frequency_optimization, iq_blobs)

    graph.add_node(readout_optimization)
    graph.connect(resonator_spectroscopy_vs_flux, readout_optimization)

    # ---- 3. Loop the optimization until the fidelity target is met ----
    graph.loop(
        readout_optimization,
        on=readout_below_threshold,
        max_iterations=parameters.max_optimization_iterations,
    )


# %% {Run}
if __name__ == "__main__":
    graph.run()

# %%
