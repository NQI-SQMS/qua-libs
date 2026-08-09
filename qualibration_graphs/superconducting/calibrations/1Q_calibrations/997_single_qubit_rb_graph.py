"""
Single-qubit RB-with-retune graph (flux-tunable transmons) — a LIGHTER sibling of
999_adaptive_graph.py.

Same adaptive pattern as 999 (RB -> route failures into a retune sub-graph -> loop until
the fidelity target is met), but the retune is intentionally lighter: it refines the
qubit *frequency* and the *pulse amplitudes* only, WITHOUT the heavy idle-flux
re-calibration (`23_ramsey_vs_flux_calibration`) that 999 runs. Use this when the flux
operating point is already trustworthy and only the gate needs a touch-up.

Workflow:
    rb_initial (with fidelity_threshold)
      |-- success --> rb_success_exit                  (no-op sink)
      `-- failure --> [retune_low_fidelity] (looped):
              ramsey                                   (refine frequency)
                -> power_rabi_error_amplification_x180 (refine x180 amplitude)
                -> power_rabi_error_amplification_x90  (refine x90 amplitude)
                -> rb_verify                           (re-benchmark; drives the loop)
The retune sub-graph repeats until each qubit's verified gate fidelity reaches
`fidelity_threshold`, or `max_retune_iterations` is hit.

This graph shows the full toolkit: sub-graph + conditional loop + `connect_on_failure`
+ the success-path sink. See 998_adaptive_graph_tutorial.ipynb for the building blocks.
For the heavier version that also re-finds the idle flux, see 999_adaptive_graph.py.
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

    fidelity_threshold: float = 0.9995
    """Single-qubit gate-fidelity acceptance threshold (1 - error_per_gate)."""

    max_retune_iterations: int = 2
    """Maximum retune-then-verify cycles per qubit before giving up."""


parameters = Parameters()


class RetuneParameters(GraphParameters):
    """Inner retune sub-graph parameters; `qubits` is filled in by the orchestrator
    with the failing targets routed from the parent."""

    qubits: Optional[List[str]] = None


# %% {Loop condition: should we keep retuning a given qubit?}
def should_keep_retuning(subgraph: QualibrationGraph, target: str) -> bool:
    """True while this qubit's verified fidelity is still below threshold (or no fit)."""
    rb_verify_node = subgraph._elements["rb_verify"]
    fit = rb_verify_node.results.get("fit_results", {}).get(target, {})
    if not fit or not fit.get("success"):
        return True
    fidelity = 1.0 - float(fit.get("error_per_gate", 1.0))
    return fidelity < parameters.fidelity_threshold


# %% {Build the graph}
with QualibrationGraph.build("single_qubit_rb_with_retune", parameters=parameters) as graph:
    # ---- 1. Initial RB on all selected qubits ----
    rb_initial = library.nodes["27_single_qubit_randomized_benchmarking"].copy(
        name="rb_initial",
        use_state_discrimination=True,
        num_random_sequences=100,
        max_circuit_depth=1024,
        delta_clifford=100,
        num_shots=20,
        log_scale=True,
        fidelity_threshold=parameters.fidelity_threshold,
    )
    graph.add_node(rb_initial)

    # ---- 2. Light retune sub-graph: frequency + amplitudes, then verify ----
    #     (No `23_ramsey_vs_flux_calibration` here — that is the 999 heavy retune.)
    with QualibrationGraph.build(
        "retune_low_fidelity",
        parameters=RetuneParameters(),
        orchestrator=BasicOrchestrator(skip_failed=False),
    ) as retune_subgraph:
        ramsey = library.nodes["12_ramsey"].copy(
            name="ramsey",
            num_shots=100,
            frequency_detuning_in_mhz=0.1,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=100_000,
            wait_time_num_points=200,
            use_state_discrimination=True,
            log_or_linear_sweep="linear",
        )
        erramp_x180 = library.nodes["11_power_rabi"].copy(
            name="power_rabi_error_amplification_x180",
            max_number_pulses_per_sweep=200,
            min_amp_factor=0.985,
            max_amp_factor=1.015,
            amp_factor_step=0.001,
            use_state_discrimination=True,
            num_shots=10,
        )
        erramp_x90 = library.nodes["11_power_rabi"].copy(
            name="power_rabi_error_amplification_x90",
            max_number_pulses_per_sweep=200,
            min_amp_factor=0.985,
            max_amp_factor=1.015,
            amp_factor_step=0.001,
            operation="x90",
            update_x90=False,
            use_state_discrimination=True,
            num_shots=10,
        )
        # The verification RB MUST be the terminal node: its fit drives the loop.
        rb_verify = library.nodes["27_single_qubit_randomized_benchmarking"].copy(
            name="rb_verify",
            use_state_discrimination=True,
            num_random_sequences=30,
            max_circuit_depth=1024,
            delta_clifford=100,
            num_shots=20,
            log_scale=True,
            fidelity_threshold=parameters.fidelity_threshold,
        )

        retune_subgraph.add_node(ramsey)
        retune_subgraph.add_node(erramp_x180)
        retune_subgraph.add_node(erramp_x90)
        retune_subgraph.add_node(rb_verify)
        retune_subgraph.connect(ramsey, erramp_x180)
        retune_subgraph.connect(erramp_x180, erramp_x90)
        retune_subgraph.connect(erramp_x90, rb_verify)

    graph.add_node(retune_subgraph)

    # ---- 3. Success-path sink (required: every node with outgoing edges needs one) ----
    success_exit = library.nodes["28_rb_success_exit"].copy(name="rb_success_exit")
    graph.add_node(success_exit)
    graph.connect(rb_initial, success_exit)

    # ---- 4. Route ONLY low-fidelity qubits into the retune sub-graph ----
    graph.connect_on_failure(rb_initial, retune_subgraph)

    # ---- 5. Loop the retune until fidelity is met (capped) ----
    graph.loop(
        retune_subgraph,
        on=should_keep_retuning,
        max_iterations=parameters.max_retune_iterations,
    )


# %% {Run}
if __name__ == "__main__":
    graph.run()

# %%
