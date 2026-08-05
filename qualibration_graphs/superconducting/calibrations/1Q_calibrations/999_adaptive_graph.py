"""
Adaptive single-qubit randomized-benchmarking graph (flux-tunable transmons).

Workflow:
    1. Run single-qubit RB on all selected qubits.
    2. Mark every qubit whose fitted gate fidelity is below `fidelity_threshold`
       as `outcomes["failed"]` (handled inside `27_single_qubit_randomized_benchmarking.py`
       when its `fidelity_threshold` parameter is set).
    3. Route only the failed qubits into the `retune_low_fidelity` subgraph via
       `connect_on_failure`. Successful qubits go to the no-op `28_rb_success_exit`
       node (required because qualibrate-core 0.5.0 demands every node with
       outgoing edges has at least one success-path connection).
    4. The retune subgraph runs:
           ramsey_vs_flux_calibration (re-find idle flux + qubit frequency)
             -> ramsey (refine qubit frequency at the new flux)
             -> power_rabi_error_amplification_x180 (refine x180 amplitude)
             -> power_rabi_error_amplification_x90  (refine x90 amplitude)
             -> rb_verify (verify gate fidelity)
       and is wrapped in a `loop(...)` driven by `rb_verify`'s results so that
       low-fid qubits keep retrying (up to `max_retune_iterations`) until
       fidelity meets the threshold.

Mirrors the structure of `81_calibration_graph_retuning_flux_tunable_transmon.py`,
with an additional `12_ramsey` step inserted between the flux calibration and
the error-amplification rounds.
"""

# %%
from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
    QualibrationNode,
)
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator


library = QualibrationLibrary.get_active_library()


# %% {Graph parameters}
class Parameters(GraphParameters):
    """Top-level parameters for the adaptive RB-with-retune graph."""

    qubits: Optional[List[str]] = None
    """Qubits to operate on. None (default) means 'all active qubits'
    (resolved by `quam.active_qubits`)."""

    fidelity_threshold: float = 0.9995
    """Single-qubit gate-fidelity acceptance threshold (1 - error_per_gate).
    Qubits below this are routed to the retune subgraph."""

    max_retune_iterations: int = 2
    """Maximum number of retune-then-verify cycles per qubit before giving up."""


parameters = Parameters()


# %% {Retune subgraph parameters}
class RetuneParameters(GraphParameters):
    """Parameters for the inner retune subgraph. The `qubits` field is filled in
    automatically by the orchestrator with the targets routed from the parent."""

    qubits: Optional[List[str]] = None


# %% {Loop condition: should we keep retuning a given qubit?}
def should_keep_retuning(subgraph: QualibrationGraph, target: str) -> bool:
    """Per-target loop predicate evaluated after each retune-subgraph iteration.

    Inspects the verification-RB's results inside the subgraph and returns True
    iff this qubit's fidelity is still below the configured threshold (or its
    fit failed). Returning False stops the loop for this qubit.
    """
    # NOTE: `_elements` is the internal mapping of subgraph nodes by name. There
    # is no public accessor for nested-graph elements in qualibrate 1.2.x; this
    # is the same mechanism the orchestrator uses internally.
    rb_verify_node = subgraph._elements["rb_verify"]
    fit = rb_verify_node.results.get("fit_results", {}).get(target, {})

    if not fit or not fit.get("success"):
        return True  # fit failed or no results -> retry

    fidelity = 1.0 - float(fit.get("error_per_gate", 1.0))
    return fidelity < parameters.fidelity_threshold


# %% {Build the graph}
with QualibrationGraph.build("adaptive_rb_with_retune", parameters=parameters) as graph:
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

    # ---- 2. Retune subgraph ----
    # Mirrors the structure of `81_calibration_graph_retuning_flux_tunable_transmon.py`
    # (lines 20-40), with an extra `12_ramsey` node inserted between the flux
    # calibration and the error-amplification rounds:
    #   ramsey_vs_flux_calibration
    #     -> ramsey                  (refine qubit frequency at the new flux)
    #     -> err_amp_x180            (refine x180 amplitude)
    #     -> err_amp_x90             (refine x90 amplitude)
    #     -> rb_verify               (verify gate fidelity; drives the loop)
    # `skip_failed=False` matches the convention of the existing 80_/81_/90_/91_
    # graphs: every node runs for every qubit even if upstream nodes failed.
    with QualibrationGraph.build(
        "retune_low_fidelity",
        parameters=RetuneParameters(),
        orchestrator=BasicOrchestrator(skip_failed=False),
    ) as retune_subgraph:
        ramsey_vs_flux_calibration = library.nodes["23_ramsey_vs_flux_calibration"].copy(
            name="ramsey_vs_flux_calibration",
            num_shots=100,
            frequency_detuning_in_mhz=5.0,
            min_wait_time_in_ns=500,
            max_wait_time_in_ns=1000,
            wait_time_step_in_ns=4,
            flux_span=0.1,
            flux_num=51,
        )
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
        # Verification RB MUST be the terminal node of the subgraph: its
        # `results["fit_results"]` is what `should_keep_retuning` inspects.
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

        retune_subgraph.add_node(ramsey_vs_flux_calibration)
        retune_subgraph.add_node(ramsey)
        retune_subgraph.add_node(erramp_x180)
        retune_subgraph.add_node(erramp_x90)
        retune_subgraph.add_node(rb_verify)

        retune_subgraph.connect(ramsey_vs_flux_calibration, ramsey)
        retune_subgraph.connect(ramsey, erramp_x180)
        retune_subgraph.connect(erramp_x180, erramp_x90)
        retune_subgraph.connect(erramp_x90, rb_verify)

    graph.add_node(retune_subgraph)

    # ---- 3. Success-path sink (required by graph validation) ----
    # qualibrate-core 0.5.0 rejects any node with outgoing edges that does not
    # have at least one success-outcome connection. The sink is a no-op node:
    # high-fidelity qubits land here and just exit.
    success_exit = library.nodes["28_rb_success_exit"].copy(name="rb_success_exit")
    graph.add_node(success_exit)
    graph.connect(rb_initial, success_exit)

    # ---- 4. Route ONLY low-fidelity qubits into the retune subgraph ----
    graph.connect_on_failure(rb_initial, retune_subgraph)

    # ---- 5. Loop the retune subgraph per-qubit until fidelity is met ----
    graph.loop(
        retune_subgraph,
        on=should_keep_retuning,
        max_iterations=parameters.max_retune_iterations,
    )


# %% {Run}
if __name__ == "__main__":
    graph.run()

# %%
