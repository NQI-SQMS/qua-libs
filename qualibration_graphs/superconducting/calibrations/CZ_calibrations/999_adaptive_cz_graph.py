"""
Adaptive two-qubit CZ-gate calibration graph (fixed couplers).

This is the 2Q analogue of `1Q_calibrations/999_adaptive_graph.py`. Same pattern,
applied per qubit pair, with the CZ-fine-calibration chain from
`101_CZ_calibration_graph.py` used as the retune subgraph.

Workflow:
    1. Run two-qubit standard RB followed by interleaved CZ RB on every active
       (or user-specified) qubit pair.
    2. The interleaved RB node marks every pair whose fitted CZ fidelity is
       below `fidelity_threshold` as `outcomes["failed"]` (handled inside
       `37b_two_qubit_interleaved_cz_rb.py` once its `fidelity_threshold`
       parameter is set).
    3. Route only the failed pairs into the `retune_low_cz_fidelity` subgraph via
       `connect_on_failure`. Successful pairs go to the no-op
       `38_cz_rb_success_exit` node (required because qualibrate-core 0.5.0
       demands every node with outgoing edges has at least one success-path
       connection).
    4. The retune subgraph runs:
           cz_conditional_phase_error_amp
             -> cz_phase_compensation
             -> cz_phase_compensation_error_amp
             -> two_qubit_standard_rb_verify
             -> two_qubit_interleaved_cz_rb_verify (drives the loop predicate)
       and is wrapped in a `loop(...)` driven by the interleaved RB's results so
       low-fidelity pairs keep retrying (up to `max_retune_iterations`) until
       fidelity meets the threshold.

Mirrors the structure of `101_CZ_calibration_graph.py`, with the leakage and
conditional-phase steps tied together as the corrective stages before the
two-step verification RB.
"""

# %%
from typing import List, Literal, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator


library = QualibrationLibrary.get_active_library()


# %% {Graph parameters}
class Parameters(GraphParameters):
    """Top-level parameters for the adaptive CZ-RB-with-retune graph."""

    targets_name = "qubit_pairs"

    qubit_pairs: Optional[List[str]] = None
    """Qubit pairs to operate on. None (default) means 'all active qubit pairs'
    (resolved by `quam.active_qubit_pairs`)."""

    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_SNZ"
    """Name of the 2Q gate macro to calibrate and benchmark. Must match a macro
    key registered on each qubit pair under `machine.qubit_pairs[<pair>].macros`.
    Propagated to every node in the graph (initial RB pair, retune subgraph, and
    verify RB pair). Default is 'cz_SNZ'."""

    fidelity_threshold: float = 0.98
    """CZ-gate-fidelity acceptance threshold (1 - error_per_CZ-gate from the
    interleaved RB). Pairs below this are routed to the retune subgraph."""

    max_retune_iterations: int = 2
    """Maximum number of retune-then-verify cycles per qubit pair before giving up."""


parameters = Parameters()


# %% {Retune subgraph parameters}
class RetuneParameters(GraphParameters):
    """Parameters for the inner retune subgraph. The `qubit_pairs` field is
    filled in automatically by the orchestrator with the targets routed from
    the parent."""

    targets_name = "qubit_pairs"

    qubit_pairs: Optional[List[str]] = None


# %% {Loop condition: should we keep retuning a given qubit pair?}
def should_keep_retuning(subgraph: QualibrationGraph, target: str) -> bool:
    """Per-target loop predicate evaluated after each retune-subgraph iteration.

    Inspects the verification interleaved-RB's results inside the subgraph and
    returns True iff this qubit pair's CZ fidelity is still below the configured
    threshold (or its fit failed). Returning False stops the loop for this pair.
    """
    # NOTE: `_elements` is the internal mapping of subgraph nodes by name. There
    # is no public accessor for nested-graph elements in qualibrate-core 0.5.0;
    # this is the same mechanism the orchestrator itself uses internally.
    verify_node = subgraph._elements["interleaved_rb_verify"]
    fit = verify_node.results.get("fit_results", {}).get(target, {})

    if not fit or not fit.get("success"):
        return True  # fit failed or no results -> retry

    fidelity = float(fit.get("fidelity", 0.0))
    return fidelity < parameters.fidelity_threshold


# %% {Build the graph}
with QualibrationGraph.build("adaptive_cz_rb_with_retune", parameters=parameters) as graph:
    # ---- 1. Initial RB on all selected qubit pairs ----
    # Two chained nodes: standard 2Q RB feeds its `StandardRB_alpha` to the
    # interleaved CZ RB through the QUAM state. Only the interleaved RB enforces
    # the fidelity threshold and sets outcomes accordingly, because it is the
    # node that produces a per-CZ-gate fidelity number.
    standard_rb_initial = library.nodes["37_two_qubit_standard_rb"].copy(
        name="standard_rb_initial",
        num_shots=200,
        use_state_discrimination=True,
        circuit_lengths=[1, 4, 16, 32, 64],
        num_circuits_per_length=5,
        seed=0,
        use_input_stream=False,
        reset_type="active",
        operation=parameters.operation,
    )
    graph.add_node(standard_rb_initial)

    interleaved_rb_initial = library.nodes["37b_two_qubit_interleaved_cz_rb"].copy(
        name="interleaved_rb_initial",
        num_shots=200,
        use_state_discrimination=True,
        circuit_lengths=[1, 4, 16, 32, 64],
        num_circuits_per_length=5,
        seed=0,
        use_input_stream=False,
        reset_type="active",
        operation=parameters.operation,
        fidelity_threshold=parameters.fidelity_threshold,
    )
    graph.add_node(interleaved_rb_initial)

    graph.connect(standard_rb_initial, interleaved_rb_initial)

    # ---- 2. Retune subgraph ----
    # Mirrors the corrective chain of `101_CZ_calibration_graph.py`:
    #   conditional_phase_error_amp
    #     -> phase_compensation
    #     -> phase_compensation_error_amp
    #     -> standard_rb_verify
    #     -> interleaved_rb_verify  (verifies CZ fidelity; drives the loop)
    # `skip_failed=False` matches the convention of the existing 99_/100_/101_
    # graphs: every node runs for every pair even if upstream nodes failed.
    with QualibrationGraph.build(
        "retune_low_cz_fidelity",
        parameters=RetuneParameters(),
        orchestrator=BasicOrchestrator(skip_failed=False),
    ) as retune_subgraph:
        conditional_phase_error_amp = library.nodes["33_cz_conditional_phase_error_amp"].copy(
            name="conditional_phase_error_amp",
            num_shots=100,
            amp_range=0.005,
            amp_step=0.0002,
            num_frame_rotations=10,
            number_of_operations=20,
            use_state_discrimination=True,
            reset_type="active",
            operation=parameters.operation,
        )
        phase_compensation = library.nodes["35_cz_phase_compensation"].copy(
            name="phase_compensation",
            num_shots=200,
            num_frames=51,
            use_state_discrimination=True,
            reset_type="active",
            operation=parameters.operation,
        )
        phase_compensation_error_amp = library.nodes["35a_cz_phase_compensation_error_amp"].copy(
            name="phase_compensation_error_amp",
            num_shots=10,
            num_frames=201,
            number_of_operations=40,
            use_state_discrimination=True,
            reset_type="active",
            operation=parameters.operation,
        )
        # Verification RB chain MUST end the subgraph: the interleaved RB's
        # `results["fit_results"]` is what `should_keep_retuning` inspects.
        standard_rb_verify = library.nodes["37_two_qubit_standard_rb"].copy(
            name="standard_rb_verify",
            num_shots=200,
            use_state_discrimination=True,
            circuit_lengths=[1, 4, 16, 32, 64],
            num_circuits_per_length=5,
            seed=0,
            use_input_stream=False,
            reset_type="active",
            operation=parameters.operation,
        )
        interleaved_rb_verify = library.nodes["37b_two_qubit_interleaved_cz_rb"].copy(
            name="interleaved_rb_verify",
            num_shots=200,
            use_state_discrimination=True,
            circuit_lengths=[1, 4, 16, 32, 64],
            num_circuits_per_length=5,
            seed=0,
            use_input_stream=False,
            reset_type="active",
            operation=parameters.operation,
            fidelity_threshold=parameters.fidelity_threshold,
        )

        retune_subgraph.add_node(conditional_phase_error_amp)
        retune_subgraph.add_node(phase_compensation)
        retune_subgraph.add_node(phase_compensation_error_amp)
        retune_subgraph.add_node(standard_rb_verify)
        retune_subgraph.add_node(interleaved_rb_verify)

        retune_subgraph.connect(conditional_phase_error_amp, phase_compensation)
        retune_subgraph.connect(phase_compensation, phase_compensation_error_amp)
        retune_subgraph.connect(phase_compensation_error_amp, standard_rb_verify)
        retune_subgraph.connect(standard_rb_verify, interleaved_rb_verify)

    graph.add_node(retune_subgraph)

    # ---- 3. Success-path sink (required by graph validation) ----
    # qualibrate-core 0.5.0 rejects any node with outgoing edges that does not
    # have at least one success-outcome connection. The sink is a no-op node:
    # high-fidelity pairs land here and just exit.
    success_exit = library.nodes["38_cz_rb_success_exit"].copy(name="cz_rb_success_exit")
    graph.add_node(success_exit)
    graph.connect(interleaved_rb_initial, success_exit)

    # ---- 4. Route ONLY low-CZ-fidelity pairs into the retune subgraph ----
    graph.connect_on_failure(interleaved_rb_initial, retune_subgraph)

    # ---- 5. Loop the retune subgraph per-pair until CZ fidelity is met ----
    graph.loop(
        retune_subgraph,
        on=should_keep_retuning,
        max_iterations=parameters.max_retune_iterations,
    )


# %% {Run}
if __name__ == "__main__":
    graph.run()

# %%
