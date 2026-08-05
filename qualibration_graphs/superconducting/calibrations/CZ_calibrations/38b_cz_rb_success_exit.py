"""
No-op success-sink node for adaptive 2Q-RB graphs.

Purpose:
    Used by `999_adaptive_cz_graph.py` (and any similar adaptive CZ graph) as the
    success-path target for the initial interleaved-RB node. It exists solely to
    satisfy the `qualibrate-core` graph-validation rule that every node with
    outgoing connections must have at least one connection on a successful outcome.

Behavior:
    - Performs no QPU operations.
    - Marks every routed qubit pair as `outcomes["successful"]` so it doesn't
      pollute the parent graph's per-target final outcomes.

This is the qubit-pair counterpart of `28_rb_success_exit.py` from
`1Q_calibrations/`.
"""

# %% {Imports}
from qualibrate import NodeParameters, QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


description = """
        CZ RB SUCCESS EXIT
A no-op terminal node used as the success-path sink for an upstream interleaved
2Q-RB node in adaptive CZ-calibration graphs. Logs the qubit pairs that reached
this node and exits.
"""


class Parameters(NodeParameters, CommonNodeParameters, QubitPairExperimentNodeParameters):
    """Only the inherited `qubit_pairs` field is used; everything else stays at
    defaults. No experiment-specific parameters.

    The framework requires the parameters class to derive from `NodeParameters`
    (a qualibrate-core requirement enforced in `QualibrationNode.__init__`).
    `CommonNodeParameters` and `QubitPairExperimentNodeParameters` mirror the
    convention of the other 2Q nodes in this folder (e.g. 37_two_qubit_standard_rb).
    """

    targets_name = "qubit_pairs"


node = QualibrationNode[Parameters, Quam](
    name="38_cz_rb_success_exit",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


# node.machine = Quam.load()


# %% {Mark routed qubit pairs as successful}
@node.run_action()
def mark_successful(node: QualibrationNode[Parameters, Quam]):
    """Log the qubit pairs routed here and mark them all as successful."""
    qubit_pairs = node.parameters.qubit_pairs or []
    if qubit_pairs:
        node.log(
            f"CZ RB success exit: qubit pairs {list(qubit_pairs)} passed the "
            f"CZ-fidelity threshold."
        )
    else:
        node.log("CZ RB success exit: no qubit pairs routed (nothing to do).")
    node.outcomes = {qp: "successful" for qp in qubit_pairs}


# %% {Save}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the (empty) results so the graph run summary records this step."""
    node.save()
