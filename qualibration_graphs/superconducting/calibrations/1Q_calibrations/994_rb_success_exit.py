"""
No-op success-sink node for adaptive RB graphs.

Purpose:
    Used by `999_adaptive_graph.py` (and any similar adaptive graph) as the
    success-path target for the initial RB node. It exists solely to satisfy
    the `qualibrate-core` graph-validation rule that every node with outgoing
    connections must have at least one connection on a successful outcome.

Behavior:
    - Performs no QPU operations.
    - Marks every routed qubit as `outcomes["successful"]` so it doesn't
      pollute the parent graph's per-target final outcomes.
"""

# %% {Imports}
from qualibrate import NodeParameters, QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


description = """
        RB SUCCESS EXIT
A no-op terminal node used as the success-path sink for an upstream RB node in
adaptive calibration graphs. Logs the qubits that reached this node and exits.
"""


class Parameters(NodeParameters, CommonNodeParameters, QubitsExperimentNodeParameters):
    """Only the inherited `qubits` field is used; everything else stays at
    defaults. No experiment-specific parameters.

    The framework requires the parameters class to derive from `NodeParameters`
    (a qualibrate-core requirement enforced in `QualibrationNode.__init__`).
    `CommonNodeParameters` and `QubitsExperimentNodeParameters` mirror the
    convention of the other nodes in this folder (e.g. 27_single_qubit_rb).
    """

    pass


node = QualibrationNode[Parameters, Quam](
    name="28_rb_success_exit",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


# Required by the framework lifecycle even though we never touch the QPU.
# node.machine = Quam.load()


# %% {Mark routed qubits as successful}
@node.run_action()
def mark_successful(node: QualibrationNode[Parameters, Quam]):
    """Log the qubits routed here and mark them all as successful."""
    qubits = node.parameters.qubits or []
    if qubits:
        node.log(f"RB success exit: qubits {list(qubits)} passed the fidelity threshold.")
    else:
        node.log("RB success exit: no qubits routed (nothing to do).")
    node.outcomes = {q: "successful" for q in qubits}


# %% {Save}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the (empty) results so the graph run summary records this step."""
    node.save()
