# %% {X180 Fine Calibration}
"""
X180 Fine Calibration Graph

Iteratively refines the x180 pulse amplitude and qubit frequency using:
  1. Power Rabi  → recalibrate x180 amplitude around its current value
  2. Ramsey      → correct the qubit frequency (f_01 / RF_frequency)

The loop repeats until the Ramsey detuning falls below ``freq_threshold_hz``
for every qubit, or until ``max_iterations`` is reached.

Failure handling:
  - If either fit fails for any qubit, the loop stops immediately and the
    pre-graph x180 amplitude and qubit frequency are restored from
    ``graph.parameters.initial_x180_amplitude`` / ``initial_qubit_f01``.

Notes:
  - The initial state is captured on the FIRST call to the condition function
    (i.e. after the first Rabi+Ramsey pair runs).  If that first run succeeds,
    "initial" corresponds to the post-first-iteration state; if it fails,
    update_state was not applied so the machine is still at the pre-graph state.
  - To enable the restore-on-failure path for a standalone run, set
    ``graph.parameters.initial_x180_amplitude`` and ``initial_qubit_f01``
    directly before calling ``graph.run()``.
"""

import logging
from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
    QualibrationNode,
)

library = QualibrationLibrary.get_active_library()
logger = logging.getLogger(__name__)


# ─── Parameters ──────────────────────────────────────────────────────────────
test_qubits = ["q4"]

class X180FineCalibrationParameters(GraphParameters):
    """Parameters for the iterative x180 fine-calibration graph."""

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # Optional bringup snapshot for rollback-on-failure.
    # Set via resolve_params when chained after qubit_calibration, or manually
    # before running the graph standalone to enable the restore-on-failure path.
    initial_x180_amplitude: Optional[float] = None
    initial_qubit_f01: Optional[float] = None
    initial_rf_frequency: Optional[float] = None

    # Power Rabi
    rabi_min_amp_factor: float = 0.001
    rabi_max_amp_factor: float = 1.99
    rabi_amp_factor_step: float = 0.005
    rabi_num_shots: int = 50
    rabi_max_number_pulses_per_sweep: int = 1

    # Ramsey
    ramsey_num_shots: int = 100
    ramsey_frequency_detuning_in_mhz: float = 1.0
    ramsey_max_wait_time_in_ns: int = 10_000
    ramsey_wait_time_num_points: int = 200
    ramsey_log_or_linear_sweep: str = "linear"

    # Convergence
    freq_threshold_hz: float = 50_000.0
    """Stop when |Ramsey detuning| < this value (Hz) for ALL qubits."""
    max_iterations: int = 10


class _RabiRamseySubgraphParameters(GraphParameters):
    """Minimal parameters for the inner Rabi→Ramsey subgraph."""

    qubits: List[str] = test_qubits


# ─── Module-level loop state ──────────────────────────────────────────────────
# Persists across condition-function calls within a single graph run.
# Reset per-qubit when that qubit converges or fails.

_loop_state: dict = {
    "initialized": {},          # qubit_name -> bool : True once initial state is stored
    "any_failed": False,        # True once any qubit signals a fit failure this run
    "initial_x180_amplitude": {},   # qubit_name -> float
    "initial_x90_amplitude": {},    # qubit_name -> float
    "initial_f01": {},              # qubit_name -> float
    "initial_rf_frequency": {},     # qubit_name -> float
    "detuning_history": {},         # qubit_name -> list[float] (|freq_offset| in Hz)
    "threshold_hz": 50_000.0,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_machine(node):
    """Return the machine from a node or subgraph.

    When looping over a subgraph (QualibrationGraph), the condition function
    receives the subgraph itself rather than a QualibrationNode.  Subgraphs
    store child elements in ``_elements`` (not ``nodes``), and the machine
    lives on the individual child nodes after they have run.
    """
    if hasattr(node, "machine") and node.machine is not None:
        return node.machine
    # node is a subgraph – iterate its _elements dict
    for child in node._elements.values():
        if hasattr(child, "machine") and child.machine is not None:
            return child.machine
        # child may itself be a nested subgraph; recurse one level
        grandchild_machine = _get_machine(child) if hasattr(child, "_elements") else None
        if grandchild_machine is not None:
            return grandchild_machine
    return None


def _restore_initial_state(machine, target: str) -> None:
    """Restore x180/x90 amplitude and f_01/RF_frequency to their pre-loop values.

    After restoring in-memory values, ``machine.save()`` is called to persist
    the restored state to the JSON file.  This is necessary because the Ramsey
    node's ``update_state`` already called ``machine.save()`` with the
    Ramsey-modified frequencies; without an explicit save here those would
    remain in the persistent state even though the in-memory values are correct.
    """
    if target not in _loop_state["initial_x180_amplitude"]:
        return  # Nothing stored yet for this qubit; nothing to restore
    q = machine.qubits[target]
    q.xy.operations["x180"].amplitude = _loop_state["initial_x180_amplitude"][target]
    q.xy.operations["x90"].amplitude = _loop_state["initial_x90_amplitude"][target]
    q.f_01 = _loop_state["initial_f01"][target]
    q.xy.RF_frequency = _loop_state["initial_rf_frequency"][target]
    logger.info(
        f"[X180 fine] {target}: Restored x180={1e3 * _loop_state['initial_x180_amplitude'][target]:.2f} mV, "
        f"f_01={_loop_state['initial_f01'][target] / 1e9:.6f} GHz."
    )
    # Persist the restored state so the JSON file reflects the rollback.
    # The Ramsey node already called machine.save() with modified frequencies;
    # without this call those would survive in state.json.
    try:
        machine.save()
    except Exception as exc:
        logger.warning(f"[X180 fine] {target}: machine.save() after restore failed: {exc}")




# ─── Condition function ───────────────────────────────────────────────────────


def should_repeat_x180_calibration(node: QualibrationNode, target: str) -> bool:
    """
    Called after each Ramsey→Rabi subgraph iteration for qubit *target*.

    Returns True  → repeat the subgraph for this qubit.
    Returns False → stop looping for this qubit (converged or failed).

    The loop framework continues overall until ALL qubits return False.
    """
    # node is the ramsey_rabi subgraph (QualibrationGraph), not a bare node,
    # so we must get the machine through a helper rather than node.machine.
    machine = _get_machine(node)

    # ── Detect start of a fresh graph run ────────────────────────────────────
    # When no qubit is "initialized" we are at the very beginning of a new run.
    if not any(_loop_state["initialized"].values()):
        _loop_state["any_failed"] = False
        _loop_state["threshold_hz"] = graph.parameters.freq_threshold_hz

    # ── Propagate "any_failed" to all remaining qubits ────────────────────────
    if _loop_state["any_failed"]:
        _restore_initial_state(machine, target)
        _loop_state["initialized"][target] = False
        return False

    q = machine.qubits[target]

    # ── Store initial state on first call for this qubit ─────────────────────
    # Read the snapshot from graph.parameters (set via resolve_params when
    # chained after qubit_calibration, or set manually for standalone runs).
    if not _loop_state["initialized"].get(target, False):
        x180_amp = graph.parameters.initial_x180_amplitude

        if x180_amp is not None:
            _loop_state["initial_x180_amplitude"][target] = x180_amp
            _loop_state["initial_x90_amplitude"][target] = x180_amp / 2
            _loop_state["initial_f01"][target] = graph.parameters.initial_qubit_f01 or float(q.f_01)
            _loop_state["initial_rf_frequency"][target] = (
                graph.parameters.initial_rf_frequency or float(q.xy.RF_frequency)
            )
            logger.info(
                f"[X180 fine] {target}: Bringup snapshot loaded – "
                f"x180={1e3 * x180_amp:.2f} mV, "
                f"f_01={_loop_state['initial_f01'][target] / 1e9:.6f} GHz "
                f"(will restore if power_rabi fails)."
            )
        else:
            # No snapshot available; restore-on-failure is disabled.
            for key in ("initial_x180_amplitude", "initial_x90_amplitude",
                        "initial_f01", "initial_rf_frequency"):
                _loop_state[key].pop(target, None)
            logger.info(f"[X180 fine] {target}: No bringup snapshot; restore-on-failure disabled.")
        _loop_state["detuning_history"][target] = []
        _loop_state["initialized"][target] = True

    # ── Check for fit failure ─────────────────────────────────────────────────
    if node.outcomes.get(target) == "failed":
        _power_rabi_node = node._elements.get("power_rabi")
        power_rabi_failed = (
            _power_rabi_node is not None
            and _power_rabi_node.outcomes.get(target) == "failed"
        )
        if power_rabi_failed and target in _loop_state["initial_x180_amplitude"]:
            logger.warning(
                f"[X180 fine] {target}: Power Rabi failed. "
                "Restoring bringup state and stopping loop."
            )
            _restore_initial_state(machine, target)
        else:
            logger.warning(
                f"[X180 fine] {target}: Fit failed "
                f"({'Ramsey' if not power_rabi_failed else 'Rabi, no backup'}). "
                "Stopping loop without restore."
            )
        _loop_state["initialized"][target] = False
        _loop_state["any_failed"] = True
        return False

    # ── Extract Ramsey detuning ────────────────────────────────────────────────
    # Read freq_offset directly from the Ramsey child node's results.
    # Results live on the child QualibrationNode, not the subgraph itself.
    _ramsey_node = node._elements.get("ramsey")
    _ramsey_results = _ramsey_node.results if _ramsey_node is not None else {}
    freq_offset = (
        _ramsey_results.get("fit_results", {})
        .get(target, {})
        .get("freq_offset", None)
    )

    if freq_offset is None:
        # Fallback: estimate detuning from change in f_01 since last iteration
        history = _loop_state["detuning_history"][target]
        last_f01 = (
            _loop_state["initial_f01"][target] - sum(history)
            if history
            else _loop_state["initial_f01"][target]
        )
        freq_offset = last_f01 - float(q.f_01)
        logger.debug(
            f"[X180 fine] {target}: freq_offset not in fit_results; "
            f"estimated from f_01 change: {freq_offset / 1e3:.2f} kHz."
        )

    abs_offset = abs(freq_offset)
    _loop_state["detuning_history"][target].append(abs_offset)

    f01 = float(q.f_01) or 1.0
    pct = abs_offset / f01 * 100.0
    logger.info(
        f"[X180 fine] {target}: |detuning| = {abs_offset / 1e3:.2f} kHz "
        f"({pct:.4f}%),  threshold = {graph.parameters.freq_threshold_hz / 1e3:.0f} kHz."
    )

    if abs_offset < graph.parameters.freq_threshold_hz:
        logger.info(f"[X180 fine] {target}: Converged after {len(_loop_state['detuning_history'][target])} iteration(s).")
        _loop_state["initialized"][target] = False
        return False  # Converged

    return True  # Continue


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "x180_fine_calibration",
    parameters=X180FineCalibrationParameters(),
) as graph:

    # Inner subgraph: Ramsey → Power Rabi (run as a unit each iteration)
    with QualibrationGraph.build(
        "ramsey_rabi",
        parameters=_RabiRamseySubgraphParameters(),
    ) as ramsey_rabi:

        # 1. Ramsey – measure and correct qubit frequency
        ramsey = library.nodes["06a_ramsey"].copy(
            name="ramsey",
            multiplexed=graph.parameters.multiplexed,
            num_shots=graph.parameters.ramsey_num_shots,
            frequency_detuning_in_mhz=graph.parameters.ramsey_frequency_detuning_in_mhz,
            max_wait_time_in_ns=graph.parameters.ramsey_max_wait_time_in_ns,
            wait_time_num_points=graph.parameters.ramsey_wait_time_num_points,
            log_or_linear_sweep=graph.parameters.ramsey_log_or_linear_sweep,
        )
        ramsey_rabi.add_node(ramsey)

        # 2. Power Rabi – recalibrate x180 pulse amplitude near its current value
        power_rabi = library.nodes["04b_power_rabi"].copy(
            name="power_rabi",
            multiplexed=graph.parameters.multiplexed,
            min_amp_factor=graph.parameters.rabi_min_amp_factor,
            max_amp_factor=graph.parameters.rabi_max_amp_factor,
            amp_factor_step=graph.parameters.rabi_amp_factor_step,
            num_shots=graph.parameters.rabi_num_shots,
            max_number_pulses_per_sweep=graph.parameters.rabi_max_number_pulses_per_sweep,
        )
        ramsey_rabi.add_node(power_rabi)

        # Ramsey must complete before Power Rabi runs
        ramsey_rabi.connect(ramsey, power_rabi)

    graph.add_node(ramsey_rabi)

    # Loop the Ramsey→Rabi pair until convergence or failure
    graph.loop(
        ramsey_rabi,
        on=should_repeat_x180_calibration,
        max_iterations=graph.parameters.max_iterations,
    )


graph.run()