# %% {GE Retuning Graph}
"""
GE Retuning Graph (96)

Iteratively refines the x180 pulse amplitude and GE qubit frequency, then
measures GE IQ blobs for state discrimination:

  x180_refinement [loop until |detuning| < freq_threshold_hz, max_iterations]:
    ramsey     → measure and correct qubit GE frequency
    power_rabi → recalibrate x180 amplitude
  → selective_rabi
  → IQ_blobs (GE)

Ramsey parameters are computed from the current T2* stored in the QUAM state:
  max_wait_time = 5 × T2*
  detuning      = 10 / (5 × T2*)   [10 full oscillation periods in the sweep]
  min_wait_time = 16 ns
"""

import logging
from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
    QualibrationNode,
)

library = QualibrationLibrary.get_active_library()
logger = logging.getLogger(__name__)

# ── Compute Ramsey defaults from QUAM T2* ─────────────────────────────────────
_DEFAULT_QUBITS = ["q1"]
_T2_GE_NS: int = 10_000  # fallback: 10 µs

try:
    from quam_config import Quam as _Quam

    _m = _Quam.load()
    _t2_vals = [
        int(_m.qubits[q].T2ramsey * 1e9)
        for q in _DEFAULT_QUBITS
        if q in _m.qubits and _m.qubits[q].T2ramsey
    ]
    if _t2_vals:
        _T2_GE_NS = min(_t2_vals)
    del _m, _t2_vals
except Exception as _e:
    logger.warning(
        f"[GE retuning] Could not load T2ramsey from QUAM ({_e}). "
        f"Using fallback {_T2_GE_NS} ns."
    )

_GE_MAX_WAIT_NS: int = 5 * _T2_GE_NS
_GE_DETUNING_MHZ: float = round(10.0 / (_GE_MAX_WAIT_NS * 1e-3), 6)  # 10 periods over 5×T2*


# ── Parameters ────────────────────────────────────────────────────────────────


class GERetuningParameters(GraphParameters):
    """Graph-flow parameters for the GE retuning graph."""

    qubits: List[str] = _DEFAULT_QUBITS
    freq_threshold_hz: float = 50_000.0
    max_iterations: int = 5


class _SubgraphParameters(GraphParameters):
    qubits: List[str] = _DEFAULT_QUBITS


# ── Loop state ────────────────────────────────────────────────────────────────

_loop_state: dict = {
    "initialized": {},
    "any_failed": False,
    "detuning_history": {},
}


def _get_machine(node):
    if hasattr(node, "machine") and node.machine is not None:
        return node.machine
    for child in node._elements.values():
        if hasattr(child, "machine") and child.machine is not None:
            return child.machine
        grandchild = _get_machine(child) if hasattr(child, "_elements") else None
        if grandchild is not None:
            return grandchild
    return None


# ── Condition function ────────────────────────────────────────────────────────


def should_repeat_ge_calibration(node: QualibrationNode, target: str) -> bool:
    """Repeat ramsey → power_rabi until |detuning| < freq_threshold_hz or failure."""
    if not any(_loop_state["initialized"].values()):
        _loop_state["any_failed"] = False

    if _loop_state["any_failed"]:
        _loop_state["initialized"][target] = False
        return False

    if not _loop_state["initialized"].get(target, False):
        _loop_state["detuning_history"][target] = []
        _loop_state["initialized"][target] = True

    if node.outcomes.get(target) == "failed":
        logger.warning(f"[GE retuning] {target}: Fit failed — stopping loop.")
        _loop_state["initialized"][target] = False
        _loop_state["any_failed"] = True
        return False

    _ramsey_node = node._elements.get("ramsey")
    freq_offset = (
        (_ramsey_node.results if _ramsey_node else {})
        .get("fit_results", {})
        .get(target, {})
        .get("freq_offset", None)
    )

    if freq_offset is None:
        _loop_state["initialized"][target] = False
        return False

    abs_offset = abs(freq_offset)
    _loop_state["detuning_history"][target].append(abs_offset)
    logger.info(
        f"[GE retuning] {target}: |detuning| = {abs_offset / 1e3:.2f} kHz, "
        f"threshold = {ge_graph.parameters.freq_threshold_hz / 1e3:.0f} kHz."
    )

    if abs_offset < ge_graph.parameters.freq_threshold_hz:
        logger.info(
            f"[GE retuning] {target}: Converged after "
            f"{len(_loop_state['detuning_history'][target])} iteration(s)."
        )
        _loop_state["initialized"][target] = False
        return False

    return True


# ── Graph construction ────────────────────────────────────────────────────────

with QualibrationGraph.build(
    "ge_retuning_graph",
    parameters=GERetuningParameters(),
) as ge_graph:

    # Inner subgraph: ramsey → power_rabi (looped until convergence)
    with QualibrationGraph.build(
        "x180_refinement",
        parameters=_SubgraphParameters(),
    ) as x180_refinement:

        ramsey = library.nodes["06a_ramsey"].copy(
            name="ramsey",
            num_shots=200,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=_GE_MAX_WAIT_NS,
            frequency_detuning_in_mhz=_GE_DETUNING_MHZ,
            wait_time_num_points=100,
            log_or_linear_sweep="linear",
            use_state_discrimination=True,
        )
        x180_refinement.add_node(ramsey)

        power_rabi = library.nodes["04b_power_rabi"].copy(
            name="power_rabi",
            min_amp_factor=0.001,
            max_amp_factor=1.9,
            amp_factor_step=0.01,
            num_shots=200,
        )
        x180_refinement.add_node(power_rabi)
        x180_refinement.connect(ramsey, power_rabi)

    ge_graph.add_node(x180_refinement)
    ge_graph.loop(
        x180_refinement,
        on=should_repeat_ge_calibration,
        max_iterations=ge_graph.parameters.max_iterations,
    )

    selective_rabi = library.nodes["04b_power_rabi"].copy(
        name="selective_rabi",
        operation="selective_x180",
        min_amp_factor=0.001,
        max_amp_factor=1.99,
        amp_factor_step=0.01,
        num_shots=200,
    )
    ge_graph.add_node(selective_rabi)
    ge_graph.connect(x180_refinement, selective_rabi)

    iq_blobs = library.nodes["07_iq_blobs"].copy(
        name="IQ_blobs",
        num_shots=2000,
    )
    ge_graph.add_node(iq_blobs)
    ge_graph.connect(selective_rabi, iq_blobs)


ge_graph.run()
