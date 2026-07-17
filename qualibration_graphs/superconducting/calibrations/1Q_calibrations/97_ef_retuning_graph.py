# %% {EF Retuning Graph}
"""
EF Retuning Graph (97)

Iteratively refines the EF_x180 pulse amplitude and EF transition frequency,
then measures GEF IQ blobs for three-state discrimination:

  ef_x180_refinement [loop until |EF detuning| < freq_threshold_hz, max_iterations]:
    ef_ramsey     → measure and correct EF transition frequency (anharmonicity)
    ef_power_rabi → recalibrate EF_x180 amplitude
  → selective_ef_rabi
  → gef_IQ_blobs (g/e/f)

Ramsey parameters are computed from the current T2*_ef stored in the QUAM state:
  max_wait_time = 5 × T2*_ef
  detuning      = 10 / (5 × T2*_ef)   [10 full oscillation periods in the sweep]
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

# ── Compute Ramsey defaults from QUAM T2*_ef ──────────────────────────────────
_DEFAULT_QUBITS = ["q1"]
_T2_EF_NS: int = 10_000  # fallback: 10 µs

try:
    from quam_config import Quam as _Quam

    _m = _Quam.load()
    _t2_vals = []
    for _q_name in _DEFAULT_QUBITS:
        if _q_name not in _m.qubits:
            continue
        _q = _m.qubits[_q_name]
        _extras = getattr(_q, "extras", None)
        _t2_ef = _extras.get("T2ramsey_ef") if isinstance(_extras, dict) else None
        if _t2_ef is not None:
            _t2_vals.append(int(_t2_ef * 1e9))
    if _t2_vals:
        _T2_EF_NS = min(_t2_vals)
    del _m, _t2_vals
except Exception as _e:
    logger.warning(
        f"[EF retuning] Could not load T2ramsey_ef from QUAM ({_e}). "
        f"Using fallback {_T2_EF_NS} ns."
    )

_EF_MAX_WAIT_NS: int = 5 * _T2_EF_NS
_EF_DETUNING_MHZ: float = round(10.0 / (_EF_MAX_WAIT_NS * 1e-3), 6)  # 10 periods over 5×T2*_ef


# ── Parameters ────────────────────────────────────────────────────────────────


class EFRetuningParameters(GraphParameters):
    """Graph-flow parameters for the EF retuning graph."""

    qubits: List[str] = _DEFAULT_QUBITS
    freq_threshold_hz: float = 50_000.0
    max_iterations: int = 5


class _SubgraphParameters(GraphParameters):
    qubits: List[str] = _DEFAULT_QUBITS


# ── Loop state ────────────────────────────────────────────────────────────────

_ef_loop_state: dict = {
    "initialized": {},
    "any_failed": False,
    "detuning_history": {},
}


# ── Condition function ────────────────────────────────────────────────────────


def should_repeat_ef_calibration(node: QualibrationNode, target: str) -> bool:
    """Repeat ef_ramsey → ef_power_rabi until |EF detuning| < freq_threshold_hz or failure."""
    if not any(_ef_loop_state["initialized"].values()):
        _ef_loop_state["any_failed"] = False

    if _ef_loop_state["any_failed"]:
        _ef_loop_state["initialized"][target] = False
        return False

    if not _ef_loop_state["initialized"].get(target, False):
        _ef_loop_state["detuning_history"][target] = []
        _ef_loop_state["initialized"][target] = True

    if node.outcomes.get(target) == "failed":
        logger.warning(f"[EF retuning] {target}: Fit failed — stopping loop.")
        _ef_loop_state["initialized"][target] = False
        _ef_loop_state["any_failed"] = True
        return False

    _ramsey_node = getattr(node, "_elements", {}).get("ef_ramsey")
    freq_offset = (
        (_ramsey_node.results if _ramsey_node else {})
        .get("fit_results", {})
        .get(target, {})
        .get("freq_offset", None)
    )

    if freq_offset is None:
        _ef_loop_state["initialized"][target] = False
        return False

    abs_offset = abs(freq_offset)
    _ef_loop_state["detuning_history"][target].append(abs_offset)
    logger.info(
        f"[EF retuning] {target}: |EF detuning| = {abs_offset / 1e3:.2f} kHz, "
        f"threshold = {ef_graph.parameters.freq_threshold_hz / 1e3:.0f} kHz."
    )

    if abs_offset < ef_graph.parameters.freq_threshold_hz:
        logger.info(
            f"[EF retuning] {target}: Converged after "
            f"{len(_ef_loop_state['detuning_history'][target])} iteration(s)."
        )
        _ef_loop_state["initialized"][target] = False
        return False

    return True


# ── Graph construction ────────────────────────────────────────────────────────

with QualibrationGraph.build(
    "ef_retuning_graph",
    parameters=EFRetuningParameters(),
) as ef_graph:

    # Inner subgraph: ef_ramsey → ef_power_rabi (looped until convergence)
    with QualibrationGraph.build(
        "ef_x180_refinement",
        parameters=_SubgraphParameters(),
    ) as ef_x180_refinement:

        ef_ramsey = library.nodes["06b_ramsey_ef"].copy(
            name="ef_ramsey",
            num_shots=200,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=_EF_MAX_WAIT_NS,
            frequency_detuning_in_mhz=_EF_DETUNING_MHZ,
            wait_time_num_points=100,
            log_or_linear_sweep="linear",
        )
        ef_x180_refinement.add_node(ef_ramsey)

        ef_power_rabi = library.nodes["13_power_rabi_ef"].copy(
            name="ef_power_rabi",
            min_amp_factor=0.001,
            max_amp_factor=1.9,
            amp_factor_step=0.01,
            num_shots=200,
        )
        ef_x180_refinement.add_node(ef_power_rabi)
        ef_x180_refinement.connect(ef_ramsey, ef_power_rabi)

    ef_graph.add_node(ef_x180_refinement)
    ef_graph.loop(
        ef_x180_refinement,
        on=should_repeat_ef_calibration,
        max_iterations=ef_graph.parameters.max_iterations,
    )

    selective_ef_rabi = library.nodes["13_power_rabi_ef"].copy(
        name="selective_ef_rabi",
        operation="selective_ef_x180",
        min_amp_factor=0.001,
        max_amp_factor=1.99,
        amp_factor_step=0.01,
        num_shots=200,
    )
    ef_graph.add_node(selective_ef_rabi)
    ef_graph.connect(ef_x180_refinement, selective_ef_rabi)

    gef_iq_blobs = library.nodes["15_iq_blobs_gef"].copy(
        name="gef_IQ_blobs",
        num_shots=2000,
    )
    ef_graph.add_node(gef_iq_blobs)
    ef_graph.connect(selective_ef_rabi, gef_iq_blobs)


ef_graph.run()
