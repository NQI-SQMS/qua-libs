"""bringup_causal.py — Adaptive bringup graph with causal orchestration.

Replaces 92_calibration_graph_bringup_fixed_frequency_transmon_adaptive.py.
The old graph (frozen as 92_BASELINE_bringup_fsm_frozen.py) used hard-coded
FSM condition functions. This graph uses BIC + BO + CausalOrchestrator instead.

Without a causal DAG (causal_dag_path=None): runs nodes in dependency order
with BO + BIC; failures retry the immediate predecessor.

With a causal DAG (causal_dag_path provided): CausalOrchestrator attributes
failures to the most likely root-cause upstream node based on learned
causal structure and GP posterior uncertainties.

Nominal execution sequence:
    02a_resonator_spectroscopy
        → 02b_resonator_punch_out     (chi-shift / punch-out characterisation)
            → 03c_qubit_spectroscopy_vs_power
                → 04b_power_rabi

Dependency-order failure routing (no causal DAG):
    04b fails → retry 03c (spectroscopy may have found wrong peak)
    03c fails → retry 02b (resonator punch-out may have used wrong power)

Causal failure routing (with learned DAG):
    Determined by the learned causal DAG — see orchestrator.py.

Usage:
    The graph is registered with QUAlibrate library via the standard
    node/graph scripts mechanism. Import and instantiation happen at
    QUAlibrate runtime; do not instantiate directly.

Node key mapping (existing qua-libs node names → BIC node types):
    "02a_resonator_spectroscopy"      → "resonator_spectroscopy"
    "02b_resonator_punch_out"         → "resonator_punch_out"
    "03c_qubit_spectroscopy_vs_power" → "qubit_spectroscopy_vs_power"
    "04b_power_rabi"                  → "power_rabi"
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

# QUAlibrate imports (only in qualibrate_graphs/)
from qualibrate import QualibrationGraph, QualibrationLibrary
from qualibrate.parameters import GraphParameters

# calib_framework imports
from calib_framework.bo.optimizer import ParameterBound
from calib_framework.bo.node_controller import BONodeController
from calib_framework.causal.discovery import CausalGraphLearner
from calib_framework.core.orchestrator import CausalOrchestrator
from calib_framework.logging.session_logger import SessionLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph parameters
# ---------------------------------------------------------------------------


class BringUpCausalParameters(GraphParameters):
    """
    Parameters for the adaptive causal bringup graph.

    Attributes:
        qubits: List of qubit names to calibrate (e.g. [\"q1\", \"q2\"]).
        bo_state_dir: Directory for GP observation files and session logs.
            Defaults to \"bo_state/\" relative to the working directory.
        causal_dag_path: Path to a JSON causal DAG file produced by
            scripts/analyze_causal_dag.py. Set to None to use dependency-order
            retry (re-run the immediate predecessor on failure). Set to a valid
            path to enable causal routing (fault attribution via the learned DAG).
        max_bo_iterations: Maximum BO iterations per node per qubit.
        success_delta_bic: ΔBIC threshold for declaring a node successful.
            Default 6.0 (= \"moderate\" evidence per Kass & Raftery 1995).

        # Resonator spectroscopy parameters
        resonator_frequency_span_mhz: Frequency sweep span in MHz.
        resonator_n_shots: Number of measurement shots.

        # Punch-out parameters
        punch_out_power_span_dbm: Drive power sweep span in dBm.

        # Qubit spectroscopy parameters
        qubit_frequency_span_mhz: Qubit frequency sweep span in MHz.
        qubit_power_shift_dbm: Drive power offset in dBm.

        # Power Rabi parameters
        rabi_amplitude: Base π-pulse amplitude.
        rabi_n_shots: Number of Rabi measurement shots.

        # Mixer calibration
        run_mixer_calibration: Whether to run IQ mixer calibration first.
    """

    qubits: list[str] = []
    bo_state_dir: str = "bo_state"
    causal_dag_path: str | None = None
    max_bo_iterations: int = 8
    success_delta_bic: float = 6.0

    # Resonator spectroscopy
    resonator_frequency_span_mhz: float = 100.0
    resonator_n_shots: int = 1000

    # Punch-out
    punch_out_power_span_dbm: float = 30.0

    # Qubit spectroscopy
    qubit_frequency_span_mhz: float = 200.0
    qubit_power_shift_dbm: float = 0.0

    # Power Rabi
    rabi_amplitude: float = 0.5
    rabi_n_shots: int = 1000

    # Mixer calibration
    run_mixer_calibration: bool = True


# ---------------------------------------------------------------------------
# Node sequence (must match the BICDiagnoser.MODEL_REGISTRY keys)
# ---------------------------------------------------------------------------

#: Canonical node IDs in execution order (used for causal DAG column ordering).
BRINGUP_NODE_SEQUENCE = [
    "02a_resonator_spectroscopy",
    "02b_resonator_punch_out",
    "03c_qubit_spectroscopy_vs_power",
    "04b_power_rabi",
]

#: Maps node_id → BICDiagnoser node_type
NODE_TYPE_MAP = {
    "02a_resonator_spectroscopy":      "resonator_spectroscopy",
    "02b_resonator_punch_out":         "resonator_punch_out",
    "03c_qubit_spectroscopy_vs_power": "qubit_spectroscopy_vs_power",
    "04b_power_rabi":                  "power_rabi",
}

#: Maps node_id → ParameterBound list for GP-BO search space
NODE_BOUNDS_MAP: dict[str, list[ParameterBound]] = {
    "02a_resonator_spectroscopy": [
        ParameterBound("frequency_span_mhz", 50.0, 500.0),
    ],
    "02b_resonator_punch_out": [
        ParameterBound("power_span_dbm", 5.0, 50.0),
    ],
    "03c_qubit_spectroscopy_vs_power": [
        ParameterBound("frequency_span_mhz", 50.0, 800.0),
        ParameterBound("power_shift_dbm", -30.0, 30.0),
    ],
    "04b_power_rabi": [
        ParameterBound("amplitude", 0.05, 0.95),
    ],
}

#: Maps node_id → (x_axis_key, y_axis_key) for BONodeController._extract_xy_data
NODE_AXIS_MAP: dict[str, tuple[str, str]] = {
    "02a_resonator_spectroscopy":      ("detuning", "R"),
    "02b_resonator_punch_out":         ("frequency", "R"),
    "03c_qubit_spectroscopy_vs_power": ("detuning", "state"),
    "04b_power_rabi":                  ("amplitude", "state"),
}

#: Maps node_id → param_map for BONodeController.
#: Each entry maps BO bound name → node.parameters attribute name.
#: All nodes here use 1:1 mappings so a plain dict suffices.
NODE_PARAM_MAP: dict[str, dict[str, str]] = {
    "02a_resonator_spectroscopy": {
        "frequency_span_mhz": "frequency_span_mhz",
    },
    "02b_resonator_punch_out": {
        "power_span_dbm": "power_span_dbm",
    },
    "03c_qubit_spectroscopy_vs_power": {
        "frequency_span_mhz": "frequency_span_mhz",
        "power_shift_dbm": "power_shift_dbm",
    },
    "04b_power_rabi": {
        "amplitude": "amplitude",
    },
}


# ---------------------------------------------------------------------------
# Graph builder function
# ---------------------------------------------------------------------------


def build_bringup_causal_graph(library: QualibrationLibrary) -> QualibrationGraph:
    """
    Build the adaptive causal bringup QualibrationGraph.

    Nodes are taken directly from the QUAlibrate library (no duplication).
    Each node is wrapped with a BONodeController as the graph.loop() condition.
    CausalOrchestrator handles dynamic re-routing at runtime.

    BO suggestions are injected directly into node.parameters before each retry —
    no QUAM temp_calibration fields are used. GaussianEstimates flow in-memory
    through the chain of BONodeController objects.

    Args:
        library: QualibrationLibrary instance (provides access to existing nodes).

    Returns:
        A configured QualibrationGraph ready for execution.
    """
    params = BringUpCausalParameters()
    bo_state_dir = Path(params.bo_state_dir)
    session_id = str(uuid.uuid4())

    # Instantiate shared logger
    sess_logger = SessionLogger(
        bo_state_dir=bo_state_dir,
        session_id=session_id,
        cooldown_id="DR3-Run009",  # TODO: read from QUAM or environment
        fridge_id="DR3",
    )

    # BONodeController objects are built incrementally so each node can reference
    # controllers for its upstream nodes.
    bo_controllers: dict[str, BONodeController] = {}

    with QualibrationGraph.build(
        "bringup_causal",
        parameters=params,
    ) as graph:

        # -----------------------------------------------------------------
        # Optional: mixer calibration (no BO, just run once)
        # -----------------------------------------------------------------
        if params.run_mixer_calibration and "mixer_calibration" in library.nodes:
            mixer_cal = library.nodes["mixer_calibration"].copy(
                name="mixer_calibration",
                qubits=graph.parameters.qubits,
            )
            graph.add_node(mixer_cal)

        # -----------------------------------------------------------------
        # Resonator spectroscopy — 02a
        # -----------------------------------------------------------------
        node_id = "02a_resonator_spectroscopy"
        res_spec = library.nodes[node_id].copy(
            name=node_id,
            qubits=graph.parameters.qubits,
            frequency_span_mhz=graph.parameters.resonator_frequency_span_mhz,
            num_averages=graph.parameters.resonator_n_shots,
        )
        graph.add_node(res_spec)
        ctrl_res_spec = _wire_bo_loop(
            graph, res_spec, node_id, bo_state_dir, session_id, sess_logger,
            graph.parameters.max_bo_iterations, graph.parameters.success_delta_bic,
            upstream_controllers={},
        )
        bo_controllers[node_id] = ctrl_res_spec

        # -----------------------------------------------------------------
        # Resonator punch-out — 02b
        # (upstream: res_spec estimate tightens frequency search)
        # -----------------------------------------------------------------
        node_id = "02b_resonator_punch_out"
        punch_out = library.nodes[node_id].copy(
            name=node_id,
            qubits=graph.parameters.qubits,
            power_span_dbm=graph.parameters.punch_out_power_span_dbm,
        )
        graph.add_node(punch_out)
        ctrl_punch_out = _wire_bo_loop(
            graph, punch_out, node_id, bo_state_dir, session_id, sess_logger,
            graph.parameters.max_bo_iterations, graph.parameters.success_delta_bic,
            upstream_controllers={"02a_resonator_spectroscopy": ctrl_res_spec},
        )
        bo_controllers[node_id] = ctrl_punch_out

        # -----------------------------------------------------------------
        # Qubit spectroscopy vs power — 03c
        # (upstream: punch-out estimate informs power and frequency range)
        # -----------------------------------------------------------------
        node_id = "03c_qubit_spectroscopy_vs_power"
        qubit_spec = library.nodes[node_id].copy(
            name=node_id,
            qubits=graph.parameters.qubits,
            frequency_span_mhz=graph.parameters.qubit_frequency_span_mhz,
            power_shift_dbm=graph.parameters.qubit_power_shift_dbm,
        )
        graph.add_node(qubit_spec)
        ctrl_qubit_spec = _wire_bo_loop(
            graph, qubit_spec, node_id, bo_state_dir, session_id, sess_logger,
            graph.parameters.max_bo_iterations, graph.parameters.success_delta_bic,
            upstream_controllers={
                "02a_resonator_spectroscopy": ctrl_res_spec,
                "02b_resonator_punch_out": ctrl_punch_out,
            },
        )
        bo_controllers[node_id] = ctrl_qubit_spec

        # -----------------------------------------------------------------
        # Power Rabi — 04b
        # (upstream: qubit spectroscopy estimate informs amplitude search)
        # -----------------------------------------------------------------
        node_id = "04b_power_rabi"
        power_rabi = library.nodes[node_id].copy(
            name=node_id,
            qubits=graph.parameters.qubits,
            amplitude=graph.parameters.rabi_amplitude,
            num_averages=graph.parameters.rabi_n_shots,
        )
        graph.add_node(power_rabi)
        ctrl_power_rabi = _wire_bo_loop(
            graph, power_rabi, node_id, bo_state_dir, session_id, sess_logger,
            graph.parameters.max_bo_iterations, graph.parameters.success_delta_bic,
            upstream_controllers={
                "02a_resonator_spectroscopy": ctrl_res_spec,
                "02b_resonator_punch_out": ctrl_punch_out,
                "03c_qubit_spectroscopy_vs_power": ctrl_qubit_spec,
            },
        )
        bo_controllers[node_id] = ctrl_power_rabi

        # -----------------------------------------------------------------
        # Sequential connections (nominal execution order)
        # -----------------------------------------------------------------
        if params.run_mixer_calibration and "mixer_calibration" in library.nodes:
            graph.connect(mixer_cal, res_spec)

        graph.connect(res_spec, punch_out)
        graph.connect(punch_out, qubit_spec)
        graph.connect(qubit_spec, power_rabi)

        # -----------------------------------------------------------------
        # Attach CausalOrchestrator as graph-level metadata
        # (actual runtime re-routing happens when orchestrator.run() is called)
        # -----------------------------------------------------------------
        causal_dag = None
        if params.causal_dag_path:
            try:
                causal_dag = CausalGraphLearner.load(params.causal_dag_path)
                logger.info("Loaded causal DAG from %s — causal routing enabled.", params.causal_dag_path)
            except Exception as e:
                logger.warning(
                    "Could not load causal DAG from %s: %s. Falling back to dependency-order retry.",
                    params.causal_dag_path, e,
                )

        # Store orchestrator on graph for external access
        graph._causal_orchestrator = CausalOrchestrator(
            node_sequence=BRINGUP_NODE_SEQUENCE,
            bo_controllers=bo_controllers,
            causal_dag=causal_dag,
            max_upstream_retries=2,
        )

        logger.info(
            "bringup_causal graph built. Mode: %s. Session %s.",
            "causal routing" if causal_dag else "dependency-order", session_id,
        )

    return graph


# ---------------------------------------------------------------------------
# Private helper: wire BO loop for a node
# ---------------------------------------------------------------------------


def _wire_bo_loop(
    graph: "QualibrationGraph",
    node: "Any",
    node_id: str,
    bo_state_dir: Path,
    session_id: str,
    sess_logger: "SessionLogger",
    max_iterations: int,
    success_delta_bic: float,
    upstream_controllers: dict[str, BONodeController],
) -> BONodeController:
    """
    Create a BONodeController for node_id and register it as the graph.loop()
    condition function.

    Args:
        graph: The QualibrationGraph being built.
        node: The QualibrationNode to wrap.
        node_id: Node identifier string.
        bo_state_dir: Directory for BO observation files.
        session_id: Current session UUID.
        sess_logger: Shared SessionLogger instance.
        max_iterations: Maximum BO iterations.
        success_delta_bic: ΔBIC success threshold.
        upstream_controllers: Dict of upstream BONodeController objects for
            in-memory estimate propagation.

    Returns:
        The created BONodeController.
    """
    node_type = NODE_TYPE_MAP.get(node_id, "resonator_spectroscopy")
    bounds = NODE_BOUNDS_MAP.get(node_id, [ParameterBound("amplitude", 0.05, 0.95)])
    x_key, y_key = NODE_AXIS_MAP.get(node_id, ("detuning", "state"))
    param_map = NODE_PARAM_MAP.get(node_id)

    controller = BONodeController(
        node_key=node_id,
        node_type=node_type,
        bounds=bounds,
        bo_state_dir=bo_state_dir,
        session_id=session_id,
        logger_inst=sess_logger,
        param_map=param_map,
        upstream_controllers=upstream_controllers,
        max_iterations=max_iterations,
        success_delta_bic=success_delta_bic,
        x_axis_key=x_key,
        y_axis_key=y_key,
    )

    graph.loop(node, on=controller.should_repeat, max_iterations=max_iterations)
    return controller


# ---------------------------------------------------------------------------
# QUAlibrate registration entry point
# ---------------------------------------------------------------------------
# When this file is placed in a directory scanned by QUAlibrate, it should
# expose a QualibrationGraph instance named `graph` or follow the convention
# expected by the local QUAlibrate installation. Adjust as needed.
#
# Example:
#   graph = build_bringup_causal_graph(library)
#
# This is intentionally left as a function call so that the graph is only
# constructed when a library is available (i.e. at runtime, not at import).
