"""CausalOrchestrator — DAG-aware calibration orchestrator.

Replaces static QualibrationGraph traversal with causal reasoning: when a
node fails, the orchestrator queries the causal DAG to identify which upstream
node is most likely responsible, then re-runs that node rather than simply
retrying the failed one.

References:
    Kelly, J. et al. (2018). "Physical qubit calibration on a directed acyclic
        graph." arXiv:1803.03226 — the DAG-based calibration framework this extends.
    Aglietti, V. et al. (2020). "Causal Bayesian Optimization." AISTATS.
        arXiv:2006.01085 — theoretical basis for DAG-aware acquisition.
    Heckerman, D. et al. (1995). "A Bayesian Approach to Causal Discovery." —
        Bayesian troubleshooting / fault attribution in graphical models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from calib_framework.core.node_result import NodeResult
    from calib_framework.bo.node_controller import BONodeController

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Execution state of a calibration node for a specific qubit."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionRecord:
    """Record of one node execution for one qubit."""
    node_id: str
    qubit: str
    state: NodeState
    result: "NodeResult | None" = None
    attempt_count: int = 0


class CausalOrchestrator:
    """
    DAG-aware calibration orchestrator.

    Dependency-order mode (causal_dag=None): on failure, re-run the immediate
        predecessor in node_sequence.
    Causal routing mode (causal_dag provided): uses the learned causal DAG + upstream
        GaussianEstimate uncertainties to identify the most likely root-cause
        node and re-run it.

    This is a tractable approximation to Causal Bayesian Optimisation
    (Aglietti et al., 2020): instead of a full interventional acquisition
    function, the learned graph + posterior uncertainty guides re-run selection.

    Args:
        node_sequence: Ordered list of node IDs (nominal execution order).
            Defines the dependency chain, not a fixed execution path.
        bo_controllers: Dict mapping node_id → BONodeController.
            Used to read GaussianEstimate uncertainties for causal attribution.
        causal_dag: Optional nx.DiGraph. Nodes are node_id strings. Edge (A → B)
            means "A's outcome causally influences B's success."
        max_upstream_retries: How many upstream re-run attempts before giving up.
    """

    def __init__(
        self,
        node_sequence: list[str],
        bo_controllers: dict[str, "BONodeController"],
        causal_dag: Any = None,  # nx.DiGraph | None
        max_upstream_retries: int = 2,
    ) -> None:
        self.node_sequence = list(node_sequence)
        self.bo_controllers = bo_controllers
        self.causal_dag = causal_dag
        self.max_upstream_retries = max_upstream_retries

        # {qubit: {node_id: ExecutionRecord}}
        self._records: dict[str, dict[str, ExecutionRecord]] = {}
        self._upstream_retry_counts: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        node_runners: dict[str, callable],
        qubits: list[str],
    ) -> dict[str, dict[str, NodeState]]:
        """
        Execute the calibration sequence for all qubits.

        Args:
            node_runners: Dict mapping node_id → callable that runs the node
                and returns the node object (with .results and .outcomes).
                In QUAlibrate this wraps node.run().
            qubits: List of qubit names to calibrate.

        Returns:
            {qubit: {node_id: NodeState}} final state map.
        """
        # Initialise records
        for qubit in qubits:
            self._records[qubit] = {
                nid: ExecutionRecord(node_id=nid, qubit=qubit, state=NodeState.PENDING)
                for nid in self.node_sequence
            }
            self._upstream_retry_counts[qubit] = {}

        # Run each node for each qubit in sequence
        for qubit in qubits:
            logger.info("CausalOrchestrator: starting qubit %s", qubit)
            self._run_qubit(node_runners, qubit)

        return self.get_state_map()

    def _run_qubit(self, node_runners: dict[str, callable], qubit: str) -> None:
        """Execute the full node sequence for a single qubit."""
        records = self._records[qubit]
        i = 0
        while i < len(self.node_sequence):
            node_id = self.node_sequence[i]
            record = records[node_id]

            logger.info("  [%s] Running node %s...", qubit, node_id)
            record.state = NodeState.RUNNING
            record.attempt_count += 1

            # Run the node
            runner = node_runners.get(node_id)
            if runner is None:
                logger.warning("  No runner for node %s — SKIPPED.", node_id)
                record.state = NodeState.SKIPPED
                i += 1
                continue

            try:
                node_obj = runner(qubit)
                outcome = self._get_qubit_outcome(node_obj, qubit)
            except Exception as e:
                logger.error("  Node %s raised exception: %s", node_id, e)
                outcome = "failed"

            if outcome == "successful":
                record.state = NodeState.SUCCESS
                i += 1
            elif outcome in {"failed", "uncertain"}:
                record.state = NodeState.FAILED

                # Attempt upstream re-run
                rerun_target = self._select_rerun_target(
                    node_id, qubit, list(records.values())
                )
                retry_key = f"{node_id}->{rerun_target}"
                count = self._upstream_retry_counts[qubit].get(retry_key, 0)

                if rerun_target is not None and count < self.max_upstream_retries:
                    self._upstream_retry_counts[qubit][retry_key] = count + 1
                    rerun_idx = self.node_sequence.index(rerun_target)
                    logger.info(
                        "  [%s] Node %s failed. Re-running upstream: %s (attempt %d/%d).",
                        qubit, node_id, rerun_target, count + 1, self.max_upstream_retries,
                    )
                    # Reset all nodes from rerun_idx onward
                    for j in range(rerun_idx, len(self.node_sequence)):
                        nid = self.node_sequence[j]
                        if records[nid].state in {NodeState.SUCCESS, NodeState.FAILED}:
                            records[nid].state = NodeState.PENDING
                    i = rerun_idx
                else:
                    logger.warning(
                        "  [%s] Node %s: max upstream retries exhausted. Stopping.",
                        qubit, node_id,
                    )
                    # Mark remaining nodes as skipped
                    for j in range(i + 1, len(self.node_sequence)):
                        records[self.node_sequence[j]].state = NodeState.SKIPPED
                    break
            else:
                # Unknown outcome — treat as success and continue
                logger.warning("  [%s] Unknown outcome '%s' for %s.", qubit, outcome, node_id)
                record.state = NodeState.SUCCESS
                i += 1

    # ------------------------------------------------------------------
    # Upstream target selection
    # ------------------------------------------------------------------

    def _select_rerun_target(
        self,
        failed_node: str,
        qubit: str,
        execution_history: list[ExecutionRecord],
    ) -> str | None:
        """
        Select which upstream node to re-run after a failure.

        Algorithm:
        Dependency-order mode (no DAG): return the immediate predecessor in node_sequence.
        Causal routing mode (DAG provided):
            1. Find all causal ancestors of failed_node in the DAG.
            2. For each ancestor, read GaussianEstimate.std from QUAM.
            3. Weight by causal edge strength (edge "weight" attribute).
            4. Return the ancestor with the highest uncertainty-weighted influence.

        Returns None if no suitable upstream node exists.
        """
        idx = self.node_sequence.index(failed_node) if failed_node in self.node_sequence else -1
        if idx <= 0:
            # No predecessor
            return None

        if self.causal_dag is None:
            # No DAG available: fall back to the immediate predecessor
            return self.node_sequence[idx - 1]

        # DAG available: attribute failure to the highest-uncertainty causal ancestor
        try:
            import networkx as nx
            ancestors = nx.ancestors(self.causal_dag, failed_node)
            # Filter to only nodes that are in our sequence and have succeeded
            candidate_records = {r.node_id: r for r in execution_history}
            candidates = [
                n for n in ancestors
                if n in candidate_records
                and candidate_records[n].state == NodeState.SUCCESS
            ]
            if not candidates:
                # Fall back to predecessor
                return self.node_sequence[idx - 1]

            # Score each candidate: uncertainty × causal edge weight
            scores: dict[str, float] = {}
            for candidate in candidates:
                uncertainty = self._get_upstream_uncertainty(candidate, qubit)
                # Causal edge weight (may be absent)
                edge_data = self.causal_dag.get_edge_data(candidate, failed_node, default={})
                edge_weight = float(edge_data.get("weight", 1.0)) if edge_data else 1.0
                scores[candidate] = uncertainty * edge_weight

            best = max(scores, key=lambda k: scores[k])
            logger.debug(
                "Causal re-run target for %s/%s: %s (score=%.3f)",
                failed_node, qubit, best, scores[best],
            )
            return best

        except Exception as e:
            logger.warning("Causal DAG attribution failed: %s. Falling back to predecessor.", e)
            return self.node_sequence[idx - 1] if idx > 0 else None

    def _get_upstream_uncertainty(self, node_id: str, qubit: str) -> float:
        """
        Read GaussianEstimate.std from the BONodeController for node_id.

        Returns 1.0 (maximum uncertainty) if no estimate is available.
        """
        ctrl = self.bo_controllers.get(node_id)
        if ctrl is None:
            return 1.0
        try:
            estimate = ctrl.get_estimate(qubit)
            if estimate is None:
                return 1.0
            return estimate.std
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_qubit_outcome(node_obj: Any, qubit: str) -> str:
        """Extract per-qubit outcome from a node object."""
        try:
            outcomes = getattr(node_obj, "outcomes", {}) or {}
            return outcomes.get(qubit, "failed")
        except Exception:
            return "failed"

    def get_state_map(self) -> dict[str, dict[str, NodeState]]:
        """Return {qubit: {node_id: NodeState}} for all qubits."""
        return {
            qubit: {nid: rec.state for nid, rec in records.items()}
            for qubit, records in self._records.items()
        }

    def summary(self) -> dict:
        """Return execution summary: states, retry counts, and totals."""
        state_map = self.get_state_map()
        total_success = sum(
            1 for qubit_states in state_map.values()
            for state in qubit_states.values()
            if state == NodeState.SUCCESS
        )
        total_failed = sum(
            1 for qubit_states in state_map.values()
            for state in qubit_states.values()
            if state == NodeState.FAILED
        )
        return {
            "state_map": {
                qubit: {nid: state.value for nid, state in states.items()}
                for qubit, states in state_map.items()
            },
            "upstream_retry_counts": dict(self._upstream_retry_counts),
            "total_success": total_success,
            "total_failed": total_failed,
        }
