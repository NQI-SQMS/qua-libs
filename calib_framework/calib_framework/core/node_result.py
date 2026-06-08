"""NodeResult — typed output of every BO-controlled calibration node.

One NodeResult is produced per (node, qubit, iteration) and written to
SessionLogger. It captures everything needed for causal discovery analysis
and paper validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calib_framework.core.bic import BICResult
from calib_framework.core.estimates import GaussianEstimate


@dataclass
class NodeResult:
    """
    Typed output of a single BO-controlled calibration node execution.

    Produced by BONodeController after each call to should_repeat().
    Written to SessionLogger as one JSONL line per execution.

    Attributes:
        node_id: QUAlibrate node identifier, e.g. "03c_qubit_spectroscopy_vs_power".
        qubit: Qubit name as used in QUAM, e.g. "q1".
        session_id: UUID string identifying the calibration session.
        timestamp: ISO 8601 timestamp of this node execution.
        parameters_used: Dict of parameter name → value as suggested by GP-BO.
        raw_fit: Raw fit outputs from the node (frequency, contrast, success, etc.).
        bic_result: BIC model selection diagnosis for this execution.
        bo_cost: Scalar BO cost registered with GPBayesianOptimizer (lower = better).
        upstream_estimates: Snapshot of QUAM GaussianEstimate inputs at execution time.
        output_estimate: GaussianEstimate produced by this node (None if failed).
        outcome: "successful" | "failed" | "uncertain"
        retry_count: How many times this node has been retried in the current session.
        n_shots: Number of measurement shots used.
        wall_clock_seconds: Elapsed wall-clock time for this node execution.
    """

    node_id: str
    qubit: str
    session_id: str
    timestamp: str
    parameters_used: dict
    raw_fit: dict
    bic_result: BICResult
    bo_cost: float
    upstream_estimates: dict[str, GaussianEstimate]
    output_estimate: GaussianEstimate | None
    outcome: str  # "successful" | "failed" | "uncertain"
    retry_count: int
    n_shots: int
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        if self.outcome not in {"successful", "failed", "uncertain"}:
            raise ValueError(
                f"outcome must be 'successful', 'failed', or 'uncertain', got {self.outcome!r}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (suitable for JSONL logging)."""
        return {
            "node_id": self.node_id,
            "qubit": self.qubit,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "parameters_used": self.parameters_used,
            "raw_fit": self.raw_fit,
            "bic_result": self.bic_result.to_dict(),
            "bo_cost": self.bo_cost,
            "upstream_estimates": {
                k: v.to_dict() for k, v in self.upstream_estimates.items()
            },
            "output_estimate": (
                self.output_estimate.to_dict() if self.output_estimate is not None else None
            ),
            "outcome": self.outcome,
            "retry_count": self.retry_count,
            "n_shots": self.n_shots,
            "wall_clock_seconds": self.wall_clock_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeResult":
        """Deserialise from a dict produced by :meth:`to_dict`."""
        upstream = {
            k: GaussianEstimate.from_dict(v)
            for k, v in d.get("upstream_estimates", {}).items()
        }
        output_est = (
            GaussianEstimate.from_dict(d["output_estimate"])
            if d.get("output_estimate") is not None
            else None
        )
        return cls(
            node_id=d["node_id"],
            qubit=d["qubit"],
            session_id=d["session_id"],
            timestamp=d["timestamp"],
            parameters_used=d["parameters_used"],
            raw_fit=d.get("raw_fit", {}),
            bic_result=BICResult.from_dict(d["bic_result"]),
            bo_cost=float(d["bo_cost"]),
            upstream_estimates=upstream,
            output_estimate=output_est,
            outcome=d["outcome"],
            retry_count=int(d["retry_count"]),
            n_shots=int(d["n_shots"]),
            wall_clock_seconds=float(d["wall_clock_seconds"]),
        )

    @staticmethod
    def outcome_from_bic(bic_result: BICResult) -> str:
        """Derive outcome string from BICResult evidence strength."""
        if bic_result.evidence_strength in {"strong", "moderate"}:
            return "successful"
        elif bic_result.evidence_strength == "weak":
            return "uncertain"
        else:
            return "failed"
