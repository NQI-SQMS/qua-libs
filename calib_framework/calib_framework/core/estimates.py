"""GaussianEstimate — a calibrated parameter value with uncertainty.

Wraps a posterior mean + standard deviation from GP-BO, together with
provenance metadata (source node, session ID, timestamp). Serializable to/from
JSON for storage in QUAM's TemporaryCalibrationData.gaussian_estimates dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class GaussianEstimate:
    """
    Calibrated parameter value with Gaussian uncertainty.

    Produced by BONodeController after a successful calibration node run.
    The ``std`` field is the GP posterior standard deviation at the best
    observed point (``GPBayesianOptimizer.posterior_std_at_best``).

    Reference: Rasmussen & Williams (2006), "Gaussian Processes for Machine
    Learning", MIT Press — posterior predictive distribution (§2.2).
    """

    mean: float
    std: float
    source_node: str
    session_id: str
    n_observations: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = field(init=False)

    def __post_init__(self) -> None:
        # Clamp std to avoid division-by-zero in search_range() and confidence.
        self.std = max(self.std, 1e-6)
        # confidence ∈ [0, 1]: 1 = perfectly confident, 0 = completely uncertain.
        raw = 1.0 - self.std / abs(self.mean + 1e-12)
        self.confidence = max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "mean": self.mean,
            "std": self.std,
            "source_node": self.source_node,
            "session_id": self.session_id,
            "n_observations": self.n_observations,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianEstimate":
        """Deserialise from a dict produced by :meth:`to_dict`."""
        obj = cls(
            mean=float(d["mean"]),
            std=float(d["std"]),
            source_node=str(d["source_node"]),
            session_id=str(d["session_id"]),
            n_observations=int(d["n_observations"]),
            timestamp=str(d.get("timestamp", datetime.now(timezone.utc).isoformat())),
        )
        return obj

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "GaussianEstimate":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def is_high_confidence(self, threshold: float = 0.95) -> bool:
        """Return True if ``confidence >= threshold``."""
        return self.confidence >= threshold

    def search_range(self, base_range: float, k: float = 3.0) -> float:
        """
        Return a search window width for downstream nodes.

        The downstream node should search over ``[mean - r/2, mean + r/2]``
        where ``r = search_range(base_range, k)``.

        Formula: ``base_range + k * std``

        Args:
            base_range: Minimum search width (e.g. default node search span).
            k: Number of standard deviations to add. Default 3σ covers 99.7 %
               of a Gaussian posterior.

        Returns:
            Adjusted search width in the same units as ``mean`` and ``std``.
        """
        return base_range + k * self.std

    def __repr__(self) -> str:
        return (
            f"GaussianEstimate(mean={self.mean:.6g}, std={self.std:.3g}, "
            f"confidence={self.confidence:.3f}, source='{self.source_node}')"
        )
