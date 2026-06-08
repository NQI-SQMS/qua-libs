"""SessionLogger — structured JSONL logging for calibration sessions.

Every node execution appends one JSON line to {bo_state_dir}/sessions.jsonl.
This file is the training data for causal discovery and the empirical validation
dataset for the paper.

File is safe for concurrent writes via filelock.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from calib_framework.core.node_result import NodeResult

logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Structured JSONL session logger.

    Every node execution appends one JSON object (line) to:
        {bo_state_dir}/sessions.jsonl

    This file is the training data for CausalGraphLearner and the
    empirical validation dataset for the paper.

    File format: one JSON object per line, fields from NodeResult.to_dict()
    plus session-level metadata (cooldown_id, fridge_id, software_version).

    Thread-safe via filelock (if installed). Falls back to unguarded writes
    with a warning if filelock is not available.

    Args:
        bo_state_dir: Directory where sessions.jsonl is written.
        session_id: UUID string identifying this session.
        cooldown_id: Human-readable cooldown identifier (e.g. "DR3-Run009").
        fridge_id: Fridge/cryostat identifier (e.g. "DR3").
    """

    SESSIONS_FILENAME = "sessions.jsonl"

    def __init__(
        self,
        bo_state_dir: str | Path,
        session_id: str,
        cooldown_id: str = "unknown",
        fridge_id: str = "unknown",
    ) -> None:
        self.bo_state_dir = Path(bo_state_dir)
        self.bo_state_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.cooldown_id = cooldown_id
        self.fridge_id = fridge_id
        self._jsonl_path = self.bo_state_dir / self.SESSIONS_FILENAME
        self._lock_path = self._jsonl_path.with_suffix(".jsonl.lock")

        # Try to import filelock; degrade gracefully if absent.
        try:
            from filelock import FileLock

            self._lock: object = FileLock(str(self._lock_path))
        except ImportError:
            logger.warning(
                "filelock not installed — SessionLogger writes are NOT thread-safe. "
                "Install with: pip install filelock"
            )
            self._lock = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append_line(self, record: dict) -> None:
        """Append one JSON line to sessions.jsonl, with optional file locking."""
        line = json.dumps(record, default=_json_default) + "\n"
        if self._lock is not None:
            with self._lock:
                self._jsonl_path.open("a", encoding="utf-8").write(line)
        else:
            self._jsonl_path.open("a", encoding="utf-8").write(line)

    def _base_record(self, record_type: str) -> dict:
        return {
            "_record_type": record_type,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "cooldown_id": self.cooldown_id,
            "fridge_id": self.fridge_id,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_node_result(self, result: "NodeResult") -> None:
        """Append NodeResult to sessions.jsonl. Thread-safe via filelock."""
        record = self._base_record("node_result")
        record.update(result.to_dict())
        self._append_line(record)

    def log_session_start(self, qubits: list[str], graph_name: str) -> None:
        """Append a session_start record."""
        record = self._base_record("session_start")
        record["qubits"] = qubits
        record["graph_name"] = graph_name
        self._append_line(record)

    def log_session_end(self, final_outcomes: dict[str, str]) -> None:
        """Append a session_end record with final per-qubit outcomes."""
        record = self._base_record("session_end")
        record["final_outcomes"] = final_outcomes
        self._append_line(record)

    # ------------------------------------------------------------------
    # Static analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_sessions(jsonl_path: str | Path) -> list[dict]:
        """
        Load all logged records from a sessions.jsonl file.

        Returns:
            List of dicts, one per line. Malformed lines are skipped with
            a warning.
        """
        path = Path(jsonl_path)
        records = []
        if not path.exists():
            logger.warning("sessions.jsonl not found at %s", path)
            return records
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line %d in %s: %s", i, path, e)
        return records

    @staticmethod
    def to_outcome_matrix(
        sessions: list[dict],
        node_ids: list[str],
        metric: str = "bo_cost",
    ) -> np.ndarray:
        """
        Convert session logs to a matrix suitable for causal discovery.

        Only records with ``_record_type == "node_result"`` are used.

        Args:
            sessions: List of dicts from :meth:`load_sessions`.
            node_ids: Ordered list of node IDs that define matrix columns.
            metric: Field name to extract as the numeric value.
                Default "bo_cost". Can also be "outcome" (encoded as float).

        Returns:
            np.ndarray of shape (n_sessions, n_nodes).
            Each row is one calibration session. NaN where a node was not
            executed in that session.
        """
        # Group node_result records by session_id
        session_data: dict[str, dict[str, float]] = {}

        for rec in sessions:
            if rec.get("_record_type") != "node_result":
                continue
            sid = rec.get("session_id", "")
            nid = rec.get("node_id", "")
            if nid not in node_ids:
                continue
            if sid not in session_data:
                session_data[sid] = {}

            if metric == "bo_cost":
                value = float(rec.get("bo_cost", float("nan")))
            elif metric == "outcome":
                outcome_map = {"successful": 0.0, "uncertain": 0.5, "failed": 1.0}
                value = outcome_map.get(rec.get("outcome", ""), float("nan"))
            else:
                raw = rec.get(metric)
                value = float(raw) if raw is not None else float("nan")

            session_data[sid][nid] = value

        if not session_data:
            return np.full((0, len(node_ids)), float("nan"))

        session_ids = sorted(session_data.keys())
        matrix = np.full((len(session_ids), len(node_ids)), float("nan"))

        for i, sid in enumerate(session_ids):
            for j, nid in enumerate(node_ids):
                if nid in session_data[sid]:
                    matrix[i, j] = session_data[sid][nid]

        return matrix


def _json_default(obj: object) -> object:
    """JSON serialiser fallback for numpy scalars and similar types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
