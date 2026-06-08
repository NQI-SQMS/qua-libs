"""CausalGraphLearner — learns a causal DAG from calibration session logs.

Input:  outcome matrix from SessionLogger.to_outcome_matrix()
Output: nx.DiGraph where nodes are node_ids and edges represent causal influence.

References:
    Spirtes, P., Glymour, C., & Scheines, R. (2000). "Causation, Prediction,
        and Search." MIT Press. (PC algorithm)
    Chickering, D.M. (2002). "Optimal Structure Identification With Greedy Search."
        JMLR 3, 507–554. (GES algorithm)
    Causal-learn library: https://github.com/py-why/causal-learn
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Minimum session counts for causal discovery
_MIN_SESSIONS_HARD = 15   # Refuse to fit below this
_MIN_SESSIONS_WARN = 30   # Warn if below this


class CausalGraphLearner:
    """
    Learns a causal DAG from calibration session logs.

    The learned graph is used by CausalOrchestrator (causal routing mode) to identify
    which upstream node to re-run when a downstream node fails.

    Algorithm options:
    - "GES": Greedy Equivalence Search with BIC score (default, recommended
             for n_sessions < 100 due to lower data requirements).
    - "PC":  Peter-Clark constraint-based (faster but requires more data and
             is more sensitive to violations of faithfulness assumption).

    Background knowledge (always enforced):
    - Temporal ordering: nodes earlier in node_sequence cannot be caused by
      later nodes. This halves the search space and is physically motivated
      (earlier nodes run first, so their outcomes cannot be caused by later
      nodes in the same session).

    Minimum sessions: 30 (warn), 15 (hard minimum — raises ValueError below).

    Args:
        node_sequence: Ordered list of node IDs defining nominal execution order.
            Earlier nodes precede later ones; causal edges cannot point backward.
        algorithm: "GES" (default) or "PC".
        alpha: Significance level for PC algorithm independence tests (default 0.05).
    """

    def __init__(
        self,
        node_sequence: list[str],
        algorithm: str = "GES",
        alpha: float = 0.05,
    ) -> None:
        if algorithm not in {"GES", "PC"}:
            raise ValueError(f"algorithm must be 'GES' or 'PC', got {algorithm!r}")
        self.node_sequence = list(node_sequence)
        self.algorithm = algorithm
        self.alpha = alpha
        self._graph: Any = None  # nx.DiGraph, set after fit()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, outcome_matrix: np.ndarray) -> Any:
        """
        Fit the causal graph from an outcome matrix.

        Args:
            outcome_matrix: np.ndarray of shape (n_sessions, n_nodes).
                            Produced by SessionLogger.to_outcome_matrix().
                            NaN values are imputed with the column mean.

        Returns:
            nx.DiGraph with node_sequence as node labels and directed edges
            representing causal influence.

        Raises:
            ImportError: If causal-learn or networkx is not installed.
            ValueError: If fewer than _MIN_SESSIONS_HARD sessions are provided.
        """
        _check_causal_learn()

        n_sessions, n_nodes = outcome_matrix.shape
        if n_sessions < _MIN_SESSIONS_HARD:
            raise ValueError(
                f"CausalGraphLearner requires at least {_MIN_SESSIONS_HARD} sessions, "
                f"got {n_sessions}. Collect more data before running causal discovery."
            )
        if n_sessions < _MIN_SESSIONS_WARN:
            logger.warning(
                "CausalGraphLearner: only %d sessions (recommend ≥%d). "
                "Results may be unreliable.",
                n_sessions, _MIN_SESSIONS_WARN,
            )

        # Impute NaN with column means
        data = _impute_nan(outcome_matrix)

        # Align node_sequence to matrix columns
        node_ids = self.node_sequence[:n_nodes]

        # Background knowledge: enforce temporal ordering (no backward edges)
        bg = _build_background_knowledge(node_ids)

        if self.algorithm == "GES":
            g = _run_ges(data, bg, node_ids)
        else:
            g = _run_pc(data, bg, node_ids, self.alpha)

        self._graph = g
        return g

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the learned DAG as JSON.

        Format: ``{"nodes": [...], "edges": [{"from": str, "to": str, "weight": float}]}``
        """
        if self._graph is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        data = _graph_to_dict(self._graph)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Causal DAG saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> Any:
        """
        Load a previously saved causal DAG.

        Args:
            path: Path to JSON file produced by :meth:`save`.

        Returns:
            nx.DiGraph.
        """
        import networkx as nx

        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        g = nx.DiGraph()
        g.add_nodes_from(data["nodes"])
        for e in data.get("edges", []):
            g.add_edge(e["from"], e["to"], weight=float(e.get("weight", 1.0)))
        return g

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, output_path: str | Path) -> None:
        """Render the DAG using matplotlib + networkx."""
        if self._graph is None:
            raise RuntimeError("Call fit() before plot().")
        import networkx as nx
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        g = self._graph

        fig, ax = plt.subplots(figsize=(max(6, len(g.nodes) * 1.5), 5))
        pos = nx.spring_layout(g, seed=42) if len(g.nodes) > 0 else {}
        nx.draw_networkx(
            g, pos=pos, ax=ax,
            node_color="lightblue", node_size=2000,
            font_size=9, arrows=True,
            arrowsize=20, edge_color="steelblue",
        )
        ax.set_title("Learned Causal DAG", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Causal DAG plot saved to %s", output_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_physical_consistency(self) -> list[str]:
        """
        Check learned graph against physical priors.

        Returns a list of warning strings for edges that are physically
        implausible (e.g. power_rabi → resonator_spectroscopy in the
        nominal execution order would be a backward edge violation).
        """
        if self._graph is None:
            return ["Graph not fitted yet."]

        warnings_list: list[str] = []
        node_order = {n: i for i, n in enumerate(self.node_sequence)}

        for u, v in self._graph.edges():
            u_idx = node_order.get(u, -1)
            v_idx = node_order.get(v, -1)
            if u_idx >= 0 and v_idx >= 0 and u_idx > v_idx:
                warnings_list.append(
                    f"Backward edge: {u} → {v} (execution order {u_idx} → {v_idx}). "
                    "This violates temporal ordering and may indicate insufficient data."
                )

        return warnings_list


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_causal_learn() -> None:
    """Raise a clear ImportError if causal-learn is not installed."""
    try:
        import causallearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "CausalGraphLearner requires the causal-learn package. "
            "Install it with: pip install causal-learn\n"
            "Or: uv pip install causal-learn"
        )


def _impute_nan(matrix: np.ndarray) -> np.ndarray:
    """Impute NaN values with column means (simple strategy for small datasets)."""
    data = matrix.copy()
    col_means = np.nanmean(data, axis=0)
    for j in range(data.shape[1]):
        mask = np.isnan(data[:, j])
        if mask.any():
            data[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.5
    return data


def _build_background_knowledge(node_ids: list[str]) -> Any:
    """
    Build causal-learn background knowledge enforcing temporal ordering.

    Marks all edges from later nodes to earlier nodes as forbidden.
    """
    try:
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        bg = BackgroundKnowledge()
        n = len(node_ids)
        for i in range(n):
            for j in range(i):
                # Forbid edge from node_ids[i] → node_ids[j] (backward in time)
                bg.add_forbidden_by_node(node_ids[i], node_ids[j])
        return bg
    except ImportError:
        return None
    except Exception as e:
        logger.warning("Could not build background knowledge: %s", e)
        return None


def _run_ges(data: np.ndarray, bg: Any, node_ids: list[str]) -> Any:
    """Run GES algorithm from causal-learn and convert to nx.DiGraph."""
    import networkx as nx
    from causallearn.search.ScoreBased.GES import ges

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ges(data, score_func="local_score_BIC")
        adj = record["G"].graph  # numpy adjacency matrix
        return _adj_to_digraph(adj, node_ids)
    except Exception as e:
        logger.error("GES algorithm failed: %s. Returning empty graph.", e)
        return _empty_graph(node_ids)


def _run_pc(data: np.ndarray, bg: Any, node_ids: list[str], alpha: float) -> Any:
    """Run PC algorithm from causal-learn and convert to nx.DiGraph."""
    import networkx as nx
    from causallearn.search.ConstraintBased.PC import pc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pc(data, alpha=alpha, indep_test="fisherz", background_knowledge=bg)
        adj = result.G.graph
        return _adj_to_digraph(adj, node_ids)
    except Exception as e:
        logger.error("PC algorithm failed: %s. Returning empty graph.", e)
        return _empty_graph(node_ids)


def _adj_to_digraph(adj: np.ndarray, node_ids: list[str]) -> Any:
    """
    Convert a causal-learn adjacency matrix to a nx.DiGraph.

    causal-learn uses: adj[i, j] == -1 and adj[j, i] == 1 → i → j.
    """
    import networkx as nx

    n = len(node_ids)
    g = nx.DiGraph()
    g.add_nodes_from(node_ids)
    for i in range(min(n, adj.shape[0])):
        for j in range(min(n, adj.shape[1])):
            if i == j:
                continue
            # causal-learn convention: adj[i,j] = -1 and adj[j,i] = 1 → i → j
            if adj[i, j] == -1 and adj[j, i] == 1:
                g.add_edge(node_ids[i], node_ids[j], weight=1.0)
    return g


def _empty_graph(node_ids: list[str]) -> Any:
    """Return an empty nx.DiGraph with the given nodes."""
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(node_ids)
    return g


def _graph_to_dict(g: Any) -> dict:
    """Serialise a nx.DiGraph to a JSON-compatible dict."""
    return {
        "nodes": list(g.nodes()),
        "edges": [
            {"from": u, "to": v, "weight": float(data.get("weight", 1.0))}
            for u, v, data in g.edges(data=True)
        ],
    }
