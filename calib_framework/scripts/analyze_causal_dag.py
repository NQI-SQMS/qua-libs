"""analyze_causal_dag.py — CLI tool to learn a causal DAG from session logs.

Reads a sessions.jsonl file produced by SessionLogger, learns a causal DAG
from the calibration outcome history, saves the DAG as JSON, and optionally
plots it.

Usage:
    python scripts/analyze_causal_dag.py \\
        --sessions bo_state/sessions.jsonl \\
        --output causal_dag.json \\
        --algorithm GES \\
        --min_sessions 30 \\
        --plot causal_dag.png

Output:
    - causal_dag.json: Learned DAG (load with CausalGraphLearner.load()).
    - causal_dag.png:  Visualisation (if --plot specified).
    - stdout:          Summary of found edges + physical consistency warnings.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("analyze_causal_dag")


# ---------------------------------------------------------------------------
# Node sequence (must match bringup_causal.BRINGUP_NODE_SEQUENCE)
# ---------------------------------------------------------------------------

DEFAULT_NODE_SEQUENCE = [
    "02a_resonator_spectroscopy",
    "02b_resonator_punch_out",
    "03c_qubit_spectroscopy_vs_power",
    "04b_power_rabi",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learn a causal DAG from calibration session logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sessions",
        required=True,
        type=Path,
        help="Path to sessions.jsonl produced by SessionLogger.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the learned causal DAG JSON.",
    )
    parser.add_argument(
        "--algorithm",
        default="GES",
        choices=["GES", "PC"],
        help="Causal discovery algorithm (default: GES).",
    )
    parser.add_argument(
        "--min_sessions",
        type=int,
        default=30,
        help="Minimum sessions required. Script exits with error if fewer.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output path for DAG visualisation PNG (optional).",
    )
    parser.add_argument(
        "--metric",
        default="bo_cost",
        choices=["bo_cost", "outcome"],
        help="Metric to use for the outcome matrix (default: bo_cost).",
    )
    parser.add_argument(
        "--node_sequence",
        nargs="+",
        default=None,
        help=(
            "Override the default node sequence (space-separated node IDs). "
            f"Default: {' '.join(DEFAULT_NODE_SEQUENCE)}"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for PC algorithm (default: 0.05).",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Import calib_framework modules
    # ------------------------------------------------------------------
    try:
        from calib_framework.logging.session_logger import SessionLogger
        from calib_framework.causal.discovery import CausalGraphLearner
    except ImportError as e:
        logger.error(
            "Cannot import calib_framework: %s\n"
            "Install with: pip install -e . (from the calib_framework directory)",
            e,
        )
        return 1

    node_sequence = args.node_sequence or DEFAULT_NODE_SEQUENCE

    # ------------------------------------------------------------------
    # Load sessions
    # ------------------------------------------------------------------
    logger.info("Loading sessions from %s ...", args.sessions)
    sessions = SessionLogger.load_sessions(args.sessions)

    node_results = [s for s in sessions if s.get("_record_type") == "node_result"]
    n_sessions = len({s.get("session_id") for s in node_results})

    logger.info("Found %d node-result records from %d unique sessions.", len(node_results), n_sessions)

    if n_sessions < args.min_sessions:
        logger.error(
            "Only %d sessions found (minimum required: %d). "
            "Collect more calibration data before running causal discovery.",
            n_sessions, args.min_sessions,
        )
        return 2

    # ------------------------------------------------------------------
    # Build outcome matrix
    # ------------------------------------------------------------------
    logger.info("Building outcome matrix (metric='%s') ...", args.metric)
    outcome_matrix = SessionLogger.to_outcome_matrix(
        sessions=sessions,
        node_ids=node_sequence,
        metric=args.metric,
    )
    logger.info(
        "Outcome matrix shape: %s (sessions × nodes). "
        "NaN fraction: %.1f%%",
        outcome_matrix.shape,
        100.0 * float(np.isnan(outcome_matrix).mean()) if hasattr(outcome_matrix, "shape") else 0,
    )

    # ------------------------------------------------------------------
    # Fit causal graph
    # ------------------------------------------------------------------
    import numpy as np

    learner = CausalGraphLearner(
        node_sequence=node_sequence,
        algorithm=args.algorithm,
        alpha=args.alpha,
    )

    logger.info("Fitting causal DAG with %s algorithm ...", args.algorithm)
    try:
        dag = learner.fit(outcome_matrix)
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return 3
    except ValueError as e:
        logger.error("Insufficient data: %s", e)
        return 2
    except Exception as e:
        logger.exception("Causal discovery failed: %s", e)
        return 4

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LEARNED CAUSAL DAG SUMMARY")
    print("=" * 60)
    print(f"Algorithm:  {args.algorithm}")
    print(f"Sessions:   {n_sessions}")
    print(f"Nodes ({len(list(dag.nodes()))}): {', '.join(dag.nodes())}")
    print(f"\nEdges ({len(list(dag.edges()))}):")
    if dag.edges():
        for u, v, data in dag.edges(data=True):
            weight = data.get("weight", 1.0)
            print(f"  {u} → {v}  (weight={weight:.3f})")
    else:
        print("  (no edges found — collect more sessions or check data quality)")

    warnings = learner.validate_physical_consistency()
    if warnings:
        print(f"\nPhysical consistency warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("\nPhysical consistency: OK (no implausible edges)")

    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Save DAG
    # ------------------------------------------------------------------
    learner.save(args.output)
    logger.info("Causal DAG saved to %s", args.output)

    # ------------------------------------------------------------------
    # Plot (optional)
    # ------------------------------------------------------------------
    if args.plot:
        try:
            learner.plot(args.plot)
            logger.info("DAG visualisation saved to %s", args.plot)
        except ImportError as e:
            logger.warning("Cannot plot: %s. Install matplotlib.", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
