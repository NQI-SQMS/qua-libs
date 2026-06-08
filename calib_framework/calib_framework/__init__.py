"""calib_framework — Autonomous SC qubit calibration via sequential Bayesian inference.

BIC model selection + GP-BO retry replace hand-coded FSM error codes.
When a causal DAG is available, CausalOrchestrator routes failures to
the most likely upstream root-cause node rather than the immediate predecessor.

Hardware-agnostic: no QUA/OPX/qualibrate imports in this package.
Those live exclusively in qualibrate_graphs/.
"""
__version__ = "0.1.0"
