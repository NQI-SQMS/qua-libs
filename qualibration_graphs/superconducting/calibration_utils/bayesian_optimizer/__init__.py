"""
bayesian_optimizer — General-purpose GP-BO engine for adaptive calibration nodes.

The BOOptimizer class provides Gaussian-process Bayesian optimisation with
Expected Improvement / UCB acquisition, Latin Hypercube seeding, and a clean
register / suggest interface.

The BONodeController wraps BOOptimizer as a qualibrate loop condition, handling
observation registration, convergence checking, parameter suggestion, and
cross-session persistence.
"""

from calibration_utils.bayesian_optimizer.bo_optimizer import BOOptimizer
from calibration_utils.bayesian_optimizer.bo_node_controller import BONodeController

__all__ = ["BOOptimizer", "BONodeController"]
