"""
time_rabi_bo — Gaussian-process Bayesian optimisation bootstrap for qubit bring-up.

Replaces the spec_vs_power → qubit_spec → power_rabi nested retry FSM with a
single clean BO loop over the Wolff et al. time-Rabi cost function.

Public API
----------
TimeRabiBoParameters   — NodeParameters subclass for the bootstrap node
fit_time_rabi          — fit a time-Rabi trace, return Ω_R / A / SNR
compute_cost           — evaluate Wolff cost function C
rabi_fit_curve         — evaluate fitted sinusoidal model on a time axis
BOOptimizer            — sklearn GP-BO with LHS init and EI/UCB acquisition
plot_bo_results        — three-panel figure: convergence, landscape, best trace
"""

from calibration_utils.time_rabi_bo.parameters import TimeRabiBoParameters
from calibration_utils.time_rabi_bo.analysis import (
    fit_time_rabi, compute_cost, TimeRabiFitResult, rabi_fit_curve,
)
from calibration_utils.bayesian_optimizer.bo_optimizer import BOOptimizer
from calibration_utils.time_rabi_bo.plotting import plot_bo_results

__all__ = [
    "TimeRabiBoParameters",
    "fit_time_rabi",
    "compute_cost",
    "rabi_fit_curve",
    "TimeRabiFitResult",
    "BOOptimizer",
    "plot_bo_results",
]
