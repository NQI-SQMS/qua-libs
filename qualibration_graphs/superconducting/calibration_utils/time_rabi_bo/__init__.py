"""
time_rabi_bo — Time-Rabi fitting and cost functions for qubit bring-up.

Public API
----------
TimeRabiBoParameters   — NodeParameters subclass for the bootstrap node
fit_time_rabi          — fit a time-Rabi trace, return Ω_R / A / SNR
compute_cost           — evaluate Wolff cost function C
rabi_fit_curve         — evaluate fitted sinusoidal model on a time axis
plot_bo_results        — three-panel figure: convergence, landscape, best trace
"""

from calibration_utils.time_rabi_bo.parameters import TimeRabiBoParameters
from calibration_utils.time_rabi_bo.analysis import (
    fit_time_rabi, compute_cost, TimeRabiFitResult, rabi_fit_curve,
)
from calibration_utils.time_rabi_bo.plotting import plot_bo_results

__all__ = [
    "TimeRabiBoParameters",
    "fit_time_rabi",
    "compute_cost",
    "rabi_fit_curve",
    "TimeRabiFitResult",
    "plot_bo_results",
]
