"""
qubit_bo_bootstrap — GP-Bayesian Optimisation qubit bring-up utilities.

Public API
──────────
    Parameters    : NodeParameters class for the 03e BO bootstrap node
    BOOptimizer   : Gaussian-process BO optimiser (sklearn GP + EI/UCB)
    BoFitResult   : Dataclass for a single time-domain Rabi fit result
    fit_rabi_trace: Robust sinusoidal fit with Lomb-Scargle seed
    compute_cost  : Wolff cost function (Ω_R, amplitude, SNR terms)

Usage example
─────────────
    from calibration_utils.qubit_bo_bootstrap import (
        Parameters,
        BOOptimizer,
        fit_rabi_trace,
        compute_cost,
    )
"""

from calibration_utils.qubit_bo_bootstrap.parameters import Parameters, NodeSpecificParameters
from calibration_utils.qubit_bo_bootstrap.bo_optimizer import BOOptimizer
from calibration_utils.qubit_bo_bootstrap.analysis import BoFitResult, fit_rabi_trace, compute_cost

__all__ = [
    "Parameters",
    "NodeSpecificParameters",
    "BOOptimizer",
    "BoFitResult",
    "fit_rabi_trace",
    "compute_cost",
]
