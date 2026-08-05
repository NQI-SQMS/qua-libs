"""SNZ conditional phase measurement utilities.

This package provides parameter definitions, oscillation fitting,
and plotting for the SNZ conditional phase experiment that combines
the t_phi_eff scan with a Ramsey-like phase tomography on the target qubit.
"""

from .analysis import FitResults, fit_raw_data, log_fitted_results, process_raw_dataset
from .parameters import Parameters
from .plotting import plot_snz_conditional_phase

__all__ = [
    "Parameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "FitResults",
    "plot_snz_conditional_phase",
]
