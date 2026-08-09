"""SNZ t_phi_eff scan calibration utilities.

This package provides parameter definitions, waveform construction,
data analysis, and plotting for the Di Carlo Sudden Net-Zero (SNZ)
experiment scanning the effective idle time of a bipolar flux pulse.
"""

from .analysis import FitResults, fit_raw_data, log_fitted_results, process_raw_dataset
from .parameters import Parameters, decompose_t_phi_eff, snz_factory
from .plotting import plot_snz_raw, plot_snz_waveforms

__all__ = [
    "Parameters",
    "snz_factory",
    "decompose_t_phi_eff",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "FitResults",
    "plot_snz_raw",
    "plot_snz_waveforms",
]
