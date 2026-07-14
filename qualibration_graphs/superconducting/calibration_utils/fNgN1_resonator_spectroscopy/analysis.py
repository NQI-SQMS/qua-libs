"""Analysis for resonator spectroscopy at Fock |N⟩.

Delegates fitting to the standard resonator_spectroscopy analysis (Lorentzian dip).
Adds a helper to split the combined fock+vacuum raw dataset into two standard datasets.
"""
import xarray as xr

from calibration_utils.resonator_spectroscopy.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    FitParameters,
)

__all__ = [
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "FitParameters",
    "split_fock_vacuum",
]


def split_fock_vacuum(ds_raw: xr.Dataset):
    """Split the combined raw dataset into two standard-format datasets.

    XarrayDataFetcher groups per-qubit streams by stripping the trailing digit,
    so stream names I1/Q1 → variable "I"/"Q" and I_vac1/Q_vac1 → "I_vac"/"Q_vac",
    all with a qubit dimension.  This function returns (ds_fock, ds_vac) where
    both use "I"/"Q" naming so that process_raw_dataset and fit_raw_data work
    on either without modification.
    """
    ds_fock = ds_raw[["I", "Q"]]
    ds_vac  = ds_raw[["I_vac", "Q_vac"]].rename({"I_vac": "I", "Q_vac": "Q"})
    return ds_fock, ds_vac
