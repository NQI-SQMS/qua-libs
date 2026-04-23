import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Tuple
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class T1EfFit:
    """Stores the relevant T1_ef experiment fit parameters for a single qubit."""

    t1: float
    """T1 of the |f⟩ state in nanoseconds."""
    t1_error: float
    success: bool


def log_fitted_results(ds: xr.Dataset, log_callable=None):
    """
    Log the fitted T1_ef results for all qubits.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'tau', 'tau_error', 'success' coordinates.
    log_callable : callable, optional
        Function for logging. Defaults to logging.info.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in ds.qubit.values:
        tau_us = 1e-3 * ds.sel(qubit=q).tau.values
        tau_err_us = 1e-3 * ds.sel(qubit=q).tau_error.values
        status = "SUCCESS!" if ds.sel(qubit=q).success.values else "FAIL!"
        log_callable(
            f"T1_ef for qubit {q}: {tau_us:.2f} ± {tau_err_us:.2f} µs --> {status}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, T1EfFit]]:
    """
    Fit the T1_ef relaxation time for each qubit using an exponential decay.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the raw I/Q data vs idle_time.
    node : QualibrationNode

    Returns
    -------
    xr.Dataset
        Dataset with fit results merged in.
    dict[str, T1EfFit]
        Per-qubit fit parameters.
    """
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset):
    """Add metadata to the dataset and extract fit results."""
    fit.attrs = {"long_name": "time", "units": "ns"}

    tau = -1 / fit.fit_data.sel(fit_vals="decay")
    fit = fit.assign_coords(tau=("qubit", tau.data))
    fit.tau.attrs = {"long_name": "T1_ef", "units": "ns"}

    tau_error = -tau * (np.sqrt(fit.fit_data.sel(fit_vals="decay_decay")) / fit.fit_data.sel(fit_vals="decay"))
    fit = fit.assign_coords(tau_error=("qubit", tau_error.data))
    fit.tau_error.attrs = {"long_name": "T1_ef error", "units": "ns"}

    success_criteria = (tau.data > 16) & (tau_error.data / tau.data < 1)
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: T1EfFit(
            t1=fit.sel(qubit=q).tau.values.__float__(),
            t1_error=fit.sel(qubit=q).tau_error.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
