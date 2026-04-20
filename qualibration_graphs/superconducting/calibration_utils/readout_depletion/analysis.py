import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V, apply_confusion_correction_to_dataset
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class FitParameters:
    """Fit parameters for the readout depletion measurement."""

    depletion_time_ns: float
    depletion_time_error_ns: float
    success: bool


def log_fitted_results(ds_fit: xr.Dataset, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in ds_fit.qubit.values:
        tau = float(ds_fit.sel(qubit=q).tau.values)
        tau_err = float(ds_fit.sel(qubit=q).tau_error.values)
        success = bool(ds_fit.sel(qubit=q).success.values)
        status = "SUCCESS!" if success else "FAIL!"
        log_callable(
            f"Depletion time for qubit {q}: {tau:.0f} +/- {tau_err:.0f} ns --> {status}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit the exponential decay of the Ramsey state population vs depletion wait time.

    The resonator photon number decays as n(tau) = n0 * exp(-tau / T_depletion).
    The Ramsey excited-state probability reflects this and also follows an
    exponential, from which we extract the depletion time constant T_depletion.
    """
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(
    fit: xr.Dataset,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    # decay is negative (exponential decay), depletion_time = -1/decay
    tau = -1 / fit.fit_data.sel(fit_vals="decay")
    fit = fit.assign_coords(tau=("qubit", tau.data))
    fit.tau.attrs = {"long_name": "Depletion time", "units": "ns"}

    tau_error = tau * (
        np.sqrt(fit.fit_data.sel(fit_vals="decay_decay"))
        / np.abs(fit.fit_data.sel(fit_vals="decay"))
    )
    fit = fit.assign_coords(tau_error=("qubit", tau_error.data))
    fit.tau_error.attrs = {"long_name": "Depletion time error", "units": "ns"}

    success_criteria = (tau.data > 16) & (tau_error.data / tau.data < 1)
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: FitParameters(
            depletion_time_ns=float(fit.sel(qubit=q).tau.values),
            depletion_time_error_ns=float(fit.sel(qubit=q).tau_error.values),
            success=bool(fit.sel(qubit=q).success.values),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
