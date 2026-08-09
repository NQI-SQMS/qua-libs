import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant AC Stark detuning fit parameters for a single qubit."""

    detuning: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted AC Stark detuning for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_detuning = f"\tAC Stark detuning: {1e-6 * fit_results[q]['detuning']:.3f} MHz\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_detuning)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    if "base_detuning" not in ds.coords:
        # Backwards compatibility with datasets saved before the sweep was centered on the
        # pre-existing operation detuning.
        ds = ds.assign_coords(base_detuning=("qubit", np.zeros(ds.sizes["qubit"])))
    ds = ds.assign_coords(full_detuning=ds["detuning"] + ds["base_detuning"])
    ds.full_detuning.attrs = {"long_name": "absolute pulse detuning", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    ds_fit = ds
    if node.parameters.use_state_discrimination:
        ds_fit["averaged_data"] = ds.state.mean(dim="nb_of_pulses")
    else:
        ds_fit["averaged_data"] = ds.I.mean(dim="nb_of_pulses")
    ds_fit["optimal_detuning"] = ds.full_detuning.isel(detuning=ds_fit["averaged_data"].argmin("detuning"))

    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    nan_success = np.isnan(fit.optimal_detuning)
    snr_success = (
        np.abs(
            (fit["averaged_data"].min("detuning") - fit["averaged_data"].mean("detuning"))
            / fit["averaged_data"].std("detuning")
        )
        > 2
    )
    success_criteria = ~nan_success & snr_success
    fit = fit.assign({"success": success_criteria})
    fit_results = {
        q: FitParameters(
            detuning=float(fit.sel(qubit=q)["optimal_detuning"]),
            success=bool(fit.sel(qubit=q).success),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
