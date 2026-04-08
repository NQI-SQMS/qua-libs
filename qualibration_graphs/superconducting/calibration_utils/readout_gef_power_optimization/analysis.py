import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fit results for a single qubit's GEF readout power optimisation."""

    optimal_amp_factor: float
    """Optimal amplitude scale factor relative to current readout amplitude."""
    optimal_amp: float
    """Optimal readout pulse amplitude [V]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"GEF readout power optimisation for qubit {q}: {status}\n"
            f"\tOptimal amp factor = {res['optimal_amp_factor']:.3f}  "
            f"(amplitude = {1e3 * res['optimal_amp']:.2f} mV)"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ to volts and compute pairwise centroid distances vs amplitude."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
    ds = ds.assign(
        {
            "Dge": np.sqrt((ds.Ig - ds.Ie) ** 2 + (ds.Qg - ds.Qe) ** 2),
            "Def": np.sqrt((ds.Ie - ds.If) ** 2 + (ds.Qe - ds.Qf) ** 2),
            "Dgf": np.sqrt((ds.Ig - ds.If) ** 2 + (ds.Qg - ds.Qf) ** 2),
        }
    )
    # Worst-case discrimination metric
    ds["Distance"] = ds[["Dge", "Def", "Dgf"]].to_array().min("variable")
    return ds


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Find the amplitude that maximises the worst-case GEF centroid separation."""
    qubits = node.namespace["qubits"]
    fit_results = {}
    ds_fit_list = []

    for qubit in qubits:
        q = qubit.name
        ds_q = ds.sel(qubit=q)

        # Light smoothing to reduce shot noise
        dist_smooth = ds_q.Distance.rolling(amp_prefactor=3, center=True, min_periods=1).mean()
        idx = int(dist_smooth.argmax(dim="amp_prefactor"))
        opt_factor = float(ds_q.amp_prefactor.isel(amp_prefactor=idx).values)
        current_amp = float(qubit.resonator.operations[node.parameters.operation].amplitude)
        opt_amp = opt_factor * current_amp

        success = np.isfinite(opt_factor) and opt_factor > 0

        fit_results[q] = FitParameters(
            optimal_amp_factor=opt_factor,
            optimal_amp=opt_amp,
            success=success,
        )

        ds_q_fit = xr.Dataset(
            {
                "optimal_amp_factor": xr.DataArray(opt_factor),
                "optimal_amp": xr.DataArray(opt_amp),
                "success": xr.DataArray(success),
            }
        )
        ds_fit_list.append(ds_q_fit.expand_dims(qubit=[q]))

    ds_fit = xr.concat(ds_fit_list, dim="qubit")
    return ds_fit, fit_results
