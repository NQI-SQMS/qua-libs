"""Analysis for the selective power Rabi node (14b).

Fits a cosine oscillation to state vs amplitude sweep and extracts the pi amplitude.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")


@dataclass
class FitParameters:
    """Fit results for a single qubit's selective power Rabi measurement."""

    pi_amp: float
    """Calibrated pi-pulse amplitude [V]."""
    pi_amp_prefactor: float
    """Amplitude pre-factor at the pi point."""
    chi2: float
    """Residual chi-squared (< 2 → oscillation detected)."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"[14b] {q}: {status} | pi_amp = {1e3 * res['pi_amp']:.3f} mV "
            f"(x{res['pi_amp_prefactor']:.3f}) | chi2 = {res['chi2']:.3f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)

    operation = node.parameters.operation
    full_amp = np.array(
        [ds.amp_prefactor * q.xy.operations[operation].amplitude
         for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    from qualibration_libs.analysis import oscillation

    operation = node.parameters.operation

    if node.parameters.use_state_discrimination:
        fit_vals = fit_oscillation(ds.state, "amp_prefactor")
    else:
        fit_vals = fit_oscillation(ds.I, "amp_prefactor")

    ds_fit = xr.merge([ds, fit_vals.rename("fit")])

    # Find pi amplitude (first maximum of fitted sinusoid)
    amp_axis = ds.amp_prefactor.values
    fit_results = {}
    for q in ds_fit.qubit.values:
        fit_q = ds_fit.sel(qubit=q)
        a = float(fit_q.fit.sel(fit_vals="a").item())
        f = float(fit_q.fit.sel(fit_vals="f").item())
        phi = float(fit_q.fit.sel(fit_vals="phi").item())
        offset = float(fit_q.fit.sel(fit_vals="offset").item())

        # chi2 check
        if np.isfinite(a) and a > 0 and len(amp_axis) > 4:
            fitted = oscillation(amp_axis, a, f, phi, offset)
            if node.parameters.use_state_discrimination:
                data = fit_q.state.values
            else:
                data = fit_q.I.values
            SS_res = float(np.sum((data - fitted) ** 2))
            chi2 = SS_res / ((len(amp_axis) - 4) * a ** 2)
        else:
            chi2 = float("inf")

        success = np.isfinite(a) and np.isfinite(f) and chi2 <= 2.0

        if success:
            fitted_curve = oscillation(amp_axis, a, f, phi, offset)
            max_idx = np.argmax(fitted_curve)
            pi_prefactor = float(amp_axis[max_idx])
        else:
            pi_prefactor = float("nan")

        q_obj = next(qb for qb in node.namespace["qubits"] if qb.name == q)
        base_amp = q_obj.xy.operations[operation].amplitude
        pi_amp = pi_prefactor * base_amp if success else float("nan")

        fit_results[q] = FitParameters(
            pi_amp=pi_amp,
            pi_amp_prefactor=pi_prefactor,
            chi2=chi2,
            success=success,
        )

    return ds_fit, fit_results
