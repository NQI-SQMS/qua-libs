"""Analysis for the T1-versus-flux node.

For every flux-bias point an independent exponential decay ``a*exp(t*decay)+offset``
is fitted along the idle-time axis, yielding T1(flux) = -1/decay. The fit is fully
vectorised over the (qubit, flux_bias) dimensions and is robust: any curve that
cannot be fitted is returned as NaN instead of raising or popping a figure.

Note: the shared ``qualibration_libs.analysis.fit_decay_exp`` helper assumes a 2-D
(qubit, time) input in its amplitude-guess step (``dat[:, -10:]``), so it cannot be
applied directly to the 3-D (qubit, flux, time) data here. We therefore wrap
``scipy.optimize.curve_fit`` with ``xr.apply_ufunc(..., vectorize=True)`` and reuse the
library ``decay_exp`` model so plotting stays consistent with the plain T1 node.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import decay_exp, guess


@dataclass
class T1VsFluxFit:
    """Stores the T1-versus-flux summary for a single qubit."""

    t1_max: float
    """Longest fitted T1 over the flux sweep, in ns."""
    t1_max_error: float
    """Uncertainty on ``t1_max``, in ns."""
    flux_at_max: float
    """Flux bias (V) at which the longest T1 is observed."""
    num_valid_flux: int
    """Number of flux points that produced a physical (finite, positive) T1."""
    success: bool
    """Whether enough flux points were successfully fitted."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the per-qubit T1-versus-flux summary."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        status = "SUCCESS" if r["success"] else "FAIL"
        log_callable(
            f"T1 vs flux for {q} : max T1 = {1e-3 * r['t1_max']:.2f} +/- "
            f"{1e-3 * r['t1_max_error']:.2f} us at flux = {r['flux_at_max']:.4f} V "
            f"({r['num_valid_flux']} valid flux points) --> {status}!"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ data to voltage if state discrimination is not used."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, T1VsFluxFit]]:
    """Fit T1 for every flux-bias point and extract the per-qubit summary."""
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_data = _fit_decay_vs_flux(ds[signal_name], time_dim="idle_time")
    ds_fit = xr.merge([ds, fit_data])
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _fit_decay_vs_flux(da: xr.DataArray, time_dim: str = "idle_time") -> xr.DataArray:
    """Fit ``a*exp(t*decay)+offset`` along ``time_dim`` for each (qubit, flux_bias).

    Returns a DataArray named ``fit_data`` with an extra ``fit_vals`` dimension holding
    ``[a, offset, decay, decay_err]`` (matching the convention used by the plain T1 node,
    plus the 1-sigma error on ``decay``)."""
    t = da[time_dim]

    def _fit_one(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 4:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        x, y = x[finite], y[finite]
        # Robust initial guesses
        offset0 = float(np.mean(y[-max(1, len(y) // 5):]))
        a0 = float(y[0] - offset0)
        try:
            decay0 = float(guess.exp_decay(x, y))
        except Exception:
            span = max(x.max() - x.min(), 1.0)
            decay0 = -1.0 / span
        if not np.isfinite(decay0) or decay0 == 0:
            span = max(x.max() - x.min(), 1.0)
            decay0 = -1.0 / span
        try:
            popt, pcov = curve_fit(decay_exp, x, y, p0=[a0, offset0, decay0], maxfev=10000)
            perr = np.sqrt(np.abs(np.diag(pcov)))
            return np.array([popt[0], popt[1], popt[2], perr[2]])
        except Exception:
            return np.array([np.nan, np.nan, np.nan, np.nan])

    fit = xr.apply_ufunc(
        _fit_one,
        t,
        da,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    fit = fit.assign_coords(fit_vals=("fit_vals", ["a", "offset", "decay", "decay_err"]))
    return fit.rename("fit_data")


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, T1VsFluxFit]]:
    """Turn fitted decay rates into T1(flux) and per-qubit summaries."""
    decay = ds_fit.fit_data.sel(fit_vals="decay")
    decay_err = ds_fit.fit_data.sel(fit_vals="decay_err")

    tau = -1.0 / decay
    tau_error = np.abs(tau) * (decay_err / np.abs(decay))
    # Keep only physical, well-constrained relaxation times. Beyond finite & positive we
    # require tau above the ~1-clock-cycle (16 ns) floor and a relative error < 1, exactly
    # as the plain T1 node does (calibration_utils/T1/analysis.py). This discards noise-only
    # / near-flat curves that fit with decay ~ 0 -> runaway tau with an astronomical error,
    # so they are never counted as valid nor written to qubit.extras.
    rel_err = np.abs(tau_error / tau)
    valid = np.isfinite(tau) & (tau > 16) & np.isfinite(tau_error) & (rel_err < 1)
    tau = tau.where(valid)
    tau_error = tau_error.where(valid)

    ds_fit["tau"] = tau
    ds_fit["tau"].attrs = {"long_name": "T1", "units": "ns"}
    ds_fit["tau_error"] = tau_error
    ds_fit["tau_error"].attrs = {"long_name": "T1 error", "units": "ns"}

    n_flux = ds_fit.sizes.get("flux_bias", 0)
    min_valid = max(3, n_flux // 2)

    fit_results = {}
    for q in ds_fit.qubit.values:
        tau_q = ds_fit.tau.sel(qubit=q)
        err_q = ds_fit.tau_error.sel(qubit=q)
        num_valid = int(np.isfinite(tau_q).sum())
        if num_valid == 0:
            fit_results[str(q)] = T1VsFluxFit(
                t1_max=float("nan"),
                t1_max_error=float("nan"),
                flux_at_max=float("nan"),
                num_valid_flux=0,
                success=False,
            )
            continue
        flux_at_max = float(tau_q.idxmax(dim="flux_bias", skipna=True).values)
        t1_max = float(tau_q.max(skipna=True).values)
        t1_max_error = float(err_q.sel(flux_bias=flux_at_max).values)
        fit_results[str(q)] = T1VsFluxFit(
            t1_max=t1_max,
            t1_max_error=t1_max_error,
            flux_at_max=flux_at_max,
            num_valid_flux=num_valid,
            success=bool(num_valid >= min_valid),
        )
    return ds_fit, fit_results
