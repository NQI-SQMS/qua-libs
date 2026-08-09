"""Analysis for the T2-echo-versus-flux node.

For every flux-bias point an independent exponential decay ``a*exp(t*decay)+offset``
is fitted along the idle-time axis, yielding T2_echo(flux) = -1/decay. The fit is
vectorised over (qubit, flux_bias) and robust (failed curves -> NaN, no figures).

See the T1_vs_flux analysis docstring for why the shared
``qualibration_libs.analysis.fit_decay_exp`` (which assumes a 2-D input in its
amplitude-guess step) is not used directly on this 3-D data.
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
class T2EchoVsFluxFit:
    """Stores the T2-echo-versus-flux summary for a single qubit."""

    t2_echo_max: float
    """Longest fitted T2 echo over the flux sweep, in seconds."""
    t2_echo_max_error: float
    """Uncertainty on ``t2_echo_max``, in seconds."""
    flux_at_max: float
    """Flux bias (V) at which the longest T2 echo is observed."""
    num_valid_flux: int
    """Number of flux points that produced a physical (finite, positive) T2 echo."""
    success: bool
    """Whether enough flux points were successfully fitted."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the per-qubit T2-echo-versus-flux summary."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        status = "SUCCESS" if r["success"] else "FAIL"
        log_callable(
            f"T2 echo vs flux for {q} : max T2e = {1e6 * r['t2_echo_max']:.2f} +/- "
            f"{1e6 * r['t2_echo_max_error']:.2f} us at flux = {r['flux_at_max']:.4f} V "
            f"({r['num_valid_flux']} valid flux points) --> {status}!"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ data to voltage if state discrimination is not used."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, T2EchoVsFluxFit]]:
    """Fit T2 echo for every flux-bias point and extract the per-qubit summary."""
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_data = _fit_decay_vs_flux(ds[signal_name], time_dim="idle_time")
    ds_fit = xr.merge([ds, fit_data])
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _fit_decay_vs_flux(da: xr.DataArray, time_dim: str = "idle_time") -> xr.DataArray:
    """Fit ``a*exp(t*decay)+offset`` along ``time_dim`` for each (qubit, flux_bias).

    Returns a DataArray named ``fit_data`` with an extra ``fit_vals`` dimension holding
    ``[a, offset, decay, decay_err]``."""
    t = da[time_dim]

    def _fit_one(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 4:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        x, y = x[finite], y[finite]
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


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, T2EchoVsFluxFit]]:
    """Turn fitted decay rates into T2_echo(flux) and per-qubit summaries."""
    decay = ds_fit.fit_data.sel(fit_vals="decay")
    decay_err = ds_fit.fit_data.sel(fit_vals="decay_err")

    tau = -1.0 / decay
    tau_error = np.abs(tau) * (decay_err / np.abs(decay))
    # Keep only physical, well-constrained coherence times: finite & positive, above the
    # ~1-clock-cycle (16 ns) floor, and with a relative error < 1. This discards noise-only
    # / near-flat echo curves that fit with decay ~ 0 -> runaway T2 with a huge error, so a
    # bad qubit cannot pass the count-based success gate nor get written to qubit.extras.
    rel_err = np.abs(tau_error / tau)
    valid = np.isfinite(tau) & (tau > 16) & np.isfinite(tau_error) & (rel_err < 1)
    tau = tau.where(valid)
    tau_error = tau_error.where(valid)

    # Stored in ns on the dataset (consistent with the time axis units)
    ds_fit["T2_echo"] = tau
    ds_fit["T2_echo"].attrs = {"long_name": "T2 echo", "units": "ns"}
    ds_fit["T2_echo_error"] = tau_error
    ds_fit["T2_echo_error"].attrs = {"long_name": "T2 echo error", "units": "ns"}

    n_flux = ds_fit.sizes.get("flux_bias", 0)
    min_valid = max(3, n_flux // 2)

    fit_results = {}
    for q in ds_fit.qubit.values:
        tau_q = ds_fit["T2_echo"].sel(qubit=q)
        err_q = ds_fit["T2_echo_error"].sel(qubit=q)
        num_valid = int(np.isfinite(tau_q).sum())
        if num_valid == 0:
            fit_results[str(q)] = T2EchoVsFluxFit(
                t2_echo_max=float("nan"),
                t2_echo_max_error=float("nan"),
                flux_at_max=float("nan"),
                num_valid_flux=0,
                success=False,
            )
            continue
        flux_at_max = float(tau_q.idxmax(dim="flux_bias", skipna=True).values)
        t2_max_ns = float(tau_q.max(skipna=True).values)
        t2_max_err_ns = float(err_q.sel(flux_bias=flux_at_max).values)
        fit_results[str(q)] = T2EchoVsFluxFit(
            t2_echo_max=1e-9 * t2_max_ns,
            t2_echo_max_error=1e-9 * t2_max_err_ns,
            flux_at_max=flux_at_max,
            num_valid_flux=num_valid,
            success=bool(num_valid >= min_valid),
        )
    return ds_fit, fit_results
