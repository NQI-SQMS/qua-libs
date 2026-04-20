import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from lmfit import Model, Parameter

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V, apply_confusion_correction_to_dataset
from qualibration_libs.analysis import oscillation


@dataclass
class FitParameters:
    """Stores the EF time Rabi fit results for a single qubit."""

    pi_duration_ns: float
    """EF pi-pulse duration in nanoseconds (rounded to nearest 4 ns)."""
    num_periods: float
    """Number of Rabi oscillation periods across the duration sweep."""
    chi2: float
    """Residual chi-squared: SS_res / ((N-4)*a²). chi2 > 2 indicates no oscillation."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted EF time Rabi results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        s_qubit = f"Results for qubit {q}: "
        s_dur = f"\tEF pi-pulse duration: {result['pi_duration_ns']:.0f} ns | "
        s_chi2 = f"Chi2: {result['chi2']:.3f} | "
        s_periods = f"Periods: {result['num_periods']:.2f}\n "
        if result["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_dur + s_chi2 + s_periods)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts and add a duration_ns coordinate (4 ns per clock cycle)."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    duration_ns = 4 * ds.duration_cc
    ds = ds.assign_coords(duration_ns=(["duration_cc"], duration_ns.values))
    ds.duration_ns.attrs = {"long_name": "EF pulse duration", "units": "ns"}
    return ds


def _fit_oscillation_time(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Fit a sinusoidal oscillation along `dim` with an FFT-based initial guess
    suited for time-domain Rabi sweeps.

    Uses the full positive-frequency FFT spectrum (no lower-bound cutoff) so
    that slow oscillations with periods >> 40 ns are correctly identified.
    """

    def _fit_single(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_sub = y - np.mean(y)
        N = len(x)
        if N < 5:
            return np.full(4, np.nan)

        T = float(x[1] - x[0])

        yf = np.fft.rfft(y_sub)
        xf = np.fft.rfftfreq(N, T)

        magnitudes = np.abs(yf[1:])
        freqs = xf[1:]

        if len(freqs) == 0 or np.max(magnitudes) == 0:
            return np.full(4, np.nan)

        idx = int(np.argmax(magnitudes))
        f0 = float(freqs[idx])
        a0 = float(2 * magnitudes[idx] / N)
        phi0 = float(np.angle(yf[1:][idx]))
        offset0 = float(np.mean(y))

        nyquist = float(xf[-1])
        f_min = max(float(freqs[0]), 0.3 * f0)
        f_max = min(nyquist, 3.0 * f0 + 1e-12)

        try:
            model = Model(oscillation, independent_vars=["t"])
            result = model.fit(
                y,
                t=x,
                a=Parameter("a", value=a0, min=0),
                f=Parameter("f", value=f0, min=f_min, max=f_max),
                phi=Parameter("phi", value=phi0),
                offset=Parameter("offset", value=offset0),
            )
            return np.array([result.values[k] for k in ["a", "f", "phi", "offset"]])
        except Exception:
            return np.full(4, np.nan)

    fit_res = xr.apply_ufunc(
        _fit_single,
        da[dim],
        da,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(fit_vals=("fit_vals", ["a", "f", "phi", "offset"]))


def _compute_chi2(fit: xr.Dataset) -> xr.DataArray:
    """
    Compute the residual chi-squared for each qubit's oscillation fit:

        chi2 = SS_res / ((N - 4) * a²)

    chi2 ≤ 2 → oscillation detected; chi2 > 2 → residuals dominate.
    """
    x = fit.duration_cc.values
    N = len(x)
    P = 4

    chi2_values = []
    for q in fit.qubit.values:
        fit_q = fit.sel(qubit=q)
        i_data = fit_q.I.values
        a = float(fit_q.fit.sel(fit_vals="a").item())
        f = float(fit_q.fit.sel(fit_vals="f").item())
        phi = float(fit_q.fit.sel(fit_vals="phi").item())
        offset = float(fit_q.fit.sel(fit_vals="offset").item())

        if not np.isfinite(a) or a <= 0.0 or N <= P:
            chi2_values.append(float("inf"))
            continue

        fitted = oscillation(x, a, f, phi, offset)
        SS_res = float(np.nansum((i_data - fitted) ** 2))
        chi2_values.append(SS_res / ((N - P) * a ** 2))

    return xr.DataArray(chi2_values, dims=["qubit"], coords={"qubit": fit.qubit})


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit EF Rabi oscillations in I vs duration_cc for each qubit.

    Returns ds_fit (with fit variables merged in) and a dict of FitParameters.
    """
    if node.parameters.use_state_discrimination:
        fit_vals = _fit_oscillation_time(ds.state, "duration_cc")
    else:
        fit_vals = _fit_oscillation_time(ds.I, "duration_cc")

    ds_fit = xr.merge([ds, fit_vals.rename("fit")])

    chi2 = _compute_chi2(ds_fit)
    sweep_range = float(ds.duration_cc.values[-1] - ds.duration_cc.values[0])

    fit_results = {}
    for q in ds_fit.qubit.values:
        fit_q = ds_fit.sel(qubit=q)
        a = float(fit_q.fit.sel(fit_vals="a").item())
        f = float(fit_q.fit.sel(fit_vals="f").item())
        q_chi2 = float(chi2.sel(qubit=q).item())

        if np.isfinite(f) and abs(f) > 0:
            pi_cc = 1.0 / (2.0 * abs(f))
            pi_duration_ns = float(round(pi_cc) * 4)
            num_periods = abs(f) * sweep_range
        else:
            pi_duration_ns = float("nan")
            num_periods = float("nan")

        nan_fail = not (np.isfinite(a) and np.isfinite(f) and np.isfinite(pi_duration_ns))
        chi2_fail = q_chi2 > 2.0
        success = not nan_fail and not chi2_fail

        fit_results[q] = FitParameters(
            pi_duration_ns=pi_duration_ns,
            num_periods=num_periods,
            chi2=q_chi2,
            success=success,
        )

    return ds_fit, fit_results
