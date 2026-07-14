"""
Analysis utilities for the f0g1 time Rabi experiment.

The sequence prepares the qubit in |f>, drives the f0g1 sideband with a variable
duration pulse, swaps back (pi_ef) and measures.  The qubit state oscillates as a
function of drive duration (Rabi-like fringe). The pi-pulse duration is extracted
from the first minimum of the fitted sinusoid.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from lmfit import Model, Parameter

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")
from qualibration_libs.analysis import oscillation


@dataclass
class FitParameters:
    """Fit results for a single qubit's f0g1 time Rabi experiment."""
    pi_duration_ns: float
    """Calibrated pi-pulse duration [ns] for the f0g1 channel."""
    num_periods: float
    """Number of Rabi oscillation periods across the duration sweep."""
    chi2: float
    """Residual chi-squared: SS_res / ((N-4)*a^2). chi2 > 2 indicates no oscillation."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tf0g1 pi-duration: {result['pi_duration_ns']:.0f} ns | "
            f"chi2: {result['chi2']:.3f} | "
            f"periods: {result['num_periods']:.2f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts (if needed) and add a duration_ns coordinate."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    duration_ns = 4 * ds.duration_cc
    ds = ds.assign_coords(duration_ns=(["duration_cc"], duration_ns.values))
    ds.duration_ns.attrs = {"long_name": "pulse duration", "units": "ns"}
    return ds


def _fit_oscillation_time(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Fit a sinusoidal oscillation along `dim` using an FFT-based initial guess
    without a hard frequency lower-bound cutoff (suited for long-period time-Rabi).
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


def _compute_chi2(ds_fit: xr.Dataset, signal_name: str) -> xr.DataArray:
    """chi2 = SS_res / ((N-4) * a^2). chi2 <= 2 indicates a valid oscillation."""
    x = ds_fit.duration_cc.values
    N = len(x)
    P = 4

    chi2_values = []
    for q in ds_fit.qubit.values:
        fit_q = ds_fit.sel(qubit=q)
        y_data = getattr(fit_q, signal_name).values
        a = float(fit_q.fit.sel(fit_vals="a").item())
        f = float(fit_q.fit.sel(fit_vals="f").item())
        phi = float(fit_q.fit.sel(fit_vals="phi").item())
        offset = float(fit_q.fit.sel(fit_vals="offset").item())

        if not np.isfinite(a) or a <= 0.0 or N <= P:
            chi2_values.append(float("inf"))
            continue

        fitted = oscillation(x, a, f, phi, offset)
        SS_res = float(np.nansum((y_data - fitted) ** 2))
        chi2_values.append(SS_res / ((N - P) * a ** 2))

    return xr.DataArray(chi2_values, dims=["qubit"], coords={"qubit": ds_fit.qubit})


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit Rabi oscillations in state (or I) vs duration_cc for each qubit."""
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    signal = getattr(ds, signal_name)

    fit_vals = _fit_oscillation_time(signal, "duration_cc")
    ds_fit = xr.merge([ds, fit_vals.rename("fit")])

    chi2 = _compute_chi2(ds_fit, signal_name)
    sweep_range = float(ds.duration_cc.values[-1] - ds.duration_cc.values[0])

    fit_results = {}
    for q in ds_fit.qubit.values:
        fit_q = ds_fit.sel(qubit=q)
        a = float(fit_q.fit.sel(fit_vals="a").item())
        f = float(fit_q.fit.sel(fit_vals="f").item())
        phi = float(fit_q.fit.sel(fit_vals="phi").item())
        q_chi2 = float(chi2.sel(qubit=q).item())

        if np.isfinite(f) and abs(f) > 0:
            # First minimum of a·cos(2π·f·t + φ) + offset within the sweep window.
            # The naive formula gives the first minimum from t=0, which may be before
            # the sweep start (min_duration_ns). We shift by whole periods to find
            # the first minimum that falls within the measured range.
            period_cc = 1.0 / abs(f)
            t_start_cc = float(ds.duration_cc.values[0])
            pi_cc = ((np.pi - phi) % (2 * np.pi)) / (2 * np.pi * abs(f))
            if pi_cc < t_start_cc:
                n_shift = int(np.ceil((t_start_cc - pi_cc) / period_cc))
                pi_cc += n_shift * period_cc
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
