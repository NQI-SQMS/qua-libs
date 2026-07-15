"""
Analysis utilities for the cavity mode T2 Ramsey experiment.

The sequence creates a coherent state |α⟩ via a displacement pulse, waits a
variable time tau with an artificial detuning imprinted via frame rotation, then
applies a reverse displacement and reads the qubit with a standard π pulse.
The qubit state oscillates as a decaying sinusoid from which T2ramsey of the
cavity mode is extracted.

Fitting convention (mirrors calibration_utils/ramsey/analysis.py):
  idle_time is stored in ns.
  fit_oscillation_decay_exp returns parameters in consistent units:
    decay [1/ns]  — exponential decay RATE  →  T2 [ns] = 1/decay
    f     [1/ns]  — oscillation frequency   →  f [Hz]  = f * 1e9

Unlike the qubit ge Ramsey (which benefits from ±detuning to stabilise the FFT
frequency guess), the cavity T2 is a single trace.  When T2 >> scan range the
data contains <<1 full oscillation cycle and the generic FFT-based guess picks
up noise peaks at the wrong frequency.  We therefore seed the frequency from
ramsey_detuning_hz and use scipy.curve_fit directly.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
    _HAS_CONFUSION_CORRECTION = True
except ImportError:
    _HAS_CONFUSION_CORRECTION = False
from qualibration_libs.analysis.models import oscillation_decay_exp as _osc_decay


@dataclass
class FitParameters:
    """Fit results for a single qubit's cavity mode T2 Ramsey experiment."""
    T2ramsey_ns: float
    """Cavity mode T2 Ramsey time [ns]."""
    T2ramsey_uncertainty_ns: float
    """1-sigma uncertainty on T2ramsey from the covariance matrix [ns]."""
    freq_offset_hz: float
    """Fitted oscillation frequency offset from the applied detuning [Hz]."""
    success: bool
    chi2: float = float("nan")
    """Residual chi-squared SS_res / ((N-5)·a²); >2 → no oscillation detected."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        T2_us = result["T2ramsey_ns"] * 1e-3
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tCavity T2ramsey: {T2_us:.1f} us | "
            f"freq offset: {result['freq_offset_hz'] * 1e-3:.3f} kHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts, or build a 'state' DataArray from stateN variables."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    elif _HAS_CONFUSION_CORRECTION:
        ds = apply_confusion_correction_to_dataset(ds, node)
    else:
        qubits = node.namespace["qubits"]
        state_arrays = [
            ds[f"state{i}"].assign_coords(qubit=q.name)
            for i, q in enumerate(qubits, start=1)
            if f"state{i}" in ds
        ]
        if state_arrays:
            ds = ds.assign(state=xr.concat(state_arrays, dim="qubit"))
    return ds


def _fit_single_trace(
    t: np.ndarray,
    y: np.ndarray,
    f0_hz: float,
) -> Tuple[np.ndarray, bool]:
    """Fit a * exp(-decay * t) * cos(2π*f*t + phi) + offset to a single trace.

    t in ns, f0_hz is the expected Ramsey frequency in Hz.
    Seeds f from f0_hz to avoid FFT picking up noise peaks when < 1 cycle is
    visible in the scan window (common when T2 >> scan range).

    Returns (popt, pcov, converged) where popt = [a, f, phi, offset, decay].
    """
    f0 = f0_hz * 1e-9          # expected frequency in 1/ns
    a0 = float((y.max() - y.min()) / 2)
    off0 = float(y.mean())
    # T2 initial guess: half the scan range
    T2_guess = max(float(t[-1] - t[0]) / 2, 1.0)
    decay0 = 1.0 / T2_guess

    # Allow f to vary up to 100× the expected detuning to capture cavity IF
    # offsets that can easily exceed the applied detuning by 10-50×.
    # Cap at 100 MHz (1e-1 / ns) to prevent the fitter chasing noise peaks.
    f_max = max(100.0 * f0, 1e-4)  # floor at 100 Hz equivalent (1e-4 / ns)

    p0 = [a0, f0, 0.0, off0, decay0]
    bounds = (
        [0.0,  0.0,   -np.pi, -np.inf, 0.0],
        [np.inf, f_max, np.pi,  np.inf, np.inf],
    )

    try:
        popt, pcov = curve_fit(
            _osc_decay, t, y, p0=p0, bounds=bounds, maxfev=20000,
        )
        return popt, pcov, True
    except Exception:
        return np.array(p0), np.full((5, 5), np.inf), False


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit decaying sinusoid seeded from ramsey_detuning_hz for each qubit."""
    signal = ds.state if node.parameters.use_state_discrimination else ds.I
    t = ds.idle_time.values.astype(float)   # ns
    f0_hz = float(node.parameters.ramsey_detuning_hz)
    N = len(t)

    fit_data = {}   # q → dict of fit DataArrays
    fit_results = {}

    for q in signal.qubit.values:
        y = signal.sel(qubit=q).values.astype(float)
        popt, pcov, converged = _fit_single_trace(t, y, f0_hz)
        a, f, phi, off, decay = popt

        fit_data[q] = {"a": a, "f": f, "phi": phi, "offset": off, "decay": decay}

        if converged and a > 0 and decay > 0:
            T2ramsey_ns = float(1.0 / decay)           # decay [1/ns] → T2 [ns]
            # σ_T2 = σ_decay / decay²  (error propagation: T2 = 1/decay)
            decay_var = float(pcov[4, 4])
            if np.isfinite(decay_var) and decay_var >= 0:
                T2ramsey_uncertainty_ns = float(np.sqrt(decay_var)) / (decay ** 2)
            else:
                T2ramsey_uncertainty_ns = float("nan")
            freq_offset_hz = float(f * 1e9 - f0_hz)    # f [1/ns] → Hz, then offset
            if N > 5:
                y_pred = _osc_decay(t, a, f, phi, off, decay)
                chi2 = float(np.sum((y - y_pred) ** 2) / ((N - 5) * a ** 2))
            else:
                chi2 = float("inf")
            success = bool(chi2 <= 2.0)
        else:
            T2ramsey_ns = float("nan")
            T2ramsey_uncertainty_ns = float("nan")
            freq_offset_hz = float("nan")
            chi2 = float("nan")
            success = False

        fit_results[q] = FitParameters(
            T2ramsey_ns=T2ramsey_ns,
            T2ramsey_uncertainty_ns=T2ramsey_uncertainty_ns,
            freq_offset_hz=freq_offset_hz,
            success=success,
            chi2=chi2,
        )

    # Build a ds_fit that carries the fit parameters so the plotting module can
    # reconstruct the fitted curve without needing to re-run the fitter.
    fit_param_names = ["a", "f", "phi", "offset", "decay"]
    qubit_coords = list(fit_data.keys())
    fit_array = np.array(
        [[fit_data[q][p] for p in fit_param_names] for q in qubit_coords]
    )  # shape (n_qubits, 5)
    fit_da = xr.DataArray(
        fit_array,
        dims=["qubit", "fit_vals"],
        coords={"qubit": qubit_coords, "fit_vals": fit_param_names},
    )
    ds_fit = xr.merge([ds, fit_da.rename("fit")])

    return ds_fit, fit_results
