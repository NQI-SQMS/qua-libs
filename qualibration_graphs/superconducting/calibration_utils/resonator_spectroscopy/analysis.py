import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit."""

    frequency: float
    fwhm: float
    success: bool
    chi2: float = float("nan")
    """Residual chi-squared from Lorentzian dip fit: SS_res / ((N-4)*amp²).
    chi2 ≤ threshold → real dip; chi2 > threshold → fitting noise."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz"
        chi2_val = fit_results[q].get("chi2", float("nan"))
        if np.isfinite(chi2_val):
            s_fwhm += f" | Residual chi2: {chi2_val:.3f}"
        s_fwhm += "\n"
        s_qubit += " SUCCESS!\n" if fit_results[q]["success"] else " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit the resonator spectroscopy dip for each qubit."""
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, ds, node)
    return fit_data, fit_results


def _lorentzian_dip(f, f0, gamma, amp, offset):
    """Lorentzian dip: offset - amp*(γ/2)²/((f-f0)²+(γ/2)²).  gamma = FWHM."""
    return offset - amp * (0.5 * gamma) ** 2 / ((f - f0) ** 2 + (0.5 * gamma) ** 2)


def _fit_lorentzian_chi2(ds: xr.Dataset, fit: xr.Dataset, node: QualibrationNode) -> xr.DataArray:
    """Fit a Lorentzian dip to ds.IQ_abs and return chi2 = SS_res / ((N-4)*amp²) per qubit.

    Uses peaks_dips output (fit.position, fit.width, fit.amplitude) as initial guesses.
    Returns inf when the position is NaN, the fit fails to converge, or N ≤ 4.
    """
    detuning = ds.detuning.values  # (N,) in Hz
    N = len(detuning)
    P = 4  # f0, gamma, amp, offset

    chi2_values = []
    for q in ds.qubit.values:
        data = ds.IQ_abs.sel(qubit=q).values

        pos = float(fit.position.sel(qubit=q).values)
        width = float(fit.width.sel(qubit=q).values)
        amp_pd = float(fit.amplitude.sel(qubit=q).values)

        if not np.isfinite(pos) or not np.isfinite(width) or width <= 0 or N <= P:
            chi2_values.append(float("inf"))
            continue

        # Initial guesses
        n_edge = max(1, N // 10)
        offset_init = float(np.mean(np.concatenate([data[:n_edge], data[-n_edge:]])))
        amp_init = max(offset_init - float(np.min(data)), abs(amp_pd), 1e-12)

        try:
            popt, _ = curve_fit(
                _lorentzian_dip,
                detuning,
                data,
                p0=[pos, width, amp_init, offset_init],
                maxfev=3000,
                bounds=([-np.inf, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            )
            amp_fit = popt[2]
            if amp_fit <= 0 or not np.isfinite(amp_fit):
                chi2_values.append(float("inf"))
                continue
            SS_res = float(np.sum((data - _lorentzian_dip(detuning, *popt)) ** 2))
            chi2_values.append(SS_res / ((N - P) * amp_fit ** 2))
        except Exception:
            chi2_values.append(float("inf"))

    return xr.DataArray(chi2_values, dims=["qubit"], coords={"qubit": ds.qubit})


def _extract_relevant_fit_parameters(fit: xr.Dataset, ds: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and extract fit results."""
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    freq_success = np.abs(res_freq.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    position_found = ~np.isnan(fit.position.data)

    chi2 = _fit_lorentzian_chi2(ds, fit, node)
    chi2_threshold = getattr(node.parameters, "chi2_threshold", 3.0)
    fit_quality_ok = chi2.values <= chi2_threshold

    success_criteria = freq_success & fwhm_success & position_found & fit_quality_ok
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {}
    for q in fit.qubit.values:
        q_success = bool(fit.sel(qubit=q).success.values)
        fit_results[q] = FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.item(),
            fwhm=fit.sel(qubit=q).fwhm.values.item(),
            success=q_success,
            chi2=float(chi2.sel(qubit=q).item()),
        )

    return fit, fit_results
