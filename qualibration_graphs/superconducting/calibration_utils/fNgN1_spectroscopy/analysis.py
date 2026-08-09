"""
Analysis utilities for the cavity fNgN1 sideband spectroscopy experiment.

The sideband shows a DIP in the qubit state population when the drive is on
resonance (the qubit transitions from |f,n⟩ to |g,n+1⟩ and is left in |g⟩
after the back-swap π_ef pulse).  We therefore negate the signal before calling
peaks_dips so that the Lorentzian detection algorithm finds the correct feature.

When use_gaussian_fit=True a Gaussian is fitted to the negated signal instead.
The Gaussian parameters (amplitude, sigma, offset_neg) are stored as extra data
variables in ds_fit so the plotting module can overlay the fitted curve.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from calibration_utils.shared import apply_confusion_matrix_correction


def apply_confusion_correction_to_dataset(ds, node):
    return apply_confusion_matrix_correction(ds, node.namespace["qubits"])
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Fit results for a single qubit's cavity fNgN1 sideband spectroscopy experiment."""
    frequency: float
    """Fitted sideband frequency [Hz]."""
    fwhm: float
    """Fitted FWHM of the sideband dip [Hz]."""
    success: bool
    # Gaussian-fit extras — NaN when peaks_dips (Lorentzian) path was used
    gaussian_amplitude: float = float("nan")
    """Amplitude of the Gaussian fitted to the negated signal (> 0)."""
    gaussian_sigma_hz: float = float("nan")
    """Gaussian sigma [Hz]; FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.355*sigma."""
    gaussian_offset_neg: float = float("nan")
    """Baseline of the Gaussian fitted to the negated signal."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tSideband frequency: {1e-9 * res['frequency']:.4f} GHz | "
            f"FWHM: {1e-3 * res['fwhm']:.1f} kHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw I/Q counts to Volts (if not using state discrimination)."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    # Add the full RF frequency as a coordinate.
    # The caller is responsible for patching sideband_drive.RF_frequency to the
    # correct centre RF for this transition before calling this function.
    sideband_drive = _get_sideband_drive(node)
    full_freq = ds.detuning + sideband_drive.RF_frequency
    ds = ds.assign_coords(full_freq=("detuning", full_freq.values))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def _fit_gaussian_dip(x: np.ndarray, y_neg: np.ndarray):
    """Fit a Gaussian peak to *y_neg*, the negated original signal (dip → peak).

    Model: y_neg = offset + amplitude * exp(-0.5 * ((x - center) / sigma)^2)
    with amplitude > 0.

    Returns (center_hz, fwhm_hz, amplitude, sigma_hz, offset, success).
    To reconstruct the curve over the *original* signal:
        y_orig_fit = -(offset + amplitude * gauss(x))
    """
    x = np.asarray(x, float)
    y_neg = np.asarray(y_neg, float)

    # Initial guess from the maximum of y_neg (peak of the negated dip)
    offset0 = float(np.percentile(y_neg, 10))
    i_max = int(np.argmax(y_neg))
    center0 = float(x[i_max])
    amplitude0 = max(float(np.max(y_neg)) - offset0, 1e-12)

    # Estimate sigma from the half-height width of the peak
    half_height = offset0 + 0.5 * amplitude0
    above = y_neg > half_height
    dx = abs(x[1] - x[0]) if len(x) > 1 else float(abs(x[-1] - x[0]))
    if np.count_nonzero(above) >= 2:
        idx = np.where(above)[0]
        left = float(x[max(idx[0] - 1, 0)])
        right = float(x[min(idx[-1] + 1, len(x) - 1)])
        sigma0 = max((right - left) / (2.0 * np.sqrt(2.0 * np.log(2.0))), dx)
    else:
        sigma0 = max(float((x[-1] - x[0]) / 6.0), dx)

    def _gauss(xi, amplitude, center, sigma, offset):
        return offset + amplitude * np.exp(-0.5 * ((xi - center) / sigma) ** 2)

    try:
        popt, _ = curve_fit(
            _gauss, x, y_neg,
            p0=[amplitude0, center0, sigma0, offset0],
            bounds=([0, x[0] - abs(x[-1] - x[0]), 0, -np.inf],
                    [np.inf, x[-1] + abs(x[-1] - x[0]), np.inf, np.inf]),
            maxfev=5000,
        )
        amplitude, center, sigma, offset = popt
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(sigma)
        return float(center), float(fwhm), float(amplitude), float(sigma), float(offset), True
    except Exception:
        fwhm0 = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma0
        return center0, fwhm0, amplitude0, sigma0, offset0, False


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the cavity sideband dip for each qubit.

    The sideband appears as a DIP in the state population (or in I_rot for raw IQ
    measurements).  We negate the signal so that the peak-detection algorithm finds
    the correct feature.

    When node.parameters.use_gaussian_fit is True a Gaussian is fitted instead of
    the default peaks_dips (Lorentzian-based) detector.  The fitted Gaussian
    parameters are stored as extra data variables in ds_fit so the plotting module
    can draw the fitted curve.
    """
    if node.parameters.use_state_discrimination:
        signal = ds.state
    else:
        shifts = np.abs(ds.IQ_abs - ds.IQ_abs.mean(dim="detuning")).idxmax(dim="detuning")
        angle = np.arctan2(
            ds.sel(detuning=shifts).Q - ds.Q.mean(dim="detuning"),
            ds.sel(detuning=shifts).I - ds.I.mean(dim="detuning"),
        )
        I_rot = ds.I * np.cos(angle) + ds.Q * np.sin(angle)
        ds = ds.assign({"I_rot": I_rot})
        signal = I_rot

    use_gaussian = getattr(node.parameters, "use_gaussian_fit", False)

    if use_gaussian:
        # Fit a Gaussian to the negated signal (dip → peak) for each qubit
        qubit_coords = signal.qubit.values
        x = ds.detuning.values.astype(float)
        positions, widths, g_amps, g_sigmas, g_offsets_neg = [], [], [], [], []
        for q in qubit_coords:
            y_neg = -signal.sel(qubit=q).values.astype(float)
            center, fwhm, amp, sigma, offset_neg, _ = _fit_gaussian_dip(x, y_neg)
            positions.append(center)
            widths.append(fwhm)
            g_amps.append(amp)
            g_sigmas.append(sigma)
            g_offsets_neg.append(offset_neg)

        fit_vals = xr.Dataset(
            {
                "position": xr.DataArray(positions, dims=["qubit"], coords={"qubit": qubit_coords}),
                "width": xr.DataArray(widths, dims=["qubit"], coords={"qubit": qubit_coords}),
                "gaussian_amplitude": xr.DataArray(g_amps, dims=["qubit"], coords={"qubit": qubit_coords}),
                "gaussian_sigma_hz": xr.DataArray(g_sigmas, dims=["qubit"], coords={"qubit": qubit_coords}),
                "gaussian_offset_neg": xr.DataArray(g_offsets_neg, dims=["qubit"], coords={"qubit": qubit_coords}),
            }
        )
    else:
        # Original path: negate to turn dip → peak and use peaks_dips (Lorentzian)
        fit_vals = peaks_dips(-signal, dim="detuning", prominence_factor=3)

    ds_fit = xr.merge([ds, fit_vals])
    ds_fit, fit_results = _extract_fit_parameters(ds_fit, node, use_gaussian=use_gaussian)
    return ds_fit, fit_results


def _get_pair_and_drive(node: QualibrationNode):
    """Return (pair, sideband_drive) for the cavity_transmon_pair whose
    cavity_mode_name matches node.parameters.mode_name."""
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            return pair, pair.sideband_drive
    raise KeyError(f"No cavity_transmon_pair with cavity_mode_name='{mode_name}'")


def _get_sideband_drive(node: QualibrationNode):
    """Return the sideband_drive channel for the cavity_transmon_pair whose
    cavity_mode_name matches node.parameters.mode_name."""
    _, sideband_drive = _get_pair_and_drive(node)
    return sideband_drive


def _extract_fit_parameters(fit: xr.Dataset, node: QualibrationNode, use_gaussian: bool = False):
    """Extract the sideband frequency and FWHM from the fitted dataset."""
    sideband_drive = _get_sideband_drive(node)
    centre_rf_freq = sideband_drive.RF_frequency

    span_hz = node.parameters.frequency_span_in_mhz * 1e6

    fit_results = {}
    for q in fit.qubit.values:
        pos = float(fit.position.sel(qubit=q).values)
        width = float(fit.width.sel(qubit=q).values)

        freq = centre_rf_freq + pos
        fwhm = abs(width)

        position_found = bool(np.isfinite(pos))
        freq_in_range = bool(abs(pos) < span_hz)
        success = position_found and freq_in_range

        # Pull Gaussian extras when available
        g_amp = float(fit.gaussian_amplitude.sel(qubit=q).values) if "gaussian_amplitude" in fit else float("nan")
        g_sigma = float(fit.gaussian_sigma_hz.sel(qubit=q).values) if "gaussian_sigma_hz" in fit else float("nan")
        g_offset = float(fit.gaussian_offset_neg.sel(qubit=q).values) if "gaussian_offset_neg" in fit else float("nan")

        fit_results[q] = FitParameters(
            frequency=freq,
            fwhm=fwhm,
            success=success,
            gaussian_amplitude=g_amp,
            gaussian_sigma_hz=g_sigma,
            gaussian_offset_neg=g_offset,
        )

    fit = fit.assign(
        success=xr.DataArray(
            [fit_results[q].success for q in fit.qubit.values],
            dims=["qubit"],
            coords={"qubit": fit.qubit},
        )
    )
    return fit, fit_results
