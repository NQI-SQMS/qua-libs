import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips, lorentzian_peak
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    relative_freq: float
    fwhm: float
    iw_angle: float
    saturation_amp: float
    x180_amp: float
    chi2: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        s_angle = f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
        s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
        s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
        s_chi2 = f"Residual chi2: {fit_results[q].get('chi2', float('nan')):.3f}\n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm + s_angle + s_saturation + s_x180 + s_chi2)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def _compute_chi2_lorentzian(fit: xr.Dataset, signal: xr.DataArray) -> xr.DataArray:
    """
    Compute the residual chi-squared for each qubit's Lorentzian fit:

        chi2 = SS_res / ((N - 4) * amplitude^2)

    chi2 <= 2 -> good fit (peak detected); chi2 > 2 -> residuals dominate (no peak).
    """
    chi2_values = []
    for q in fit.qubit.values:
        fit_q = fit.sel(qubit=q)
        data_q = signal.sel(qubit=q).values
        N = len(data_q)
        P = 4
        amplitude = float(fit_q.amplitude.values)
        position = float(fit_q.position.values)
        width = float(fit_q.width.values)
        baseline = float(fit_q.base_line.mean().values)
        if not np.isfinite(amplitude) or amplitude <= 0 or N <= P:
            chi2_values.append(float("inf"))
            continue
        fitted = lorentzian_peak(fit.detuning.values, amplitude, position, width / 2, baseline)
        SS_res = float(np.nansum((data_q - fitted) ** 2))
        chi2_values.append(SS_res / ((N - P) * amplitude ** 2))
    return xr.DataArray(chi2_values, dims=["qubit"], coords={"qubit": fit.qubit})


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    When ``node.parameters.signal_source == 'IQ_abs'`` the magnitude is used for
    fitting directly (no IQ rotation is needed and the integration weight angle is
    left unchanged).  When ``signal_source == 'I_rot'`` (default) the IQ data is
    rotated via PCA to maximize qubit-induced variance in a single quadrature.
    """
    signal_source = getattr(node.parameters, "signal_source", "I_rot")
    is_dip = getattr(node.parameters, "find_dip", False)

    ds_fit = ds

    # Always compute I_rot so it is available for inspection even when IQ_abs is used.
    shifts = np.abs((ds_fit.IQ_abs - ds_fit.IQ_abs.mean(dim="detuning"))).idxmax(dim="detuning")
    angle = np.arctan2(
        ds_fit.sel(detuning=shifts).Q - ds_fit.Q.mean(dim="detuning"),
        ds_fit.sel(detuning=shifts).I - ds_fit.I.mean(dim="detuning"),
    )
    ds_fit = ds_fit.assign({"iw_angle": angle})
    ds_fit = ds_fit.assign({"I_rot": ds_fit.I * np.cos(ds_fit.iw_angle) + ds_fit.Q * np.sin(ds_fit.iw_angle)})

    # Select the signal passed to peaks_dips
    if signal_source == "IQ_abs":
        signal_for_fit = ds_fit.IQ_abs
    elif signal_source == "I":
        signal_for_fit = -ds_fit.I if is_dip else ds_fit.I
    elif is_dip:
        signal_for_fit = -ds_fit.I_rot
    else:
        signal_for_fit = ds_fit.I_rot

    fit_vals = peaks_dips(signal_for_fit, dim="detuning", prominence_factor=5)
    ds_fit = xr.merge([ds_fit, fit_vals])
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and extract fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    full_freq = np.array([q.xy.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    rel_freq = fit.position
    fit = fit.assign({"res_freq": ("qubit", res_freq.data)})
    fit = fit.assign({"relative_freq": ("qubit", rel_freq.data)})
    fit.res_freq.attrs = {"long_name": "qubit xy frequency", "units": "Hz"}

    fwhm = np.abs(fit.width)
    fit = fit.assign({"fwhm": fwhm})
    fit.fwhm.attrs = {"long_name": "qubit fwhm", "units": "Hz"}

    signal_source = getattr(node.parameters, "signal_source", "I_rot")
    prev_angles = np.array(
        [q.resonator.operations["readout"].integration_weights_angle for q in node.namespace["qubits"]]
    )
    if signal_source in ("IQ_abs", "I"):
        # IQ_abs and I give no new phase information — keep the existing angle unchanged.
        fit = fit.assign({"iw_angle": ("qubit", prev_angles)})
    else:
        fit = fit.assign({"iw_angle": (prev_angles + fit.iw_angle) % (2 * np.pi)})
    fit.iw_angle.attrs = {"long_name": "integration weight angle", "units": "rad"}

    x180_length = np.array([q.xy.operations["x180"].length * 1e-9 for q in node.namespace["qubits"]])
    used_amp = np.array(
        [q.xy.operations["saturation"].amplitude * node.parameters.operation_amplitude_factor
         for q in node.namespace["qubits"]]
    )
    factor_cw = node.parameters.target_peak_width / fit.width
    fit = fit.assign({"saturation_amplitude": factor_cw * used_amp / node.parameters.operation_amplitude_factor})
    factor_x180 = np.pi / (fit.width * x180_length)
    fit = fit.assign({"x180_amplitude": factor_x180 * used_amp})

    freq_success = np.abs(res_freq) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq

    # Chi-square check (hard failure) — use the same signal that was fitted
    is_dip = getattr(node.parameters, "find_dip", False)
    if signal_source == "IQ_abs":
        signal_for_chi2 = fit.IQ_abs
    elif signal_source == "I":
        signal_for_chi2 = -fit.I if is_dip else fit.I
    elif is_dip:
        signal_for_chi2 = -fit.I_rot
    else:
        signal_for_chi2 = fit.I_rot
    chi2 = _compute_chi2_lorentzian(fit, signal_for_chi2)
    chi2_success = chi2 <= 2.0

    success_criteria = freq_success & fwhm_success & chi2_success
    fit = fit.assign({"success": success_criteria})
    fit = fit.assign({"chi2": chi2})

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            relative_freq=fit.sel(qubit=q).relative_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            iw_angle=fit.sel(qubit=q).iw_angle.values.__float__(),
            saturation_amp=fit.sel(qubit=q).saturation_amplitude.values.__float__(),
            x180_amp=fit.sel(qubit=q).x180_amplitude.values.__float__(),
            chi2=float(chi2.sel(qubit=q).values),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
