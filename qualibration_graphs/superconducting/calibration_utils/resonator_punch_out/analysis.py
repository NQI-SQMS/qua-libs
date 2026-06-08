import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


# =========================
# Fit parameter container
# =========================

@dataclass
class FitParameters:
    success: bool
    resonator_frequency: float  # legacy: freq_low_abs (kept for logging compatibility)
    frequency_shift: float
    optimal_power: float
    freq_low_abs: float = 0.0   # absolute resonator frequency at LOW power (use this to update state)
    chi2_low: float = float("nan")
    """Residual chi-squared from Lorentzian dip fit at low power: SS_res / ((N-4)*amp²)."""
    chi2_high: float = float("nan")
    """Residual chi-squared from Lorentzian dip fit at high power: SS_res / ((N-4)*amp²)."""


# =========================
# Logging
# =========================

def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {fit_results[q]['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.3f} MHz)\n"

        chi2_low = fit_results[q].get("chi2_low", float("nan"))
        chi2_high = fit_results[q].get("chi2_high", float("nan"))
        s_chi2 = ""
        if np.isfinite(chi2_low) or np.isfinite(chi2_high):
            s_chi2 = f"Lorentzian chi2: low={chi2_low:.3f}, high={chi2_high:.3f}\n"

        s_qubit += " SUCCESS!\n" if fit_results[q]["success"] else " FAIL!\n"
        log_callable(s_qubit + s_power + s_freq + s_shift + s_chi2)


# =========================
# Raw data processing
# =========================

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):

    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )

    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    ds = ds.assign(
        IQ_abs_norm=ds.IQ_abs / ds.IQ_abs.mean(dim="detuning")
    )

    return ds


# =========================
# Lorentzian fit helpers
# =========================

def _lorentzian_dip(f, f0, gamma, amp, offset):
    """Lorentzian dip: offset - amp*(γ/2)²/((f-f0)²+(γ/2)²). gamma = FWHM."""
    return offset - amp * (0.5 * gamma) ** 2 / ((f - f0) ** 2 + (0.5 * gamma) ** 2)


def _fit_lorentzian(detuning: np.ndarray, data: np.ndarray):
    """Fit a Lorentzian dip to (detuning, data).

    Returns (f0_fit, chi2) where:
      f0_fit : fitted center detuning (Hz), or nan on failure.
      chi2   : SS_res / ((N-4)*amp²), or inf on failure.

    Uses peaks_dips for the initial f0 estimate (baseline-subtracted, prominence-gated)
    so the warm start is robust to sloping baselines and low-SNR data.  If peaks_dips
    finds no prominent dip, returns (nan, inf) immediately — the caller interprets this
    as a failed fit and chi2 = inf, which correctly triggers a punch-out retry.
    """
    from qualibration_libs.analysis import peaks_dips
    import xarray as xr

    N = len(detuning)
    P = 4  # f0, gamma, amp, offset
    if N <= P:
        return np.nan, float("inf")

    # Build a minimal DataArray so peaks_dips can operate on it.
    da = xr.DataArray(data, coords={"detuning": detuning}, dims=["detuning"])
    pd = peaks_dips(da, "detuning")
    f0_init = float(pd.position.item())
    width_init = float(pd.width.item())

    # peaks_dips returns nan when no prominent peak/dip is found.
    # At low SNR this is the correct outcome — flag as failed immediately.
    if not np.isfinite(f0_init) or not np.isfinite(width_init) or width_init <= 0:
        return np.nan, float("inf")

    n_edge = max(1, N // 10)
    offset_init = float(np.mean(np.concatenate([data[:n_edge], data[-n_edge:]])))
    amp_init = max(offset_init - float(np.min(data)), 1e-12)
    gamma_init = width_init

    try:
        popt, _ = curve_fit(
            _lorentzian_dip,
            detuning,
            data,
            p0=[f0_init, gamma_init, amp_init, offset_init],
            maxfev=3000,
            bounds=([-np.inf, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        )
        f0_fit, _gamma, amp_fit, _offset = popt
        if amp_fit <= 0 or not np.isfinite(f0_fit):
            return np.nan, float("inf")
        SS_res = float(np.sum((data - _lorentzian_dip(detuning, *popt)) ** 2))
        return f0_fit, SS_res / ((N - P) * amp_fit ** 2)
    except Exception:
        return np.nan, float("inf")


# =========================
# Shift-based fit logic
# =========================

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:

    powers = ds.power.values
    if len(powers) != 2:
        raise ValueError(
            "Shift-based resonator analysis requires exactly 2 power points."
        )

    P_low, P_high = float(powers[0]), float(powers[-1])
    detuning = ds.detuning.values
    qubit_names = ds.qubit.values
    n_qubits = len(qubit_names)
    n_powers = len(powers)

    f0_array = np.full((n_qubits, n_powers), np.nan)
    chi2_array = np.full((n_qubits, n_powers), np.inf)

    # Fit Lorentzian at each (qubit, power) combination.
    # power index 0 → low power (isel=0), power index 1 → high power (isel=-1).
    # peaks_dips inside _fit_lorentzian handles baseline removal, so a sloping
    # background does not shift the fitted f0.
    for i, q in enumerate(node.namespace["qubits"]):
        for j, p_isel in enumerate([0, -1]):
            data = ds.sel(qubit=q.name).isel(power=p_isel).IQ_abs.squeeze().values
            f0, chi2 = _fit_lorentzian(detuning, data)
            f0_array[i, j] = f0
            chi2_array[i, j] = chi2

    lorentz_f0 = xr.DataArray(
        f0_array,
        dims=["qubit", "power"],
        coords={"qubit": ds.qubit, "power": powers},
        attrs={"long_name": "Lorentzian fit center detuning", "units": "Hz"},
    )
    lorentz_chi2 = xr.DataArray(
        chi2_array,
        dims=["qubit", "power"],
        coords={"qubit": ds.qubit, "power": powers},
        attrs={"long_name": "Lorentzian fit residual chi-squared"},
    )

    freq_low = lorentz_f0.isel(power=0)    # fitted dip detuning at low power  (n_qubits,)
    freq_high = lorentz_f0.isel(power=-1)  # fitted dip detuning at high power (n_qubits,)
    freq_shift = freq_high - freq_low

    shift_threshold = node.parameters.frequency_shift_threshold_in_hz
    large_shift = abs(freq_shift) > shift_threshold
    # NaN freq_shift → large_shift is False → optimal_power = P_high, but success will be
    # False anyway due to no_nans / chi2_ok checks in _extract_relevant_fit_parameters.
    optimal_power = xr.where(large_shift, P_low, P_high)

    ds_fit = ds.assign(
        lorentz_f0=lorentz_f0,
        lorentz_chi2=lorentz_chi2,
    ).assign_coords(
        freq_shift=("qubit", freq_shift.data),
        freq_low=("qubit", freq_low.data),
        optimal_power=("qubit", optimal_power.data),
    )

    return _extract_relevant_fit_parameters(ds_fit, node)


# =========================
# Result extraction
# =========================

def _extract_relevant_fit_parameters(
    fit: xr.Dataset, node: QualibrationNode
):
    """Extract fit parameters and determine success based on punch-out detection."""

    base_freq = np.array(
        [q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    freq_low_abs = fit.freq_low + base_freq

    fit = fit.assign_coords(
        freq_low_abs=("qubit", freq_low_abs.data),
    )
    fit.freq_low_abs.attrs = {"long_name": "low-power resonator frequency", "units": "Hz"}

    chi2_threshold = getattr(node.parameters, "chi2_threshold", 3.0)

    # Both power points must have a valid Lorentzian fit (chi2 ≤ threshold).
    # lorentz_chi2 is inf for failed fits, so this correctly rejects them.
    chi2_ok = (fit.lorentz_chi2 <= chi2_threshold).all(dim="power").values

    no_nans = ~(np.isnan(fit.freq_shift.data) | np.isnan(fit.optimal_power.data))
    freq_in_range = np.abs(fit.freq_shift.data) < node.parameters.frequency_span_in_mhz * 1e6
    punchout_detected = np.abs(fit.freq_shift.data) > node.parameters.frequency_shift_threshold_in_hz

    success = no_nans & freq_in_range & punchout_detected & chi2_ok
    fit = fit.assign_coords(success=("qubit", success))

    fit_results = {}
    for q in fit.qubit.values:
        q_success = bool(fit.sel(qubit=q).success)
        q_shift = float(fit.freq_shift.sel(qubit=q))
        q_chi2_low = float(fit.lorentz_chi2.sel(qubit=q).isel(power=0).item())
        q_chi2_high = float(fit.lorentz_chi2.sel(qubit=q).isel(power=-1).item())

        fit_results[q] = FitParameters(
            success=q_success,
            resonator_frequency=float(fit.freq_low_abs.sel(qubit=q)),
            frequency_shift=q_shift,
            optimal_power=float(fit.optimal_power.sel(qubit=q)),
            freq_low_abs=float(fit.freq_low_abs.sel(qubit=q)),
            chi2_low=q_chi2_low,
            chi2_high=q_chi2_high,
        )

    return fit, fit_results
