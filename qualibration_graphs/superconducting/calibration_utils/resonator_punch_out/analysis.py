import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from calibration_utils.error_codes import (
    ResonatorPunchOutErrorCode,
    ResonatorPunchOutCorrectiveAction,
)


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
    error_code: int = ResonatorPunchOutErrorCode.SUCCESS
    corrective_action: int = ResonatorPunchOutCorrectiveAction.NONE
    action_magnitude: float = 0.0
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
        error_code = ResonatorPunchOutErrorCode(fit_results[q].get("error_code", 0))
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {fit_results[q]['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.3f} MHz)\n"
        s_error = f"Error code: {error_code.name} ({error_code.value})\n"

        chi2_low = fit_results[q].get("chi2_low", float("nan"))
        chi2_high = fit_results[q].get("chi2_high", float("nan"))
        s_chi2 = ""
        if np.isfinite(chi2_low) or np.isfinite(chi2_high):
            s_chi2 = f"Lorentzian chi2: low={chi2_low:.3f}, high={chi2_high:.3f}\n"

        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"

        log_callable(s_qubit + s_error + s_power + s_freq + s_shift + s_chi2)


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

    Returns (f0_fit, chi2, gamma_fit, amp_fit, offset_fit) where:
      f0_fit  : fitted center detuning (Hz), or nan on failure.
      chi2    : SS_res / ((N-4)*amp²), or inf on failure.
      gamma_fit, amp_fit, offset_fit : remaining Lorentzian parameters (nan on failure),
        kept so the fitted curve can be reconstructed for plotting.

    Uses peaks_dips for the initial f0 estimate (baseline-subtracted, prominence-gated)
    so the warm start is robust to sloping baselines and low-SNR data.  If peaks_dips
    finds no prominent dip, returns nan/inf immediately — the caller interprets this
    as a failed fit and chi2 = inf, which correctly triggers a punch-out retry.
    """
    from qualibration_libs.analysis import peaks_dips
    import xarray as xr

    N = len(detuning)
    P = 4  # f0, gamma, amp, offset
    if N <= P:
        return np.nan, float("inf"), np.nan, np.nan, np.nan

    # Build a minimal DataArray so peaks_dips can operate on it.
    da = xr.DataArray(data, coords={"detuning": detuning}, dims=["detuning"])
    pd = peaks_dips(da, "detuning")
    f0_init = float(pd.position.item())
    width_init = float(pd.width.item())

    # peaks_dips returns nan when no prominent peak/dip is found.
    # At low SNR this is the correct outcome — flag as failed immediately.
    if not np.isfinite(f0_init) or not np.isfinite(width_init) or width_init <= 0:
        return np.nan, float("inf"), np.nan, np.nan, np.nan

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
        f0_fit, gamma_fit, amp_fit, offset_fit = popt
        if amp_fit <= 0 or not np.isfinite(f0_fit):
            return np.nan, float("inf"), np.nan, np.nan, np.nan
        SS_res = float(np.sum((data - _lorentzian_dip(detuning, *popt)) ** 2))
        return f0_fit, SS_res / ((N - P) * amp_fit ** 2), gamma_fit, amp_fit, offset_fit
    except Exception:
        return np.nan, float("inf"), np.nan, np.nan, np.nan


# =========================
# Bifurcation-based fit logic (dense power sweep, >2 points)
# =========================

def _first_threshold_crossing(diff_arr: np.ndarray, factor: float = 3.0) -> int:
    """Index of the first step where `diff_arr` exceeds `factor` times its median.

    Returns `len(diff_arr)` (one past the last valid step index) if the median is
    zero/non-finite or no step crosses the threshold — i.e. no bifurcation was
    observed within the swept range, so the whole range is treated as "safe".
    """
    if diff_arr.size == 0:
        return 0
    median = np.median(diff_arr)
    if not np.isfinite(median) or median == 0:
        return len(diff_arr)
    idxs = np.flatnonzero(diff_arr > factor * median)
    return int(idxs[0]) if idxs.size else len(diff_arr)


def _fit_bifurcation_dense(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Locate the punch-out bifurcation from a dense power sweep.

    For each power, the resonance is taken as the detuning of the magnitude minimum
    (argmin of IQ_abs_norm). The bifurcation is the first power step where the
    resonance detuning or phase jumps by more than 3x the median step size; the
    "safe" (linear-regime) operating power is taken 2 steps before that.
    """
    powers = ds.power.values
    detuning = ds.detuning.values
    n_powers = len(powers)
    qubits = node.namespace["qubits"]
    n_qubits = len(qubits)

    min_freq = np.full((n_qubits, n_powers), np.nan)
    min_phase = np.full((n_qubits, n_powers), np.nan)

    for i, q in enumerate(qubits):
        mag = ds.sel(qubit=q.name).IQ_abs_norm.transpose("power", "detuning").values
        phase = ds.sel(qubit=q.name).phase.transpose("power", "detuning").values
        idx = np.argmin(mag, axis=1)
        min_freq[i] = detuning[idx]
        min_phase[i] = phase[np.arange(n_powers), idx]

    base_freq = np.array([q.resonator.RF_frequency for q in qubits])

    bif_idx = np.zeros(n_qubits, dtype=int)
    safe_idx = np.zeros(n_qubits, dtype=int)
    bifurcation_found = np.zeros(n_qubits, dtype=bool)
    for i in range(n_qubits):
        freq_step = np.abs(np.diff(min_freq[i]))
        phase_step = np.abs(np.diff(min_phase[i]))
        idx_freq = _first_threshold_crossing(freq_step)
        idx_phase = _first_threshold_crossing(phase_step)
        bif_idx[i] = min(idx_freq, idx_phase)
        bifurcation_found[i] = bif_idx[i] < (n_powers - 1)
        safe_idx[i] = max(0, bif_idx[i] - 2)

    safe_power = powers[safe_idx]
    safe_freq_detuning = min_freq[np.arange(n_qubits), safe_idx]
    safe_freq_abs = safe_freq_detuning + base_freq
    freq_shift = min_freq[:, -1] - min_freq[:, 0]

    ds_fit = ds.assign(
        min_freq=xr.DataArray(min_freq, dims=["qubit", "power"], coords={"qubit": ds.qubit, "power": powers}),
        min_phase=xr.DataArray(min_phase, dims=["qubit", "power"], coords={"qubit": ds.qubit, "power": powers}),
    ).assign_coords(
        bif_idx=("qubit", bif_idx),
        safe_idx=("qubit", safe_idx),
        safe_power=("qubit", safe_power),
        safe_freq_abs=("qubit", safe_freq_abs),
    )

    fit_results = {}
    for i, q in enumerate(qubits):
        success = bool(bifurcation_found[i])
        error_code = (
            ResonatorPunchOutErrorCode.SUCCESS if success else ResonatorPunchOutErrorCode.NO_SHIFT_DETECTED
        )
        fit_results[q.name] = FitParameters(
            success=success,
            resonator_frequency=float(safe_freq_abs[i]),
            frequency_shift=float(freq_shift[i]),
            optimal_power=float(safe_power[i]),
            freq_low_abs=float(safe_freq_abs[i]),
            error_code=int(error_code),
        )

    ds_fit = ds_fit.assign_coords(success=("qubit", bifurcation_found))
    return ds_fit, fit_results


# =========================
# Shift-based fit logic (2 power points)
# =========================

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:

    powers = ds.power.values
    if len(powers) != 2:
        # Dense diagnostic sweep: locate the punch-out bifurcation directly
        # instead of comparing only two power points.
        return _fit_bifurcation_dense(ds, node)

    P_low, P_high = float(powers[0]), float(powers[-1])
    detuning = ds.detuning.values
    qubit_names = ds.qubit.values
    n_qubits = len(qubit_names)
    n_powers = len(powers)

    f0_array = np.full((n_qubits, n_powers), np.nan)
    chi2_array = np.full((n_qubits, n_powers), np.inf)
    gamma_array = np.full((n_qubits, n_powers), np.nan)
    amp_array = np.full((n_qubits, n_powers), np.nan)
    offset_array = np.full((n_qubits, n_powers), np.nan)

    # Fit Lorentzian at each (qubit, power) combination.
    # power index 0 → low power (isel=0), power index 1 → high power (isel=-1).
    # peaks_dips inside _fit_lorentzian handles baseline removal, so a sloping
    # background does not shift the fitted f0.
    for i, q in enumerate(node.namespace["qubits"]):
        for j, p_isel in enumerate([0, -1]):
            data = ds.sel(qubit=q.name).isel(power=p_isel).IQ_abs.squeeze().values
            f0, chi2, gamma, amp, offset = _fit_lorentzian(detuning, data)
            f0_array[i, j] = f0
            chi2_array[i, j] = chi2
            gamma_array[i, j] = gamma
            amp_array[i, j] = amp
            offset_array[i, j] = offset

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
    lorentz_gamma = xr.DataArray(
        gamma_array,
        dims=["qubit", "power"],
        coords={"qubit": ds.qubit, "power": powers},
        attrs={"long_name": "Lorentzian fit FWHM", "units": "Hz"},
    )
    lorentz_amp = xr.DataArray(
        amp_array,
        dims=["qubit", "power"],
        coords={"qubit": ds.qubit, "power": powers},
        attrs={"long_name": "Lorentzian fit dip amplitude"},
    )
    lorentz_offset = xr.DataArray(
        offset_array,
        dims=["qubit", "power"],
        coords={"qubit": ds.qubit, "power": powers},
        attrs={"long_name": "Lorentzian fit baseline offset"},
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
        lorentz_gamma=lorentz_gamma,
        lorentz_amp=lorentz_amp,
        lorentz_offset=lorentz_offset,
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
    qubit_list = fit.qubit.values.tolist()
    for q in fit.qubit.values:
        q_idx = qubit_list.index(q)
        q_success = bool(fit.sel(qubit=q).success)
        q_shift = float(fit.freq_shift.sel(qubit=q))
        q_abs_shift = abs(q_shift)
        q_chi2_ok = bool(chi2_ok[q_idx])
        q_no_nans = bool(no_nans[q_idx])
        q_freq_in_range = bool(freq_in_range[q_idx])
        q_chi2_low = float(fit.lorentz_chi2.sel(qubit=q).isel(power=0).item())
        q_chi2_high = float(fit.lorentz_chi2.sel(qubit=q).isel(power=-1).item())

        if q_success:
            error_code = ResonatorPunchOutErrorCode.SUCCESS
        elif not q_chi2_ok:
            # Lorentzian fit failed at one or both power points → noisy data
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif not q_no_nans:
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif not q_freq_in_range:
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif q_abs_shift < 1e3:
            error_code = ResonatorPunchOutErrorCode.NO_SHIFT_DETECTED
        else:
            error_code = ResonatorPunchOutErrorCode.SHIFT_BELOW_THRESHOLD

        fit_results[q] = FitParameters(
            success=q_success,
            resonator_frequency=float(fit.freq_low_abs.sel(qubit=q)),
            frequency_shift=q_shift,
            optimal_power=float(fit.optimal_power.sel(qubit=q)),
            freq_low_abs=float(fit.freq_low_abs.sel(qubit=q)),
            error_code=int(error_code),
            chi2_low=q_chi2_low,
            chi2_high=q_chi2_high,
        )

    return fit, fit_results
