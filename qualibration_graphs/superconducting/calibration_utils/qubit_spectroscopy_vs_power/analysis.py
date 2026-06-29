import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips, lorentzian_peak
from calibration_utils.error_codes import (
    QubitSpectroscopyErrorCode,
    QubitSpectroscopyCorrectiveAction,
)


@dataclass
class FitParameters:
    """Spectroscopy vs power fit results for a single qubit."""

    selected_power: float
    rough_qubit_frequency: float
    linewidth: float
    iw_angle: float
    success: bool
    over_saturated: bool = False
    error_code: int = QubitSpectroscopyErrorCode.SUCCESS
    corrective_action: int = QubitSpectroscopyCorrectiveAction.NONE
    action_magnitude: float = 0.0
    # x180/saturation power: derived from linewidth-doubling power + T_spec/T_pi scaling
    x180_power_dbm: float = float("nan")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts and add full RF frequency coordinate."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def _mask_blacklisted_detunings(ds, machine, freq_tolerance_hz=5e6, power_tolerance_dbm=3.0):
    """Set IQ_abs and I_rot to NaN near any blacklisted (qubit_freq_hz, drive_power_dbm) pair."""
    if machine is None or not hasattr(machine, "temp_calibration") or machine.temp_calibration is None:
        return ds
    power_vals = ds.power.values
    vars_to_mask = [v for v in ["IQ_abs", "I_rot"] if v in ds]
    if not vars_to_mask:
        return ds
    masked_arrays = {v: ds[v].copy() for v in vars_to_mask}
    for qubit_name in ds.qubit.values:
        try:
            bl_points = machine.temp_calibration[qubit_name].blacklisted_qubit_points
        except (KeyError, TypeError, AttributeError):
            bl_points = None
        if not bl_points:
            continue
        full_freq_q = ds.full_freq.sel(qubit=qubit_name).values
        for bl_freq, bl_power in bl_points:
            freq_mask = np.abs(full_freq_q - bl_freq) <= freq_tolerance_hz
            power_mask = np.abs(power_vals - bl_power) <= power_tolerance_dbm
            if not np.any(freq_mask) or not np.any(power_mask):
                continue
            to_null = np.outer(power_mask, freq_mask)
            keep = xr.DataArray(~to_null, dims=["power", "detuning"],
                                coords={"power": ds.power, "detuning": ds.detuning})
            for v in vars_to_mask:
                masked_arrays[v].loc[dict(qubit=qubit_name)] = (
                    masked_arrays[v].sel(qubit=qubit_name).where(keep)
                )
    return ds.assign(**masked_arrays)


def _peak_index(i_rot, min_height):
    y = np.asarray(i_rot)
    if np.all(np.isnan(y)):
        return -1
    baseline = np.nanmin(y)
    idx = int(np.nanargmax(y))
    if y[idx] - baseline < min_height:
        return -1
    return idx


def _apply_persistence_filter(peak_indices, detuning, lookahead, freq_tolerance_hz):
    """Remove isolated peaks that do not persist at higher power levels."""
    n_power = len(peak_indices)
    filtered = peak_indices.copy()
    for i in range(n_power):
        if peak_indices[i] < 0:
            continue
        n_higher = n_power - i - 1
        if n_higher == 0:
            continue
        n_to_check = min(lookahead, n_higher)
        freq_i = detuning[peak_indices[i]]
        found = any(
            peak_indices[j] >= 0 and abs(detuning[peak_indices[j]] - freq_i) <= freq_tolerance_hz
            for j in range(i + 1, i + 1 + n_to_check)
        )
        if not found:
            filtered[i] = -1
    return filtered


def _compute_fwhm_around_peak(detuning, signal, peak_idx):
    """Compute FWHM with sub-step accuracy via linear interpolation."""
    if peak_idx < 0:
        return np.nan
    x = np.asarray(detuning, dtype=float)
    y = np.asarray(signal, dtype=float)
    if np.all(np.isnan(y)):
        return np.nan
    y = y - np.nanmin(y)
    half_max = 0.5 * float(np.nanmax(y))
    above = y >= half_max
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    left_i, right_i = int(idx[0]), int(idx[-1])
    if left_i > 0 and not above[left_i - 1]:
        dy = y[left_i] - y[left_i - 1]
        left_x = (x[left_i - 1] + (half_max - y[left_i - 1]) / dy * (x[left_i] - x[left_i - 1])
                  if dy > 0 else x[left_i])
    else:
        left_x = x[left_i]
    if right_i < len(x) - 1 and not above[right_i + 1]:
        dy = y[right_i + 1] - y[right_i]
        right_x = (x[right_i] + (half_max - y[right_i]) / dy * (x[right_i + 1] - x[right_i])
                   if dy < 0 else x[right_i])
    else:
        right_x = x[right_i]
    return right_x - left_x


def _check_high_baseline(signal, fwhm0_hz, detuning_step):
    """Return True if signal is consistently elevated over more than 10x the intrinsic linewidth."""
    y = np.asarray(signal)
    if np.all(np.isnan(y)):
        return False
    baseline, peak = np.nanmin(y), np.nanmax(y)
    threshold = baseline + 0.2 * (peak - baseline)
    above_threshold = y >= threshold
    if not np.any(above_threshold):
        return False
    changes = np.diff(np.concatenate([[False], above_threshold, [False]]).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        return False
    return bool(np.max(ends - starts) * detuning_step > 10 * fwhm0_hz)


def _compute_chi2_lorentzian(i_rot_slice, detuning):
    """
    Fit a Lorentzian to a 1D I_rot slice and compute residual chi-squared:
        chi2 = SS_res / ((N - 4) * amplitude^2)
    chi2 <= 2 -> good fit; chi2 > 2 -> no clear peak.
    """
    try:
        da = xr.DataArray(i_rot_slice, dims=["detuning"], coords={"detuning": detuning})
        fit = peaks_dips(da, "detuning")
        amplitude = float(fit.amplitude.values)
        position = float(fit.position.values)
        width = float(fit.width.values)
        baseline = float(fit.base_line.mean().values)
        if not np.isfinite(amplitude) or amplitude <= 0:
            return float("inf")
        N, P = len(i_rot_slice), 4
        if N <= P:
            return float("inf")
        fitted = lorentzian_peak(detuning, amplitude, position, width / 2, baseline)
        SS_res = float(np.nansum((i_rot_slice - fitted) ** 2))
        return SS_res / ((N - P) * amplitude ** 2)
    except Exception:
        return float("inf")


def detect_qubit_by_gradient_score(
    data: np.ndarray,
    freq: np.ndarray,
) -> tuple:
    """
    Detect the most likely qubit frequency from a 2D spectroscopy vs power map
    using three complementary detectors combined via a normalized weighted sum.

    Algorithm:
    1.  Gaussian smoothing (sigma_power=1, sigma_freq=2) to reduce noise.
    2.  Remove row-wise DC offset (zero-mean each power trace).
    3.  Ridge detector   — vertical-ridge convolution kernel; highlights columns
                           that stand out from their immediate neighbours.
    4.  Persistence det. — counts how many power rows show a gradient magnitude
                           above 1σ at each frequency; filters single-row artefacts.
    5.  Variance det.    — variance across power; suppresses flat background.
    6.  Normalize each score to [0, 1].
    7.  Weighted sum: 0.4 × ridge + 0.4 × persistence + 0.2 × variance.
    8.  Suppress the first / last 5 frequency bins (edge artefacts from convolution).
    9.  Detected frequency = argmax(combined score).

    Design principle: use a weighted sum rather than multiplication so that
    no single weak detector can veto a strong combined signal.

    Args:
        data : 2D array (n_powers, n_freqs) — spectroscopy signal (any units).
        freq : 1D array (n_freqs,)          — RF frequencies in Hz.

    Returns:
        f_qubit           : float               — detected qubit frequency in Hz.
        combined_score    : ndarray (n_freqs,)  — final combined score.
        data_norm         : ndarray (n_powers, n_freqs) — smoothed, offset-subtracted.
        ridge_score_norm  : ndarray (n_freqs,)  — normalised ridge score.
        persist_score_norm: ndarray (n_freqs,)  — normalised persistence score.
    """
    data = np.asarray(data, dtype=float)
    freq = np.asarray(freq, dtype=float)

    # Step 1 — Gaussian smoothing (NaN-safe normalized convolution).
    # `gaussian_filter` does not ignore NaNs: a single NaN (e.g. a
    # blacklisted detuning/power point) poisons every pixel within the
    # kernel's support, which with sigma=1 on the power axis can wipe out
    # most of a map with only ~10 power points. Convolve the valid-data
    # mask alongside the data and divide back out so masked holes stay
    # local instead of spreading.
    valid = np.isfinite(data)
    data_filled = np.where(valid, data, 0.0)
    weight = gaussian_filter(valid.astype(float), sigma=[1, 2])
    data_smooth = gaussian_filter(data_filled, sigma=[1, 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        data_smooth = np.where(weight > 1e-8, data_smooth / weight, np.nan)

    # Step 2 — remove row-wise DC offset
    data_norm = data_smooth - np.nanmean(data_smooth, axis=1, keepdims=True)

    # Downstream detectors can't handle NaNs (convolve2d/gradient/var would
    # propagate them); fill any remaining masked pixels with 0 (already
    # the row mean) just for score computation, but keep `data_norm` itself
    # with NaNs so the plotted heatmap still shows the masked points as gaps.
    data_norm_safe = np.nan_to_num(data_norm, nan=0.0)

    # Step 3 — Ridge detector: vertical-ridge convolution kernel
    # Each column is compared to its immediate left/right neighbours;
    # a vertical stripe standing above background scores highly.
    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]], dtype=float)
    ridge_map = convolve2d(data_norm_safe, kernel, mode="same", boundary="symm")
    ridge_score_raw = np.sum(np.abs(ridge_map), axis=0)

    # Step 4 — Persistence detector: how many power rows show a strong gradient
    grad_f = np.gradient(data_norm_safe, axis=1)
    threshold = 1.0 * np.std(grad_f)
    persist_score_raw = np.sum(np.abs(grad_f) > threshold, axis=0).astype(float)

    # Step 5 — Variance detector: feature must vary with power
    var_score_raw = np.var(data_norm_safe, axis=0)

    # Step 6 — Normalize each score to [0, 1]
    def _norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if (hi - lo) > 1e-30 else np.zeros_like(s)

    ridge_score_norm  = _norm(ridge_score_raw)
    persist_score_norm = _norm(persist_score_raw)
    var_score_norm    = _norm(var_score_raw)

    # Step 7 — Weighted combination (sum, not product — robust to weak detectors)
    combined_score = (
        0.4 * ridge_score_norm
        + 0.4 * persist_score_norm
        + 0.2 * var_score_norm
    )

    # Step 8 — Suppress edge bins (convolution boundary artefacts)
    search_score = combined_score.copy()
    edge = 5
    search_score[:edge]  = 0.0
    search_score[-edge:] = 0.0

    # Step 9 — Detect
    f_qubit = float(freq[np.argmax(search_score)])

    return f_qubit, combined_score, data_norm, ridge_score_norm, persist_score_norm


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Qubit spectroscopy vs power analysis with PCA-based IQ rotation.

    1. Flatten all (power, detuning) IQ points per qubit and compute the 2×2
       covariance matrix.  The eigenvector of the largest eigenvalue gives the
       direction of maximum variance in the IQ plane → rotation angle θ.
       If PCA explained-variance ratio < 0.6 (nearly circular cloud) → no signal.
    2. Rotate: I_rot = I·cos θ + Q·sin θ,  Q_rot = −I·sin θ + Q·cos θ.
       Sign-correct so the dominant feature is a positive peak in I_rot.
    3. Detect the qubit frequency via gradient-score (primary) or per-power
       argmax (fallback).
    4. Select the drive power just below the onset of power broadening.
    5. Verify Lorentzian fit quality via chi-squared (hard failure if chi² > 2).
    6. On success, store the rotation angle for saving to the QUAM state.
    """
    p = node.parameters
    machine = node.machine

    ds = _mask_blacklisted_detunings(ds, machine)

    # ── IQ rotation via PCA ────────────────────────────────────────────────────
    # Flatten all (power, detuning) IQ points per qubit and find the direction
    # of maximum variance in the IQ plane.  This direction is qubit-independent
    # (no assumption about where the qubit sits in frequency) and is robust to
    # features that appear at any power level.
    #
    # The PCA eigenvector sign is arbitrary — a subsequent flip step ensures the
    # dominant spectral feature appears as a positive peak in I_rot.
    angle_list, pca_var_ratio_list = [], []
    for q in ds.qubit.values:
        I_flat = ds.I.sel(qubit=q).values.ravel()
        Q_flat = ds.Q.sel(qubit=q).values.ravel()
        valid = np.isfinite(I_flat) & np.isfinite(Q_flat)
        I_v, Q_v = I_flat[valid], Q_flat[valid]
        if len(I_v) < 4:
            angle_list.append(0.0)
            pca_var_ratio_list.append(0.5)
            continue
        X = np.array([I_v, Q_v])                          # 2 × N
        C = np.cov(X)                                      # 2 × 2 covariance
        eigenvalues, eigenvectors = np.linalg.eigh(C)     # ascending eigenvalues
        v = eigenvectors[:, -1]                            # principal eigenvector
        angle_list.append(float(np.arctan2(v[1], v[0])))
        total = float(np.sum(eigenvalues))
        pca_var_ratio_list.append(float(eigenvalues[-1] / total) if total > 0 else 0.5)

    angle = xr.DataArray(angle_list, dims=["qubit"], coords={"qubit": ds.qubit})
    pca_variance_ratio = xr.DataArray(
        pca_var_ratio_list, dims=["qubit"], coords={"qubit": ds.qubit},
        attrs={"long_name": "PCA explained variance ratio (principal component)"},
    )
    ds["pca_variance_ratio"] = pca_variance_ratio

    ds["iw_angle"] = angle
    ds["I_rot"] = ds.I * np.cos(angle) + ds.Q * np.sin(angle)
    ds["Q_rot"] = -ds.I * np.sin(angle) + ds.Q * np.cos(angle)

    # Sign fix: PCA eigenvector direction is arbitrary.  Ensure the dominant
    # spectral feature appears as a positive peak (not a dip) in I_rot.
    # Compare the global positive and negative excursions from the mean:
    # if the negative excursion dominates, the rotation is off by π → flip.
    i_rot_dev = ds["I_rot"] - ds["I_rot"].mean(dim=["power", "detuning"])
    pos_peak = i_rot_dev.max(dim=["power", "detuning"])
    neg_peak = (-i_rot_dev).max(dim=["power", "detuning"])
    needs_flip = neg_peak > pos_peak
    if needs_flip.any():
        flipped_angle = xr.where(needs_flip, angle + np.pi, angle)
        ds["iw_angle"] = flipped_angle
        ds["I_rot"] = ds.I * np.cos(flipped_angle) + ds.Q * np.sin(flipped_angle)
        ds["Q_rot"] = -ds.I * np.sin(flipped_angle) + ds.Q * np.cos(flipped_angle)

    # Re-apply blacklist masking now that I_rot exists
    ds = _mask_blacklisted_detunings(ds, machine)

    # Select working signal for peak detection
    signal_source = getattr(p, "signal_source", "I_rot")
    if signal_source == "IQ_abs":
        raw_signal = ds.IQ_abs
    elif signal_source == "phase":
        raw_signal = ds.phase
    else:
        raw_signal = ds.I_rot

    # Ensure working signal has dominant feature as a positive peak.
    i_sig_dev = raw_signal - raw_signal.mean(dim=["power", "detuning"])
    sig_pos = i_sig_dev.max(dim=["power", "detuning"])
    sig_neg = (-i_sig_dev).max(dim=["power", "detuning"])
    sig_needs_flip = sig_neg > sig_pos
    if sig_needs_flip.any():
        raw_signal = xr.where(sig_needs_flip, -raw_signal, raw_signal)

    ds["working_signal"] = raw_signal

    # ── Multi-detector frequency detection (primary estimator) ────────────────
    # Three complementary detectors (ridge, persistence, variance) are combined
    # via a normalized weighted sum.  Running on the full 2D map is much more
    # robust than per-power argmax.
    grad_score_list, data_norm_list, grad_freq_list = [], [], []
    ridge_score_list, persist_score_list = [], []
    for q in ds.qubit.values:
        sig_2d = ds.working_signal.sel(qubit=q).transpose("power", "detuning").values  # (n_power, n_freq)
        full_freq_q = ds.full_freq.sel(qubit=q).values                                # (n_freq,)
        fq, score_q, data_norm_q, ridge_q, persist_q = detect_qubit_by_gradient_score(sig_2d, full_freq_q)
        grad_freq_list.append(fq)
        grad_score_list.append(score_q)
        data_norm_list.append(data_norm_q)
        ridge_score_list.append(ridge_q)
        persist_score_list.append(persist_q)

    ds["gradient_score"] = xr.DataArray(
        np.array(grad_score_list),
        dims=["qubit", "detuning"],
        coords={"qubit": ds.qubit, "detuning": ds.detuning},
        attrs={"long_name": "Combined score (ridge+persistence+variance)", "units": "a.u."},
    )
    ds["ridge_score"] = xr.DataArray(
        np.array(ridge_score_list),
        dims=["qubit", "detuning"],
        coords={"qubit": ds.qubit, "detuning": ds.detuning},
        attrs={"long_name": "Ridge score (normalized)", "units": "a.u."},
    )
    ds["persistence_score"] = xr.DataArray(
        np.array(persist_score_list),
        dims=["qubit", "detuning"],
        coords={"qubit": ds.qubit, "detuning": ds.detuning},
        attrs={"long_name": "Persistence score (normalized)", "units": "a.u."},
    )
    ds["data_norm"] = xr.DataArray(
        np.array(data_norm_list),
        dims=["qubit", "power", "detuning"],
        coords={"qubit": ds.qubit, "power": ds.power, "detuning": ds.detuning},
        attrs={"long_name": "Smoothed, offset-subtracted signal"},
    )
    ds["gradient_frequency"] = xr.DataArray(
        np.array(grad_freq_list),
        dims=["qubit"],
        coords={"qubit": ds.qubit},
        attrs={"long_name": "Detected qubit frequency (multi-detector)", "units": "Hz"},
    )

    # Peak finding on the working signal
    sig_baseline = ds.working_signal.min(dim=["power", "detuning"])
    sig_max = ds.working_signal.max(dim=["power", "detuning"])
    min_peak_height = 0.1 * (sig_max - sig_baseline)

    peak_index = xr.apply_ufunc(
        _peak_index, ds.working_signal, min_peak_height,
        input_core_dims=[["detuning"], []],
        vectorize=True, output_dtypes=[int],
    )
    ds["peak_index"] = peak_index

    if int(p.peak_persistence_lookahead) > 0:
        peak_index = xr.apply_ufunc(
            _apply_persistence_filter, peak_index, ds.detuning,
            kwargs={"lookahead": int(p.peak_persistence_lookahead),
                    "freq_tolerance_hz": float(p.peak_persistence_freq_tolerance_hz)},
            input_core_dims=[["power"], ["detuning"]],
            output_core_dims=[["power"]],
            vectorize=True, output_dtypes=[int],
        )
        ds["peak_index"] = peak_index

    ds["peak_height"] = xr.where(peak_index >= 0, ds.working_signal.isel(detuning=peak_index), np.nan)

    linewidth = xr.apply_ufunc(
        _compute_fwhm_around_peak, ds.detuning, ds.working_signal, peak_index,
        input_core_dims=[["detuning"], ["detuning"], []],
        vectorize=True, output_dtypes=[float],
    )
    ds["linewidth"] = linewidth

    # FWHM0 per qubit: median of the 3 lowest-power valid half-max FWHMs.
    # Used to detect when linewidth has doubled (broadening onset, Omega^2*T1*T2*~3).
    fwhm0_per_qubit = {}
    for q in ds.qubit.values:
        lw_q = ds.linewidth.sel(qubit=q)
        pi_q = ds.peak_index.sel(qubit=q)
        valid_lw = lw_q.where(pi_q >= 0).dropna("power")
        if len(valid_lw) >= 1:
            fwhm0_per_qubit[q] = float(np.nanmedian(valid_lw.sortby("power").values[:3]))
        else:
            fwhm0_per_qubit[q] = float(np.nanmin(lw_q.values)) if not np.all(np.isnan(lw_q.values)) else 1e6

    ds["fwhm0"] = xr.DataArray(
        [fwhm0_per_qubit[q] for q in ds.qubit.values],
        dims=["qubit"], coords={"qubit": ds.qubit},
        attrs={"long_name": "Low-power linewidth FWHM0", "units": "Hz"},
    )

    # Power selection: use the linewidth threshold parameter.
    # The highest power where linewidth stays at or below linewidth_threshold_hz is
    # chosen as the spectroscopy operating point.  The same power (before the safety
    # buffer) is the reference for x180/saturation amplitude scaling.
    valid_power = (ds.peak_index >= 0) & (ds.linewidth <= p.linewidth_threshold_hz)
    primary_selected = ds.linewidth.where(valid_power).idxmax(dim="power", skipna=True)
    fallback_selected = ds.linewidth.where(ds.peak_index >= 0).idxmin(dim="power", skipna=True)
    used_fallback = ~np.isfinite(primary_selected)
    # p_threshold: the threshold-crossing power (before safety buffer).
    # Used as the reference for x180/saturation power scaling.
    p_threshold = primary_selected.where(~used_fallback, other=fallback_selected)
    ds["p_threshold"] = p_threshold
    selected_power = p_threshold - p.power_buffer_db
    ds["selected_power"] = selected_power
    ds["used_fallback_power"] = used_fallback

    # Primary frequency estimate: gradient-score result (robust across all powers).
    # Fallback: argmax of working_signal at the selected power level.
    def _peak_frequency_fallback(full_freq, signal_data, power, target_power):
        if np.isnan(target_power):
            return np.nan
        diff = np.abs(power - target_power)
        if np.all(np.isnan(diff)):
            return np.nan
        idx = int(np.nanargmin(diff))
        spectrum = signal_data[idx]
        return full_freq[int(np.nanargmax(spectrum))] if not np.all(np.isnan(spectrum)) else np.nan

    fallback_freq = xr.apply_ufunc(
        _peak_frequency_fallback, ds.full_freq, ds.working_signal, ds.power, ds.selected_power,
        input_core_dims=[["detuning"], ["power", "detuning"], ["power"], []],
        vectorize=True, output_dtypes=[float],
    )
    # Use gradient frequency when finite; otherwise fall back
    rough_freq = ds["gradient_frequency"].where(
        np.isfinite(ds["gradient_frequency"]), other=fallback_freq
    )
    ds["rough_qubit_frequency"] = rough_freq

    detuning_step = float(np.diff(ds.detuning.values).mean())
    fit_results = {}

    for q in ds.qubit.values:
        qubit_data = ds.sel(qubit=q)

        # PCA variance ratio < 0.6 → IQ cloud is nearly circular → no dominant
        # qubit-induced direction → treat as "no peak found".
        no_phase_peak = bool(float(pca_variance_ratio.sel(qubit=q)) < 0.6)

        fwhm0_q = fwhm0_per_qubit[q]
        over_saturated = bool(all(
            _check_high_baseline(
                qubit_data.working_signal.isel(power=pi).values,
                fwhm0_q, detuning_step,
            )
            for pi in range(len(qubit_data.power))
        ))

        success = (
            not no_phase_peak
            and bool(np.isfinite(qubit_data.selected_power))
            and bool(np.isfinite(qubit_data.rough_qubit_frequency))
        )

        chi2 = float("nan")
        if success:
            p_sel = float(qubit_data.selected_power)
            p_idx = int(np.argmin(np.abs(qubit_data.power.values - p_sel)))
            signal_slice = qubit_data.working_signal.isel(power=p_idx).values
            chi2 = _compute_chi2_lorentzian(signal_slice, ds.detuning.values)
            if chi2 > 2.0:
                success = False

        used_fallback_q = bool(ds["used_fallback_power"].sel(qubit=q).item())

        if no_phase_peak:
            error_code = QubitSpectroscopyErrorCode.NO_PEAK_FOUND
        elif success and (over_saturated or used_fallback_q):
            error_code = QubitSpectroscopyErrorCode.OVER_SATURATED_SUCCESS
        elif success:
            error_code = QubitSpectroscopyErrorCode.SUCCESS
        elif over_saturated:
            error_code = QubitSpectroscopyErrorCode.OVER_SATURATED
        else:
            error_code = QubitSpectroscopyErrorCode.NO_PEAK_FOUND

        iw_angle_q = float(ds["iw_angle"].sel(qubit=q).item()) if success else float("nan")

        # x180/saturation power via inverse-proportionality scaling (no T2* needed).
        # At the threshold power (where linewidth reaches linewidth_threshold_hz), the
        # spectroscopy saturation pulse of T_spec ns is the reference.  Since Omega = kappa * A
        # and A * T_pi = const (for a fixed rotation angle):
        #   P_x180 = P_threshold + 20 * log10(T_spec / T_pi_target)
        try:
            op_len_ns = getattr(p, "operation_len_in_ns", None)
            if op_len_ns is None:
                qubit_obj = node.machine.qubits[q]
                op_len_ns = float(qubit_obj.xy.operations[getattr(p, "operation", "saturation")].length)
            T_spec_ns = float(op_len_ns)
        except Exception:
            T_spec_ns = float("nan")

        T_pi_target_ns = float(getattr(p, "rabi_sweep_max_duration_ns", 300.0)) / (
            2.0 * max(1, int(getattr(p, "rabi_target_periods", 3)))
        )
        p_threshold_q = float(ds["p_threshold"].sel(qubit=q).values)

        if np.isfinite(p_threshold_q) and np.isfinite(T_spec_ns) and T_spec_ns > 0 and T_pi_target_ns > 0:
            x180_power_q = p_threshold_q + 20.0 * np.log10(T_spec_ns / T_pi_target_ns)
            # Enforce hardware limits: Octave gain range [-20, +20 dB].
            # volts2dBm(max_amplitude_opx) + 20 is the max deliverable power;
            # volts2dBm(min_amplitude_opx) - 20 is the min detectable power.
            try:
                from qualang_tools.units import unit as _u_cls
                _u = _u_cls(coerce_to_integer=True)
                max_amp = float(getattr(p, "max_amplitude_opx", 0.5))
                min_amp = float(getattr(p, "min_amplitude_opx", 0.001))
                max_hw_dbm = _u.volts2dBm(max_amp) + 20.0
                min_hw_dbm = _u.volts2dBm(min_amp) - 20.0
                if x180_power_q > max_hw_dbm:
                    x180_power_q = max_hw_dbm
                elif x180_power_q < min_hw_dbm:
                    x180_power_q = float("nan")
            except Exception:
                pass
        else:
            x180_power_q = float("nan")

        fit_results[q] = FitParameters(
            selected_power=float(qubit_data.selected_power.values),
            rough_qubit_frequency=float(qubit_data.rough_qubit_frequency.values),
            linewidth=float(qubit_data.linewidth.min(dim="power").values),
            iw_angle=iw_angle_q,
            success=success,
            over_saturated=over_saturated,
            error_code=int(error_code),
            x180_power_dbm=x180_power_q,
        )

    return ds, fit_results


def log_fitted_results(fit_results: Dict[str, Dict], log_callable=None):
    """Log the fitted results for each qubit."""
    if log_callable is None:
        log_callable = print
    for qubit_name, result in fit_results.items():
        success = result.get("success", False)
        over_saturated = result.get("over_saturated", False)
        error_code = QubitSpectroscopyErrorCode(result.get("error_code", 0))
        x180_pwr = result.get("x180_power_dbm", float("nan"))
        x180_str = f"{x180_pwr:.1f} dBm" if np.isfinite(x180_pwr) else "N/A"

        if success:
            status = "SUCCESS" + (" (OVER-SATURATED)" if over_saturated else "")
            log_callable(
                f"[{qubit_name}] {status} - Error code: {error_code.name} ({error_code.value})\n"
                f"  Selected power:  {result['selected_power']:.2f} dBm\n"
                f"  Qubit frequency: {result['rough_qubit_frequency'] / 1e9:.6f} GHz\n"
                f"  Min linewidth:   {result['linewidth'] / 1e6:.2f} MHz\n"
                f"  IW angle:        {result.get('iw_angle', float('nan')):.4f} rad\n"
                f"  x180/sat power:  {x180_str}  (linewidth-doubling + T_spec/T_pi scaling)"
            )
        else:
            log_callable(
                f"[{qubit_name}] FAILED - Error code: {error_code.name} ({error_code.value})\n"
                f"  No valid peak found in phase / I_rot"
            )
