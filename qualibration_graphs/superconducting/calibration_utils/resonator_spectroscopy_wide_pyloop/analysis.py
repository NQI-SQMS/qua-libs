"""Analysis utilities for wide-band resonator spectroscopy.

The node feeds in a dataset whose IQ traces span an absolute RF range
(potentially several hundred MHz to several GHz) covered by multiple
MW-FEM LO segments. The analysis:

1. Converts IQ to voltage and derives amplitude / phase.
2. Detects all dip candidates above a prominence threshold per qubit.
3. Greedily assigns each qubit to the candidate nearest its current
   `resonator.RF_frequency` (within `proximity_tolerance_mhz`).
4. Refits a narrow-window Lorentzian-with-linear-background around each
   assigned dip for a clean f0 / FWHM.

Qubits with no nearby candidate are marked failed and the full candidate
list is exposed so the user can re-fit specific qubits with a manual
window via `re_fit_resonators` / `re_fit_centers_ghz` / `re_fit_span_mhz`
(re-fit runs from a saved dataset via `load_data_id`, no hardware re-run).
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


# ---------------------------------------------------------------------------
# Fitting model
# ---------------------------------------------------------------------------

def lorentzian_dip_linbg(f, f0, fwhm, amp, bg0, bg1):
    """Inverted Lorentzian with linear background.

    R(f) = [bg0 + bg1*(f - fc)] - amp / [1 + ((f - f0)/(fwhm/2))^2]

    `fc` is the centre of the frequency array so that `bg1` has units of V/Hz
    without absorbing a large offset into `bg0`.
    """
    fc = f.mean()
    return (bg0 + bg1 * (f - fc)) - amp / (1 + ((f - f0) / (fwhm / 2)) ** 2)


# ---------------------------------------------------------------------------
# Dip candidate detection
# ---------------------------------------------------------------------------

@dataclass
class DipCandidate:
    """A candidate resonator dip found in the wide trace."""

    rf_hz: float
    prominence_db: float
    fwhm_hz: float
    index: int


def find_dip_candidates(
    rf_hz: np.ndarray,
    amplitude: np.ndarray,
    min_prominence_db: float = 2.0,
    smooth_window: int = 21,
) -> List[DipCandidate]:
    """Find all dip candidates in a wide amplitude trace.

    Operates in dB to make prominence thresholds physically meaningful.
    Returns candidates sorted by descending prominence.
    """
    rf_hz = np.asarray(rf_hz, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    # Guard against zeros before dB conversion
    amp_clip = np.maximum(amplitude, 1e-15)
    amp_db = 20.0 * np.log10(amp_clip)

    win = min(smooth_window, len(amp_db) // 3 * 2 - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    try:
        amp_db_smooth = savgol_filter(amp_db, win, 3)
    except Exception:
        amp_db_smooth = amp_db.copy()

    # Dips are peaks of the inverted dB trace
    peaks, props = find_peaks(-amp_db_smooth, prominence=min_prominence_db)
    if len(peaks) == 0:
        return []

    widths_samples, *_ = peak_widths(-amp_db_smooth, peaks, rel_height=0.5)
    # Convert sample width to Hz using the local sample spacing at each peak
    df_per_sample = np.gradient(rf_hz)
    fwhm_hz = widths_samples * df_per_sample[peaks]

    candidates = [
        DipCandidate(
            rf_hz=float(rf_hz[p]),
            prominence_db=float(props["prominences"][i]),
            fwhm_hz=float(fwhm_hz[i]),
            index=int(p),
        )
        for i, p in enumerate(peaks)
    ]
    candidates.sort(key=lambda c: c.prominence_db, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Proximity assignment (greedy nearest-neighbour)
# ---------------------------------------------------------------------------

def assign_dips_to_qubits(
    qubit_names: List[str],
    qubit_init_rf_hz: List[float],
    candidates: List[DipCandidate],
    tolerance_hz: float,
) -> Tuple[Dict[str, Optional[DipCandidate]], List[DipCandidate]]:
    """Greedily assign each qubit to its closest unassigned candidate.

    Walks qubits in ascending order of |init_rf_hz - nearest_candidate_rf_hz|
    so that the easiest-to-place qubits go first, reducing conflicts. Returns:

        assignments: dict[qubit_name -> DipCandidate or None]
        leftovers:   list[DipCandidate] of unassigned candidates
    """
    remaining = list(candidates)
    assignments: Dict[str, Optional[DipCandidate]] = {n: None for n in qubit_names}

    # Build (qubit_name, init_rf, distance_to_nearest_remaining) tuples and
    # process in ascending distance order. After each assignment, re-evaluate
    # distances for the remaining qubits (cheap for small N).
    pending = list(zip(qubit_names, qubit_init_rf_hz))
    while pending and remaining:
        # For each pending qubit, find its nearest remaining candidate
        best = None  # (qubit_idx_in_pending, candidate_idx_in_remaining, distance)
        for qi, (qname, qrf) in enumerate(pending):
            dists = [abs(c.rf_hz - qrf) for c in remaining]
            ci = int(np.argmin(dists))
            d = dists[ci]
            if best is None or d < best[2]:
                best = (qi, ci, d)
        qi, ci, d = best
        qname, qrf = pending.pop(qi)
        cand = remaining.pop(ci)
        if d <= tolerance_hz:
            assignments[qname] = cand
        else:
            # Closest available candidate is outside tolerance: leave qubit
            # unassigned and return the candidate to the pool for other qubits
            # to consider.
            remaining.append(cand)
            # If the closest is too far for this qubit, it'll also be too far
            # for any further iteration of THIS qubit (no new candidates appear).
            # So just leave it unassigned and continue with the others.
            assignments[qname] = None

    return assignments, remaining


# ---------------------------------------------------------------------------
# Narrow-window fitter (re-used by both wide-scan path and re-fit overrides)
# ---------------------------------------------------------------------------

def find_best_dip(smoothed, edge_fraction=0.04, min_prominence_fraction=0.03):
    """Return (dip_idx, is_edge_only) for the deepest inner local dip."""
    N = len(smoothed)
    edge = int(N * edge_fraction)
    A_rng = smoothed.max() - smoothed.min()
    peaks_neg, _ = find_peaks(-smoothed, prominence=A_rng * min_prominence_fraction)
    inner = [p for p in peaks_neg if edge <= p <= N - 1 - edge]
    if inner:
        return int(inner[np.argmin(smoothed[inner])]), False
    return int(np.argmin(smoothed)), True


def fit_resonator(
    freqs,
    amplitude,
    *,
    override_center_hz: Optional[float] = None,
    override_span_hz: Optional[float] = None,
    window_fwhm_factor: float = 4.0,
    min_window_mhz: float = 5.0,
    detrend_window_mhz: float = 10.0,
    max_fwhm_mhz: float = 15.0,
    r2_threshold: float = 0.85,
    min_contrast: float = 0.05,
    edge_fraction: float = 0.04,
    smooth_window: int = 11,
):
    """Lorentzian-with-linear-background fitter for a single resonator dip.

    When `override_center_hz` and `override_span_hz` are provided the data is
    pre-sliced to [center - span/2, center + span/2] before fitting. This is
    what the wide-scan flow uses to refit a clean Lorentzian around each
    assigned dip, and what the re-fit override path uses for manual windows.
    """
    _nan5 = np.full(5, np.nan)
    result = dict(
        f0=np.nan, fwhm=np.nan, r2=np.nan, success=False,
        popt=_nan5.copy(), edge_dip=False, contrast=np.nan, dip_idx=-1,
    )

    freqs = np.asarray(freqs, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)

    if override_center_hz is not None and override_span_hz is not None:
        half = override_span_hz / 2.0
        mask_ov = (freqs >= override_center_hz - half) & (freqs <= override_center_hz + half)
        if mask_ov.sum() >= 8:
            freqs = freqs[mask_ov]
            amplitude = amplitude[mask_ov]

    span_hz = freqs[-1] - freqs[0]
    N = len(freqs)
    edge_s = int(N * edge_fraction)

    win = min(smooth_window, N // 3 * 2 - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    try:
        smoothed = savgol_filter(amplitude, win, 3)
    except Exception:
        smoothed = amplitude.copy()

    dip_idx, is_edge = find_best_dip(smoothed, edge_fraction=edge_fraction)
    result["dip_idx"] = dip_idx
    result["edge_dip"] = is_edge
    if is_edge or dip_idx < edge_s or dip_idx > N - 1 - edge_s:
        return result

    f0_init = freqs[dip_idx]
    A_raw = smoothed.max() - smoothed.min()
    median_val = np.median(smoothed)
    if median_val <= 0 or A_raw / median_val < min_contrast:
        return result

    detr_half = detrend_window_mhz * 1e6 / 2.0
    detr_mask = (freqs >= f0_init - detr_half) & (freqs <= f0_init + detr_half)
    if detr_mask.sum() < 8:
        detr_mask = np.ones(N, dtype=bool)
    f_d = freqs[detr_mask]
    a_d = amplitude[detr_mask]
    b0d = (a_d[0] + a_d[-1]) / 2.0
    b1d = (a_d[-1] - a_d[0]) / (f_d[-1] - f_d[0]) if len(f_d) > 1 else 0.0
    a_detr = a_d - (b0d + b1d * (f_d - f_d.mean()))
    di2 = int(np.argmin(a_detr))
    Ar2 = a_detr.max() - a_detr.min()
    hd2 = a_detr[di2] + Ar2 / 2.0
    lc2 = np.where(a_detr[: di2 + 1] >= hd2)[0]
    rc2 = np.where(a_detr[di2:] >= hd2)[0]
    if len(lc2) and len(rc2):
        fwhm_init = f_d[di2 + rc2[0]] - f_d[lc2[-1]]
    else:
        hd = smoothed[dip_idx] + A_raw / 2.0
        lc = np.where(smoothed[: dip_idx + 1] >= hd)[0]
        rc = np.where(smoothed[dip_idx:] >= hd)[0]
        fwhm_init = (
            freqs[dip_idx + rc[0]] - freqs[lc[-1]]
            if (len(lc) and len(rc))
            else span_hz * 0.05
        )

    half_win = max(fwhm_init * window_fwhm_factor / 2.0, min_window_mhz * 1e6 / 2.0)
    fit_mask = (freqs >= f0_init - half_win) & (freqs <= f0_init + half_win)
    if fit_mask.sum() < 8:
        fit_mask = np.ones(N, dtype=bool)

    f_win = freqs[fit_mask]
    a_win = amplitude[fit_mask]

    b0 = (a_win[0] + a_win[-1]) / 2.0
    b1 = (a_win[-1] - a_win[0]) / (f_win[-1] - f_win[0]) if len(f_win) > 1 else 0.0
    Ar_win = a_win.max() - a_win.min()
    p0 = [f0_init, fwhm_init, Ar_win, b0, b1]
    bounds = (
        [f_win.min(), 0, 0, 0, -np.inf],
        [f_win.max(), span_hz, A_raw * 5, b0 * 3 if b0 > 0 else 1.0, np.inf],
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lorentzian_dip_linbg, f_win, a_win,
                p0=p0, bounds=bounds, maxfev=10000,
            )
    except Exception:
        return result

    f0_fit, fwhm_fit, amp_fit, bg0_fit, bg1_fit = popt
    y_pred = lorentzian_dip_linbg(f_win, *popt)
    ss_res = np.sum((a_win - y_pred) ** 2)
    ss_tot = np.sum((a_win - a_win.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    bg_at_f0 = bg0_fit + bg1_fit * (f0_fit - f_win.mean())
    fitted_contrast = amp_fit / bg_at_f0 if bg_at_f0 > 0 else 0.0

    success = (
        r2 >= r2_threshold
        and fwhm_fit <= max_fwhm_mhz * 1e6
        and fwhm_fit / span_hz <= 0.50
        and fwhm_fit > 0
        and fitted_contrast >= min_contrast
        and f_win.min() <= f0_fit <= f_win.max()
    )

    result.update(
        f0=f0_fit, fwhm=fwhm_fit, r2=r2,
        success=success, popt=np.array(popt),
        contrast=fitted_contrast,
    )
    return result


# ---------------------------------------------------------------------------
# FitParameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitParameters:
    """Per-qubit fitted resonator parameters."""

    frequency: float
    fwhm: float
    r2: float
    success: bool
    # Wide-scan extras (per-qubit context useful for debugging / re-fit prompts)
    init_rf_hz: float = float("nan")
    assigned_dip_rf_hz: float = float("nan")
    assignment_distance_hz: float = float("nan")
    n_candidates_total: int = 0
    n_candidates_leftover: int = 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted results for all qubits, including assignment context."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS!" if res["success"] else "FAIL!"
        f_ghz = res["frequency"] / 1e9 if not np.isnan(res["frequency"]) else float("nan")
        fwhm_khz = res["fwhm"] / 1e3 if not np.isnan(res["fwhm"]) else float("nan")
        init_ghz = res.get("init_rf_hz", float("nan")) / 1e9
        dist_mhz = res.get("assignment_distance_hz", float("nan")) / 1e6
        log_callable(
            f"Results for qubit {q}:  {status}\n"
            f"\tResonator frequency: {f_ghz:.4f} GHz | "
            f"FWHM: {fwhm_khz:.1f} kHz | R²: {res['r2']:.3f}\n"
            f"\tinit: {init_ghz:.4f} GHz | assigned distance: {dist_mhz:+.1f} MHz | "
            f"candidates total: {res.get('n_candidates_total', 0)}"
        )


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ to voltage and add amplitude/phase for the wide trace.

    Expects `ds` to have coords (`qubit`, `RF_frequency`) and data vars `I`, `Q`.
    Adds `IQ_abs` and `phase` data vars. The raw unwrapped phase is stored as-is
    (still carrying the cable-delay slope) so downstream re-analysis can apply
    its own detrending; both the wide-phase plot (global linear) and the local
    detrended-phase plot (degree-3 background) handle their own subtraction.
    """
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "RF_frequency", subtract_slope_flag=False)
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the wide trace: find candidates, assign, refit narrow window per qubit.

    Honors the re-fit override parameters (`re_fit_resonators`,
    `re_fit_centers_ghz`, `re_fit_span_mhz`) for manual per-qubit windows.

    Returns
    -------
    ds_fit : xr.Dataset
        Per-qubit fit results (f0, fwhm, r2, success, popt) plus the candidate
        list saved as `candidates_rf_hz` / `candidates_prominence_db` ragged
        per-qubit arrays (padded with NaN).
    fit_results : dict[str, FitParameters]
    """
    qubits = node.namespace["qubits"]
    params = node.parameters

    # Per-qubit override lookup
    overrides: Dict[str, Dict] = {}
    re_fit_names = params.re_fit_resonators or []
    re_fit_centers = params.re_fit_centers_ghz or []
    re_fit_spans = params.re_fit_span_mhz or []
    for name, center_ghz, span_mhz in zip(re_fit_names, re_fit_centers, re_fit_spans):
        overrides[name] = {
            "center_hz": center_ghz * 1e9,
            "span_hz": span_mhz * 1e6,
        }

    rf_hz_axis = ds.RF_frequency.values
    qubit_names = [q.name for q in qubits]
    qubit_init_rf = [float(q.resonator.RF_frequency) for q in qubits]

    # --- Step 1: per-qubit candidate detection on the wide amplitude trace ---
    candidates_by_qubit: Dict[str, List[DipCandidate]] = {}
    for q in qubits:
        amp_q = ds.sel(qubit=q.name).IQ_abs.values
        candidates_by_qubit[q.name] = find_dip_candidates(
            rf_hz_axis, amp_q,
            min_prominence_db=params.min_dip_prominence_db,
        )

    # --- Step 2: greedy proximity assignment, per qubit, using its own candidates ---
    # (Each qubit's own readout trace is the source of truth for its own dip.
    # We do NOT pool candidates across qubits, since multiplexed readouts may
    # have slightly different baselines.)
    tolerance_hz = params.proximity_tolerance_mhz * 1e6
    assignments: Dict[str, Optional[DipCandidate]] = {}
    leftovers_by_qubit: Dict[str, List[DipCandidate]] = {}
    for q in qubits:
        cands = candidates_by_qubit[q.name]
        a, leftovers = assign_dips_to_qubits(
            [q.name], [float(q.resonator.RF_frequency)], cands, tolerance_hz,
        )
        assignments[q.name] = a[q.name]
        leftovers_by_qubit[q.name] = leftovers

    # --- Step 3: refit a narrow Lorentzian window around each assigned dip ---
    f0_vals, fwhm_vals, r2_vals, success_vals, popt_vals = [], [], [], [], []
    for q in qubits:
        full_freq_q = rf_hz_axis
        amplitude_q = ds.sel(qubit=q.name).IQ_abs.values

        ov = overrides.get(q.name)
        if ov is not None:
            res = fit_resonator(
                full_freq_q, amplitude_q,
                override_center_hz=ov["center_hz"],
                override_span_hz=ov["span_hz"],
            )
        else:
            cand = assignments[q.name]
            if cand is None:
                # No assigned dip — record an empty failure
                f0_vals.append(np.nan)
                fwhm_vals.append(np.nan)
                r2_vals.append(0.0)
                success_vals.append(False)
                popt_vals.append(np.full(5, np.nan))
                continue
            # Window the wide trace around the assigned dip and refit
            span_hz = max(cand.fwhm_hz * 8.0, 8e6)  # at least 8 MHz window
            res = fit_resonator(
                full_freq_q, amplitude_q,
                override_center_hz=cand.rf_hz,
                override_span_hz=span_hz,
            )

        f0_vals.append(res["f0"])
        fwhm_vals.append(res["fwhm"])
        r2_vals.append(res["r2"] if not np.isnan(res["r2"]) else 0.0)
        success_vals.append(res["success"])
        popt_vals.append(res["popt"])

    popt_array = np.stack(popt_vals, axis=0)  # (n_qubits, 5)

    # Pack candidate lists as ragged 2D arrays (NaN-padded)
    max_cand = max((len(v) for v in candidates_by_qubit.values()), default=0)
    cand_rf = np.full((len(qubit_names), max_cand), np.nan)
    cand_prom = np.full((len(qubit_names), max_cand), np.nan)
    for i, name in enumerate(qubit_names):
        for j, c in enumerate(candidates_by_qubit[name]):
            cand_rf[i, j] = c.rf_hz
            cand_prom[i, j] = c.prominence_db

    ds_fit = xr.Dataset(
        {
            "f0": xr.DataArray(f0_vals, coords={"qubit": qubit_names}, dims="qubit",
                               attrs={"long_name": "resonator frequency", "units": "Hz"}),
            "fwhm": xr.DataArray(fwhm_vals, coords={"qubit": qubit_names}, dims="qubit",
                                 attrs={"long_name": "FWHM", "units": "Hz"}),
            "r2": xr.DataArray(r2_vals, coords={"qubit": qubit_names}, dims="qubit",
                               attrs={"long_name": "R²"}),
            "success": xr.DataArray(success_vals, coords={"qubit": qubit_names}, dims="qubit"),
            "popt": xr.DataArray(
                popt_array,
                coords={"qubit": qubit_names, "param": np.arange(5)},
                dims=["qubit", "param"],
                attrs={"long_name": "fit parameters [f0, fwhm, amp, bg0, bg1]"},
            ),
            "candidates_rf_hz": xr.DataArray(
                cand_rf,
                coords={"qubit": qubit_names, "cand_idx": np.arange(max_cand)},
                dims=["qubit", "cand_idx"],
                attrs={"long_name": "candidate dip RF", "units": "Hz"},
            ),
            "candidates_prominence_db": xr.DataArray(
                cand_prom,
                coords={"qubit": qubit_names, "cand_idx": np.arange(max_cand)},
                dims=["qubit", "cand_idx"],
                attrs={"long_name": "candidate dip prominence", "units": "dB"},
            ),
            "init_rf_hz": xr.DataArray(
                qubit_init_rf, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "initial resonator.RF_frequency", "units": "Hz"},
            ),
        }
    )

    fit_results = {}
    for q in qubits:
        name = q.name
        cand = assignments[name]
        f0 = float(ds_fit.sel(qubit=name).f0.values)
        init_rf = float(q.resonator.RF_frequency)
        fit_results[name] = FitParameters(
            frequency=f0,
            fwhm=float(ds_fit.sel(qubit=name).fwhm.values),
            r2=float(ds_fit.sel(qubit=name).r2.values),
            success=bool(ds_fit.sel(qubit=name).success.values),
            init_rf_hz=init_rf,
            assigned_dip_rf_hz=float(cand.rf_hz) if cand is not None else float("nan"),
            assignment_distance_hz=float(cand.rf_hz - init_rf) if cand is not None else float("nan"),
            n_candidates_total=len(candidates_by_qubit[name]),
            n_candidates_leftover=len(leftovers_by_qubit[name]),
        )

    return ds_fit, fit_results
