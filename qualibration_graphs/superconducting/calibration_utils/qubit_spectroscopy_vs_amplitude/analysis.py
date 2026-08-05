"""Analysis utilities for qubit spectroscopy versus drive power calibration.

Mirrors ``resonator_spectroscopy_vs_amplitude`` in structure, but for the QUBIT
drive: at each drive power the rotated quadrature shows a Lorentzian *peak*
(qubit line) that broadens with power (power broadening) and shifts slightly
(AC-Stark).  The optimal drive power for spectroscopy is the highest power whose
fitted FWHM is still close to the intrinsic (low-power) linewidth — i.e. just
below the power-broadening onset — which gives the best SNR without smearing the
line.  The qubit frequency is reported at that optimal power.

Two transitions are fitted:
  * the GE (g->e) line, near detuning 0 (the stored qubit frequency). It drives the
    optimal-power / f_01 state update, exactly as before.
  * the 2-photon g->f/2 line, which appears at higher drive power near
    detuning = -anharmonicity/2. From its position the anharmonicity is measured as
    |alpha| = 2*(f_GE - f_2photon). If the swept span does not reach the expected
    2-photon location, a warning is emitted (terminal + plot): widen the range if the
    line was not found, or "measured |alpha| smaller than stored" if it was.

The peak-finder prominence is auto-tuned per qubit (``auto_tune``), so the user does
not have to hand-set ``min_prominence_factor``.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from scipy.signal import find_peaks

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


# ---------------------------------------------------------------------------
# Internal fit hyperparameters (robust defaults; not exposed as node parameters
# so non-expert users are not faced with a wall of fit knobs). Power users can
# retune here. Each is still read via getattr from node.parameters, so a caller
# that re-adds the attribute keeps working.
# ---------------------------------------------------------------------------
_MIN_PROMINENCE_FACTOR = 5.0       # per-power peak prominence (noise sigma) when auto_tune is off
_FWHM_BROADENING_FACTOR = 2.0      # saturation onset: FWHM grown to this x intrinsic linewidth
_N_LOWPOWER_FLOOR_POINTS = 5       # lowest valid powers used to estimate the intrinsic linewidth floor
_GE_SEARCH_WINDOW_MHZ = 70.0       # half-window around detuning 0 for the GE peak search
_EF_SEARCH_WINDOW_MHZ = 40.0       # half-window around the expected 2-photon location for the EF search
_ANHARMONICITY_MIN_MHZ = 50.0      # lower sanity bound on measured |anharmonicity|
_ANHARMONICITY_MAX_MHZ = 500.0     # upper sanity bound on measured |anharmonicity|
_TARGET_PEAK_WIDTH_HZ = 3e6        # target linewidth used (with update_pulses_amplitude) to rescale amplitude
_MIN_SATURATION_TO_PI_RATIO = 5.0  # warn when drive duration < this x the x180 (pi) length
_FRINGE_DROP_FRACTION = 0.15       # fringe diagnostic: flag a peak-height drop > this fraction of running max


@dataclass
class FitParameters:
    """Per-qubit fitted parameters for qubit spectroscopy versus drive power."""

    success: bool
    frequency: float          # absolute qubit frequency at the optimal power [Hz]
    relative_freq: float      # peak detuning at the optimal power [Hz]
    optimal_power: float      # optimal drive power [dBm]
    optimal_amplitude: float  # optimal drive waveform amplitude [V]
    fwhm: float               # fitted FWHM at the optimal power [Hz]
    intrinsic_fwhm: float     # intrinsic (narrowest) linewidth estimate [Hz]
    power_warning: str        # "" | "sweep_too_hot" (line already saturating at the lowest swept power)
    short_pulse: bool         # drive duration < min_saturation_to_pi_ratio x t_pi (Rabi-fringe risk, not saturation)
    iw_angle: float           # IQ rotation angle [rad]
    saturation_amp: float     # rescaled saturation amplitude (if requested) [V]
    # Expected positions from the STORED anharmonicity (cross-check), NaN if |alpha| unknown
    twophoton_freq: float     # expected 2-photon g->f/2 freq = GE - |alpha|/2 [Hz]
    ef_freq: float            # expected e->f freq = GE - |alpha| [Hz]
    # --- 2-photon (g->f/2) FIT, used to measure the anharmonicity ---
    ef_success: bool          # the 2-photon line was found and the measured |alpha| is physical
    twophoton_freq_fitted: float  # fitted 2-photon g->f/2 absolute frequency [Hz], NaN if not found
    ef_freq_fitted: float         # implied e->f frequency = f_GE - |alpha|_measured [Hz], NaN if not found
    anharmonicity_fitted: float   # measured |alpha| MAGNITUDE = 2*(f_GE - f_2photon) [Hz, >0], NaN if not found
    anharmonicity_stored: float   # stored anharmonicity from the state [Hz, SIGNED; convention varies], NaN if absent
    ef_in_span: bool          # whether the expected 2-photon location lies inside the swept span
    ef_warning: str           # "" | "widen_range" | "anharm_smaller"


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the per-qubit fitted results, including the EF/anharmonicity outcome and any span warning."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        r = fit_results[q]
        status = "SUCCESS!" if r["success"] else "FAIL!"

        # EF / anharmonicity line
        if r.get("ef_success"):
            a_meas = r["anharmonicity_fitted"]                       # positive magnitude |alpha|
            a_store = r.get("anharmonicity_stored", np.nan)          # signed stored value (convention varies)
            cmp = ""
            if np.isfinite(a_store) and a_store != 0:
                rel = "smaller" if a_meas < abs(a_store) else "larger"
                cmp = f" ({rel} than stored {1e-6 * abs(a_store):.1f} MHz)"
            ef_note = (f" | EF (2-photon) fit: anharmonicity |alpha| = {1e-6 * a_meas:.1f} MHz{cmp}, "
                       f"f_ef @ {1e-9 * r['ef_freq_fitted']:.4f} GHz")
        elif np.isfinite(r.get("twophoton_freq", np.nan)):
            ef_note = (f" | EF (2-photon) not fitted; expected g->f/2 @ {1e-9 * r['twophoton_freq']:.4f} GHz, "
                       f"e->f @ {1e-9 * r['ef_freq']:.4f} GHz")
        else:
            ef_note = ""

        # Span-coverage warning (small but explicit, per request)
        warn = r.get("ef_warning", "")
        if warn == "widen_range":
            ef_note += ("\n\t[!] EF/anharmonicity line is OUTSIDE the swept span and was not found -> "
                        "WIDEN frequency_span_in_mhz (>= the stored |alpha|) to capture it.")
        elif warn == "anharm_smaller":
            ef_note += ("\n\t[!] EF line found INSIDE a span narrower than the expected location -> the measured "
                        "anharmonicity is SMALLER than the stored value.")

        if r.get("power_warning") == "sweep_too_hot":
            ef_note += ("\n\t[!] line already saturating at the LOWEST swept power -> the optimal power is pinned "
                        "near the floor; LOWER min_power_dbm so the sweep includes the unsaturated regime.")
        if r.get("short_pulse"):
            ef_note += ("\n\t[!] drive duration is short relative to the x180 (pi) pulse -> the amplitude sweep "
                        "may leave the saturation regime and show Rabi-nutation fringes; use a longer "
                        "operation_len_in_ns for clean saturation spectroscopy.")

        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tOptimal drive power: {r['optimal_power']:.2f} dBm "
            f"(amp {1e3 * r['optimal_amplitude']:.1f} mV) | "
            f"Qubit (GE) frequency: {1e-9 * r['frequency']:.4f} GHz | "
            f"FWHM: {1e-3 * r['fwhm']:.1f} kHz "
            f"(intrinsic {1e-3 * r['intrinsic_fwhm']:.1f} kHz){ef_note}\n"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ to V, rotate to the signal quadrature, and attach freq/amplitude coords."""
    qubits = node.namespace["qubits"]
    ds = convert_IQ_to_V(ds, qubits)
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    # Rotation angle per qubit, determined from the highest-power slice (clearest peak).
    ds_hp = ds.isel(power=-1)
    shifts = np.abs(ds_hp.IQ_abs - ds_hp.IQ_abs.mean(dim="detuning")).idxmax(dim="detuning")
    angle = np.arctan2(
        ds_hp.sel(detuning=shifts).Q - ds_hp.Q.mean(dim="detuning"),
        ds_hp.sel(detuning=shifts).I - ds_hp.I.mean(dim="detuning"),
    )
    ds = ds.assign({"iw_angle": angle})
    ds = ds.assign({"I_rot": ds.I * np.cos(ds.iw_angle) + ds.Q * np.sin(ds.iw_angle)})

    # Absolute RF frequency axis per qubit
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in qubits])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    # Drive amplitude (V) derived from the power (dBm) axis, for the dual y-axis / linear plot
    ds.attrs["max_amp"] = node.parameters.max_amp
    ds.attrs["max_power_dbm"] = node.parameters.max_power_dbm
    amp_values = ds.attrs["max_amp"] * 10 ** ((ds.power - ds.attrs["max_power_dbm"]) / 20)
    ds = ds.assign_coords(amplitude=("power", amp_values.values))
    ds.amplitude.attrs = {"long_name": "drive amplitude", "units": "V"}
    return ds


def _per_power_peak(detuning_hz, i_rot_2d, prominence_factor, mask=None):
    """Most-prominent peak per power on |I_rot - baseline|, optionally restricted to a detuning mask.

    Uses a robust MAD-based noise estimate with a relative floor (works on a narrow window, unlike an
    ALS-baseline peak finder which can absorb a peak sitting near the window centre).

    Returns (position[Hz detuning], width[Hz FWHM]) arrays of length n_power (NaN where no peak found).
    """
    det = np.asarray(detuning_hz, dtype=float)
    idxs = np.where(mask)[0] if mask is not None else np.arange(det.size)
    step = abs(det[1] - det[0]) if det.size > 1 else 1.0
    npow = i_rot_2d.shape[1]
    pos = np.full(npow, np.nan)
    wid = np.full(npow, np.nan)
    for j in range(npow):
        y = i_rot_2d[:, j].astype(float)
        y = y - np.median(y)
        s = np.abs(y)
        mad = np.median(np.abs(y)) + 1e-12
        sw = s[idxs]
        if sw.size < 3 or sw.max() <= 0:
            continue
        pk, pr = find_peaks(sw, prominence=max(prominence_factor * mad, 0.05 * sw.max()), distance=4, width=1)
        if len(pk) == 0:
            continue
        b = int(np.argmax(pr["prominences"]))
        pos[j] = det[idxs][pk[b]]
        wid[j] = pr["widths"][b] * step
    return pos, wid


def _auto_prominence_factor(detuning_hz, i_rot_2d, mask, candidates, n_floor):
    """Pick the highest (most selective) prominence factor that still detects a consistent peak.

    Sweeps ``candidates`` high->low and accepts the first that (a) finds a peak in a sufficient fraction of
    power slices and (b) whose lowest-power detections cluster tightly (low MAD). This rejects noise-driven
    peaks (which need a low prominence) while still locking onto a real, repeatable line. Falls back to the
    most permissive candidate if none qualify.
    """
    det = np.asarray(detuning_hz, dtype=float)
    npow = i_rot_2d.shape[1]
    win = float(det.max() - det.min()) if det.size > 1 else 1.0
    min_count = max(3, int(0.3 * npow))
    for pf in candidates:
        pos, _ = _per_power_peak(det, i_rot_2d, pf, mask=mask)
        vpos = pos[np.isfinite(pos)]
        if vpos.size < min_count:
            continue
        lp = vpos[: max(n_floor, 5)]
        mad = float(np.median(np.abs(lp - np.median(lp)))) if lp.size else np.inf
        if mad <= 0.15 * win:
            return float(pf)
    return float(candidates[-1])


def _rolling_median(x, w=5):
    """Rolling-median smooth of a 1-D array (NaN-aware), to reject single-point FWHM noise spikes."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = w // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = x[max(0, i - h):min(n, i + h + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _saturation_onset_optimal(power_dbm, fwhm, contrast, broadening_factor, margin_db, n_floor):
    """Pick the optimal spectroscopy drive power = ``margin_db`` below the saturation/power-broadening onset.

    The line is narrowest (intrinsic) at low power and broadens as the transition saturates. We:
      1. keep only well-detected slices (contrast >= 30% of its max) so noise-only low powers don't bias things;
      2. estimate the intrinsic linewidth as the minimum of the SMOOTHED FWHM over those slices;
      3. define the saturation onset as the first power whose smoothed FWHM exceeds
         ``broadening_factor`` x intrinsic for two consecutive points (robust to single spikes);
      4. set the optimal power ``margin_db`` dB below that onset (clean single-photon GE regime, away from the
         2-photon g->f line).

    Returns (optimal_idx, intrinsic_fwhm, onset_found, sweep_too_hot). ``sweep_too_hot`` is True when the line
    is already strongly present at the lowest swept power (the sweep does not include the unsaturated regime, so
    the onset - margin lands at/near the floor -> the user should lower ``min_power_dbm``).
    """
    power = np.asarray(power_dbm, dtype=float)
    fwhm = np.abs(np.asarray(fwhm, dtype=float))
    contrast = np.asarray(contrast, dtype=float)
    valid = np.isfinite(fwhm) & np.isfinite(contrast) & (fwhm > 0) & (contrast > 0)
    vidx = np.where(valid)[0]
    if vidx.size < max(3, n_floor):
        return None, np.nan, False, False

    cmax = float(np.max(contrast[vidx]))
    good = valid & (contrast >= 0.3 * cmax)
    gidx = np.where(good)[0]
    if gidx.size < max(3, n_floor):
        gidx = vidx

    fw = _rolling_median(fwhm, 5)
    intrinsic = float(np.nanmin(fw[gidx]))
    threshold = broadening_factor * intrinsic

    onset_idx = None
    for k in range(gidx.size - 1):
        i, j = gidx[k], gidx[k + 1]
        if fw[i] > threshold and fw[j] > threshold:
            onset_idx = i
            break
    onset_found = onset_idx is not None
    if onset_idx is None:
        onset_idx = int(gidx[-1])  # no broadening seen within the sweep -> highest power

    onset_power = power[onset_idx]
    target = onset_power - margin_db
    cand = gidx[power[gidx] <= onset_power]
    optimal_idx = int(cand[int(np.argmin(np.abs(power[cand] - target)))]) if cand.size else int(onset_idx)

    # Sweep too hot: the line is already >= half its max contrast at the lowest swept power -> no clean
    # unsaturated region, the onset is at/below the floor.
    sweep_too_hot = bool(contrast[vidx[0]] >= 0.5 * cmax)
    return optimal_idx, intrinsic, onset_found, sweep_too_hot


def _fit_ef_twophoton(detuning_hz, i_rot_2d, prominence, ge_rel, ge_win_hz, a_abs,
                      ef_win_hz, anh_min, anh_max, n_floor):
    """Fit the 2-photon g->f/2 line (below GE) and return the measured anharmonicity magnitude.

    Strategy: search a window around the EXPECTED location (-|a|/2, always below GE; the stored sign is not
    trusted), restricted to detunings below the GE window. The 2-photon line strengthens with drive power, so
    read its position at the highest powers where it is detected. If nothing physical is found there, scan the
    whole below-GE region for the next candidate. Returns (ef_success, det_2photon[Hz, relative], |alpha|[Hz, >0]).
    """
    det = np.asarray(detuning_hz, dtype=float)
    det_min = float(det.min())

    def _measure(mask):
        if not mask.any():
            return None
        pos_ef, _ = _per_power_peak(det, i_rot_2d, prominence, mask=mask)
        v = np.where(np.isfinite(pos_ef))[0]
        if v.size < 3:
            return None
        d2 = float(np.median(pos_ef[v[-max(3, n_floor):]]))  # high-power detections (clearest)
        a = 2.0 * (ge_rel - d2)                               # |alpha| magnitude (d2 < ge_rel => positive)
        if anh_min <= a <= anh_max:
            return d2, a
        return None

    below_ge = det < (ge_rel - 0.5 * ge_win_hz)
    if np.isfinite(a_abs):
        expected = ge_rel - a_abs / 2.0
        primary = below_ge & (np.abs(det - expected) <= ef_win_hz)
        res = _measure(primary)
        if res is not None:
            return True, res[0], res[1]
    # Fallback: next candidate anywhere clearly below the GE window
    res = _measure(det < (ge_rel - ge_win_hz))
    if res is not None:
        return True, res[0], res[1]
    return False, np.nan, np.nan


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Per-power peak fit of the GE line (optimal power + f_01) and of the 2-photon EF line (anharmonicity).

    GE is the qubit's directly-driven transition, so it has the lowest power threshold and sits near the
    stored qubit frequency (detuning = 0). The GE search is restricted to detuning in
    [-ge_search_window_mhz, +ge_search_window_mhz] so the fit cannot lock onto an EF/2-photon line (lower
    side, higher power) or a neighbour (higher side). The GE frequency is read at the lowest powers where
    the in-window peak is detected (least power broadening / AC-Stark). If no peak is found in the window
    the qubit is marked failed instead of returning a far, wrong peak.

    The 2-photon g->f/2 line is fitted separately (see ``_fit_ef_twophoton``) to measure the anharmonicity;
    if the swept span does not reach the expected 2-photon location, ``ef_warning`` is set accordingly.
    """
    qubits = node.namespace["qubits"]
    params = node.parameters
    qnames = [str(q) for q in ds.qubit.values]

    det_hz = ds.detuning.values.astype(float)
    det_min, det_max = float(det_hz.min()), float(det_hz.max())
    win_hz = float(getattr(params, "ge_search_window_mhz", _GE_SEARCH_WINDOW_MHZ)) * 1e6
    win_mask = np.abs(det_hz) <= win_hz       # GE search window around detuning 0 (stored frequency)
    npow = ds.sizes["power"]

    power_dbm = ds.power.values.astype(float)
    rf = {q.name: float(q.xy.RF_frequency) for q in qubits}
    anharm = {q.name: float(getattr(q, "anharmonicity", np.nan) or np.nan) for q in qubits}
    prev_angle = {q.name: float(q.resonator.operations["readout"].integration_weights_angle) for q in qubits}
    used_amp = {q.name: float(q.xy.operations[params.operation].amplitude) for q in qubits}

    # Short-pulse guard: spectroscopy drive duration vs the x180 (pi) pulse length. If the drive is short, the
    # amplitude sweep leaves the saturation regime and shows coherent Rabi-nutation fringes (see teammate note).
    _ratio = float(getattr(params, "min_saturation_to_pi_ratio", _MIN_SATURATION_TO_PI_RATIO))
    _oplen = getattr(params, "operation_len_in_ns", None)

    def _is_short(q):
        try:
            sat_len = float(_oplen) if _oplen else float(q.xy.operations[params.operation].length)
            t_pi = float(q.xy.operations["x180"].length)
            return bool(t_pi > 0 and sat_len < _ratio * t_pi)
        except Exception:
            return False

    short_pulse = {q.name: _is_short(q) for q in qubits}

    # Auto-tune / manual settings for the peak finder
    auto = bool(getattr(params, "auto_tune", True))
    prom_candidates = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0]
    n_floor = int(getattr(params, "n_lowpower_floor_points", _N_LOWPOWER_FLOOR_POINTS))
    fit_ef = bool(getattr(params, "fit_ef_transition", True))
    ef_win_hz = float(getattr(params, "ef_search_window_mhz", _EF_SEARCH_WINDOW_MHZ)) * 1e6
    anh_min = float(getattr(params, "anharmonicity_min_mhz", _ANHARMONICITY_MIN_MHZ)) * 1e6
    anh_max = float(getattr(params, "anharmonicity_max_mhz", _ANHARMONICITY_MAX_MHZ)) * 1e6

    # Full-span most-prominent peak per (qubit, power): for the plot trace.
    peak_pos_full = np.full((len(qnames), npow), np.nan)
    peak_w_full = np.full((len(qnames), npow), np.nan)

    margin_db = float(getattr(params, "power_below_saturation_db", 1.0))
    cols = {k: [] for k in ("opt_power", "opt_amp", "ge_rel", "fwhm", "intrinsic",
                            "success", "onset", "pwarn", "short", "twoph", "ef",
                            "ef_success", "twoph_fit", "ef_fit", "anh_fit", "anh_store",
                            "ef_in_span", "ef_warn")}
    for qi, qn in enumerate(qnames):
        i_rot_2d = ds.I_rot.sel(qubit=qn).transpose("detuning", "power").values
        prom = _auto_prominence_factor(det_hz, i_rot_2d, win_mask, prom_candidates, n_floor) \
            if auto else float(getattr(params, "min_prominence_factor", _MIN_PROMINENCE_FACTOR))

        # full-span trace (for plotting) and in-window GE candidate (for the fit)
        peak_pos_full[qi], peak_w_full[qi] = _per_power_peak(det_hz, i_rot_2d, prom)
        pos_q, fwhm_q = _per_power_peak(det_hz, i_rot_2d, prom, mask=win_mask)
        # in-window peak contrast per power (drives the saturation-onset selection)
        contrast_q = np.array([
            float(np.max(np.abs(i_rot_2d[:, j][win_mask] - np.median(i_rot_2d[:, j])))) for j in range(npow)
        ])
        optimal_idx, intrinsic, onset, sweep_hot = _saturation_onset_optimal(
            power_dbm, fwhm_q, contrast_q,
            float(getattr(params, "fwhm_broadening_factor", _FWHM_BROADENING_FACTOR)), margin_db, n_floor
        )
        valid_idx = np.where(np.isfinite(pos_q) & np.isfinite(fwhm_q) & (fwhm_q > 0))[0]
        a = anharm[qn]                                   # stored anharmonicity (SIGNED; convention varies per state)
        a_abs = abs(a) if np.isfinite(a) else np.nan     # magnitude used for all positions (2-photon is BELOW GE)
        if optimal_idx is None or valid_idx.size < 3:
            for k in ("opt_power", "opt_amp", "ge_rel", "fwhm", "intrinsic", "twoph", "ef",
                      "twoph_fit", "ef_fit", "anh_fit"):
                cols[k].append(np.nan)
            cols["success"].append(False); cols["onset"].append(False)
            cols["pwarn"].append("")
            cols["short"].append(short_pulse[qn])
            cols["ef_success"].append(False)
            cols["anh_store"].append(a)
            cols["ef_in_span"].append(False)
            cols["ef_warn"].append("")
            continue
        # GE detuning: low-power read (Stark-free) if requested, else at the optimal power
        if params.ge_low_power_first:
            n = max(1, n_floor)
            ge_rel = float(np.median(pos_q[valid_idx[:n]]))
        else:
            ge_rel = float(pos_q[optimal_idx])
        op = power_dbm[optimal_idx]
        cols["opt_power"].append(op)
        cols["opt_amp"].append(params.max_amp * 10 ** ((op - params.max_power_dbm) / 20))
        cols["ge_rel"].append(ge_rel)
        cols["fwhm"].append(float(np.abs(fwhm_q[optimal_idx])))
        cols["intrinsic"].append(intrinsic)
        cols["onset"].append(onset)
        cols["pwarn"].append("sweep_too_hot" if sweep_hot else "")
        cols["short"].append(short_pulse[qn])
        cols["success"].append(True)
        cols["twoph"].append(ge_rel + rf[qn] - a_abs / 2 if np.isfinite(a_abs) else np.nan)  # expected g->f/2 (abs Hz)
        cols["ef"].append(ge_rel + rf[qn] - a_abs if np.isfinite(a_abs) else np.nan)          # expected e->f (abs Hz)

        # --- 2-photon EF fit (anharmonicity). Use the magnitude: the 2-photon line is BELOW GE at -|a|/2,
        #     regardless of the stored sign (which is inconsistent across states). ---
        ef_in_span = bool(np.isfinite(a_abs) and (det_min <= (ge_rel - a_abs / 2.0) <= det_max))
        ef_ok, det_2ph, a_meas = (False, np.nan, np.nan)
        if fit_ef:
            ef_ok, det_2ph, a_meas = _fit_ef_twophoton(
                det_hz, i_rot_2d, prom, ge_rel, win_hz, a_abs, ef_win_hz, anh_min, anh_max, n_floor
            )
        twoph_fit = rf[qn] + det_2ph if ef_ok else np.nan          # fitted 2-photon abs Hz
        ef_fit = (rf[qn] + ge_rel) - a_meas if ef_ok else np.nan   # implied e->f abs Hz
        # Warning only when the expected location is OUTSIDE the swept span (per request)
        if np.isfinite(a) and not ef_in_span:
            ef_warn = "anharm_smaller" if ef_ok else "widen_range"
        else:
            ef_warn = ""
        cols["ef_success"].append(ef_ok)
        cols["twoph_fit"].append(twoph_fit)
        cols["ef_fit"].append(ef_fit)
        cols["anh_fit"].append(a_meas)
        cols["anh_store"].append(a)
        cols["ef_in_span"].append(ef_in_span)
        cols["ef_warn"].append(ef_warn)

    rf_arr = np.array([rf[qn] for qn in qnames])
    ds_fit = ds.assign(
        peak_position=(("qubit", "power"), peak_pos_full),
        peak_width=(("qubit", "power"), peak_w_full),
    )
    ds_fit = ds_fit.assign_coords(
        optimal_power=("qubit", np.array(cols["opt_power"])),
        optimal_amplitude=("qubit", np.array(cols["opt_amp"])),
        freq_shift=("qubit", np.array(cols["ge_rel"])),
        fwhm_at_optimal=("qubit", np.array(cols["fwhm"])),
        intrinsic_fwhm=("qubit", np.array(cols["intrinsic"])),
        onset_found=("qubit", np.array(cols["onset"])),
        power_warning=("qubit", np.array(cols["pwarn"], dtype=object)),
        short_pulse=("qubit", np.array(cols["short"])),
        success=("qubit", np.array(cols["success"])),
        res_freq=("qubit", np.array(cols["ge_rel"]) + rf_arr),
        twophoton_freq=("qubit", np.array(cols["twoph"])),
        ef_freq=("qubit", np.array(cols["ef"])),
        ef_success=("qubit", np.array(cols["ef_success"])),
        twophoton_freq_fitted=("qubit", np.array(cols["twoph_fit"])),
        ef_freq_fitted=("qubit", np.array(cols["ef_fit"])),
        anharmonicity_fitted=("qubit", np.array(cols["anh_fit"])),
        anharmonicity_stored=("qubit", np.array(cols["anh_store"])),
        ef_in_span=("qubit", np.array(cols["ef_in_span"])),
        ef_warning=("qubit", np.array(cols["ef_warn"], dtype=object)),
    )
    ds_fit.optimal_power.attrs = {"long_name": "optimal drive power", "units": "dBm"}
    ds_fit.res_freq.attrs = {"long_name": "qubit xy frequency (GE)", "units": "Hz"}
    ds_fit.attrs["ge_window_mhz"] = float(getattr(params, "ge_search_window_mhz", _GE_SEARCH_WINDOW_MHZ))

    sat_amp = {}
    for qn in qnames:
        fw = float(ds_fit.fwhm_at_optimal.sel(qubit=qn).values)
        _tpw = float(getattr(params, "target_peak_width", _TARGET_PEAK_WIDTH_HZ))
        sat_amp[qn] = float(_tpw / fw * used_amp[qn]) if np.isfinite(fw) and fw > 0 else np.nan

    fit_results = {
        qn: FitParameters(
            success=bool(ds_fit.success.sel(qubit=qn).values),
            frequency=float(ds_fit.res_freq.sel(qubit=qn).values),
            relative_freq=float(ds_fit.freq_shift.sel(qubit=qn).values),
            optimal_power=float(ds_fit.optimal_power.sel(qubit=qn).values),
            optimal_amplitude=float(ds_fit.optimal_amplitude.sel(qubit=qn).values),
            fwhm=float(ds_fit.fwhm_at_optimal.sel(qubit=qn).values),
            intrinsic_fwhm=float(ds_fit.intrinsic_fwhm.sel(qubit=qn).values),
            power_warning=str(ds_fit.power_warning.sel(qubit=qn).values),
            short_pulse=bool(ds_fit.short_pulse.sel(qubit=qn).values),
            iw_angle=float((prev_angle[qn] + float(ds.iw_angle.sel(qubit=qn).values)) % (2 * np.pi)),
            saturation_amp=sat_amp[qn],
            twophoton_freq=float(ds_fit.twophoton_freq.sel(qubit=qn).values),
            ef_freq=float(ds_fit.ef_freq.sel(qubit=qn).values),
            ef_success=bool(ds_fit.ef_success.sel(qubit=qn).values),
            twophoton_freq_fitted=float(ds_fit.twophoton_freq_fitted.sel(qubit=qn).values),
            ef_freq_fitted=float(ds_fit.ef_freq_fitted.sel(qubit=qn).values),
            anharmonicity_fitted=float(ds_fit.anharmonicity_fitted.sel(qubit=qn).values),
            anharmonicity_stored=float(ds_fit.anharmonicity_stored.sel(qubit=qn).values),
            ef_in_span=bool(ds_fit.ef_in_span.sel(qubit=qn).values),
            ef_warning=str(ds_fit.ef_warning.sel(qubit=qn).values),
        )
        for qn in qnames
    }
    return ds_fit, fit_results


# ---------------------------------------------------------------------------
# Coherent-nutation fringe diagnostic (data-shape only; no pi/T1/T2 needed)
# ---------------------------------------------------------------------------

def detect_amplitude_fringe(ds: xr.Dataset, node: QualibrationNode, drop_fraction: float = None) -> Dict[str, dict]:
    """Detect coherent-nutation fringing along the drive-power axis, per qubit.

    Physics: with a fixed saturation duration, the on-resonance excited-state
    population follows the coherent Rabi angle Omega*t as the amplitude grows, so
    the spectroscopy peak height rises, then DIPS when Omega*t crosses a multiple of
    2*pi (the qubit returns toward |0>). In true saturation (or in constant_angle
    mode) the peak height instead rises monotonically and plateaus. We therefore
    flag the lowest power at which the smoothed peak height, after having risen above
    its starting value, drops by more than ``drop_fraction`` of its running maximum.

    Uses only the measured signal shape (max-over-detuning deviation of IQ_abs per
    power) — no pi-pulse / T1 / T2 input, consistent with this early-stage node.

    Returns ``{qubit_name: {fringe_detected, fringe_power_dbm, peak_vs_power, power_dbm}}``.
    """
    if drop_fraction is None:
        drop_fraction = float(getattr(node.parameters, "fringe_drop_fraction", _FRINGE_DROP_FRACTION))

    results: Dict[str, dict] = {}
    for q in node.namespace["qubits"]:
        dq = ds.sel(qubit=q.name)
        # Peak height per power = max over detuning of |IQ_abs - baseline|.
        baseline = dq.IQ_abs.mean(dim="detuning")
        peak = np.abs(dq.IQ_abs - baseline).max(dim="detuning")
        # Order by ascending power so "rise then drop" is well-defined.
        peak = peak.sortby("power")
        powers = np.asarray(peak.power.values, dtype=float)
        pv = np.asarray(peak.values, dtype=float)

        # Light 3-point smoothing to avoid flagging single-point noise. Pad the edges with
        # 'edge' (not zero) before convolving: np.convolve(..., mode='same') zero-pads, which
        # makes the LAST smoothed sample ~(pv[-2]+pv[-1])/3 — a spurious ~33% drop that
        # false-flags a fringe at the top power on clean saturating curves.
        if pv.size >= 5:
            kernel = np.ones(3) / 3.0
            pvs = np.convolve(np.pad(pv, 1, mode="edge"), kernel, mode="valid")
        else:
            pvs = pv.copy()

        running_max = np.maximum.accumulate(pvs) if pvs.size else pvs
        fringe_idx = None
        for i in range(1, pvs.size):
            rm = running_max[i]
            if rm > 0 and rm > pvs[0] and (rm - pvs[i]) / rm > drop_fraction:
                fringe_idx = i
                break

        results[q.name] = {
            "fringe_detected": bool(fringe_idx is not None),
            "fringe_power_dbm": float(powers[fringe_idx]) if fringe_idx is not None else None,
            "peak_vs_power": pv.tolist(),
            "power_dbm": powers.tolist(),
        }
    return results


def log_fringe_results(fringe_results: Dict[str, dict], log_callable=None):
    """Log a one-line fringe verdict per qubit; warn when fringing is detected."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fringe_results.items():
        if r["fringe_detected"]:
            log_callable(
                f"[fringe] qubit {q}: coherent-nutation fringing detected at "
                f"~{r['fringe_power_dbm']:.1f} dBm — in 'fixed' mode increase the saturation "
                f"duration or switch duration_mode='constant_angle'."
            )
        else:
            log_callable(f"[fringe] qubit {q}: no fringing detected (peak height rises monotonically).")
