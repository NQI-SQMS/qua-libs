"""Analysis utilities for resonator spectroscopy versus power (IQ circles) calibration.

Self-tuning fit (no user knobs)
-------------------------------
``fit_raw_data`` extracts, robustly and without any user-set fitting parameters, the
dressed resonator frequency and the optimal readout power from the (power x detuning)
scan, then computes the readout-power state targets.

Algorithm (per qubit, on ``IQ_abs_norm``):
  1. For every readout power, find the resonator dip by edge-linear detrending the slice
     and taking the argmin within the central 80% of the band (``_robust_dip_features``)
     -> per-power dip detuning ``pos(P)``, depth ``c(P)`` and FWHM ``w(P)``. (This avoids
     ``peaks_dips``, whose ``position`` is NaN exactly at the deep-dip powers, and avoids
     band-edge baseline-slope artifacts.)
  2. Significant powers = ``c(P) >= 0.3 * max(c)``.
  3. Dressed resonance ``d0`` = contrast-weighted median of ``pos`` over significant
     powers (the dispersive cluster; robust to noisy/punch-out slices).
  4. "On resonance" = significant AND ``|pos - d0| <= tol`` with a TIGHT
     ``tol = max(1.5*detuning_step, 0.3*linewidth)``.
  5. ``optimal_power`` is placed conservatively below punch-out: anchor at the strongest
     on-resonance power, scan to higher power for the onset where the dip leaves the
     dressed frequency (sustained), and take the last on-resonance power minus
     ``power_below_punchout_db`` (default 3 dB). No punch-out in range -> optimal = top
     swept power minus ``power_below_punchout_db`` (success, "widen sweep" warning).
     ``success`` also requires a real DRESSED dip (SNR >= _SNR_MIN measured on the
     low-power on-resonance slices that anchored ``d0`` — not the global max over all
     powers, which a strong post-punch-out bare-cavity dip can satisfy even when the
     dressed resonance is dead) AND the chosen power staying inside the swept range.
  6. ``frequency_shift = d0`` ; ``resonator_frequency = RF_frequency + d0``.

Readout-power targets (``_compute_power_targets``) are computed per shared readout line
(resonators that share one ``full_scale_power_dbm`` port): one common full-scale on the
1 dB grid sized for the loudest tone, and a per-resonator amplitude. The node's
``update_state`` applies these (and preserves the power of line members not measured
this run). Every fit threshold is self-tuned from the data; the only operating-point knob
is ``power_below_punchout_db``.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

# ---------------------------------------------------------------------------
# MW-FEM full-scale power grid (dBm). Granularity is 1 dB (QOP >= 3.3); the range is
# -11..16 (QOP 3.3-3.6) or -11..18 (>= 3.7). We default to 16 (above 16 dBm is not
# guaranteed across the band); bump _FULL_SCALE_DBM_MAX to 18 on QOP >= 3.7.
# ---------------------------------------------------------------------------
_FULL_SCALE_DBM_MIN = -11
_FULL_SCALE_DBM_MAX = 16


def _on_grid_full_scale_dbm(power_in_dbm: float) -> int:
    """Smallest 1 dB-grid full-scale power >= ``power_in_dbm``, clamped to the allowed range."""
    return int(min(max(int(np.ceil(power_in_dbm)), _FULL_SCALE_DBM_MIN), _FULL_SCALE_DBM_MAX))


def _readout_line_label(qubit) -> str:
    """Short readout-line id from the resonator's shared full-scale port reference.

    e.g. ``#/ports/mw_outputs/con1/1/1/full_scale_power_dbm`` -> ``con1/1/1``.
    """
    ref = qubit.resonator.opx_output.get_reference(attr="full_scale_power_dbm")
    parts = ref.strip("#/").split("/")
    return "/".join(parts[2:5]) if len(parts) >= 5 else ref


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    optimal_power: float
    punchout: bool
    # Readout-power state targets (per shared readout line). full_scale_power_dbm is shared
    # across all resonators on the same line; amplitude is per resonator.
    target_full_scale_power_dbm: Optional[int]
    target_amplitude: Optional[float]
    readout_line: str


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        r = fit_results[q]
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {r['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * r['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * r['frequency_shift']:.0f} MHz)\n"
        if r["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        msg = s_qubit + s_power + s_freq + s_shift
        # Warn when no punch-out was reached in the swept range (optimal pinned to the top).
        if r["success"] and not r.get("punchout", True):
            msg += (
                f"  WARNING: no punch-out detected within the swept power range for {q}; "
                f"optimal set to the top swept power. Consider sweeping to higher power.\n"
            )
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V,
    or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["detuning"])})
    # Store power-to-amplitude conversion parameters for use in plotting
    ds.attrs["max_amp"] = node.parameters.max_amp
    ds.attrs["max_power_dbm"] = node.parameters.max_power_dbm
    # Add amplitude as a coordinate derived from power (dBm), for linear-scale plotting
    amp_values = ds.attrs["max_amp"] * 10 ** ((ds.power - ds.attrs["max_power_dbm"]) / 20)
    ds = ds.assign_coords(amplitude=("power", amp_values.values))
    ds.amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}
    return ds


def _edge_detrend(y: np.ndarray, det: np.ndarray, edge_frac: float = 1.0 / 7.0) -> np.ndarray:
    """Remove a linear baseline fitted through the outer ``edge_frac`` of each end of ``y``.

    The resonator dip sits centrally; the band edges are baseline, so a line through them
    flattens the slope without touching the dip. Robust where an ALS/global baseline leaves
    a residual slope whose minimum sits at a band edge (the failure that mislocated the dip).
    """
    n = len(det)
    e = max(3, int(n * edge_frac))
    idx = np.r_[np.arange(e), np.arange(n - e, n)]
    slope, intercept = np.polyfit(det[idx], y[idx], 1)
    return y - (slope * det + intercept)


def _robust_dip_features(da: xr.DataArray, det: np.ndarray):
    """Per-power dip detuning, depth, FWHM and noise via edge-detrend + central-region argmin.

    Returns ``(pos, depth, fwhm, noise)`` arrays over the power axis. The dip is searched
    only in the central 80% of the band (band-edge artifacts are excluded), after an
    edge-linear detrend. ``noise`` is a robust (MAD-based) baseline noise estimate per slice,
    so a per-power SNR = depth / noise can gate out flat (no-dip) traces. Unlike
    ``peaks_dips`` (whose ``position`` is NaN exactly at the deep-dip powers), ``pos`` is
    always finite where the slice is finite.
    """
    Y = da.transpose("power", "detuning").values
    n = len(det)
    c0, c1 = int(0.1 * n), int(0.9 * n)  # central 80%
    dstep = float(np.median(np.abs(np.diff(det)))) if n > 1 else 1.0
    e = max(3, int(n / 7))
    edge = np.r_[np.arange(e), np.arange(n - e, n)]
    npow = Y.shape[0]
    pos = np.full(npow, np.nan)
    depth = np.zeros(npow)
    fwhm = np.full(npow, np.nan)
    noise = np.full(npow, np.nan)
    for p in range(npow):
        y = Y[p]
        if not np.all(np.isfinite(y)):
            continue
        yd = _edge_detrend(y, det)
        seg = yd[c0:c1]
        j = c0 + int(np.argmin(seg))
        pos[p] = det[j]
        depth[p] = -yd[j]
        noise[p] = 1.4826 * np.median(np.abs(yd[edge] - np.median(yd[edge]))) + 1e-12
        if depth[p] > 0:  # FWHM: width where the detrended trace is below half the depth
            below = np.where(yd <= -depth[p] / 2.0)[0]
            if below.size >= 2:
                fwhm[p] = (below.max() - below.min()) * dstep
    return pos, depth, fwhm, noise


_SNR_MIN = 5.0  # a real dip: best-slice depth must exceed this many baseline-noise sigmas
_PUNCHOUT_BACKOFF_DB = 3.0  # fallback when power_below_punchout_db is absent; matches its default (3 dB)


def _analyze_one_qubit(pos, contrast, width, noise, powers, det, backoff_db=_PUNCHOUT_BACKOFF_DB, Y=None):
    """Robust dressed-frequency + optimal-power from per-power dip features (one qubit).

    ``pos`` (dip detuning, Hz), ``contrast`` (depth), ``width`` (FWHM, Hz) and ``noise``
    (baseline sigma) are the per-power features from ``_robust_dip_features``; ``powers``
    (dBm) and the detuning axis ``det`` (Hz) complete the inputs. ``Y`` (optional,
    ``(power, detuning)`` array of the normalized response) enables the AVERAGED
    dressed-slice SNR: at low readout power the dressed dip is stationary, so averaging
    the on-resonance low-power slices gains ~sqrt(N) in SNR — without it, low-shot scans
    whose real dip is invisible in any single slice would be rejected. Returns a dict
    with ``d0`` (dressed detuning), ``optimal_power`` (dBm), ``punchout`` (bool),
    ``success`` (bool) and ``dressed_snr`` (max of best-single-slice and averaged-slice
    dip SNR on the dressed band — the quantity the success gate certifies).

    Optimal power is placed (conservatively) BELOW punch-out: starting from the strongest
    on-resonance power (anchor), we scan to higher power for the onset where the dip leaves
    the dressed frequency (tight tolerance); the optimal is the last power still on the
    dressed resonance, minus ``backoff_db`` dB (user parameter ``power_below_punchout_db``).
    This avoids landing at/after the resonance has begun to shift.
    """
    pos = np.asarray(pos, float); contrast = np.abs(np.asarray(contrast, float))
    width = np.abs(np.asarray(width, float)); noise = np.asarray(noise, float)
    powers = np.asarray(powers, float)
    span = float(det.max() - det.min())
    dstep = float(np.median(np.abs(np.diff(det)))) if len(det) > 1 else 1.0
    fail = dict(d0=np.nan, optimal_power=np.nan, punchout=False, success=False, dressed_snr=0.0)

    good = np.isfinite(pos) & np.isfinite(contrast) & (contrast > 0)
    if good.sum() == 0:
        return fail
    cmax = float(np.nanmax(contrast[good]))
    significant = good & (contrast >= 0.3 * cmax)
    if significant.sum() == 0:
        return fail

    # Dressed resonance = the dip present at LOW readout power. After punch-out the bare-cavity
    # dip that appears at HIGH power is often much STRONGER (higher SNR) than the weak dressed
    # dip, so a contrast-weighted median over ALL powers can lock onto the bare cavity. Anchor
    # d0 on the low-power dispersive region (bottom 25% of the swept power range).
    plo_cut = powers.min() + 0.25 * (powers.max() - powers.min())
    low = significant & (powers <= plo_cut)
    if low.sum() == 0:
        low = significant  # fallback: whole range (e.g. punch-out already at the lowest power)
    f = pos[low]; w = contrast[low]
    order = np.argsort(f)
    wcum = np.cumsum(w[order])
    d0 = float(f[order][np.searchsorted(wcum, wcum[-1] / 2.0)])

    # Tight "still on the dressed resonance" tolerance (linewidth-capped, >= ~1.5 detuning steps)
    med_w = float(np.nanmedian(width[significant]))
    lw_capped = min(med_w, 0.1 * span) if np.isfinite(med_w) else 0.05 * span
    tol = max(1.5 * dstep, 0.3 * lw_capped)
    on_res = significant & (np.abs(pos - d0) <= tol)
    if on_res.sum() == 0:
        return dict(d0=d0, optimal_power=np.nan, punchout=False, success=False, dressed_snr=0.0)

    # Anchor at the strongest on-resonance power, then scan UP for the punch-out onset
    # (first higher significant power that leaves the dressed resonance, sustained).
    anchor = float(powers[on_res][np.nanargmax(contrast[on_res])])
    higher = [k for k in np.argsort(powers) if significant[k] and powers[k] > anchor]
    onset = None
    for ii, k in enumerate(higher):
        if np.abs(pos[k] - d0) > tol and (ii == len(higher) - 1 or np.abs(pos[higher[ii + 1]] - d0) > tol):
            onset = float(powers[k]); break

    if onset is not None:
        plateau = powers[on_res & (powers < onset)]
        punchout = True
    else:
        plateau = powers[on_res]
        # No frequency departure: punch-out only if the dip's contrast collapsed before the top.
        punchout = bool(float(np.max(powers[significant])) < float(np.max(powers)) - 1e-9)
    if plateau.size == 0:
        return dict(d0=d0, optimal_power=np.nan, punchout=punchout, success=False, dressed_snr=0.0)

    optimal_power = float(np.max(plateau)) - backoff_db  # backoff below the punch-out edge

    # Strict success: a real dip (SNR), held on-resonance across >= 3 powers, central (not edge),
    # AND the chosen power is still inside the swept range. A large power_below_punchout_db can
    # push optimal_power below powers.min(); without this guard _compute_power_targets would turn
    # that into a near-zero readout amplitude written to state with success=True (readout silently
    # disabled). Below-range -> fail (do not write) rather than disable readout.
    #
    # The SNR is gated on the DRESSED band — the low-power on-resonance slices that
    # anchored d0 — NOT on the global max over all powers. After punch-out the
    # bare-cavity dip at HIGH power is often far STRONGER than the dressed dip, so a
    # global-max SNR passes even when the dressed resonance itself is dead (noise-level
    # low-power dip) and a garbage d0 would be written with success=True. The gate takes
    # the max of (best single dressed slice, averaged dressed slices): the average gains
    # ~sqrt(N) for low-shot scans whose real dip no single slice resolves. On the
    # reference archive this removes 7 of 8 dead-resonator false-accepts among recorded
    # failures — and declines ~18 legacy "successes" whose low-power band provably holds
    # no dip (the legacy writes disagree with the actual data by MHz) — while keeping
    # every visually-confirmed recoverable fit (dead dressed dips score <= ~4.6, real
    # ones >= ~5.2, so _SNR_MIN = 5.0 sits exactly at the natural gap).
    p_min = float(np.min(powers))
    snr = contrast / noise
    dressed = on_res & low
    if dressed.sum() == 0:
        dressed = on_res
    dressed_snr = (
        float(np.nanmax(snr[dressed]))
        if np.any(np.isfinite(snr[dressed]))
        else 0.0
    )
    # Averaged dressed-slice SNR: the dressed dip is stationary at low power, so the
    # mean of the on-resonance low-power slices gains ~sqrt(N) — this rescues low-shot
    # scans whose (real) dip never clears the noise in any single slice.
    if Y is not None:
        idxs = np.where(dressed)[0]
        if idxs.size > 0:
            n_det = len(det)
            c0, c1 = int(0.1 * n_det), int(0.9 * n_det)
            e = max(3, int(n_det / 7))
            edge_idx = np.r_[np.arange(e), np.arange(n_det - e, n_det)]
            ybar = np.mean([_edge_detrend(np.asarray(Y[p], float), det) for p in idxs], axis=0)
            depth_avg = -float(np.min(ybar[c0:c1]))
            noise_avg = 1.4826 * float(np.median(np.abs(ybar[edge_idx] - np.median(ybar[edge_idx])))) + 1e-12
            if depth_avg > 0:
                dressed_snr = max(dressed_snr, depth_avg / noise_avg)
    success = bool(
        dressed_snr >= _SNR_MIN and on_res.sum() >= 3 and abs(d0) < 0.45 * span
        and optimal_power >= p_min
    )
    return dict(
        d0=d0, optimal_power=optimal_power, punchout=punchout, success=success,
        dressed_snr=dressed_snr,
    )


def _compute_power_targets(qubits, optimal_power, success, max_amp):
    """Per shared readout line, one common full-scale (1 dB grid) + a per-resonator amplitude.

    Resonators are grouped by their shared ``full_scale_power_dbm`` port reference. For each
    line with at least one successful qubit, the common full-scale is sized for the loudest
    (highest optimal power) tone so its amplitude stays <= ``max_amp``; quieter tones get a
    smaller amplitude at the same full-scale. Returns ``(target_fs, target_amp)`` arrays
    aligned with ``qubits`` (NaN where not set). Members not measured this run are handled in
    the node's ``update_state`` (power preserved).
    """
    n = len(qubits)
    target_fs = np.full(n, np.nan)
    target_amp = np.full(n, np.nan)
    groups: Dict[str, list] = {}
    for i, q in enumerate(qubits):
        ref = q.resonator.opx_output.get_reference(attr="full_scale_power_dbm")
        groups.setdefault(ref, []).append(i)
    for idxs in groups.values():
        ok = [i for i in idxs if success[i] and np.isfinite(optimal_power[i])]
        if not ok:
            continue
        max_opt = max(optimal_power[i] for i in ok)
        fs = _on_grid_full_scale_dbm(max_opt - 20.0 * np.log10(max_amp))
        for i in ok:
            target_fs[i] = fs
            target_amp[i] = 10 ** ((optimal_power[i] - fs) / 20.0)
    return target_fs, target_amp


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Self-tuning fit of the dressed resonance + optimal readout power (see module docstring).

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with dims (qubit, power, detuning) and ``IQ_abs_norm``.
    node : QualibrationNode
        Provides ``node.namespace['qubits']`` (for RF_frequency + line grouping) and
        ``node.parameters.max_amp`` (the per-tone amplitude cap).

    Returns
    -------
    (xr.Dataset, dict[str, FitParameters])
        The fit dataset (per-power ``res_freq_vs_power``/``contrast_vs_power`` + per-qubit
        coords) and the per-qubit fit-result dataclasses.
    """
    qubits = node.namespace["qubits"]
    names = [q.name for q in qubits]
    det = ds["detuning"].values.astype(float)
    powers = ds["power"].values.astype(float)
    span = float(det.max() - det.min())
    max_amp = float(node.parameters.max_amp)
    backoff_db = float(getattr(node.parameters, "power_below_punchout_db", _PUNCHOUT_BACKOFF_DB))

    nq, npow = ds.sizes["qubit"], len(powers)
    res_freq_vs_power = np.full((nq, npow), np.nan)
    contrast_vs_power = np.full((nq, npow), np.nan)
    d0 = np.full(nq, np.nan)
    optimal_power = np.full(nq, np.nan)
    success = np.zeros(nq, dtype=bool)
    punchout = np.zeros(nq, dtype=bool)
    dressed_snr = np.zeros(nq)

    for i, q in enumerate(qubits):
        # Per-power dip: edge-detrend + central-region argmin (robust; see _robust_dip_features).
        da = ds["IQ_abs_norm"].sel(qubit=q.name)
        pos, con, wid, noi = _robust_dip_features(da, det)
        res_freq_vs_power[i] = pos
        contrast_vs_power[i] = con
        Y_q = da.transpose("power", "detuning").values
        r = _analyze_one_qubit(pos, con, wid, noi, powers, det, backoff_db=backoff_db, Y=Y_q)
        d0[i] = r["d0"]
        optimal_power[i] = r["optimal_power"]
        success[i] = r["success"]
        punchout[i] = r["punchout"]
        dressed_snr[i] = r["dressed_snr"]

    rf = np.array([q.resonator.RF_frequency for q in qubits], dtype=float)
    target_fs, target_amp = _compute_power_targets(qubits, optimal_power, success, max_amp)
    lines = [_readout_line_label(q) for q in qubits]

    ds_fit = ds.assign(
        res_freq_vs_power=(["qubit", "power"], res_freq_vs_power),
        contrast_vs_power=(["qubit", "power"], contrast_vs_power),
    )
    ds_fit = ds_fit.assign_coords(
        freq_shift=("qubit", d0),
        optimal_power=("qubit", optimal_power),
        res_freq=("qubit", rf + d0),
        success=("qubit", success),
        punchout=("qubit", punchout),
        dressed_snr=("qubit", dressed_snr),
        target_full_scale_power_dbm=("qubit", target_fs),
        target_amplitude=("qubit", target_amp),
        readout_line=("qubit", np.array(lines, dtype=object)),
    )
    ds_fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    fit_results = {
        names[i]: FitParameters(
            success=bool(success[i]),
            resonator_frequency=float(rf[i] + d0[i]),
            frequency_shift=float(d0[i]),
            optimal_power=float(optimal_power[i]),
            punchout=bool(punchout[i]),
            target_full_scale_power_dbm=(int(target_fs[i]) if np.isfinite(target_fs[i]) else None),
            target_amplitude=(float(target_amp[i]) if np.isfinite(target_amp[i]) else None),
            readout_line=lines[i],
        )
        for i in range(nq)
    }
    return ds_fit, fit_results


# ---------------------------------------------------------------------------
# Complex (delay-removed) normalization + circle / Probst diameter-correction fit.
# These are additive helpers (used by the plotting figures); they do NOT touch the
# existing fit_raw_data pipeline. Robust per-power extraction of contrast and
# Qi / Qc / Ql, which avoids the low-SNR magnitude-minimum artifact.
# ---------------------------------------------------------------------------
def _circle_fit_kasa(x, y):
    """Algebraic (Kasa) least-squares circle fit. Returns ``(cx, cy, R)``.

    Mean-centres the data for conditioning; falls back to the centroid with ``R = nan``
    on a singular system. Inputs and outputs share the same units.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x.mean()
    ym = y.mean()
    u = x - xm
    v = y - ym
    Suu = np.dot(u, u)
    Svv = np.dot(v, v)
    Suv = np.dot(u, v)
    Suuu = np.dot(u, u * u)
    Svvv = np.dot(v, v * v)
    Suvv = np.dot(u, v * v)
    Svuu = np.dot(v, u * u)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return xm, ym, np.nan
    cx = uc + xm
    cy = vc + ym
    R = np.sqrt(max(uc * uc + vc * vc + (Suu + Svv) / len(x), 0.0))
    return cx, cy, R


def complex_normalize(f, S):
    """Remove the electrical delay and normalize a complex trace by its off-resonance baseline.

    The off-resonance phase is linear in frequency (cable delay = -2*pi*f*tau); a line fit
    on the band edges removes it. Dividing by the off-resonance magnitude pins the
    off-resonance point near (1, 0) and divides out the readout drive scaling. Phase-only
    delay removal does not change |S|.
    """
    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=complex)
    ne = max(8, len(f) // 20)
    idx = np.r_[np.arange(ne), np.arange(len(f) - ne, len(f))]
    sl, ic = np.polyfit(f[idx], np.unwrap(np.angle(S))[idx], 1)
    Sd = S * np.exp(-1j * (sl * f + ic))
    baseC = np.median(np.abs(np.r_[Sd[:ne], Sd[-ne:]]))
    if not np.isfinite(baseC) or baseC == 0:
        baseC = 1.0
    return Sd / baseC


def fit_resonator_circle(f, S):
    """Notch (hanger) complex fit of one trace S(f) via the diameter-correction method.

    Pipeline: ``complex_normalize`` -> Kasa circle fit -> phase fit
    ``theta = th0 + 2*arctan(2*Ql*(1 - f/fr))`` -> ``Qc = Ql/(2*r0)``,
    ``phi = arg(1 - z_center)``, ``1/Qi = 1/Ql - cos(phi)/Qc``. Returns a dict with
    ``fr, Ql, Qc, Qi, contrast, contrast_naive, Sn``; fit fields are ``nan`` on failure
    (e.g. resonance outside the window). ``contrast`` uses a lightly smoothed minimum to
    avoid the low-SNR magnitude-minimum artifact; ``contrast_naive`` is the raw minimum.
    """
    f = np.asarray(f, dtype=float)
    out = dict(fr=np.nan, Ql=np.nan, Qc=np.nan, Qi=np.nan,
               contrast=np.nan, contrast_naive=np.nan, Sn=None)
    if len(f) < 10:
        return out
    Sn = complex_normalize(f, S)
    out["Sn"] = Sn
    mag = np.abs(Sn)
    w = 7 if len(mag) >= 7 else len(mag)
    mags = np.convolve(mag, np.ones(w) / w, mode="same")
    imin = int(np.argmin(mags))
    out["contrast"] = float(1.0 - mags[imin])
    out["contrast_naive"] = float(1.0 - mag.min())
    out["fr"] = float(f[imin])
    # FWHM (smoothed) -> Ql guess
    half = 1.0 - out["contrast"] / 2.0
    lo = imin
    while lo > 0 and mags[lo] < half:
        lo -= 1
    hi = imin
    while hi < len(mags) - 1 and mags[hi] < half:
        hi += 1
    fwhm = abs(f[hi] - f[lo]) if hi > lo else (f[-1] - f[0]) / 10.0
    ql_guess = out["fr"] / fwhm if fwhm > 0 else 1e4
    try:
        from scipy.optimize import curve_fit

        cx, cy, r0 = _circle_fit_kasa(Sn.real, Sn.imag)
        zc = cx + 1j * cy
        theta = np.unwrap(np.angle(Sn - zc))

        def _model(ff, fr, Ql, th0):
            return th0 + 2.0 * np.arctan(2.0 * Ql * (1.0 - ff / fr))

        popt, _ = curve_fit(_model, f, theta, p0=[out["fr"], ql_guess, np.median(theta)], maxfev=20000)
        fr, Ql, _ = popt
        Ql = abs(Ql)
        Qc = Ql / (2.0 * r0) if r0 > 0 else np.nan
        phi = np.angle(1.0 - zc)
        inv_Qi = 1.0 / Ql - np.cos(phi) / Qc
        Qi = 1.0 / inv_Qi if inv_Qi > 0 else np.nan
        out.update(fr=float(fr), Ql=float(Ql), Qc=float(Qc), Qi=float(Qi))
    except Exception:
        pass
    return out


def compute_quality_factors(ds: xr.Dataset) -> xr.Dataset:
    """Per (qubit, power) complex-fit quality factors and contrast.

    Loops over qubits and powers, fitting each ``S(detuning) = I + iQ`` trace (using
    ``full_freq`` as the absolute frequency axis) with ``fit_resonator_circle``. Returns a
    Dataset with data vars ``Qi, Qc, Ql, res_freq, contrast, contrast_naive`` on dims
    ``(qubit, power)``.
    """
    qubits = [str(q) for q in np.atleast_1d(ds.qubit.values)]
    powers = np.asarray(ds.power.values, dtype=float)
    nq, npow = len(qubits), len(powers)
    Qi = np.full((nq, npow), np.nan)
    Qc = np.full((nq, npow), np.nan)
    Ql = np.full((nq, npow), np.nan)
    res = np.full((nq, npow), np.nan)
    con = np.full((nq, npow), np.nan)
    con_n = np.full((nq, npow), np.nan)
    for iq, qn in enumerate(qubits):
        ff = np.asarray(ds["full_freq"].sel(qubit=qn).values, dtype=float)
        for k in range(npow):
            I = ds["I"].sel(qubit=qn).isel(power=k).values
            Q = ds["Q"].sel(qubit=qn).isel(power=k).values
            r = fit_resonator_circle(ff, np.asarray(I) + 1j * np.asarray(Q))
            Qi[iq, k] = r["Qi"]
            Qc[iq, k] = r["Qc"]
            Ql[iq, k] = r["Ql"]
            res[iq, k] = r["fr"]
            con[iq, k] = r["contrast"]
            con_n[iq, k] = r["contrast_naive"]
    return xr.Dataset(
        {
            "Qi": (["qubit", "power"], Qi),
            "Qc": (["qubit", "power"], Qc),
            "Ql": (["qubit", "power"], Ql),
            "res_freq": (["qubit", "power"], res),
            "contrast": (["qubit", "power"], con),
            "contrast_naive": (["qubit", "power"], con_n),
        },
        coords={"qubit": qubits, "power": powers},
    )
