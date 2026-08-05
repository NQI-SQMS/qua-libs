"""Shared, pure helpers for 2D flux-map analyses (resonator/qubit spectroscopy vs flux).

Both vs-flux nodes share the same skeleton — per-flux-slice feature extraction
(dip for the resonator, peak for the qubit) followed by a model fit of the
feature ridge across flux — and historically both used UNGATED per-slice
extractors (raw ``idxmin`` / library ``peaks_dips``): slices with no real
feature contributed noise positions, silently corrupting the ridge model and
the sweet spot derived from it. These helpers centralise the hardened logic so
the two nodes cannot drift apart:

* :func:`extract_feature_ridge` — noise-σ-gated per-slice feature positions
  with sub-bin refinement and a validity mask (no silent noise points).
* :func:`fit_ridge_sinusoid` — the canonical ``a·cos(2πf·x+φ)+offset`` fit
  (FFT-seeded, canonicalised) plus the QUALITY numbers a fit-to-noise cannot
  fake: R², amplitude significance vs ridge scatter, and coverage.

Everything here is numpy-pure (no xarray, no node state) and unit-testable
offline.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


# ---------------------------------------------------------------------------
# Per-slice feature extraction
# ---------------------------------------------------------------------------

def _slice_feature(freqs, sig, kind, min_snr, smooth_window):
    """Most significant peak/dip of one 1D slice: (position, snr) or (nan, 0).

    Edge-linear detrend (the resonator/qubit feature sits centrally; band edges
    are baseline) + noise-σ-relative prominence + 3-point parabolic sub-bin
    refinement. ``kind`` is 'peak' or 'dip'.
    """
    n = len(sig)
    if n < 8:
        return np.nan, 0.0
    win = min(smooth_window, n // 3 * 2 - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    try:
        sm = savgol_filter(sig, win, 3)
    except Exception:
        sm = np.asarray(sig, dtype=float)
    resid = sig - sm
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-15

    # Edge-linear detrend so a sloped/curved baseline cannot masquerade as the feature
    e = max(3, n // 7)
    edge_idx = np.r_[np.arange(e), np.arange(n - e, n)]
    slope, intercept = np.polyfit(freqs[edge_idx], sm[edge_idx], 1)
    det = sm - (slope * freqs + intercept)
    y = det if kind == "peak" else -det

    pk, props = find_peaks(y, prominence=min_snr * sigma)
    inner = [(p, pr) for p, pr in zip(pk, props["prominences"]) if e <= p <= n - 1 - e]
    if not inner:
        return np.nan, 0.0
    p, pr = max(inner, key=lambda t: t[1])
    # 3-point parabolic sub-bin refinement
    pos = float(freqs[p])
    if 0 < p < n - 1:
        y0, y1, y2 = y[p - 1], y[p], y[p + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0:  # concave down around a maximum of y
            delta = 0.5 * (y0 - y2) / denom
            if -1.0 <= delta <= 1.0:
                pos = float(freqs[p] + delta * (freqs[p + 1] - freqs[p]))
    return pos, float(pr / sigma)


def extract_feature_ridge(freqs, data2d, kind="dip", *, min_snr=5.0, smooth_window=11):
    """Gated feature ridge of a (flux, detuning)-shaped map.

    Parameters
    ----------
    freqs : (n_det,) detuning axis.
    data2d : (n_flux, n_det) array — one spectroscopy slice per flux point.
    kind : 'dip' (resonator) or 'peak' (qubit).
    min_snr : per-slice significance gate (prominence / noise sigma). Slices
        with no feature this significant contribute NaN + mask=False instead of
        a noise position — the single most important difference from the legacy
        raw-argmin / ungated extractors.

    Returns dict(position (n_flux,), snr (n_flux,), mask (n_flux,) bool,
    coverage float).
    """
    freqs = np.asarray(freqs, dtype=float)
    data2d = np.asarray(data2d, dtype=float)
    n_flux = data2d.shape[0]
    position = np.full(n_flux, np.nan)
    snr = np.zeros(n_flux)
    for i in range(n_flux):
        row = data2d[i]
        if not np.all(np.isfinite(row)):
            continue
        position[i], snr[i] = _slice_feature(freqs, row, kind, min_snr, smooth_window)
    mask = np.isfinite(position)
    return dict(position=position, snr=snr, mask=mask, coverage=float(mask.mean()) if n_flux else 0.0)


# ---------------------------------------------------------------------------
# Ridge model: canonical sinusoid + quality
# ---------------------------------------------------------------------------

def _sinusoid(x, a, f, phi, offset):
    return a * np.cos(2 * np.pi * f * x + phi) + offset


@dataclass
class RidgeSinusoidFit:
    """Canonical sinusoid parameters + the quality numbers a noise fit cannot fake."""

    a: float = float("nan")
    f: float = float("nan")
    phi: float = float("nan")
    offset: float = float("nan")
    r2: float = float("nan")            # of the sinusoid vs the (weighted) ridge points
    amp_snr: float = 0.0                # |a| / robust residual scatter — flat/noise ridge -> ~0-2
    period: float = float("nan")        # 1/f in flux units
    n_points: int = 0
    success: bool = False               # finite params + minimally constrained period
    flat_response: bool = False         # amplitude indistinguishable from scatter -> phase (and
    #                                     hence any sweet-spot flux) is meaningless


def fit_ridge_sinusoid(
    flux,
    position,
    *,
    weights: Optional[np.ndarray] = None,
    min_amp_snr: float = 3.0,
) -> RidgeSinusoidFit:
    """Fit ``a·cos(2πf·x+φ)+offset`` to a gated ridge and grade it honestly.

    FFT-seeded initial guess on a uniform re-grid; canonicalised to ``a>0,
    f>0, φ∈[-π,π)``. ``weights`` (e.g. per-slice SNR) weight the least squares.

    QUALITY: ``amp_snr = |a| / MAD-σ(residuals)``. A ridge with no real flux
    dependence (dead line, decoupled qubit) still returns *some* sinusoid —
    but its amplitude is comparable to the residual scatter, so
    ``flat_response`` is raised and ``success`` stays False: the phase of a
    noise sinusoid is uniformly random and writing a sweet-spot flux from it
    would be confident garbage (the legacy behaviour this replaces).
    """
    out = RidgeSinusoidFit()
    flux = np.asarray(flux, dtype=float)
    position = np.asarray(position, dtype=float)
    finite = np.isfinite(flux) & np.isfinite(position)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        finite &= np.isfinite(weights) & (weights > 0)
    x = flux[finite]
    y = position[finite]
    w = weights[finite] if weights is not None else np.ones(x.size)
    out.n_points = int(x.size)
    if x.size < 8:
        return out

    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]
    span = float(x.max() - x.min())
    if span <= 0:
        return out

    offset0 = float(np.average(y, weights=w))
    amp0 = float((np.max(y) - np.min(y)) / 2.0) or 1e-15

    n_uniform = max(64, 2 * x.size)
    xu = np.linspace(x.min(), x.max(), n_uniform)
    yu = np.interp(xu, x, y) - offset0
    fft_vals = np.fft.rfft(yu)
    fft_freqs = np.fft.rfftfreq(n_uniform, d=span / (n_uniform - 1))
    if fft_vals.size > 1:
        k = int(np.argmax(np.abs(fft_vals[1:]))) + 1
        f0 = float(fft_freqs[k]) or 1.0 / span
        phi0 = float(np.angle(fft_vals[k]))
    else:
        f0, phi0 = 1.0 / span, 0.0

    try:
        popt, _ = curve_fit(
            _sinusoid, x, y, p0=[amp0, f0, phi0, offset0],
            sigma=1.0 / np.sqrt(w), absolute_sigma=False, maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return out

    a, f, phi, offset = popt
    if f < 0:
        f, phi = -f, -phi
    if a < 0:
        a, phi = -a, phi + np.pi
    phi = float(((phi + np.pi) % (2 * np.pi)) - np.pi)

    resid = y - _sinusoid(x, a, f, phi, offset)
    scatter = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-15
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
    out.a, out.f, out.phi, out.offset = float(a), float(f), phi, float(offset)
    out.r2 = 1.0 - ss_res / ss_tot
    out.amp_snr = float(a / scatter)
    out.period = 1.0 / f if f > 0 else float("nan")
    out.flat_response = out.amp_snr < min_amp_snr
    # Period must be at least minimally constrained by the sweep: with less
    # than ~half a period visible the FFT/fit degenerate into a drift line.
    period_ok = np.isfinite(out.period) and out.period <= 2.5 * span
    out.success = bool(
        np.all(np.isfinite([a, f, phi, offset])) and f > 0
        and not out.flat_response and period_ok
    )
    return out
