"""Plotting utilities for wide-band resonator spectroscopy.

Each figure shows one panel per qubit on a QubitGrid. Panels share the
same absolute RF axis (the wide scan range). Per-qubit overlays:

- Red dashed vertical at the assigned dip's f0
- Gray dashed verticals at unassigned candidate dips
- Red dashed curve = local Lorentzian-with-linear-background fit, drawn
  only within the fit window (so it doesn't drag through the wide trace)
"""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import lorentzian_dip_linbg

u = unit(coerce_to_integer=True)

_FS_SUPTITLE = 28
_FS_TITLE    = 24
_FS_LABEL    = 24
_FS_TICK     = 22
_FS_LEGEND   = 18
_FS_CBAR     = 20

_TITLE_PAD = 8

# Per-panel sizing for QubitGrid figures. Width per column is generous so
# legends and axis labels don't collide; height per row leaves room for the
# title above and tick labels below.
_PER_COL_INCH = 6.0
_PER_ROW_INCH = 4.5
_MIN_WIDTH_INCH = 8.0
_MIN_HEIGHT_INCH = 6.0


def _apply_tick_fontsize(ax: Axes, size: int = _FS_TICK) -> None:
    ax.tick_params(axis="both", labelsize=size)


def _setup_grid_figure(grid) -> None:
    """Resize a QubitGrid figure to its panel count and apply tight_layout.

    `grid.all_axes` is the underlying 2D ndarray of axes returned by
    `plt.subplots(*shape, squeeze=False)` in `QubitGrid.__init__`. Use its
    `.shape` to scale the figure so each panel has consistent breathing
    room regardless of how many qubits are plotted.

    constrained_layout was tried but it crashes (ZeroDivisionError in
    _layoutgrid.grid_constraints) when combined with `axis("off")` cells
    that QubitGrid uses for empty grid slots, and qualang_tools' data
    handler calls `savefig(..., bbox_inches="tight")` which triggers a
    layout pass at save time. tight_layout with explicit padding works.
    """
    nrows, ncols = grid.all_axes.shape
    width = max(_MIN_WIDTH_INCH, ncols * _PER_COL_INCH)
    height = max(_MIN_HEIGHT_INCH, nrows * _PER_ROW_INCH)
    grid.fig.set_size_inches(width, height)
    grid.fig.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.0)


def _candidates_for_qubit(fits: xr.Dataset, qubit_name: str) -> List[float]:
    """Return all candidate RF frequencies (Hz) for a qubit, dropping NaN padding."""
    if fits is None or "candidates_rf_hz" not in fits:
        return []
    arr = fits.sel(qubit=qubit_name).candidates_rf_hz.values
    return [float(x) for x in arr if not np.isnan(x)]


def _per_segment_baselines(rf_hz: np.ndarray, values: np.ndarray, boundaries: List[float]) -> np.ndarray:
    """Return a per-sample baseline = the median of the segment each sample falls in.

    Each segment uses a different LO, so its absolute |I+jQ| baseline differs.
    Subtracting the per-segment median centers each segment around 0 and makes
    the segment-boundary jump in the wide trace go away. Median is robust to
    the resonator dips themselves.
    """
    if not boundaries:
        return np.full_like(values, np.median(values))
    edges = [-np.inf, *boundaries, np.inf]
    baseline = np.empty_like(values)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rf_hz >= lo) & (rf_hz < hi)
        if mask.any():
            baseline[mask] = np.median(values[mask])
    return baseline


def _per_segment_linear_detrend(
    rf_hz: np.ndarray, phase: np.ndarray, boundaries: List[float]
) -> np.ndarray:
    """Remove the cable-delay slope PER LO SEGMENT, not with one global line.

    The wide trace is stitched from several LO segments; each segment is
    phase-referenced to its OWN LO, so its effective electrical-delay slope and
    absolute phase offset differ slightly. A single global linear fit cannot
    flatten that piecewise structure and leaves a large V/W-shaped residual that
    is NOT a resonator feature — its vertex sits on a segment boundary, the same
    shape on every qubit (verified on real KRISS runs: a band3-only scan shows it
    too, so it is not a band-transition effect). Fitting and subtracting a line
    WITHIN each segment (and unwrapping within the segment, so inter-segment
    offset jumps never enter the unwrap) removes the cable slope and the
    stitching offsets, leaving only the O(1 rad) dispersive resonator features.

    Mirrors `_per_segment_baselines` (the per-segment median already used to flatten
    the amplitude trace). Falls back to the old global linear detrend when no
    segment boundaries are available (single-segment scan / older saved data).
    """
    if not boundaries:
        ph = np.unwrap(phase)
        slope, intercept = np.polyfit(rf_hz, ph, 1)
        return ph - (slope * rf_hz + intercept)
    edges = [-np.inf, *boundaries, np.inf]
    out = np.full_like(phase, np.nan, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rf_hz >= lo) & (rf_hz < hi)
        if mask.sum() < 2:
            if mask.any():
                out[mask] = 0.0
            continue
        ph_seg = np.unwrap(phase[mask])
        slope, intercept = np.polyfit(rf_hz[mask], ph_seg, 1)
        out[mask] = ph_seg - (slope * rf_hz[mask] + intercept)
    return out


def _draw_candidate_markers(
    ax: Axes,
    fits: xr.Dataset,
    qubit_name: str,
    assigned_f0_hz: float,
) -> None:
    """Draw gray dashed verticals at unassigned candidates, red at the assigned dip."""
    for c_hz in _candidates_for_qubit(fits, qubit_name):
        is_assigned = (
            not np.isnan(assigned_f0_hz)
            and abs(c_hz - assigned_f0_hz) < 1e6  # within 1 MHz = same dip
        )
        if is_assigned:
            continue
        ax.axvline(c_hz / u.GHz, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)


def plot_raw_amplitude_with_fit(
    ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset
) -> Figure:
    """One panel per qubit: wide amplitude trace + assigned/candidate markers + local fit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), fits)
    grid.fig.suptitle("Wide resonator spectroscopy (amplitude + fit)", fontsize=_FS_SUPTITLE)
    _setup_grid_figure(grid)
    return grid.fig


def plot_individual_amplitude_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fit_q: xr.Dataset,
    fits_all: xr.Dataset = None,
) -> None:
    """Single-qubit amplitude panel with assignment overlays."""
    ds_q = ds.loc[qubit]
    qubit_name = qubit["qubit"]
    rf_hz = ds_q.RF_frequency.values
    rf_ghz = rf_hz / u.GHz
    amp_mv_raw = ds_q.IQ_abs.values / u.mV

    boundaries_hz = list(ds.attrs.get("segment_boundaries_hz", []))
    baseline_mv = _per_segment_baselines(rf_hz, amp_mv_raw, boundaries_hz)
    amp_mv = amp_mv_raw - baseline_mv

    ax.plot(rf_ghz, amp_mv, color="steelblue", linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("RF frequency [GHz]", fontsize=_FS_LABEL)
    ax.set_ylabel(r"$R - \mathrm{seg\ median}$ [mV]", fontsize=_FS_LABEL)
    _apply_tick_fontsize(ax)

    f0_hz = float(fit_q.f0.values) if fit_q is not None else float("nan")
    _draw_candidate_markers(ax, fits_all, qubit_name, f0_hz)

    if fit_q is not None and not np.isnan(f0_hz):
        popt = fit_q.popt.values  # [f0, fwhm, amp, bg0, bg1]
        if not np.any(np.isnan(popt)):
            fwhm_hz = popt[1]
            # Fit curve drawn only over ±4 FWHM around f0 so it doesn't span the wide axis
            half_win = max(fwhm_hz * 4.0, 5e6)
            fit_mask = (rf_hz >= f0_hz - half_win) & (rf_hz <= f0_hz + half_win)
            if fit_mask.sum() >= 4:
                f_win = rf_hz[fit_mask]
                fit_curve_mv = lorentzian_dip_linbg(f_win, *popt) / u.mV
                # Subtract the same per-segment median so the overlay aligns
                # with the corrected blue trace.
                fit_curve_mv = fit_curve_mv - baseline_mv[fit_mask]
                ax.plot(
                    f_win / u.GHz, fit_curve_mv,
                    "r--", linewidth=1.5,
                    label=f"f₀={f0_hz / u.GHz:.4f} GHz, FWHM={fwhm_hz / u.MHz:.2f} MHz",
                )
            ax.axvline(
                f0_hz / u.GHz, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
            )
            ax.legend(fontsize=_FS_LEGEND, loc="upper right")
        else:
            # Assignment failed for this qubit; annotate inside the panel
            # rather than burying it in the title.
            ax.text(
                0.5, 0.95, "UNASSIGNED",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=_FS_LEGEND, color="gray",
            )

    ax.set_title(qubit_name, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD)


def plot_raw_phase(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
) -> Figure:
    """One panel per qubit: wide phase trace (per-segment cable slope removed) with markers."""
    boundaries_hz = list(ds.attrs.get("segment_boundaries_hz", []))
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]
        rf_hz = ds_q.RF_frequency.values
        # The cable-delay slope (2π × TOF × Δf) reaches O(10^3) rad over a wide span.
        # Detrend PER LO SEGMENT (not one global line): each segment is referenced to
        # its own LO, so a single global fit leaves a large V/W artifact whose vertex
        # sits on a segment boundary — not a resonator feature. See
        # `_per_segment_linear_detrend`.
        phase_detrended = _per_segment_linear_detrend(rf_hz, ds_q.phase.values, boundaries_hz)

        ax.plot(rf_hz / u.GHz, phase_detrended, color="steelblue", linewidth=1.0)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("RF frequency [GHz]", fontsize=_FS_LABEL)
        ax.set_ylabel("phase − per-segment linear fit [rad]", fontsize=_FS_LABEL)
        ax.set_title(qubit_name, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD)
        _apply_tick_fontsize(ax)

        f0_hz = float("nan")
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            f0_hz = float(fit_q.f0.values)
            _draw_candidate_markers(ax, fits, qubit_name, f0_hz)
            if not np.isnan(f0_hz):
                ax.axvline(
                    f0_hz / u.GHz, color="red", linestyle="--", linewidth=1.0,
                    label=f"f₀={f0_hz / u.GHz:.4f} GHz",
                )
                ax.legend(fontsize=_FS_LEGEND, loc="upper right")

    grid.fig.suptitle("Wide resonator spectroscopy (phase, per-segment linear-detrended)", fontsize=_FS_SUPTITLE)
    _setup_grid_figure(grid)
    return grid.fig


def plot_local_amplitude_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
) -> Figure:
    """Per-qubit amplitude trace zoomed to ±5×FWHM around the fit's f₀.

    Mirrors `plot_detrended_phase` / `plot_iq_circle` (same local window per
    qubit) but plots raw `IQ_abs` in mV with the Lorentzian + linear-bg fit
    overlay and an FWHM band. Useful as a sanity check that each fit caught
    the real dip — the wide amplitude figure can mask the dip when the
    per-segment baseline is large.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]
        rf_hz = ds_q.RF_frequency.values
        amp_mv = ds_q.IQ_abs.values / u.mV

        ax.set_xlabel("RF frequency [GHz]", fontsize=_FS_LABEL)
        ax.set_ylabel(r"$R=\sqrt{I^2+Q^2}$ [mV]", fontsize=_FS_LABEL)
        ax.set_title(qubit_name, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD)
        _apply_tick_fontsize(ax)

        if fits is None:
            ax.plot(rf_hz / u.GHz, amp_mv, color="steelblue", linewidth=1.0)
            continue

        fit_q = fits.sel(qubit=qubit_name)
        f0_hz = float(fit_q.f0.values)
        fwhm_hz = float(fit_q.fwhm.values)
        popt = fit_q.popt.values

        if (
            np.isnan(f0_hz) or np.isnan(fwhm_hz) or fwhm_hz <= 0
            or np.any(np.isnan(popt))
        ):
            ax.text(
                0.5, 0.5, "UNASSIGNED",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_FS_LABEL, color="gray",
            )
            continue

        # Same window as plot_detrended_phase / plot_iq_circle so reviewers
        # can compare the three local figures side-by-side per qubit.
        half_win = max(fwhm_hz * 5.0, 10e6)
        mask = (rf_hz >= f0_hz - half_win) & (rf_hz <= f0_hz + half_win)
        if mask.sum() < 8:
            ax.text(
                0.5, 0.5, "fit window too narrow",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_FS_LABEL,
            )
            continue

        rf_win = rf_hz[mask]
        amp_win = amp_mv[mask]
        ax.plot(rf_win / u.GHz, amp_win, color="steelblue", linewidth=1.0)

        # Lorentzian + linear-bg fit, evaluated only inside the local window.
        fit_curve_mv = lorentzian_dip_linbg(rf_win, *popt) / u.mV
        ax.plot(
            rf_win / u.GHz, fit_curve_mv, "r--", linewidth=1.5,
            label=f"f₀={f0_hz/u.GHz:.4f} GHz, FWHM={fwhm_hz/u.MHz:.2f} MHz",
        )
        ax.axvspan(
            (f0_hz - fwhm_hz / 2) / u.GHz,
            (f0_hz + fwhm_hz / 2) / u.GHz,
            alpha=0.15, color="red",
        )
        ax.axvline(
            f0_hz / u.GHz, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
        )
        ax.legend(fontsize=_FS_LEGEND, loc="upper right")

    grid.fig.suptitle(
        "Wide resonator spectroscopy (amplitude + fit, local)",
        fontsize=_FS_SUPTITLE,
    )
    _setup_grid_figure(grid)
    return grid.fig


def plot_detrended_phase(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
) -> Figure:
    """Per-qubit phase with a local degree-3 polynomial background subtracted.

    Detrending is done only inside the assigned fit window (±5×FWHM around f0).
    For unassigned qubits, no detrend is performed and a notice is shown.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]
        rf_hz = ds_q.RF_frequency.values
        phase = ds_q.phase.values

        if fits is None:
            ax.plot(rf_hz / u.GHz, phase, color="steelblue", linewidth=1.0)
            ax.set_title(qubit_name, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD)
            ax.set_xlabel("RF frequency [GHz]", fontsize=_FS_LABEL)
            ax.set_ylabel("phase [rad]", fontsize=_FS_LABEL)
            _apply_tick_fontsize(ax)
            continue

        fit_q = fits.sel(qubit=qubit_name)
        f0_hz = float(fit_q.f0.values)
        fwhm_hz = float(fit_q.fwhm.values)

        if not np.isnan(f0_hz) and not np.isnan(fwhm_hz) and fwhm_hz > 0:
            half_win = max(fwhm_hz * 5.0, 10e6)
            mask = (rf_hz >= f0_hz - half_win) & (rf_hz <= f0_hz + half_win)
            if mask.sum() >= 8:
                rf_win = rf_hz[mask]
                phase_win = phase[mask]
                # Background mask: exclude ±3×FWHM around f0
                bg_excl = np.abs(rf_win - f0_hz) > 3 * fwhm_hz
                bg_rf = rf_win[bg_excl] if bg_excl.sum() >= 4 else rf_win
                bg_phase = phase_win[bg_excl] if bg_excl.sum() >= 4 else phase_win
                coeffs = np.polyfit(bg_rf, bg_phase, deg=3)
                phase_bg = np.polyval(coeffs, rf_win)
                ax.plot(rf_win / u.GHz, phase_win - phase_bg, color="steelblue", linewidth=1.0)
                ax.axvline(f0_hz / u.GHz, color="red", linestyle="--", linewidth=1.0,
                           label=f"f₀={f0_hz / u.GHz:.4f} GHz")
                ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
                ax.legend(fontsize=_FS_LEGEND, loc="upper right")
            else:
                ax.text(0.5, 0.5, "fit window too narrow", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_FS_LABEL)
        else:
            ax.text(0.5, 0.5, "UNASSIGNED", ha="center", va="center",
                    transform=ax.transAxes, fontsize=_FS_LABEL, color="gray")

        ax.set_title(qubit_name, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD)
        ax.set_xlabel("RF frequency [GHz]", fontsize=_FS_LABEL)
        ax.set_ylabel("phase residual [rad]", fontsize=_FS_LABEL)
        _apply_tick_fontsize(ax)

    grid.fig.suptitle("Wide resonator spectroscopy (detrended phase, local)", fontsize=_FS_SUPTITLE)
    _setup_grid_figure(grid)
    return grid.fig


def plot_iq_circle(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
) -> Figure:
    """IQ parametric trace per qubit, colour-coded by RF, with star at assigned f0.

    For a wide scan most of the trace is far off-resonance, so we restrict the
    drawn IQ trace to the assigned dip's local window (±5×FWHM). Unassigned
    qubits show the full trace as a fallback.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]
        rf_hz = ds_q.RF_frequency.values
        I_mV = ds_q["I"].values / u.mV
        Q_mV = ds_q["Q"].values / u.mV

        mask = np.ones(len(rf_hz), dtype=bool)
        f0_hz = float("nan")
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            f0_hz = float(fit_q.f0.values)
            fwhm_hz = float(fit_q.fwhm.values)
            if not np.isnan(f0_hz) and not np.isnan(fwhm_hz) and fwhm_hz > 0:
                half_win = max(fwhm_hz * 5.0, 10e6)
                mask = (rf_hz >= f0_hz - half_win) & (rf_hz <= f0_hz + half_win)
                if mask.sum() < 8:
                    mask = np.ones(len(rf_hz), dtype=bool)

        I_w, Q_w = I_mV[mask], Q_mV[mask]
        rf_w_mhz = rf_hz[mask] / u.MHz

        ax.plot(I_w, Q_w, color="gray", linewidth=0.8, alpha=0.35, zorder=1)
        sc = ax.scatter(I_w, Q_w, c=rf_w_mhz, cmap="plasma", s=8, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlabel("I [mV]", fontsize=_FS_LABEL)
        ax.set_ylabel("Q [mV]", fontsize=_FS_LABEL)
        _apply_tick_fontsize(ax)

        cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("RF [MHz]", fontsize=_FS_CBAR)
        cbar.ax.tick_params(labelsize=_FS_CBAR)

        ax.set_title(qubit_name, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD)

        if not np.isnan(f0_hz):
            idx_full = int(np.argmin(np.abs(rf_hz - f0_hz)))
            ax.scatter(
                I_mV[idx_full], Q_mV[idx_full],
                color="red", marker="*", s=200, zorder=5,
                label=f"f₀={f0_hz / u.GHz:.4f} GHz",
            )
            ax.legend(fontsize=_FS_LEGEND, loc="upper right")

    grid.fig.suptitle("Wide resonator spectroscopy (IQ circle, local)", fontsize=_FS_SUPTITLE)
    _setup_grid_figure(grid)
    return grid.fig
