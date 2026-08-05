"""Plotting utilities for resonator spectroscopy versus power (IQ circles) calibration.

Self-contained copy of ``resonator_spectroscopy_vs_amplitude/plotting.py`` (the two
heatmap plots), plus a new ``plot_iq_circles_vs_power`` that overlays one raw I/Q circle
per readout power on a single axes per qubit, colour-coded by power [dBm]. The production
module is intentionally left untouched.
"""

from types import SimpleNamespace
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import complex_normalize, compute_quality_factors

u = unit(coerce_to_integer=True)

# ---------------------------------------------------------------------------
# Grid-size-aware figure scaling (mirrors resonator_spectroscopy_new/plotting.py)
# ---------------------------------------------------------------------------
# The original module hard-coded a fixed 15x9 in figure AND large absolute font
# sizes tuned for a *single* panel.  On a multiplexed grid the figure stayed
# 15x9 while each panel shrank, so the 24 pt text / per-subplot colorbars
# collided and the grid looked mismatched as the qubit count grew.  Here every
# figure derives a single scale factor ``s`` from the grid layout
# (``grid.all_axes.shape``) and scales the figure size AND all text/marker sizes
# together, so each panel keeps the same font-to-panel ratio as the single-qubit
# plot (1 qubit -> ``s = 1`` -> the original 15x9 + base-font look).
# ---------------------------------------------------------------------------
# Reference single-panel geometry + bounds (tune the figure look here)
_REF_PANEL_W = 15.0   # inches, width of one reference panel
_REF_PANEL_H = 9.0    # inches, height of one reference panel
# Maximum total figure size for a multiplexed grid.  ``s`` never exceeds 1, so
# panels are never larger than the reference; large grids are capped here and
# fonts shrink to fit.
_TARGET_W = 24.0      # inches, max total figure width
_TARGET_H = 15.0      # inches, max total figure height

# Base (gold) font / pad sizes — applied unscaled to a single panel
_BASE_FS_SUPTITLE = 28   # figure suptitle
_BASE_FS_TITLE    = 24   # per-subplot title
_BASE_FS_LABEL    = 24   # axis label (xlabel / ylabel)
_BASE_FS_TICK     = 22   # tick-label size
_BASE_FS_LEGEND   = 20   # legend text
_BASE_FS_CBAR     = 20   # colorbar label / ticks
# Extra vertical padding (pts) for subplot titles on plots with a twiny() top
# x-axis: the top ticks + label need clearing.  Plain plots use the small pad.
_BASE_TITLE_PAD_TWINY = 40   # plots with twiny top x-axis
_BASE_TITLE_PAD       =  8   # plots without a top x-axis


def _clip(value: float, floor: float) -> float:
    """Return *value* but never below *floor* (keeps thin lines/markers visible)."""
    return value if value >= floor else floor


def _style_for_grid(grid) -> SimpleNamespace:
    """Derive a grid-size-aware style from a constructed :class:`QubitGrid`.

    ``grid.all_axes`` is the full ``(nrows, ncols)`` array of axes created by
    ``plt.subplots`` inside ``QubitGrid``, so its shape gives the layout.  The
    scale factor ``s`` is chosen so the total figure fits within
    ``_TARGET_W`` x ``_TARGET_H`` while never exceeding the reference panel size
    (``s <= 1``).  All font / pad / line / marker sizes scale with ``s``.
    """
    nrows, ncols = grid.all_axes.shape
    s = min(
        1.0,
        _TARGET_W / (_REF_PANEL_W * ncols),
        _TARGET_H / (_REF_PANEL_H * nrows),
    )
    return SimpleNamespace(
        s=s,
        nrows=nrows,
        ncols=ncols,
        # Total figure size (bounded by the targets; >= a single reference panel)
        fig_w=_REF_PANEL_W * s * ncols,
        fig_h=_REF_PANEL_H * s * nrows,
        # Fonts scale linearly with the panel
        fs_suptitle=_BASE_FS_SUPTITLE * s,
        fs_title=_BASE_FS_TITLE * s,
        fs_label=_BASE_FS_LABEL * s,
        fs_tick=_BASE_FS_TICK * s,
        fs_legend=_BASE_FS_LEGEND * s,
        fs_cbar=_BASE_FS_CBAR * s,
        # Title padding scales with the panel too (it must clear the top axis)
        pad_twiny=_BASE_TITLE_PAD_TWINY * s,
        pad=_BASE_TITLE_PAD * s,
        # Line / marker sizes scale linearly (with visibility floors). The fit-overlay
        # lines (migration / optimal / dressed-freq) use the thicker floors below so they
        # stay clearly visible on the viridis heatmap even on a multi-qubit grid.
        lw_line=_clip(1.4 * s, 1.0),
        lw_thin=_clip(1.2 * s, 0.9),
        lw_hair=_clip(0.5 * s, 0.4),
        lw_highlight=_clip(3.2 * s, 2.2),
        scatter_s=_clip(40.0 * s, 12.0),
        scatter_s_small=_clip(28.0 * s, 10.0),
    )


def _base_style() -> SimpleNamespace:
    """Single-panel (gold) style, for helpers called without a grid style (s = 1)."""
    return SimpleNamespace(
        s=1.0, nrows=1, ncols=1,
        fig_w=_REF_PANEL_W, fig_h=_REF_PANEL_H,
        fs_suptitle=_BASE_FS_SUPTITLE, fs_title=_BASE_FS_TITLE, fs_label=_BASE_FS_LABEL,
        fs_tick=_BASE_FS_TICK, fs_legend=_BASE_FS_LEGEND, fs_cbar=_BASE_FS_CBAR,
        pad_twiny=_BASE_TITLE_PAD_TWINY, pad=_BASE_TITLE_PAD,
        lw_line=1.4, lw_thin=1.2, lw_hair=0.5, lw_highlight=3.2,
        scatter_s=40.0, scatter_s_small=28.0,
    )


def _apply_tick_fontsize(ax: Axes, size: float) -> None:
    """Set tick-label fontsize on both axes of *ax*."""
    ax.tick_params(axis="both", labelsize=size)


def _finalize(grid, st: SimpleNamespace, suptitle: str) -> "Figure":
    """Apply the shared figure-level styling (size, suptitle, layout)."""
    grid.fig.suptitle(suptitle, fontsize=st.fs_suptitle)
    grid.fig.set_size_inches(st.fig_w, st.fig_h)
    # Reserve a little headroom for the (scaled) suptitle so it never collides
    # with the per-subplot titles / twiny top axes.
    grid.fig.tight_layout(rect=[0, 0, 1, 0.97])
    return grid.fig


def _power_update_annotation(fit) -> str:
    """Text summarizing the state values that will be written for this resonator.

    ``full_scale_power_dbm`` is SHARED across all resonators on the same readout line, so it
    is labelled with the line id. Returns "fit failed" when the fit was unsuccessful.
    """
    try:
        success = bool(fit.success)
    except Exception:
        success = False
    if not success:
        # Surface WHY: the success gate certifies the low-power DRESSED dip's SNR
        # (see analysis._analyze_one_qubit) — show it so a weak/dead dressed
        # resonance is distinguishable from a sweep/power-range problem.
        try:
            snr = float(fit.dressed_snr)
            if np.isfinite(snr):
                return f"fit failed (dressed dip SNR {snr:.1f} < 5)"
        except Exception:
            pass
        return "fit failed"
    lines = [f"optimal = {float(fit.optimal_power):.1f} dBm"]
    line_id = str(fit.readout_line.values) if hasattr(fit.readout_line, "values") else str(fit.readout_line)
    fs = float(fit.target_full_scale_power_dbm)
    amp = float(fit.target_amplitude)
    if np.isfinite(fs):
        lines.append(f"full-scale = {int(fs)} dBm  (shared: {line_id})")
    if np.isfinite(amp):
        lines.append(f"amplitude = {amp:.4f} V")
    try:
        if not bool(fit.punchout):
            lines.append("no punch-out: widen sweep")
    except Exception:
        pass
    return "\n".join(lines)


def _annotate_update(ax: Axes, fit, st: SimpleNamespace) -> None:
    """Draw the per-resonator state-update annotation box (top-left, semi-transparent)."""
    ax.text(
        0.02, 0.98, _power_update_annotation(fit), transform=ax.transAxes,
        va="top", ha="left", fontsize=st.fs_legend, zorder=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )


def _circle_fit_kasa(x, y):
    """Algebraic (Kasa) least-squares circle fit. Returns ``(cx, cy, R)``.

    Fits the circle minimising the algebraic distance to the points ``(x, y)``. Data are
    mean-centred first for numerical conditioning. Falls back to the centroid with
    ``R = nan`` if the normal equations are singular (e.g. near-collinear points). Inputs
    and outputs share the same units.
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


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the raw data with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), st)

    return _finalize(grid, st, "Resonator spectroscopy vs power")


def plot_individual_raw_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, st: SimpleNamespace = None
):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    if st is None:
        st = _base_style()
    qubit_name = qubit["qubit"]

    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax,
        add_colorbar=False,
        x="freq_GHz",
        y="power",
        linewidth=st.lw_hair,
    )
    ax.set_xlabel("RF frequency [GHz]", fontsize=st.fs_label)
    ax.set_ylabel("Power (dBm)", fontsize=st.fs_label)
    _apply_tick_fontsize(ax, st.fs_tick)
    ax2 = ax.twiny()
    ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True
    )
    ax2.set_xlabel("Detuning [MHz]", fontsize=st.fs_label)
    ax2.set_title("")  # clear xarray auto-title on the twin axis
    _apply_tick_fontsize(ax2, st.fs_tick)
    # Centered subplot title — pad clears the twiny top-axis label
    ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad_twiny)
    # Per-power dressed resonance trace (orange)
    ax2.plot(
        fit.res_freq_vs_power * 1e-6,
        fit.power,
        color="orange",
        linewidth=st.lw_thin,
        alpha=0.9,
    )
    # Optimal readout power (red, high-contrast vs the viridis background) and the
    # dressed resonance frequency (cyan dashed).
    if bool(fit.success):
        ax2.axhline(y=float(fit.optimal_power), color="red", linestyle="-", linewidth=st.lw_highlight, alpha=0.9)
        ax2.axvline(x=float(fit.freq_shift) * 1e-6, color="deepskyblue", linestyle="--", linewidth=st.lw_highlight, alpha=0.9)
    # Annotate the state values that will be written (full-scale is shared per readout line).
    # Draw on ax2 (the twiny top axes) so it is not hidden behind ax2's heatmap.
    _annotate_update(ax2, fit, st)
    max_amp = ds.attrs.get("max_amp")
    max_power_dbm = ds.attrs.get("max_power_dbm")
    if max_amp is not None and max_power_dbm is not None:
        def dbm_to_amp(p):
            return max_amp * 10 ** ((p - max_power_dbm) / 20)

        def amp_to_dbm(a):
            return max_power_dbm + 20 * np.log10(np.maximum(a, 1e-12) / max_amp)

        ax_right = ax.secondary_yaxis("right", functions=(dbm_to_amp, amp_to_dbm))
        ax_right.set_ylabel("Readout amplitude (V)", fontsize=st.fs_label)
        ax_right.tick_params(axis="y", labelsize=st.fs_tick)
        # Place ticks at log-spaced amplitude values so they are evenly readable
        y_min, y_max = ax.get_ylim()
        amp_min_val = max(dbm_to_amp(y_min), 1e-9)
        amp_max_val = dbm_to_amp(y_max)
        if amp_min_val < amp_max_val:
            amp_ticks = np.geomspace(amp_min_val, amp_max_val, 8)
            ax_right.set_yticks(amp_ticks)
        ax_right.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))


def plot_raw_data_amp_linear(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> "Figure":
    """
    Plots the IQ response as a 2D colormap with linear amplitude (V) on the y-axis.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data (must have an ``amplitude`` coordinate).
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_amp_linear(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), st)

    return _finalize(grid, st, "Resonator spectroscopy vs power (linear amplitude axis)")


def plot_individual_raw_data_amp_linear(
    ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None, st: SimpleNamespace = None
):
    """
    Plots a single qubit's IQ response with linear amplitude (V) on the y-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot.
    ds : xr.Dataset
        The dataset containing the quadrature data (must have an ``amplitude`` coordinate).
    qubit : dict
        Mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters.
    """
    if st is None:
        st = _base_style()
    qubit_name = qubit["qubit"]
    max_amp = ds.attrs.get("max_amp")
    max_power_dbm = ds.attrs.get("max_power_dbm")

    ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
        ax=ax,
        add_colorbar=False,
        x="detuning_MHz",
        y="amplitude",
        robust=True,
    )
    ax.set_ylabel("Readout amplitude (V)", fontsize=st.fs_label)
    ax.set_xlabel("Detuning [MHz]", fontsize=st.fs_label)
    _apply_tick_fontsize(ax, st.fs_tick)
    ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)

    if fit is not None:
        # Per-power dressed resonance trace in detuning space (MHz)
        ax.plot(
            fit.res_freq_vs_power.values * 1e-6,
            ds.loc[qubit].amplitude.values,
            color="orange",
            linewidth=st.lw_thin,
            alpha=0.9,
        )
        if bool(fit.success) and max_amp is not None and max_power_dbm is not None:
            optimal_amp = max_amp * 10 ** ((float(fit.optimal_power) - max_power_dbm) / 20)
            ax.axhline(y=optimal_amp, color="red", linestyle="-", linewidth=st.lw_highlight, alpha=0.9)
            ax.axvline(x=float(fit.freq_shift) * 1e-6, color="deepskyblue", linestyle="--", linewidth=st.lw_highlight, alpha=0.9)
        # Annotate the state values that will be written (full-scale is shared per readout line)
        _annotate_update(ax, fit, st)


def plot_iq_circles_vs_power(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
    num_circles: int = 12,
) -> Figure:
    """Overlay raw I/Q circles, one per readout power, on a single axes per qubit.

    Each readout-power slice of the vs-power dataset is, in effect, a single resonator
    spectroscopy: as the detuning is swept, ``(I(f), Q(f))`` traces a circle in the I/Q
    plane.  This plot draws those circles together — colour-coded by readout power [dBm] —
    so that the radius growth with power and its deformation in the punch-out (non-linear)
    regime are visible at a glance.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset; must contain raw ``I`` and ``Q`` (in V) with a ``power`` dim.
    qubits : list of AnyTransmon
        Qubits to plot (one subplot each, laid out by ``grid_location``).
    fits : xr.Dataset, optional
        Fit dataset.  When provided and the fit succeeded, the circle at the fitted
        ``optimal_power`` is highlighted as a thick red trace.
    num_circles : int
        Number of powers to draw, evenly sub-sampled from the full power axis. Default 12.

    Returns
    -------
    Figure
        The matplotlib figure object containing the per-qubit grid.

    Notes
    -----
    The centre of each drawn circle is also marked, via two estimators: an algebraic
    circle fit (``o`` markers) and the plain centroid (``x`` markers), both coloured by
    power.  Because the centre drift is small on this full-scale view, see the companion
    ``plot_iq_circle_centers_vs_power`` for a zoomed centre-locus plot.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)

    powers = ds.power.values  # dBm
    n_pow = len(powers)
    # Evenly sub-sample power indices (unique, so num_circles > n_pow is harmless)
    sel = np.unique(np.linspace(0, n_pow - 1, min(num_circles, n_pow)).astype(int))
    norm = mcolors.Normalize(vmin=float(powers.min()), vmax=float(powers.max()))
    cmap = plt.get_cmap("viridis")

    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]

        # One raw I/Q circle per sub-sampled power (coloured by power), and its centre
        cf_pts, cen_pts, sel_powers = [], [], []
        for k in sel:
            I_mV = ds_q["I"].isel(power=k).values / u.mV
            Q_mV = ds_q["Q"].isel(power=k).values / u.mV
            ax.plot(
                I_mV, Q_mV,
                color=cmap(norm(float(powers[k]))),
                linewidth=st.lw_line, alpha=0.85, zorder=2,
            )
            cfx, cfy, _ = _circle_fit_kasa(I_mV, Q_mV)
            cf_pts.append((cfx, cfy))
            cen_pts.append((float(I_mV.mean()), float(Q_mV.mean())))
            sel_powers.append(float(powers[k]))
        cf_pts = np.array(cf_pts)
        cen_pts = np.array(cen_pts)

        # Centre of each drawn circle: 'o' = circle fit, 'x' = centroid (coloured by power)
        ax.scatter(cf_pts[:, 0], cf_pts[:, 1], c=sel_powers, cmap=cmap, norm=norm,
                   marker="o", s=st.scatter_s, edgecolors="black", linewidths=st.lw_hair, zorder=4)
        ax.scatter(cen_pts[:, 0], cen_pts[:, 1], c=sel_powers, cmap=cmap, norm=norm,
                   marker="x", s=st.scatter_s, linewidths=st.lw_thin, zorder=4)

        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
                   markeredgecolor="black", markersize=9, label="circle-fit center"),
            Line2D([0], [0], marker="x", color="gray", linestyle="none",
                   markersize=9, label="centroid"),
        ]

        # Highlight the circle at the fitted optimal readout power
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            if bool(fit_q.success.values):
                opt_dbm = float(fit_q.optimal_power.values)
                kopt = int(np.argmin(np.abs(powers - opt_dbm)))
                ax.plot(
                    ds_q["I"].isel(power=kopt).values / u.mV,
                    ds_q["Q"].isel(power=kopt).values / u.mV,
                    color="red", linewidth=st.lw_highlight, zorder=5,
                )
                legend_handles.append(
                    Line2D([0], [0], color="red", lw=st.lw_highlight, label=f"optimal = {opt_dbm:.1f} dBm"))

        ax.set_aspect("equal")
        ax.set_xlabel("I [mV]", fontsize=st.fs_label)
        ax.set_ylabel("Q [mV]", fontsize=st.fs_label)
        _apply_tick_fontsize(ax, st.fs_tick)

        # Per-subplot colorbar mapped to readout power (keeps subplot titles centered)
        cbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        cbar.set_label("Readout power [dBm]", fontsize=st.fs_cbar)
        cbar.ax.tick_params(labelsize=st.fs_cbar)

        ax.legend(handles=legend_handles, fontsize=st.fs_legend, loc="upper right")
        # Title set AFTER colorbar so it is centered over the (now-final) axes width
        ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)

    return _finalize(grid, st, "Resonator spectroscopy vs power (IQ circles)")


def plot_iq_circle_centers_vs_power(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
) -> Figure:
    """Plot the locus of I/Q circle centres versus readout power (zoomed).

    For every readout power the I/Q trace is reduced to a single centre point with two
    estimators — an algebraic (Kasa) circle fit (``o``) and the plain centroid (``x``) —
    and the centres are drawn together, colour-coded by power [dBm] and connected in power
    order so the drift direction is visible.  The centre drift is typically tiny compared
    with the circle radius, so the axes auto-scale to the centre locus only: this is a
    deliberate zoom and is NOT on the same scale as ``plot_iq_circles_vs_power``.  Axis
    labels are in mV so the absolute drift is readable; a text box reports the circle-fit
    drift span (peak-to-peak) in I and Q.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with raw ``I`` and ``Q`` (in V) and a ``power`` dim.
    qubits : list of AnyTransmon
        Qubits to plot (one subplot each, laid out by ``grid_location``).
    fits : xr.Dataset, optional
        Fit dataset; when the fit succeeded the centre at ``optimal_power`` is ringed red.

    Returns
    -------
    Figure
        The matplotlib figure object containing the per-qubit grid.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    powers = ds.power.values  # dBm
    norm = mcolors.Normalize(vmin=float(powers.min()), vmax=float(powers.max()))
    cmap = plt.get_cmap("viridis")

    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]

        # Centre of every power slice, via both estimators (units: mV)
        cf = np.full((len(powers), 2), np.nan)
        cen = np.full((len(powers), 2), np.nan)
        for k in range(len(powers)):
            I_mV = ds_q["I"].isel(power=k).values / u.mV
            Q_mV = ds_q["Q"].isel(power=k).values / u.mV
            cfx, cfy, _ = _circle_fit_kasa(I_mV, Q_mV)
            cf[k] = (cfx, cfy)
            cen[k] = (float(I_mV.mean()), float(Q_mV.mean()))

        # Connecting lines (power order) reveal the drift direction
        ax.plot(cf[:, 0], cf[:, 1], color="0.6", lw=st.lw_thin, alpha=0.6, zorder=1)
        ax.plot(cen[:, 0], cen[:, 1], color="0.6", lw=st.lw_thin, ls="--", alpha=0.5, zorder=1)
        # Centres coloured by power
        ax.scatter(cf[:, 0], cf[:, 1], c=powers, cmap=cmap, norm=norm,
                   marker="o", s=st.scatter_s_small, edgecolors="black", linewidths=st.lw_hair, zorder=3)
        ax.scatter(cen[:, 0], cen[:, 1], c=powers, cmap=cmap, norm=norm,
                   marker="x", s=st.scatter_s_small, linewidths=st.lw_line, zorder=3)

        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
                   markeredgecolor="black", markersize=9, label="circle-fit center"),
            Line2D([0], [0], marker="x", color="gray", linestyle="none",
                   markersize=9, label="centroid"),
        ]

        # Ring the centre at the optimal readout power
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            if bool(fit_q.success.values):
                opt_dbm = float(fit_q.optimal_power.values)
                kopt = int(np.argmin(np.abs(powers - opt_dbm)))
                ax.scatter(cf[kopt, 0], cf[kopt, 1], s=st.scatter_s_small * 7, facecolors="none",
                           edgecolors="red", linewidths=st.lw_highlight, zorder=5)
                legend_handles.append(
                    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                           markeredgecolor="red", markersize=12, markeredgewidth=2.0,
                           label=f"optimal = {opt_dbm:.1f} dBm"))

        # Quantify the (small) drift from the circle-fit centres
        valid = ~np.isnan(cf[:, 0])
        if valid.sum() > 1:
            dI = float(np.nanmax(cf[valid, 0]) - np.nanmin(cf[valid, 0]))
            dQ = float(np.nanmax(cf[valid, 1]) - np.nanmin(cf[valid, 1]))
            ax.text(0.02, 0.98, f"drift: dI={dI:.2f}, dQ={dQ:.2f} mV",
                    transform=ax.transAxes, va="top", ha="left", fontsize=st.fs_legend,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        ax.set_xlabel("I center [mV]", fontsize=st.fs_label)
        ax.set_ylabel("Q center [mV]", fontsize=st.fs_label)
        _apply_tick_fontsize(ax, st.fs_tick)
        cbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        cbar.set_label("Readout power [dBm]", fontsize=st.fs_cbar)
        cbar.ax.tick_params(labelsize=st.fs_cbar)
        ax.legend(handles=legend_handles, fontsize=st.fs_legend, loc="upper right")
        ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)

    return _finalize(grid, st, "Resonator spectroscopy vs power (IQ circle centers)")


def plot_dip_traces_vs_power(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
    num_circles: int = 12,
    normalize: bool = True,
) -> Figure:
    """Overlay the resonator dip trace for each selected readout power.

    Uses the SAME evenly sub-sampled powers as ``plot_iq_circles_vs_power`` (controlled by
    ``num_circles``), so each circle there has a matching dip trace here.  One line per power
    is plotted against detuning [MHz], colour-coded by power [dBm], and the trace at the
    fitted optimal power is highlighted in red.

    ``normalize`` selects the y-axis quantity:

    - ``True`` (default): ``IQ_abs_norm = |IQ| / mean_over_detuning(|IQ|)`` — the readout
      drive-amplitude scaling (which grows with power) is divided out, so the RELATIVE dip
      depth/contrast is directly comparable across powers.
    - ``False``: raw ``R = sqrt(I^2 + Q^2)`` in mV — the literal amplitude, so traces stack
      vertically with power (drive grows) but the absolute dip is visible.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset; must contain ``IQ_abs_norm`` (normalize=True) or ``IQ_abs``
        (normalize=False) with a ``power`` dim.
    qubits : list of AnyTransmon
        Qubits to plot (one subplot each, laid out by ``grid_location``).
    fits : xr.Dataset, optional
        Fit dataset; when the fit succeeded the trace at ``optimal_power`` is drawn in red.
    num_circles : int
        Number of powers to draw, evenly sub-sampled from the full power axis (kept equal to
        the circle plot's value so the two figures show the same powers). Default 12.
    normalize : bool
        Plot normalized contrast (True) or raw |IQ| in mV (False). Default True.

    Returns
    -------
    Figure
        The matplotlib figure object containing the per-qubit grid.
    """
    var = "IQ_abs_norm" if normalize else "IQ_abs"
    y_scale = 1.0 if normalize else u.mV
    y_label = "normalized |IQ| / mean(|IQ|)" if normalize else "R = sqrt(I^2 + Q^2) [mV]"
    sup = (
        "Resonator spectroscopy vs power (dip traces, normalized)"
        if normalize
        else "Resonator spectroscopy vs power (dip traces, raw |IQ|)"
    )

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)

    powers = ds.power.values  # dBm
    n_pow = len(powers)
    sel = np.unique(np.linspace(0, n_pow - 1, min(num_circles, n_pow)).astype(int))
    norm = mcolors.Normalize(vmin=float(powers.min()), vmax=float(powers.max()))
    cmap = plt.get_cmap("viridis")

    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        qubit_name = qubit["qubit"]
        detuning_mhz = ds_q.detuning.values / u.MHz

        # One dip trace per sub-sampled power, coloured by power [dBm]
        for k in sel:
            y = ds_q[var].isel(power=k).values / y_scale
            ax.plot(detuning_mhz, y, color=cmap(norm(float(powers[k]))),
                    linewidth=st.lw_line, alpha=0.85, zorder=2)

        # Highlight the dip trace at the fitted optimal readout power
        legend_handles = []
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            if bool(fit_q.success.values):
                opt_dbm = float(fit_q.optimal_power.values)
                kopt = int(np.argmin(np.abs(powers - opt_dbm)))
                ax.plot(detuning_mhz, ds_q[var].isel(power=kopt).values / y_scale,
                        color="red", linewidth=st.lw_highlight, zorder=5)
                legend_handles.append(
                    Line2D([0], [0], color="red", lw=st.lw_highlight, label=f"optimal = {opt_dbm:.1f} dBm"))

        ax.set_xlabel("Detuning [MHz]", fontsize=st.fs_label)
        ax.set_ylabel(y_label, fontsize=st.fs_label)
        _apply_tick_fontsize(ax, st.fs_tick)
        cbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        cbar.set_label("Readout power [dBm]", fontsize=st.fs_cbar)
        cbar.ax.tick_params(labelsize=st.fs_cbar)
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=st.fs_legend, loc="upper right")
        ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)

    return _finalize(grid, st, sup)


def plot_normalized_complex_response(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    num_circles: int = 12,
) -> Figure:
    """Overlay the delay-removed, off-resonance-normalized complex response per qubit.

    For sub-sampled readout powers, ``S_norm = (I+iQ)/<I+iQ>_offres`` (with the electrical
    delay removed) is drawn in the complex plane, zoomed to the resonance.  This is the
    clean version of the customer-style "normalized complex response": the off-resonance
    point sits at ~(1, 0) and each power traces a clean circle.  Circles overlapping across
    power means the contrast (and the resonator response) is essentially power-independent
    in that range; a shrinking/shifting circle flags a real power dependence.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with raw ``I``, ``Q`` (V) and a ``full_freq`` coordinate.
    qubits : list of AnyTransmon
        Qubits to plot (one subplot each, laid out by ``grid_location``).
    num_circles : int
        Number of powers to draw, evenly sub-sampled from the full power axis. Default 12.

    Returns
    -------
    Figure
        The matplotlib figure object containing the per-qubit grid.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    powers = ds.power.values
    n_pow = len(powers)
    sel = np.unique(np.linspace(0, n_pow - 1, min(num_circles, n_pow)).astype(int))
    norm = mcolors.Normalize(vmin=float(powers.min()), vmax=float(powers.max()))
    cmap = plt.get_cmap("viridis")

    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        ff = np.asarray(ds["full_freq"].sel(qubit=qubit_name).values, dtype=float)
        for k in sel:
            I = ds["I"].sel(qubit=qubit_name).isel(power=k).values
            Q = ds["Q"].sel(qubit=qubit_name).isel(power=k).values
            Sn = complex_normalize(ff, np.asarray(I) + 1j * np.asarray(Q))
            im = int(np.argmin(np.abs(Sn)))
            m = np.abs(ff - ff[im]) <= 5e5  # +/- 0.5 MHz around resonance
            ax.plot(Sn[m].real, Sn[m].imag, color=cmap(norm(float(powers[k]))),
                    linewidth=st.lw_line, alpha=0.85, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(S_norm)", fontsize=st.fs_label)
        ax.set_ylabel("Im(S_norm)", fontsize=st.fs_label)
        _apply_tick_fontsize(ax, st.fs_tick)
        cbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        cbar.set_label("Readout power [dBm]", fontsize=st.fs_cbar)
        cbar.ax.tick_params(labelsize=st.fs_cbar)
        ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)

    return _finalize(grid, st, "Resonator spectroscopy vs power (normalized complex response)")


def plot_quality_factors_vs_power(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
) -> Figure:
    """Per qubit, plot complex-fit Q factors (Qi/Qc/Ql, log) and contrast versus power.

    Computed by ``compute_quality_factors`` (delay removal + complex normalization +
    circle/phase diameter-correction fit). Qi rising with power is the TLS signature; a
    contrast that collapses (with the resonance shifting, see the heatmap figure) is
    punch-out. The naive-min contrast is overlaid to expose the low-SNR artifact — it dives
    at low power while the robust (smoothed) contrast does not.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with raw ``I``, ``Q`` (V) and a ``full_freq`` coordinate.
    qubits : list of AnyTransmon
        Qubits to plot (one subplot each, laid out by ``grid_location``).

    Returns
    -------
    Figure
        The matplotlib figure object containing the per-qubit grid.
    """
    qf = compute_quality_factors(ds)
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    powers = np.asarray(ds.power.values, dtype=float)

    for ax, qubit in grid_iter(grid):
        qn = qubit["qubit"]
        q = qf.sel(qubit=qn)
        ax.semilogy(powers, q["Qi"].values, "o-", color="tab:green", label="Qi")
        ax.semilogy(powers, q["Qc"].values, "s-", color="tab:orange", label="|Qc|")
        ax.semilogy(powers, q["Ql"].values, "^-", color="tab:blue", label="Ql")
        ax.set_xlabel("Readout power [dBm]", fontsize=st.fs_label)
        ax.set_ylabel("quality factor", fontsize=st.fs_label)
        _apply_tick_fontsize(ax, st.fs_tick)
        ax.grid(alpha=0.3, which="both")

        axc = ax.twinx()
        axc.plot(powers, q["contrast"].values, "d-", color="black", label="contrast (robust)")
        axc.plot(powers, q["contrast_naive"].values, "x--", color="gray", alpha=0.7,
                 label="contrast (naive min)")
        axc.set_ylabel("contrast", fontsize=st.fs_label)
        axc.set_ylim(0, 1.05)
        axc.tick_params(axis="y", labelsize=st.fs_tick)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axc.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=st.fs_legend, loc="best")
        ax.set_title(f"qubit = {qn}", loc="center", fontsize=st.fs_title, pad=st.pad)

    return _finalize(grid, st, "Resonator spectroscopy vs power (complex-fit Qi/Qc/Ql + contrast)")
