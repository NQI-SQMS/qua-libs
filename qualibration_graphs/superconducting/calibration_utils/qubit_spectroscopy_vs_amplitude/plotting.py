"""Plotting utilities for qubit spectroscopy versus drive power calibration.

Mirrors ``resonator_spectroscopy_vs_amplitude/plotting.py``: a 2D colormap of the
rotated quadrature vs (drive-frequency, drive-power), the fitted peak-position
trace vs power, the optimal-power line, and a dual y-axis (power dBm + drive
amplitude V).  A second figure renders the same data on a linear amplitude axis.

All fit overlays are drawn as HIGH-CONTRAST lines (bright core + black halo via
path_effects) so they stay visible against the viridis colormap, whose mid-tones
are green/teal — a thin plain-coloured line used to disappear there.
"""

from typing import Dict, List

import numpy as np
import matplotlib.ticker
import matplotlib.patheffects as pe
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

# High-contrast styling for fit overlays (bright core + black outline), so they pop on viridis.
_HALO = [pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()]
_HALO_THIN = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]
_PEAK_COLOR = "#FFE600"   # bright yellow-white: the GE peak-position trace
_OPT_COLOR = "#FF8000"    # bright orange: the optimal-power line
_GE_COLOR = "#FFFFFF"     # white: the chosen GE detuning line
_EF_COLOR = "#00E5FF"     # bright cyan: the FITTED 2-photon (EF) line

_WARN_TEXT = {
    "widen_range": "EF/anharm. outside span -> widen frequency_span",
    "anharm_smaller": "measured |alpha| < stored (EF inside narrow span)",
}


def _annotate_ef_warning(ax: Axes, fit: xr.Dataset):
    """Draw a small flag in the panel when the swept span does not reach the expected 2-photon location."""
    try:
        warn = str(fit.ef_warning.values)
    except Exception:
        warn = ""
    msg = _WARN_TEXT.get(warn)
    if msg:
        ax.text(
            0.02, 0.02, "[!] " + msg, transform=ax.transAxes, fontsize=7, color="black",
            va="bottom", ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFD400", ec="black", alpha=0.92),
        )


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Grid of per-qubit panels: rotated-I colormap vs (RF frequency, drive power) with fit overlay."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    grid.fig.suptitle("Qubit spectroscopy vs drive power")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None):
    """Single-qubit panel: rotated-I colormap, peak trace vs power, optimal-power line, dual y-axis."""
    (ds.assign_coords(freq_GHz=ds.full_freq / u.GHz).loc[qubit].I_rot / u.mV).plot(
        ax=ax, add_colorbar=False, x="freq_GHz", y="power", robust=True,
    )
    ax.set_ylabel("Power (dBm)")
    ax.set_xlabel("RF frequency [GHz]")

    ax2 = ax.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].I_rot / u.mV).plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True,
    )
    ax2.set_xlabel("Detuning [MHz]")
    ax2.set_title("")

    if fit is not None:
        # GE search window (edges) + expected EF / 2-photon positions (from stored anharmonicity, dim ref lines)
        win = fit.attrs.get("ge_window_mhz")
        if win:
            for edge in (-float(win), float(win)):
                ax2.axvline(edge, color="limegreen", linestyle="--", linewidth=1.0, alpha=0.7)
        rf = float((ds.full_freq.loc[qubit] - ds.detuning).mean())
        for name, color in (("twophoton_freq", "magenta"), ("ef_freq", "brown")):
            v = float(fit[name].values)
            if np.isfinite(v):
                ax2.axvline((v - rf) * 1e-6, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        # Fitted GE peak position for each drive power (HIGH CONTRAST)
        ax2.plot(fit.peak_position * 1e-6, fit.power, color=_PEAK_COLOR, linewidth=2.0,
                 path_effects=_HALO, zorder=6)
        if bool(fit.success):
            ax2.axhline(y=float(fit.optimal_power), color=_OPT_COLOR, linestyle="-",
                        linewidth=2.0, path_effects=_HALO_THIN, zorder=6)
            ax2.axvline(x=float(fit.freq_shift) * 1e-6, color=_GE_COLOR, linestyle="--",
                        linewidth=1.8, path_effects=_HALO_THIN, zorder=6)
        # Fitted 2-photon (EF) line -> measured anharmonicity (HIGH CONTRAST, distinct colour)
        if bool(fit.ef_success):
            d2 = (float(fit.twophoton_freq_fitted) - rf) * 1e-6
            ax2.axvline(x=d2, color=_EF_COLOR, linestyle="-", linewidth=1.8,
                        path_effects=_HALO_THIN, zorder=6)
        # annotate on ax2 (the topmost / twiny axes) so the flag is not hidden behind its colormap
        _annotate_ef_warning(ax2, fit)

    max_amp = ds.attrs.get("max_amp")
    max_power_dbm = ds.attrs.get("max_power_dbm")
    if max_amp is not None and max_power_dbm is not None:
        def dbm_to_amp(p):
            return max_amp * 10 ** ((p - max_power_dbm) / 20)

        def amp_to_dbm(a):
            return max_power_dbm + 20 * np.log10(np.maximum(a, 1e-12) / max_amp)

        ax_right = ax.secondary_yaxis("right", functions=(dbm_to_amp, amp_to_dbm))
        ax_right.set_ylabel("Drive amplitude (V)")
        y_min, y_max = ax.get_ylim()
        amp_min_val = max(dbm_to_amp(y_min), 1e-9)
        amp_max_val = dbm_to_amp(y_max)
        if amp_min_val < amp_max_val:
            ax_right.set_yticks(np.geomspace(amp_min_val, amp_max_val, 8))
        ax_right.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))

    ax.set_title(qubit["qubit"])


def plot_raw_data_amp_linear(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Grid of per-qubit panels on a linear drive-amplitude (V) y-axis."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_amp_linear(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    grid.fig.suptitle("Qubit spectroscopy vs drive power (linear amplitude axis)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_amp_linear(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None):
    """Single-qubit panel with linear drive amplitude (V) on the y-axis."""
    max_amp = ds.attrs.get("max_amp")
    max_power_dbm = ds.attrs.get("max_power_dbm")

    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].I_rot / u.mV).plot(
        ax=ax, add_colorbar=False, x="detuning_MHz", y="amplitude", robust=True,
    )
    ax.set_ylabel("Drive amplitude (V)")
    ax.set_xlabel("Detuning [MHz]")

    if fit is not None:
        win = fit.attrs.get("ge_window_mhz")
        if win:
            for edge in (-float(win), float(win)):
                ax.axvline(edge, color="limegreen", linestyle="--", linewidth=1.0, alpha=0.7)
        rf = float((ds.full_freq.loc[qubit] - ds.detuning).mean())
        for name, color in (("twophoton_freq", "magenta"), ("ef_freq", "brown")):
            v = float(fit[name].values)
            if np.isfinite(v):
                ax.axvline((v - rf) * 1e-6, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        # Fitted GE peak position vs amplitude (HIGH CONTRAST)
        ax.plot(fit.peak_position * 1e-6, ds.loc[qubit].amplitude.values, color=_PEAK_COLOR,
                linewidth=2.0, path_effects=_HALO, zorder=6)
        if bool(fit.success) and max_amp is not None and max_power_dbm is not None:
            optimal_amp = max_amp * 10 ** ((float(fit.optimal_power) - max_power_dbm) / 20)
            ax.axhline(y=optimal_amp, color=_OPT_COLOR, linestyle="-",
                       linewidth=2.0, path_effects=_HALO_THIN, zorder=6)
            ax.axvline(x=float(fit.freq_shift) * 1e-6, color=_GE_COLOR, linestyle="--",
                       linewidth=1.8, path_effects=_HALO_THIN, zorder=6)
        if bool(fit.ef_success):
            d2 = (float(fit.twophoton_freq_fitted) - rf) * 1e-6
            ax.axvline(x=d2, color=_EF_COLOR, linestyle="-", linewidth=1.8,
                       path_effects=_HALO_THIN, zorder=6)
        _annotate_ef_warning(ax, fit)
    ax.set_title(qubit["qubit"])


def plot_raw_data_no_fit(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Raw 2D colormap vs (RF frequency, drive power) with NO fit overlay.

    Many users want to see the bare data without the peak trace / optimal-power line /
    GE & EF lines drawn on top (which can hide the underlying spectroscopy). This reuses
    ``plot_individual_raw_data_with_fit`` with ``fit=None`` so the panel keeps the same
    dual frequency/detuning axes and the secondary drive-amplitude axis, just without any
    overlay.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(ax, ds, qubit, None)
    grid.fig.suptitle("Qubit spectroscopy vs drive power (raw, no fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_peak_height_vs_power(ds, qubits: List[AnyTransmon], fringe_results: Dict[str, dict]) -> Figure:
    """Per-qubit on-resonance peak height vs drive power, with the fringe marker.

    A monotonic-rising-then-plateau curve indicates clean (saturated / constant_angle)
    behaviour; a dip flags coherent-nutation fringing (vertical dashed line). Compare the
    ``fixed`` and ``constant_angle`` duration modes side by side with this curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        r = fringe_results.get(name, {})
        powers = np.asarray(r.get("power_dbm", []), dtype=float)
        pv = np.asarray(r.get("peak_vs_power", []), dtype=float)
        if powers.size and pv.size:
            ax.plot(powers, pv * 1e3, ".-", color="steelblue", linewidth=1.0, markersize=3)
        ax.set_xlabel("drive power [dBm]")
        ax.set_ylabel("peak height |IQ_abs - base| [mV]")
        if r.get("fringe_detected"):
            fp = r.get("fringe_power_dbm")
            if fp is not None:
                ax.axvline(fp, color="crimson", linestyle="--", linewidth=1.2)
                ax.set_title(f"{name}  (fringe @ {fp:.1f} dBm)", color="crimson")
        else:
            ax.set_title(f"{name}  (no fringe)")

    grid.fig.suptitle("Peak height vs drive power (fringe diagnostic)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
