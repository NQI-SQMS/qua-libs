"""Plotting utilities for the qubit RPM (Rabi Population Measurement) experiment."""
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_rpm(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    qubits,
    fit_results: dict,
) -> Figure:
    """Plot both RPM sweeps ('g' and 'e') with fitted amplitudes and P_th annotation."""
    grid = QubitGrid(ds_g, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        _plot_single(
            ax,
            ds_g.sel(qubit=q_name),
            ds_e.sel(qubit=q_name),
            fit_results.get(q_name),
        )
    grid.fig.suptitle("Qubit RPM — Thermal Population Measurement")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_g, ds_e, fit_params):
    x = ds_g.amp_factor.values.astype(float)

    sig = "state" if "state" in ds_g.data_vars else "I"
    scale = 1.0 if sig == "state" else 1e3
    unit  = "" if sig == "state" else " (mV)"

    y_g = getattr(ds_g, sig).values.astype(float) * scale
    y_e = getattr(ds_e, sig).values.astype(float) * scale

    ax.plot(x, y_g, "o", ms=3, color="C0", label="from |g⟩")
    ax.plot(x, y_e, "s", ms=3, color="C1", label="from thermal")

    if fit_params is not None and fit_params.success:
        p_th = fit_params.thermal_population
        t_eff = fit_params.effective_temperature_mk
        ax.text(
            0.98, 0.05,
            f"P_th = {100*p_th:.2f}%\nT_eff = {t_eff:.1f} mK",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        )

    ax.set_xlabel("EF amplitude scale factor")
    ax.set_ylabel(f"State population{unit}")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.15)
