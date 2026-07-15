"""Plotting utilities for the cavity active reset test."""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot P(0) vs reset duration for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q_fit = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], q_fit)

    grid.fig.suptitle(f"Cavity Active Reset Test — {mode_name}")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit_params=None):
    t_us = ds_q["reset_duration"].values * 1e-3  # ns → µs

    if "P0" in ds_q.data_vars:
        y = ds_q["P0"].values
    elif "state" in ds_q.data_vars:
        y = 1.0 - ds_q["state"].values
    else:
        return

    ax.plot(t_us, y, ".", ms=4, color="C1", label="With reset")
    ax.axhline(0.95, color="grey", ls=":", lw=1, label="P(0) = 0.95")

    if fit_params is not None:
        t95_us = fit_params.get("t95_ns", float("nan")) * 1e-3
        P0_max = fit_params.get("P0_max", float("nan"))
        success = fit_params.get("success", False)
        status = "SUCCESS" if success else "FAILED"

        if np.isfinite(t95_us):
            ax.axvline(t95_us, color="C2", ls="--", lw=1.5, label=f"t95={t95_us:.0f} µs")

        title = f"P0_max={P0_max:.2f}  [{status}]"
        if np.isfinite(t95_us):
            title += f"\nt95={t95_us:.0f} µs"
        ax.set_title(title, fontsize=9, color="green" if success else "red")

    ax.set_xlabel("Reset duration (µs)")
    ax.set_ylabel("P(0) = 1 − P(|e⟩)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
