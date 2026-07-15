"""Plotting utilities for the cavity Fock |1⟩ T1 experiment."""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import decay_exp


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits: xr.Dataset,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot cavity Fock |1⟩ T1 data and exponential decay fit for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q_fit_params = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], fits.sel(qubit=q_name), q_fit_params)

    grid.fig.suptitle(f"Fock |1⟩ T1 — {mode_name}")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit, fit_params=None):
    x_ns = ds_q.idle_time.values  # in ns

    if "state" in ds_q.data_vars:
        y = ds_q.state.values
        ylabel = "State population"
    elif "I" in ds_q.data_vars:
        y = ds_q.I.values * 1e3
        ylabel = "I (mV)"
    else:
        return

    ax.plot(x_ns * 1e-3, y, ".", ms=4, color="C0", label="data")

    # Lock y-limits to the data range before adding the fit, so the fit
    # curve cannot push the axis beyond what the data shows.
    y_finite = y[np.isfinite(y)]
    if y_finite.size > 0:
        y_margin = max((y_finite.max() - y_finite.min()) * 0.1, 0.05)
        ax.set_ylim(y_finite.min() - y_margin, y_finite.max() + y_margin)

    if fit is not None:
        a     = float(fit.fit.sel(fit_vals="a").item())
        off   = float(fit.fit.sel(fit_vals="offset").item())
        decay = float(fit.fit.sel(fit_vals="decay").item())

        if np.isfinite(decay) and decay < 0:
            x_dense = np.linspace(x_ns.min(), x_ns.max(), 500)
            y_fit = decay_exp(x_dense, a, off, decay)
            if "state" not in ds_q.data_vars and "I" in ds_q.data_vars:
                y_fit = y_fit * 1e3
            ax.plot(x_dense * 1e-3, y_fit, "-", lw=1.5, color="C1", label="fit")

    if fit_params is not None:
        T1_us = fit_params.get("T1_ns", float("nan")) * 1e-3
        T1_err_us = fit_params.get("T1_error_ns", float("nan")) * 1e-3
        success = fit_params.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        T1_str = f"{T1_us:.1f} ± {T1_err_us:.1f}" if np.isfinite(T1_us) else "nan"
        ax.set_title(f"T1 = {T1_str} µs  [{status}]", fontsize=9,
                     color="green" if success else "red")
        if np.isfinite(T1_us):
            ax.text(0.97, 0.97, f"T1 = {T1_str} µs",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="green" if success else "red")

    ax.set_xlabel("Idle time (µs)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
