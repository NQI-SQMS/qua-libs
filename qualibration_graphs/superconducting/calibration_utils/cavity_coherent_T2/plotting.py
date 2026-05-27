"""Plotting utilities for the cavity mode T2 Ramsey experiment."""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis.models import oscillation_decay_exp


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits: xr.Dataset,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot cavity mode T2 Ramsey data and decaying sinusoid fit for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q_fit_params = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], fits.sel(qubit=q_name), q_fit_params)

    grid.fig.suptitle(f"Cavity Mode T2 Ramsey — {mode_name}")
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

    if fit is not None:
        a     = float(fit.fit.sel(fit_vals="a").item())
        f     = float(fit.fit.sel(fit_vals="f").item())
        phi   = float(fit.fit.sel(fit_vals="phi").item())
        off   = float(fit.fit.sel(fit_vals="offset").item())
        decay = float(fit.fit.sel(fit_vals="decay").item())

        if np.isfinite(a) and np.isfinite(decay) and decay > 0:
            # fit used idle_time in ns; oscillation_decay_exp uses same units
            x_dense = np.linspace(x_ns.min(), x_ns.max(), 500)
            y_fit = oscillation_decay_exp(x_dense, a, f, phi, off, decay)
            if "I" in ds_q.data_vars:
                y_fit = y_fit * 1e3
            ax.plot(x_dense * 1e-3, y_fit, "-", lw=1.5, color="C1", label="fit")

    if fit_params is not None:
        T2_us = fit_params.get("T2ramsey_ns", float("nan")) * 1e-3
        success = fit_params.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        T2_str = f"{T2_us:.1f}" if np.isfinite(T2_us) else "nan"
        ax.set_title(f"T2* = {T2_str} us  [{status}]", fontsize=9,
                     color="green" if success else "red")
        if np.isfinite(T2_us):
            T2_ns = T2_us * 1e3
            ax.axvline(T2_ns * 1e-3, color="C2", ls="--", lw=1, label=f"T2*={T2_str} us")

    ax.set_xlabel("Idle time (us)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
