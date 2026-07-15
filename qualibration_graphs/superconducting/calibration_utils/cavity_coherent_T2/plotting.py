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

    # Lock y-limits to the data range so a bad fit cannot obscure the data.
    y_range = float(y.max() - y.min())
    margin = max(0.15 * y_range, 1e-4)
    ax.set_ylim(float(y.min()) - margin, float(y.max()) + margin)

    if fit is not None:
        a     = float(fit.fit.sel(fit_vals="a").item())
        f     = float(fit.fit.sel(fit_vals="f").item())
        phi   = float(fit.fit.sel(fit_vals="phi").item())
        off   = float(fit.fit.sel(fit_vals="offset").item())
        decay = float(fit.fit.sel(fit_vals="decay").item())

        if np.isfinite(a) and np.isfinite(decay) and decay > 0:
            # Evaluate the fit at the actual data points (mirrors the ge Ramsey
            # plotting approach).  This avoids the aliasing artifact where a dense
            # grid produces a solid orange block when the fit frequency is high.
            y_fit = oscillation_decay_exp(x_ns, a, f, phi, off, decay)
            if "I" in ds_q.data_vars:
                y_fit = y_fit * 1e3
            ax.plot(x_ns * 1e-3, y_fit, "-", lw=1.5, color="C1", label="fit")

    if fit_params is not None:
        T2_us = fit_params.get("T2ramsey_ns", float("nan")) * 1e-3
        T2_err_us = fit_params.get("T2ramsey_uncertainty_ns", float("nan")) * 1e-3
        chi2 = fit_params.get("chi2", float("nan"))
        success = fit_params.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        T2_str = f"{T2_us:.1f}" if np.isfinite(T2_us) else "nan"
        err_str = f" ± {T2_err_us:.1f}" if np.isfinite(T2_err_us) else ""
        chi2_str = f"χ²={chi2:.2f}" if np.isfinite(chi2) else ""
        ax.set_title(
            f"T2* = ({T2_str}{err_str}) us  [{status}]  {chi2_str}",
            fontsize=9,
            color="green" if success else "red",
        )

    ax.set_xlabel("Idle time (us)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
