"""Plotting utilities for the f0g1 time Rabi experiment."""
from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import oscillation


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits: xr.Dataset,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot f0g1 time Rabi oscillations vs pulse duration for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q_fit_params = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], fits.sel(qubit=q_name), q_fit_params)

    grid.fig.suptitle(f"f0g1 Time Rabi — {mode_name}")
    grid.fig.set_size_inches(12, 7)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit, fit_params=None):
    if "duration_ns" in ds_q.coords:
        x = ds_q.duration_ns.values
    else:
        x = ds_q.duration_cc.values * 4
    x_cc = ds_q.duration_cc.values

    if "state" in ds_q.data_vars:
        y = ds_q.state.values
        ylabel = "State population"
        scale = 1.0
    elif "I" in ds_q.data_vars:
        y = ds_q.I.values * 1e3
        ylabel = "I (mV)"
        scale = 1e3
    else:
        return

    ax.plot(x, y, ".", ms=4, color="C0", label="data")

    if fit is not None:
        a   = float(fit.fit.sel(fit_vals="a").item())
        f   = float(fit.fit.sel(fit_vals="f").item())
        phi = float(fit.fit.sel(fit_vals="phi").item())
        off = float(fit.fit.sel(fit_vals="offset").item())

        if np.isfinite(a) and np.isfinite(f):
            fitted = oscillation(x_cc, a, f, phi, off) * scale
            ax.plot(x, fitted, "-", lw=1.5, color="C1", label="fit")

            if abs(f) > 0:
                period_cc = 1.0 / abs(f)
                t_start_cc = float(x_cc[0])
                pi_cc = ((np.pi - phi) % (2 * np.pi)) / (2 * np.pi * abs(f))
                if pi_cc < t_start_cc:
                    n_shift = int(np.ceil((t_start_cc - pi_cc) / period_cc))
                    pi_cc += n_shift * period_cc
                pi_ns = round(pi_cc) * 4
                ax.axvline(pi_ns, color="C2", ls="--", lw=1, label=f"π: {pi_ns:.0f} ns")

    if fit_params is not None:
        chi2 = fit_params.get("chi2", float("nan"))
        success = fit_params.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        chi2_str = f"{chi2:.2f}" if np.isfinite(chi2) else "nan"
        ax.set_title(f"chi2={chi2_str}  [{status}]", fontsize=8,
                     color="green" if success else "red")

    ax.set_xlabel("Pulse duration [ns]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
