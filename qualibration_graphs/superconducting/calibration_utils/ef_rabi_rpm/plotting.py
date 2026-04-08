"""Plotting utilities for the EF Rabi RPM (f-state preparation check) experiment."""
import numpy as np
from matplotlib.figure import Figure

import xarray as xr
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset,
    qubits,
    fit_results: dict,
) -> Figure:
    """Plot EF Rabi RPM oscillation vs amplitude with fit and π-amp marker."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        _plot_single(
            ax,
            ds.sel(qubit=q_name),
            ds_fit.sel(qubit=q_name) if ds_fit is not None else None,
            fit_results.get(q_name),
        )
    grid.fig.suptitle("EF Rabi RPM — f-state preparation check")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit_q, fit_params):
    x = ds_q.amp_factor.values.astype(float)
    sig = "state" if "state" in ds_q.data_vars else "I"
    scale = 1.0 if sig == "state" else 1e3
    unit  = "" if sig == "state" else " (mV)"

    y = getattr(ds_q, sig).values.astype(float) * scale
    ax.plot(x, y, ".", ms=4, color="C0", label="data")

    if fit_params is not None and np.isfinite(fit_params.fit_A) and fit_params.fit_f > 0:
        A, f, phi, off = fit_params.fit_A, fit_params.fit_f, fit_params.fit_phi, fit_params.fit_offset
        x_dense = np.linspace(x.min(), x.max(), 500)
        y_fit = A * np.cos(2 * np.pi * f * x_dense + phi) + off
        ax.plot(x_dense, y_fit * scale, "-", lw=1.5, color="C1", label="fit")

    if fit_params is not None and fit_params.success:
        a_pi = fit_params.pi_amp_factor
        ax.axvline(a_pi, color="C2", ls="--", lw=1.5,
                   label=f"π: a={a_pi:.3f}  ({1e3*fit_params.pi_amp:.1f} mV)")
        ax.text(
            0.98, 0.95,
            f"π factor = {a_pi:.3f}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color="green" if abs(a_pi - 1.0) < 0.1 else "orange",
        )

    ax.set_xlabel("EF amplitude scale factor")
    ax.set_ylabel(f"State population{unit}")
    ax.legend(fontsize=8)
    if sig == "state":
        ax.set_ylim(-0.05, 1.15)
    else:
        margin = 0.1 * (y.max() - y.min()) if y.max() != y.min() else 0.1
        ax.set_ylim(y.min() - margin, y.max() + margin)
