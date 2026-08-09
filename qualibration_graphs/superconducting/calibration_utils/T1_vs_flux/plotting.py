"""Plotting for the T1-versus-flux node."""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _signal_name(ds: xr.Dataset) -> str:
    if hasattr(ds, "state"):
        return "state"
    if hasattr(ds, "I"):
        return "I"
    raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot the 2-D relaxation map (idle time vs flux bias) for each qubit.

    A horizontal dashed line marks the flux bias giving the longest T1.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_map(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("T1 vs flux (relaxation map)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_t1_vs_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot the extracted T1 versus flux bias (with error bars) for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_curve(ax, fits.sel(qubit=qubit["qubit"]), qubit)

    grid.fig.suptitle("T1 vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_map(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None):
    signal = _signal_name(ds)
    da = ds.sel(qubit=qubit["qubit"])[signal]
    if signal == "I":
        da = da * 1e3
        cbar_label = "Trans. amp. I [mV]"
    else:
        cbar_label = "State"

    # A 2-D map needs both axes to have >1 point; otherwise (e.g. a single flux point)
    # degrade gracefully to a 1-D line of signal vs idle time instead of crashing.
    two_d = da.sizes.get("flux_bias", 1) > 1 and da.sizes.get("idle_time", 1) > 1
    if two_d:
        da.plot(ax=ax, x="idle_time", y="flux_bias", add_colorbar=True, cbar_kwargs={"label": cbar_label})
        if fit is not None:
            valid = np.isfinite(fit.tau.values)
            if valid.any():
                idx = int(np.nanargmax(np.where(valid, fit.tau.values, -np.inf)))
                best_flux = float(fit.flux_bias.values[idx])
                ax.axhline(best_flux, color="red", linestyle="--", linewidth=1, label="max T1")
    else:
        da.squeeze().plot(ax=ax)

    # Set labels AFTER plotting so they override xarray's auto-generated labels
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle time [ns]")
    ax.set_ylabel("Flux bias [V]" if two_d else cbar_label)


def _plot_individual_curve(ax: Axes, fit: xr.Dataset, qubit: dict):
    flux = fit.flux_bias.values
    tau_us = fit.tau.values * 1e-3
    err_us = fit.tau_error.values * 1e-3

    ax.errorbar(flux, tau_us, yerr=err_us, fmt="o-", capsize=3, markersize=4)

    valid = np.isfinite(tau_us)
    if valid.any():
        idx = int(np.nanargmax(np.where(valid, tau_us, -np.inf)))
        best_flux = float(flux[idx])
        best_t1 = float(tau_us[idx])
        ax.set_ylim([0, best_t1*1.1])
        ax.axvline(best_flux, color="red", linestyle="--", linewidth=1)
        ax.text(
            0.05,
            0.95,
            f"max T1 = {best_t1:.1f} us\n@ flux = {best_flux:.4f} V",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.5},
        )

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux bias [V]")
    ax.set_ylabel("T1 [us]")
