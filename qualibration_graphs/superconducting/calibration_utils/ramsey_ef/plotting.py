from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import oscillation_decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the EF Ramsey oscillations with fitted curves for each qubit.

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
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("EF Ramsey (rotated 'I' quadrature + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit EF Ramsey data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    if fit is not None:
        fitted_ramsey_data = oscillation_decay_exp(
            ds.idle_time,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="f"),
            fit.sel(fit_vals="phi"),
            fit.sel(fit_vals="offset"),
            fit.sel(fit_vals="decay"),
        )
    else:
        fitted_ramsey_data = None

    if hasattr(fit, "state"):
        _plot_state(ax, fit, qubit, fitted_ramsey_data)
        ax.set_ylabel("State Population")
    elif hasattr(fit, "I"):
        _plot_transmission_amplitude(ax, fit, qubit, fitted_ramsey_data)
        ax.set_ylabel("Trans. amp. I [mV]")
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        _add_fit_text(ax, fit)
    ax.legend()


def _add_fit_text(ax, fit):
    """Add T2*_ef fit results text to the axis corner."""
    fit_da = fit["fit"] if isinstance(fit, xr.Dataset) else fit
    decay = float(fit_da.sel(fit_vals="decay").mean(dim="detuning_signs"))
    decay_res = float(fit_da.sel(fit_vals="decay_decay").mean(dim="detuning_signs"))
    tau_us = 1e-3 / decay  # T2*_ef in µs (idle_time in ns, so decay is in 1/ns)
    tau_err_us = tau_us * (np.sqrt(decay_res) / abs(decay))
    ax.text(
        0.98,
        0.98,
        f"T2*_ef = {tau_us:.2f} ± {tau_err_us:.2f} µs",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )


def _plot_state(ax, ds, qubit, fitted=None):
    """Plot state data for a qubit."""
    ds.sel(detuning_signs=1).state.plot(ax=ax, x="idle_time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +")
    ds.sel(detuning_signs=-1).state.plot(ax=ax, x="idle_time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -")
    if fitted is not None:
        ax.plot(ds.idle_time, fitted.fit.sel(detuning_signs=1), c="C0", ls="-", lw=1)
        ax.plot(ds.idle_time, fitted.fit.sel(detuning_signs=-1), c="C1", ls="-", lw=1)


def _plot_transmission_amplitude(ax, ds, qubit, fitted=None):
    """Plot transmission amplitude for a qubit."""
    (ds.sel(detuning_signs=1).I * 1e3).plot(
        ax=ax, x="idle_time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
    )
    (ds.sel(detuning_signs=-1).I * 1e3).plot(
        ax=ax, x="idle_time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
    )
    if fitted is not None:
        ax.plot(ds.idle_time, 1e3 * fitted.fit.sel(detuning_signs=1), c="C0", ls="-", lw=1)
        ax.plot(ds.idle_time, 1e3 * fitted.fit.sel(detuning_signs=-1), c="C1", ls="-", lw=1)
