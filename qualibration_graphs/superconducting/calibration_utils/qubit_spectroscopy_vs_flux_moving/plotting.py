"""Plotting utilities for moving-qubit spectroscopy vs flux calibration."""

from typing import Dict, List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data(ds: xr.Dataset, qubit_pairs: List, quantity: str = "IQ_abs", size: int = 8) -> Dict[str, Figure]:
    """Plot the moving and stationary qubit readouts as two distinct 2D (frequency vs flux) figures.

    One subplot per qubit pair, using the same subplot layout for both figures so
    they can be compared side by side.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing ``I_moving``/``Q_moving``/``IQ_abs_moving`` (and the
        ``_stationary`` equivalents) plus the ``full_freq``/``full_flux`` axes.
    qubit_pairs : list
        The qubit pairs being plotted.
    quantity : str
        Which quadrature to plot: ``"IQ_abs"`` (default, always positive), ``"I"``, or ``"Q"``.
    size : int
        Size (in inches) of each subplot. Default is 8.

    Returns
    -------
    dict[str, Figure]
        ``{"moving": <figure>, "stationary": <figure>}``.
    """
    g_names, qp_names = grid_pair_names(qubit_pairs)

    grid_moving = QubitPairGrid(g_names, qp_names, size=size)
    for ax, qubit in grid_iter(grid_moving):
        _plot_individual(ax, ds, qubit, f"{quantity}_moving")
    grid_moving.fig.suptitle(f"Qubit spectroscopy vs flux around idle — moving qubit readout ({quantity})")
    grid_moving.fig.tight_layout()

    grid_stationary = QubitPairGrid(g_names, qp_names, size=size)
    for ax, qubit in grid_iter(grid_stationary):
        _plot_individual(ax, ds, qubit, f"{quantity}_stationary")
    grid_stationary.fig.suptitle(f"Qubit spectroscopy vs flux around idle — stationary qubit readout ({quantity})")
    grid_stationary.fig.tight_layout()

    return {"moving": grid_moving.fig, "stationary": grid_stationary.fig}


def _plot_individual(ax: Axes, ds: xr.Dataset, qubit: dict, data_var: str):
    """Plot a single qubit pair's 2D heatmap (frequency vs moving-qubit flux) for the given data variable."""
    ds_plot = ds.assign_coords(freq_GHz=ds.full_freq / 1e9, flux_mV=ds.full_flux * 1e3).loc[qubit]
    ds_plot[data_var].plot(ax=ax, add_colorbar=True, x="flux_mV", y="freq_GHz", robust=True)
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Moving qubit flux (mV)")
    ax.set_ylabel("Frequency (GHz)")
