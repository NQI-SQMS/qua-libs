"""Plotting utilities for conditional qubit spectroscopy vs flux calibration (control in |f>)."""

from typing import Dict, List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data(ds: xr.Dataset, qubit_pairs: List, quantity: str = "IQ_abs", size: int = 8) -> Dict[str, Figure]:
    """Plot the control and target qubit readouts as two distinct 2D (frequency vs flux) figures.

    One subplot per qubit pair, using the same subplot layout for both figures so
    they can be compared side by side.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing ``I_control``/``Q_control``/``IQ_abs_control`` (and the ``_target``
        equivalents) plus the ``full_freq``/``flux_bias`` axes.
    qubit_pairs : list
        The qubit pairs being plotted.
    quantity : str
        Which quadrature to plot: ``"IQ_abs"`` (default, always positive), ``"I"``, or ``"Q"``.
    size : int
        Size (in inches) of each subplot. Default is 8.

    Returns
    -------
    dict[str, Figure]
        ``{"control": <figure>, "target": <figure>}``.
    """
    g_names, qp_names = grid_pair_names(qubit_pairs)

    grid_control = QubitPairGrid(g_names, qp_names, size=size)
    for ax, qubit in grid_iter(grid_control):
        _plot_individual(ax, ds, qubit, f"{quantity}_control")
    grid_control.fig.suptitle(f"Conditional qubit spectroscopy vs flux (control in |f>) — control qubit readout ({quantity})")
    grid_control.fig.tight_layout()

    grid_target = QubitPairGrid(g_names, qp_names, size=size)
    for ax, qubit in grid_iter(grid_target):
        _plot_individual(ax, ds, qubit, f"{quantity}_target")
    grid_target.fig.suptitle(f"Conditional qubit spectroscopy vs flux (control in |f>) — target qubit readout ({quantity})")
    grid_target.fig.tight_layout()

    return {"control": grid_control.fig, "target": grid_target.fig}


def _plot_individual(ax: Axes, ds: xr.Dataset, qubit: dict, data_var: str):
    """Plot a single qubit pair's 2D heatmap (frequency vs target flux) for the given data variable."""
    ds_plot = ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit]
    ds_plot[data_var].plot(ax=ax, add_colorbar=True, x="flux_bias", y="freq_GHz", robust=True)
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Target flux (V)")
    ax.set_ylabel("Frequency (GHz)")
