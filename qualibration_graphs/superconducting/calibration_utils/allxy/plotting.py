from typing import List, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import IDEAL_POPULATION, SEQUENCE_LABELS


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """
    Plots the outcome of the 21 AllXY sequences for each qubit, overlaid with the ideal
    (ground / equal-superposition / excited state) reference populations.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the raw quadrature (or state) data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the normalized population and AllXY error.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("AllXY")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict, fit: Optional[xr.Dataset] = None):
    """Plots the AllXY outcome and ideal reference for a single qubit."""
    sequence_index = np.arange(len(SEQUENCE_LABELS))

    if fit is not None and "normalized_population" in fit:
        population = fit["normalized_population"]
        ax.set_ylabel("Normalized population")
    else:
        ds_q = ds.sel(qubit=qubit["qubit"])
        population = ds_q.state if "state" in ds_q else ds_q.I
        ax.set_ylabel("State population" if "state" in ds_q else "I quadrature [a.u.]")

    ax.plot(sequence_index, population, "o", color="C0", ms=5, label="Data")
    ax.plot(sequence_index, IDEAL_POPULATION, "-", color="C1", lw=1.2, label="Ideal")
    ax.set_xticks(sequence_index)
    ax.set_xticklabels(SEQUENCE_LABELS, rotation=90, fontsize=6)
    ax.set_title(qubit["qubit"])

    if fit is not None and "allxy_error" in fit:
        ax.text(
            0.02,
            0.98,
            f"Mean error = {float(fit['allxy_error']):.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    ax.legend(loc="center right", fontsize=8)
