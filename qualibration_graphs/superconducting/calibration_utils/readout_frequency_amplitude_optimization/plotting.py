from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_fidelity_2d(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the readout fidelity as a function of the readout frequency detuning and amplitude for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the fit data (fidelity vs detuning and amp_prefactor).
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters (optimal detuning and amplitude).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains a 2D heatmap of the readout fidelity with the optimal point marked by a red star.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_fidelity_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Readout frequency and amplitude optimization")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_fidelity_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots the 2D readout fidelity map (detuning vs readout amplitude) for a single qubit, with the optimal point
    marked by a red star.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the fit data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    fidelity = 100 * ds.fit_data.sel(fit_vals="meas_fidelity").loc[qubit]
    fidelity = fidelity.assign_coords(
        detuning_MHz=("detuning", ds.detuning.values / 1e6),
        readout_amplitude_mV=("amp_prefactor", 1e3 * ds.readout_amplitude.loc[qubit].values),
    )
    fidelity.plot(
        ax=ax,
        x="detuning_MHz",
        y="readout_amplitude_mV",
        cmap="viridis",
        add_colorbar=True,
        cbar_kwargs={"label": "Fidelity [%]"},
    )
    ax.plot(
        1e-6 * float(fit.optimal_detuning),
        1e3 * float(fit.optimal_amplitude),
        marker="*",
        color="red",
        markersize=15,
        markeredgecolor="k",
        linestyle="None",
        label="optimum",
    )
    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])
    ax.legend(loc="upper right")
