from typing import List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("AC Stark detuning")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    if hasattr(fit, "state"):
        fit.assign_coords(detuning_MHz=fit.full_detuning / u.MHz).state.plot(
            ax=ax, x="detuning_MHz", y="nb_of_pulses"
        )
        ax.set_ylabel("State population")
        ax.set_title(qubit["qubit"] + " - state population")
    elif hasattr(fit, "I"):
        (fit.assign_coords(detuning_MHz=fit.full_detuning / u.MHz).I * 1e3).plot(
            ax=ax, x="detuning_MHz", y="nb_of_pulses"
        )
        ax.set_ylabel("Trans. amp. I [mV]")
        ax.set_title(qubit["qubit"] + " - I quadrature [mV]")
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    ax.axvline(float(fit["optimal_detuning"]) / u.MHz, color="r")
    ax.set_ylabel("number of pulses")
    ax.set_xlabel("detuning [MHz]")
