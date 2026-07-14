from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_fidelity_map(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the readout fidelity as a function of readout frequency and amplitude for the given qubits, with the
    optimal point marked by a star.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The dataset containing the raw data together with the fidelity proxy ("fit_data") and the optimal
        ("optimal_detuning", "optimal_amp_prefactor") coordinates, as returned by ``fit_raw_data``.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_fidelity_map(ax, ds_fit, qubit)
    grid.fig.suptitle("Readout optimisation - fidelity map")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_fidelity_map(ax: Axes, ds_fit: xr.Dataset, qubit: dict[str, str]):
    """
    Plots the fidelity map for a single qubit on a given axis, with the optimal readout frequency/amplitude
    point marked by a star.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds_fit : xr.Dataset
        The dataset containing the raw data, the fidelity proxy and the optimal point coordinates.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    """
    ds_q = ds_fit.sel(qubit=qubit["qubit"])
    fidelity = ds_q.fidelity_ge.transpose("amp_prefactor", "detuning")
    freq_ghz = 1e-9 * ds_q.full_freq
    amp_mv = 1e3 * ds_q.readout_amplitude

    c = ax.pcolor(freq_ghz, amp_mv, fidelity, shading="auto")
    ax.figure.colorbar(c, ax=ax, label="Fidelity")

    best_freq_ghz = 1e-9 * float(ds_q.full_freq.sel(detuning=ds_q["optimal_detuning"], method="nearest"))
    best_amp_mv = 1e3 * float(ds_q.readout_amplitude.sel(amp_prefactor=ds_q["optimal_amp_prefactor"], method="nearest"))
    best_fidelity = float(
        ds_q.fit_data.sel(
            fit_vals="meas_fidelity",
            detuning=ds_q["optimal_detuning"],
            amp_prefactor=ds_q["optimal_amp_prefactor"],
            method="nearest",
        )
    )
    ax.plot(
        best_freq_ghz,
        best_amp_mv,
        "r*",
        markersize=14,
        label=f"best: {best_freq_ghz:.5f} GHz / {best_amp_mv:.1f} mV / {best_fidelity * 100:.1f}%",
    )
    ax.set_xlabel("Readout frequency [GHz]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])
    ax.legend(fontsize=8, loc="lower center")
