from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the raw data with fitted curves for the given qubits.

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

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy vs power")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None
):

    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax,
        add_colorbar=False,
        x="freq_GHz",
        y="power",
        linewidth=0.5,
    )

    ax.set_ylabel("Power (dBm)")

    ax2 = ax.twiny()
    ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
        ax=ax2,
        add_colorbar=False,
        x="detuning_MHz",
        y="power",
        robust=True,
    )
    ax2.set_xlabel("Detuning [MHz]")

    if fit is None:
        return

    # --- NEW: plot the two resonance points only ---
    powers = ds.power.values
    P_low, P_high = powers[0], powers[1]

    # resonance positions at the two powers
    f_low = (
        ds.sel(power=P_low)
        .loc[qubit]
        .IQ_abs_norm
        .idxmin("detuning")
        .item()
        / 1e6
    )

    f_high = (
        ds.sel(power=P_high)
        .loc[qubit]
        .IQ_abs_norm
        .idxmin("detuning")
        .item()
        / 1e6
    )

    ax2.plot(
        [f_low, f_high],
        [P_low, P_high],
        "o-",
        color="orange",
        linewidth=1,
        markersize=4,
        label="Resonance shift",
    )

    # Optimal readout power
    if fit.success:
        ax2.axhline(
            y=fit.optimal_power,
            color="g",
            linestyle="-",
            label="Optimal power",
        )

        detuning_opt_MHz = fit.freq_shift * 1e-6

        ax2.axvline(
            x=-detuning_opt_MHz,
            color="blue",
            linestyle="--",
        )
    

