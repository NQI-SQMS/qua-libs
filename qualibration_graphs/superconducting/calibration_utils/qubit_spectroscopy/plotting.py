from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_peak
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    find_dip: bool = False,
    signal_source: str = "I_rot",
):
    """
    Plots the qubit spectroscopy signal with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    find_dip : bool
        When True and signal_source='I_rot', the fit was performed on -I_rot so
        the baseline must be sign-corrected before plotting.
    signal_source : str
        'I_rot' (default) or 'IQ_abs'.  Selects which signal to display.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(
            ax, ds, qubit, fits.sel(qubit=qubit["qubit"]),
            find_dip=find_dip, signal_source=signal_source,
        )

    signal_label = {"IQ_abs": "IQ_abs", "I": "I"}.get(signal_source, "I_rot")
    grid.fig.suptitle(f"Qubit spectroscopy ({signal_label} + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
    find_dip: bool = False,
    signal_source: str = "I_rot",
):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    find_dip : bool
        When True and signal_source='I_rot', negate the baseline so it is
        expressed in I_rot space (peaks_dips fitted on -I_rot).
    signal_source : str
        'I_rot' (default) or 'IQ_abs' — selects which variable to plot.
    """
    use_iq_abs = (signal_source == "IQ_abs")
    use_raw_i  = (signal_source == "I")
    # When find_dip=True, signal_for_fit = -I_rot (or -I). Display negated so the
    # plot shows a peak and the Lorentzian overlay aligns.
    negate = find_dip and not use_iq_abs
    if use_iq_abs:
        plot_var = "IQ_abs"
        ylabel   = "IQ_abs [mV]"
    elif use_raw_i:
        plot_var = "I"
        ylabel   = "-I [mV]" if negate else "I [mV]"
    else:
        plot_var = "I_rot"
        ylabel   = "-I_rot [mV]" if negate else "I_rot [mV]"
    data_sign = -1.0 if negate else 1.0

    if fit is not None:
        fitted_data = lorentzian_peak(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Primary x-axis: full RF frequency in GHz
    (data_sign * fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz)[plot_var] / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(ylabel)
    # Secondary x-axis: detuning in MHz
    ax2 = ax.twiny()
    (data_sign * fit.assign_coords(detuning_MHz=fit.detuning / u.MHz)[plot_var] / u.mV).plot(ax=ax2, x="detuning_MHz", label="")
    ax2.set_xlabel("Detuning [MHz]")
    # Overlay the Lorentzian fit
    if fitted_data is not None:
        ax2.plot(fit.detuning / u.MHz, fitted_data / u.mV, "r--")
