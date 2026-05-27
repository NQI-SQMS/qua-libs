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
    Plots the EF qubit spectroscopy signal with fitted curves for the given qubits.

    The bottom x-axis shows the actual RF frequency swept in the EF experiment:
        RF = f_ge + detuning + anharmonicity
    stored in the dataset coordinate ``full_freq``.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data (must have ``full_freq`` coord).
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    find_dip : bool
        Must match the node parameter used during analysis.
    signal_source : str
        'I_rot' (default), 'I' (raw I quadrature), or 'IQ_abs' (magnitude).

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
    grid.fig.suptitle(f"Qubit EF spectroscopy ({signal_label} + fit)")
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
    Plots individual qubit EF data on a given axis with optional fit.

    The bottom axis shows the actual EF RF frequency (``full_freq``, which
    already accounts for the anharmonicity offset applied in the QUA program).
    The top axis shows the detuning relative to the scan centre.

    Parameters
    ----------
    find_dip : bool
        When True, negate the signal so peaks_dips sees a peak.
    signal_source : str
        'I_rot' (default), 'I' (raw I quadrature), or 'IQ_abs' (magnitude).
    """
    use_iq_abs = (signal_source == "IQ_abs")
    use_raw_I = (signal_source == "I")
    negate = find_dip and not use_iq_abs
    if use_iq_abs:
        plot_var, ylabel = "IQ_abs", "IQ_abs [mV]"
    elif use_raw_I:
        plot_var, ylabel = "I", ("-I [mV]" if negate else "I [mV]")
    else:
        plot_var, ylabel = "I_rot", ("-I_rot [mV]" if negate else "I_rot [mV]")
    data_sign = -1.0 if negate else 1.0

    if fit is not None:
        baseline_sign = -1.0 if negate else 1.0
        fitted_data = lorentzian_peak(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            baseline_sign * float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Bottom axis: actual EF RF frequency (f_ge + detuning + anharmonicity)
    (data_sign * fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz)[plot_var] / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("EF RF frequency [GHz]")
    ax.set_ylabel(ylabel)
    # Top axis: detuning relative to scan centre
    ax2 = ax.twiny()
    (data_sign * fit.assign_coords(detuning_MHz=fit.detuning / u.MHz)[plot_var] / u.mV).plot(ax=ax2, x="detuning_MHz", label="")
    ax2.set_xlabel("Detuning from EF centre [MHz]")
    # Fitted curve
    if fitted_data is not None:
        ax2.plot(fit.detuning / u.MHz, fitted_data / u.mV, "r--")
