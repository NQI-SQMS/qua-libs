from typing import List, Dict
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from .circle_fit import plot_circle_fit, CircleFitParameters

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the raw phase data for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        # Create a first x-axis for full_freq_GHz
        ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        # Create a second x-axis for detuning_MHz
        ax2 = ax1.twiny()
        ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (phase)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

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
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    if fit:
        fitted_data = lorentzian_dip(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Create a first x-axis for full_freq_GHz
    (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs / u.mV).plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the fitted data
    if fitted_data is not None:
        ax2.plot(ds.detuning / u.MHz, fitted_data / u.mV, "r--")


def plot_circle_fit_result(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fit_results: Dict,
    circle_fit_results: Dict[str, CircleFitParameters] | None = None,
) -> Figure:
    """4-panel circle-fit diagnostic figure for each qubit, arranged in a grid.

    Parameters
    ----------
    ds : xr.Dataset — raw dataset (must have I, Q, detuning dimensions)
    qubits : list of AnyTransmon
    fit_results : dict — node.results["fit_results"] (asdict of FitParameters)
    circle_fit_results : dict — node.namespace["circle_fit_results"] (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(qubits)
    fig = plt.figure(figsize=(14, 3.5 * n))
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.55, wspace=0.4)

    for row, q in enumerate(qubits):
        freq_Hz = ds.detuning.values + q.resonator.RF_frequency
        S21 = (ds["I"].sel(qubit=q.name).values
               + 1j * ds["Q"].sel(qubit=q.name).values)
        freq_GHz = freq_Hz * 1e-9

        cf: CircleFitParameters | None = None
        if circle_fit_results:
            cf = circle_fit_results.get(q.name)

        delay_ns = cf.delay_ns if cf and np.isfinite(cf.delay_ns) else 0.0
        S21_corr = S21 * np.exp(2j * np.pi * freq_GHz * delay_ns)

        # Panel A — raw IQ
        ax = fig.add_subplot(gs[row, 0])
        ax.scatter(S21.real, S21.imag, c=freq_GHz, cmap="plasma",
                   s=10, edgecolors="none")
        ax.plot(S21.real, S21.imag, "-", color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_title(f"{q.name} — raw IQ", fontsize=9)
        ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3)

        # Panel B — delay-corrected IQ + circle
        ax = fig.add_subplot(gs[row, 1])
        ax.scatter(S21_corr.real, S21_corr.imag, c=freq_GHz, cmap="plasma",
                   s=10, edgecolors="none")
        ax.plot(S21_corr.real, S21_corr.imag, "-", color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_title(f"τ = {delay_ns:.2f} ns", fontsize=9)
        ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3)

        # Panel C — magnitude
        ax = fig.add_subplot(gs[row, 2])
        ax.plot(freq_GHz, 20 * np.log10(np.abs(S21)), ".", ms=3,
                color="steelblue")
        fr = fit_results.get(q.name, {}).get("frequency", None)
        if fr and np.isfinite(fr):
            ax.axvline(fr * 1e-9, color="r", lw=1.0, ls="--",
                       label=f"f₀={fr*1e-9:.5f}")
            ax.legend(fontsize=7)
        ax.set_xlabel("Freq (GHz)"); ax.set_ylabel("|S21| dB")
        ax.set_title("Magnitude", fontsize=9); ax.grid(True, alpha=0.3)

        # Panel D — phase (delay-corrected)
        ax = fig.add_subplot(gs[row, 3])
        ax.plot(freq_GHz, np.degrees(np.unwrap(np.angle(S21_corr))),
                "-", color="darkorange", lw=1.2)
        ax.set_xlabel("Freq (GHz)"); ax.set_ylabel("Phase (deg)")
        ax.set_title("Corr. phase", fontsize=9); ax.grid(True, alpha=0.3)

        # Q-value annotation
        if cf and cf.success:
            fig.text(
                0.5, 1.0 - row / n - 0.01,
                f"Q_L={cf.Q_loaded:.0f}  Q_i={cf.Q_internal:.0f}"
                f"  Q_e={cf.Q_external:.0f}  κ/2π={cf.kappa_Hz*1e-3:.1f} kHz",
                ha="center", va="top", fontsize=9,
                transform=fig.transFigure,
            )

    fig.suptitle("Resonator circle fit — 02a", fontsize=11, y=1.02)
    return fig