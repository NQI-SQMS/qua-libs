from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the CZ chevron for each qubit pair, with two columns per pair.

    - Left  : the chevron-signal qubit (``state_target``) with the fitted CZ point.
    - Right : the |2>-excited qubit (``state_control``, GEF readout) showing leakage,
              with the same fitted CZ point overlaid so the leakage at the chosen gate
              amplitude/time can be judged at a glance.

    Both qubits are read out by the node; the signal qubit carries the chevron arcs that
    are fitted, while the excited qubit (driven into |2> at the avoided crossing) carries
    the leakage information.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the measured data.
    qubit_pairs : list of AnyTransmon
        A list of qubit pairs to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - One row of subplots per qubit pair; left = signal, right = leakage.
    """
    n = len(qubit_pairs)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(16, 4.5 * n), squeeze=False)
    for ii, qp in enumerate(qubit_pairs):
        # Try to get fit data for this qubit pair, handle if missing
        try:
            fit_data = fits.sel(qubit_pair=qp.id) if fits is not None else None
        except (KeyError, ValueError):
            # If this qubit pair is not in the fit results, set fit_data to None
            fit_data = None

        # Left: chevron-signal qubit (state_target) + fitted CZ point
        plot_individual_data_with(axs[ii, 0], ds, qp.id, fit_data)
        # Right: |2>-excited qubit (state_control) leakage + same CZ point
        plot_leakage_data(axs[ii, 1], ds, qp.id, fit_data)

    fig.suptitle("CZ Chevron - signal qubit (left)  |  |2> leakage (right)")
    fig.tight_layout()

    return fig


def plot_individual_data_with(ax: Axes, ds: xr.Dataset, qubit_pair: str, fit: xr.Dataset = None):
    """
    Plots the chevron-signal qubit (``state_target``) for one pair, with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the measured data.
    qubit_pair : str
        The qubit pair id to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted CZ point is marked.
    """

    if hasattr(ds, "state_target"):
        # If the dataset has 'state_target', use it for plotting
        data = ds.state_target
    else:
        data = ds.I_target

    data.sel(qubit_pair=qubit_pair).plot(y="amp_full", ax=ax)

    # Only plot fit results if they exist and are valid
    if fit is not None:
        try:
            # Check if fit values exist and are valid (not NaN)
            cz_len = float(fit.cz_len)
            cz_amp = float(fit.cz_amp)
            if not (np.isnan(cz_len) or np.isnan(cz_amp)):
                ax.scatter(cz_len, cz_amp, color="red", label="Fitted", marker="*", s=100)
                ax.set_title(f"{qubit_pair} - signal (Fit Successful)")
                ax.legend()
            else:
                ax.set_title(f"{qubit_pair} - signal (Fit Failed: Invalid Parameters)")
        except (ValueError, TypeError, AttributeError):
            ax.set_title(f"{qubit_pair} - signal (Fit Failed)")
    else:
        ax.set_title(f"{qubit_pair} - signal (No Fit Data)")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Flux pulse amplitude [V]")


def plot_leakage_data(ax: Axes, ds: xr.Dataset, qubit_pair: str, fit: xr.Dataset = None):
    """
    Plots the |2>-excited qubit leakage (``state_control`` from the GEF readout) for one pair.

    The excited qubit is the one driven into |2> at the avoided crossing (the ``*_control``
    streams, branch-resolved). Higher values indicate more |2> population, i.e. leakage. If a
    fit is available, the chosen CZ (amplitude, time) point is overlaid so the leakage at the
    gate operating point can be read directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the measured data.
    qubit_pair : str
        The qubit pair id to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    if hasattr(ds, "state_control"):
        # GEF readout of the excited qubit -> mean state in [0, 2]; higher = more |2> leakage
        data = ds.state_control
    else:
        data = ds.I_control

    data.sel(qubit_pair=qubit_pair).plot(y="amp_full", ax=ax)

    # Overlay the same fitted CZ point so leakage at the gate operating point is visible.
    if fit is not None:
        try:
            cz_len = float(fit.cz_len)
            cz_amp = float(fit.cz_amp)
            if not (np.isnan(cz_len) or np.isnan(cz_amp)):
                ax.scatter(cz_len, cz_amp, color="red", label="CZ point", marker="*", s=100)
                ax.legend()
        except (ValueError, TypeError, AttributeError):
            pass

    ax.set_title(f"{qubit_pair} - |2> leakage (excited qubit)")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Flux pulse amplitude [V]")
