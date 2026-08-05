"""Plotting utilities for resonator spectroscopy versus coupler flux calibration (``_new``).

Self-contained copy of ``resonator_spectroscopy_vs_flux/plotting.py``.  The grid
is keyed by the unique **qubit-pair** name and built as a plain subplot grid
(one panel per pair).  This avoids ``QubitGrid``, which keys by ``grid_location``
and would collide when two pairs share the same measured qubit (same location).
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """
    Plot the raw data with fitted curves, one panel per qubit pair.

    The ``qubit`` dimension of ``ds``/``fits`` is keyed by the unique qubit-pair
    name, so panels are selected by pair name and laid out on a plain subplot
    grid (robust when two pairs share the same measured qubit).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data (``qubit`` dim = pair names).
    qubits : list of AnyTransmon
        Unused for layout (kept for call-site compatibility); the grid is built
        from the pairs present in ``ds``.
    fits : xr.Dataset
        The dataset containing the fit parameters (``qubit`` dim = pair names).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    pair_names = [str(q) for q in ds.qubit.values]
    n = len(pair_names)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, name in enumerate(pair_names):
        ax = axes[idx // ncols][idx % ncols]
        plot_individual_raw_data_with_fit(ax, ds, {"qubit": name}, fits.sel(qubit=name))

    # Hide any unused cells
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Resonator spectroscopy vs coupler flux")
    fig.tight_layout()

    # Single shared, de-duplicated legend at the figure level
    seen: dict[str, object] = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label and not label.startswith("_") and label not in seen:
                seen[label] = handle
    if seen:
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            loc="lower center",
            ncol=len(seen),
            bbox_to_anchor=(0.5, 0.0),
            fontsize="small",
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.12)
    return fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots a single qubit pair's data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit-pair panel to plot, e.g. ``{"qubit": "q4-5"}``.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    ax2 = ax.twiny()
    # Plot using the attenuated current x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax2,
        add_colorbar=False,
        x="attenuated_current",
        y="freq_GHz",
        robust=True,
    )
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("Freq (GHz)")
    ax2.set_title("")
    # Move ax2 behind ax
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    # Plot using the flux x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax, add_colorbar=False, x="flux_bias", y="freq_GHz", robust=True
    )
    if fit is not None:
        rf_frequency = float((ds.full_freq.loc[qubit] - ds.detuning).mean())
        dip_positions = fit.peak_freq.dropna(dim="flux_bias")
        ax.scatter(
            dip_positions.flux_bias.values,
            (dip_positions.values + rf_frequency) * 1e-9,
            s=18,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label="dip positions",
            zorder=3,
        )

        if "peak_freq_fit" in fit:
            peak_freq_fit = fit.peak_freq_fit.dropna(dim="fit_flux_bias")
            ax.plot(
                peak_freq_fit.fit_flux_bias.values,
                (peak_freq_fit.values + rf_frequency) * 1e-9,
                color="magenta",
                linewidth=2.2,
                label="sinusoidal fit",
                zorder=5,
            )

    if fit is not None and fit.fit_results.success.values:
        ax.axvline(
            fit.fit_results.idle_offset,
            linestyle="dashed",
            linewidth=2,
            color="r",
            label="max offset",
        )
        if np.isfinite(float(fit.fit_results.flux_min)):
            ax.axvline(
                fit.fit_results.flux_min,
                linestyle="dashed",
                linewidth=2,
                color="orange",
                label="min offset",
            )
        # Location of the current resonator frequency
        ax.plot(
            fit.fit_results.idle_offset.values,
            fit.fit_results.sweet_spot_frequency.values * 1e-9,
            "r*",
            markersize=10,
        )
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux (V)")
