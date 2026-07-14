"""Plotting utilities for cavity fNgN1 sideband spectroscopy."""
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits: xr.Dataset,
    mode_name: str = "alice",
    transition_label: str = "f0g1",
) -> Figure:
    """Plot cavity sideband spectroscopy data with fit overlay.

    When Gaussian fit parameters are present in *fits* (use_gaussian_fit=True),
    the Gaussian curve is drawn over the data.  Otherwise a vertical line marks
    the centre found by the peaks_dips detector.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        _plot_single(ax, ds.sel(qubit=q_name), fits.sel(qubit=q_name))

    grid.fig.suptitle(f"{transition_label} spectroscopy — {mode_name} (state dip + fit)")
    grid.fig.set_size_inches(12, 7)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax: Axes, ds: xr.Dataset, fit: xr.Dataset):
    """Plot one qubit panel."""
    if "full_freq" in ds.coords:
        x_plot = ds.full_freq.values * 1e-9  # → GHz
        xlabel = "RF frequency (GHz)"
        f_rf = float(ds.full_freq.isel(detuning=0).values) - float(ds.detuning.isel(detuning=0).values)
        def _det_to_plot(det_hz):
            return (f_rf + det_hz) * 1e-9
    else:
        x_plot = ds.detuning.values * 1e-6  # → MHz
        xlabel = "Detuning (MHz)"
        def _det_to_plot(det_hz):
            return det_hz * 1e-6

    if "state" in ds.data_vars:
        y = ds.state.values
        ylabel = "State population"
    elif "I_rot" in ds.data_vars:
        y = ds.I_rot.values * 1e3
        ylabel = "I_rot (mV)"
    elif "I_rot" in fit.data_vars:
        # I_rot is computed in fit_raw_data and lives in ds_fit, not ds_raw
        y = fit.I_rot.values * 1e3
        ylabel = "I_rot (mV)"
    else:
        return

    ax.plot(x_plot, y, ".", ms=4, color="C0", label="data")

    pos = float(fit.position.values) if "position" in fit else np.nan

    # --- Gaussian overlay (use_gaussian_fit=True) ---
    has_gaussian = (
        "gaussian_sigma_hz" in fit.data_vars
        and np.isfinite(float(fit.gaussian_sigma_hz.values))
    )
    if has_gaussian and np.isfinite(pos):
        amp = float(fit.gaussian_amplitude.values)
        sigma = float(fit.gaussian_sigma_hz.values)
        offset_neg = float(fit.gaussian_offset_neg.values)
        fwhm_hz = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

        det_fine = np.linspace(float(ds.detuning.values.min()), float(ds.detuning.values.max()), 500)
        y_gauss_neg = offset_neg + amp * np.exp(-0.5 * ((det_fine - pos) / sigma) ** 2)
        y_gauss = -y_gauss_neg  # back to original signal space
        if "I_rot" in ds.data_vars or "I_rot" in fit.data_vars:
            y_gauss = y_gauss * 1e3  # convert to mV

        xpos = _det_to_plot(pos)
        fwhm_label = fwhm_hz * 1e-6
        ax.plot(_det_to_plot(det_fine), y_gauss, "-", color="C1", lw=1.5,
                label=f"Gaussian fit\ncenter={xpos:.4f}\nFWHM={fwhm_label:.3f} MHz")
        ax.axvline(xpos, color="C1", lw=1.0, ls="--", alpha=0.6)

    # --- Vertical-line overlay (peaks_dips / Lorentzian, default) ---
    elif np.isfinite(pos):
        xpos = _det_to_plot(pos)
        ax.axvline(xpos, color="C1", lw=1.5, ls="--", label=f"fit: {xpos:.4f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
