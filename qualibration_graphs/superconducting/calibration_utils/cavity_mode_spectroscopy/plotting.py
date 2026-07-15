"""Plotting utilities for the cavity mode spectroscopy measurement (node 27)."""
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def _lorentzian_dip(x, amplitude, center, hwhm, offset):
    return offset - amplitude * hwhm ** 2 / (hwhm ** 2 + (x - center) ** 2)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results=None,
    mode_name: str = "alice",
) -> Figure:
    """Plot cavity mode spectroscopy: qubit excitation vs cavity drive frequency.

    A dip in excitation probability marks the cavity resonance frequency.
    When Gaussian fit parameters are present in *fit_results* (use_gaussian_fit=True),
    the Gaussian curve is drawn over the data in addition to the vertical marker.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        fp = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], fp)

    grid.fig.suptitle(f"Cavity Mode Spectroscopy — {mode_name}")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit_params=None):
    use_full_freq = "full_freq" in ds_q.coords
    det_hz = ds_q.detuning.values.astype(float)
    if use_full_freq:
        x_plot = ds_q.full_freq.values * 1e-9
        xlabel = "RF frequency (GHz)"
        rf_freq_hz = float(ds_q.full_freq.values[0]) - float(det_hz[0])
        def _det_to_plot(d_hz):
            return (rf_freq_hz + d_hz) * 1e-9
    else:
        x_plot = det_hz * 1e-6
        xlabel = "Detuning (MHz)"
        def _det_to_plot(d_hz):
            return d_hz * 1e-6

    signal_name = "state" if "state" in ds_q.data_vars else "I"
    if signal_name == "state":
        y = ds_q.state.values
        ylabel = "State population"
    else:
        y = ds_q.I.values * 1e3
        ylabel = "I (mV)"

    ax.plot(x_plot, y, ".", ms=4, color="C0", alpha=0.8, label="data")

    if fit_params is not None:
        success = fit_params.get("success", False)
        freq_hz = fit_params.get("frequency_hz", float("nan"))
        fwhm_hz = fit_params.get("fwhm_hz", float("nan"))
        center_det = fit_params.get("center_detuning_hz", float("nan"))

        # --- Gaussian curve overlay (use_gaussian_fit=True) ---
        g_amp = fit_params.get("gaussian_amplitude", float("nan"))
        g_sigma = fit_params.get("gaussian_sigma_hz", float("nan"))
        g_offset_neg = fit_params.get("gaussian_offset_neg", float("nan"))
        has_gaussian = np.isfinite(g_sigma) and np.isfinite(g_amp)

        if success and np.isfinite(freq_hz):
            freq_plot = _det_to_plot(center_det)
            ax.axvline(freq_plot, color="r", ls="--", lw=1.5,
                       label=f"f_cav={freq_hz*1e-9:.4f} GHz")

            if has_gaussian:
                det_fine = np.linspace(det_hz.min(), det_hz.max(), 500)
                y_gauss_neg = g_offset_neg + g_amp * np.exp(-0.5 * ((det_fine - center_det) / g_sigma) ** 2)
                y_gauss = -y_gauss_neg  # back to original signal space
                if signal_name == "I":
                    y_gauss = y_gauss * 1e3  # mV
                ax.plot(_det_to_plot(det_fine), y_gauss, "-", color="r", lw=1.5,
                        label=f"Gaussian fit\nFWHM={fwhm_hz*1e-6:.3f} MHz")

            fwhm_mhz = fwhm_hz * 1e-6
            fit_type = "Gaussian" if has_gaussian else "Lorentzian"
            ax.set_title(
                f"f_cav = {freq_hz*1e-9:.5f} GHz\n"
                f"FWHM = {fwhm_mhz:.2f} MHz  [{fit_type} — SUCCESS]",
                fontsize=9, color="green",
            )
        else:
            ax.set_title("[FAILED]", fontsize=9, color="red")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
