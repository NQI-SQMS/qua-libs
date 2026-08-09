"""Plotting utilities for qubit spectroscopy versus flux calibration."""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot the raw IQ amplitude vs flux/frequency map and overlay the fitted
    parabola curve plus sweet-spot markers."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(
            ax, ds, qubit, fits.sel(qubit=qubit["qubit"])
        )

    grid.fig.suptitle("Qubit spectroscopy vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _safe_float(value, default=float("nan")):
    """Coerce an xarray scalar / numpy scalar / float to a Python float, with
    a fallback if the value is missing or not coerceable."""
    try:
        return float(np.asarray(value).item())
    except Exception:  # noqa: BLE001
        return default


def _is_finite(value) -> bool:
    return np.isfinite(_safe_float(value))


def plot_individual_raw_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fit: xr.Dataset = None,
):
    """Plot single-qubit data with an optional parabola fit overlay.

    The fitted ``c2*x^2 + c1*x + c0`` is drawn only inside the arch window
    (``[arch_flux_min, arch_flux_max]``); the arch window edges are drawn as
    light grey vertical lines.
    """
    ax2 = ax.twiny()
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
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax, add_colorbar=False, x="flux_bias", y="freq_GHz", robust=True
    )

    success = bool(_safe_float(fit.success, default=0.0)) if fit is not None else False
    idle_offset = _safe_float(fit.idle_offset) if fit is not None else float("nan")
    if success and _is_finite(idle_offset):
        ax.axvline(
            idle_offset,
            linestyle="dashed",
            linewidth=2,
            color="r",
            label="idle offset",
        )

    if fit is not None:
        _overlay_parabola(ax, ds, fit, qubit)

        if _is_finite(fit.upper_sweet_spot_flux):
            usf = _safe_float(fit.upper_sweet_spot_flux)
            usf_freq = _safe_float(fit.upper_sweet_spot_frequency)
            ax.plot(
                usf,
                usf_freq * 1e-9,
                marker="D",
                color="lime",
                markersize=10,
                zorder=5,
                label=f"upper SS ({usf_freq * 1e-9:.3f} GHz)",
            )
        if _is_finite(fit.lower_sweet_spot_flux):
            lsf = _safe_float(fit.lower_sweet_spot_flux)
            lsf_freq = _safe_float(fit.lower_sweet_spot_frequency)
            ax.plot(
                lsf,
                lsf_freq * 1e-9,
                marker="D",
                color="cyan",
                markersize=10,
                zorder=5,
                label=f"lower SS ({lsf_freq * 1e-9:.3f} GHz)",
            )

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux (V)")
    ax.legend(fontsize=7)


def _overlay_parabola(ax: Axes, ds: xr.Dataset, fit: xr.Dataset, qubit: dict):
    """Overlay the fitted parabola curve only within the arch window."""
    c2 = _safe_float(getattr(fit, "c2", None))
    c1 = _safe_float(getattr(fit, "c1", None))
    c0 = _safe_float(getattr(fit, "c0", None))
    if not (np.isfinite(c2) and np.isfinite(c1) and np.isfinite(c0)):
        return

    flux_min = _safe_float(getattr(fit, "arch_flux_min", None))
    flux_max = _safe_float(getattr(fit, "arch_flux_max", None))
    if not (np.isfinite(flux_min) and np.isfinite(flux_max)) or flux_max <= flux_min:
        flux_axis = ds.flux_bias.values
        flux_min = float(flux_axis.min())
        flux_max = float(flux_axis.max())

    # c0/c1/c2 are in DETUNING units (the parabola was fit to peak-position vs flux),
    # but the y-axis is the absolute RF frequency (full_freq). Add the RF reference so
    # the curve lands on the data arch instead of near 0 Hz.
    rf_freq = 0.0
    if "full_freq" in ds.coords:
        try:
            rf_freq = float(ds.full_freq.sel(qubit=qubit["qubit"]).values[0])
            rf_freq -= float(ds.detuning.values[0])
        except Exception:  # noqa: BLE001
            rf_freq = 0.0

    flux_dense = np.linspace(flux_min, flux_max, 200)
    freq_dense = c2 * flux_dense**2 + c1 * flux_dense + c0 + rf_freq
    ax.plot(
        flux_dense,
        freq_dense * 1e-9,
        color="tab:orange",
        linewidth=2,
        linestyle="-",
        alpha=0.5,
        label="Parabola fit",
    )

    for edge in (flux_min, flux_max):
        ax.axvline(edge, color="grey", linestyle=":", linewidth=1, alpha=0.5)
