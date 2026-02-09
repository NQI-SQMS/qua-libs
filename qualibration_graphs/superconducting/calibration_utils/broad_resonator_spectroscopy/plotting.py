from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax1, qubit in grid_iter(grid):
        ds.assign_coords(
            full_freq_GHz=ds.full_freq / u.GHz
        ).loc[qubit].phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")

        ax2 = ax1.twiny()
        ds.assign_coords(
            detuning_MHz=ds.detuning / u.MHz
        ).loc[qubit].phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")

    grid.fig.suptitle("Resonator spectroscopy (phase)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_raw_amplitude_with_fit(
    ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset
) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_individual_amplitude_with_fit(
            ax,
            ds,
            qubit,
            fits.sel(qubit=qubit["qubit"]) if fits is not None else None,
        )

    grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
):
    # ---- Optional Lorentzian (legacy) ----
    fitted_data = None
    if fit is not None and all(k in fit for k in ("amplitude", "position", "width", "base_line")):
        try:
            fitted_data = lorentzian_dip(
                ds.detuning,
                float(fit.amplitude.values),
                float(fit.position.values),
                float(fit.width.values) / 2,
                float(fit.base_line.mean().values),
            )
        except Exception:
            fitted_data = None

    # ---- Main plot: RF frequency ----
    (
        ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz)
        .loc[qubit]
        .IQ_abs
        / u.mV
    ).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")

    # ---- Secondary axis: detuning ----
    ax2 = ax.twiny()
    (
        ds.assign_coords(detuning_MHz=ds.detuning / u.MHz)
        .loc[qubit]
        .IQ_abs
        / u.mV
    ).plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]")

    if fitted_data is not None:
        ax2.plot(ds.detuning / u.MHz, fitted_data / u.mV, "r--", linewidth=1)

    # ---- Plot chosen resonator frequency (SINGLE RED DOT) ----
    if fit is not None and "res_freq" in fit and "success" in fit:
        try:
            if bool(fit["success"].values):
                full_freq = np.asarray(ds.full_freq.values).ravel()

                raw = ds.loc[qubit].IQ_abs.values
                raw = np.asarray(raw).squeeze().ravel()   # <-- CRITICAL FIX

                f_chosen = float(np.asarray(fit["res_freq"].values).squeeze())

                # interpolate magnitude at fitted frequency
                y_at_pos = np.interp(f_chosen, full_freq, raw) / u.mV

                ax.scatter(
                    f_chosen / u.GHz,
                    y_at_pos,
                    s=30,
                    c="red",
                    marker="o",
                    zorder=20,
                )

        except Exception as e:
            print(f"[plot] failed to plot red dot for {qubit}: {e}")

