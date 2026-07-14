from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from calibration_utils.resonator_punch_out.analysis import _lorentzian_dip

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator punch-out data for the given qubits.

    With exactly 2 power points (the default shift-based calibration), this shows
    the normalized magnitude and phase at low/high power with the fitted Lorentzian
    dip overlaid. With more than 2 power points (a dense diagnostic sweep), this
    instead shows a 2D heatmap of magnitude/phase vs. frequency and power, plus the
    extracted resonance-vs-power curve with the detected bifurcation marked.

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
    """
    if len(ds.power) > 2:
        return _plot_bifurcation_dense(ds, qubits, fits)

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy at low/high readout power")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


# =========================
# Dense sweep: 2D heatmap + bifurcation curve
# =========================

def _plot_bifurcation_dense(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    n_qubits = len(qubits)
    fig, axes = plt.subplots(n_qubits, 4, figsize=(20, 4.5 * n_qubits), squeeze=False)

    freqs_mhz = ds.detuning.values / u.MHz
    powers = ds.power.values

    for row, qubit in enumerate(qubits):
        ax_mag, ax_phase, ax_freq, ax_phase_vs_power = axes[row]
        ds_q = ds.sel(qubit=qubit.name)
        fit_q = fits.sel(qubit=qubit.name)

        mag = ds_q.IQ_abs_norm.transpose("power", "detuning").values
        phase = ds_q.phase.transpose("power", "detuning").values

        c1 = ax_mag.pcolormesh(freqs_mhz, powers, mag, shading="auto")
        fig.colorbar(c1, ax=ax_mag, label="|S| normalized")
        ax_mag.set_xlabel("Detuning [MHz]")
        ax_mag.set_ylabel("Readout power [dBm]")
        ax_mag.set_title(f"{qubit.name}: magnitude")

        c2 = ax_phase.pcolormesh(freqs_mhz, powers, phase, shading="auto", cmap="RdBu")
        fig.colorbar(c2, ax=ax_phase, label="Phase [rad]")
        ax_phase.set_xlabel("Detuning [MHz]")
        ax_phase.set_ylabel("Readout power [dBm]")
        ax_phase.set_title(f"{qubit.name}: phase")

        f_center_ghz = qubit.resonator.RF_frequency / 1e9
        for ax in [ax_mag, ax_phase]:
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda x, fc=f_center_ghz: x / 1e3 + fc,
                    lambda x, fc=f_center_ghz: (x - fc) * 1e3,
                ),
            )
            secax.set_xlabel("RF frequency [GHz]")

        min_freq_mhz = fit_q.min_freq.values / u.MHz
        min_phase = fit_q.min_phase.values
        safe_power = float(fit_q.safe_power)
        bif_idx = int(fit_q.bif_idx)
        success = bool(fit_q.success)

        marker_color = "lime" if success else "gray"
        ax_mag.axhline(y=safe_power, color=marker_color, linestyle="--", linewidth=1.5)
        ax_phase.axhline(y=safe_power, color=marker_color, linestyle="--", linewidth=1.5)

        ax_freq.plot(powers, min_freq_mhz, "o-", color="C0")
        ax_freq.axvline(safe_power, color=marker_color, linestyle="--",
                         label=f"safe power = {safe_power:.1f} dBm")
        if 0 <= bif_idx < len(powers):
            ax_freq.axvline(powers[bif_idx], color="red", linestyle=":", label="bifurcation")
        ax_freq.set_xlabel("Readout power [dBm]")
        ax_freq.set_ylabel("Resonance detuning [MHz]")
        ax_freq.set_title(f"{qubit.name}: resonance vs. power")
        ax_freq.legend(fontsize="x-small")

        ax_phase_vs_power.plot(powers, min_phase, "s-", color="C1")
        ax_phase_vs_power.axvline(safe_power, color=marker_color, linestyle="--",
                                   label=f"safe power = {safe_power:.1f} dBm")
        if 0 <= bif_idx < len(powers):
            ax_phase_vs_power.axvline(powers[bif_idx], color="red", linestyle=":", label="bifurcation")
        ax_phase_vs_power.set_xlabel("Readout power [dBm]")
        ax_phase_vs_power.set_ylabel("Phase at resonance [rad]")
        ax_phase_vs_power.set_title(f"{qubit.name}: phase vs. power")
        ax_phase_vs_power.legend(fontsize="x-small")

    fig.suptitle("Resonator punch-out: dense power sweep")
    fig.tight_layout()
    return fig


# =========================
# Shift-based (2 power points): line plot + fit overlay
# =========================

def _plot_fit_curve(ax: Axes, detuning_hz: np.ndarray, fit_power: xr.Dataset, norm_factor: float, color: str):
    """Overlay the smooth fitted Lorentzian curve (normalized) on `ax`, if the fit succeeded."""
    f0 = float(fit_power.lorentz_f0)
    gamma = float(fit_power.lorentz_gamma)
    amp = float(fit_power.lorentz_amp)
    offset = float(fit_power.lorentz_offset)
    if not all(np.isfinite([f0, gamma, amp, offset])):
        return
    f_grid = np.linspace(detuning_hz.min(), detuning_hz.max(), 10 * len(detuning_hz))
    curve_norm = _lorentzian_dip(f_grid, f0, gamma, amp, offset) / norm_factor
    ax.plot(f_grid / u.MHz, curve_norm, color=color, linestyle="-", linewidth=2, zorder=3)
    ax.axvline(x=f0 / u.MHz, color=color, linestyle=":", linewidth=1)


def plot_individual_raw_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None
):
    powers = ds.power.values
    P_low = float(powers[0])
    P_high = float(powers[-1])

    ds_q = ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit]

    # --- magnitude (normalized), left axis ---
    # Raw data is dimmed (faint markers) since it's noisy; the fitted curve (below)
    # carries the visual signal so the dip shape stays legible.
    ds_q.isel(power=0).IQ_abs_norm.plot(
        ax=ax, x="detuning_MHz", color="blue", marker=".", linestyle="none",
        markersize=3, alpha=0.35, label=f"|S| @ {P_low:.1f} dBm",
    )
    ds_q.isel(power=-1).IQ_abs_norm.plot(
        ax=ax, x="detuning_MHz", color="red", marker=".", linestyle="none",
        markersize=3, alpha=0.35, label=f"|S| @ {P_high:.1f} dBm",
    )
    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Normalized magnitude")

    # --- phase, dashed lines, right axis, dimmed so it stays secondary to magnitude ---
    ax2 = ax.twinx()
    ds_q.isel(power=0).phase.plot(
        ax=ax2, x="detuning_MHz", color="blue", linestyle="--", linewidth=0.8, alpha=0.4,
        label=f"phase @ {P_low:.1f} dBm",
    )
    ds_q.isel(power=-1).phase.plot(
        ax=ax2, x="detuning_MHz", color="red", linestyle="--", linewidth=0.8, alpha=0.4,
        label=f"phase @ {P_high:.1f} dBm",
    )
    ax2.set_ylabel("Phase [rad]")

    ax.set_title("")
    ax2.set_title("")

    handles = ax.get_lines() + ax2.get_lines()
    ax.legend(handles=handles, labels=[h.get_label() for h in handles], fontsize="x-small", loc="best")

    if fit is None or "lorentz_f0" not in fit:
        return

    # Overlay the fitted Lorentzian curve and mark its center, for each power.
    norm_low = float(ds_q.isel(power=0).IQ_abs.mean())
    norm_high = float(ds_q.isel(power=-1).IQ_abs.mean())
    _plot_fit_curve(ax, ds_q.detuning.values, fit.isel(power=0), norm_low, color="blue")
    _plot_fit_curve(ax, ds_q.detuning.values, fit.isel(power=-1), norm_high, color="red")
