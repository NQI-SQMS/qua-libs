"""Plotting for displacement calibration via PNRS (node 28).

Produces two subplots per qubit:
  1. 2D colormesh: displacement amplitude (x) vs qubit detuning (y, kHz) vs state (colour).
     Fitted peak positions for each photon number n are overlaid as dashed lines.
  2. n̄ vs A: raw (A, n̄) scatter + fitted n̄ = k·A² curve + vertical marker at A₁ph.
"""
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    num_qubits = len(list(qubits))

    fig = plt.figure(figsize=(12, 5 * num_qubits))
    outer = gridspec.GridSpec(num_qubits, 1, figure=fig, hspace=0.5)

    for qubit_idx, qubit in enumerate(qubits):
        q_name = qubit.name
        ds_q = ds.sel(qubit=q_name)

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        if signal_name == "state":
            data = ds_q.state.values          # shape (n_amps, n_dfs)
            clabel = "State population"
        else:
            data = ds_q.I.values * 1e3
            clabel = "I (mV)"

        power_dbm = ds_q.power.values
        amps = 10 ** ((power_dbm - power_dbm.max()) / 20)  # linear scale factors
        dfs_khz = ds_q.detuning.values * 1e-3

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[qubit_idx], wspace=0.35
        )
        ax2d = fig.add_subplot(inner[0])
        ax_nb = fig.add_subplot(inner[1])

        # --- 2D colormesh ---
        pcm = ax2d.pcolormesh(
            power_dbm, dfs_khz, data.T, shading="auto", cmap="viridis"
        )
        fig.colorbar(pcm, ax=ax2d, label=clabel)
        ax2d.set_xlabel("Displacement power (dBm)")
        ax2d.set_ylabel("Qubit detuning (kHz)")
        ax2d.set_title(f"{q_name} — {mode_name}: 2D PNRS map")

        # Overlay fitted peak positions if available
        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["success"] and np.isfinite(res["chi_hz"]) and res["chi_hz"] > 0:
                chi_khz = res["chi_hz"] * 1e-3
                # Draw one horizontal line per detected peak (n=0,1,2,...)
                n_overlay = max(res.get("num_peaks_used", 4), 4)
                for n_idx in range(n_overlay):
                    peak_khz = -n_idx * chi_khz
                    if dfs_khz.min() <= peak_khz <= dfs_khz.max():
                        ax2d.axhline(
                            peak_khz, color=f"C{n_idx % 10}", ls="--", lw=1.2,
                            label=f"n={n_idx}"
                        )
                ax2d.legend(fontsize=7, loc="upper right")

        # --- n̄ vs A ---
        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["nbar_vs_amp"]:
                A_raw = np.array([v[0] for v in res["nbar_vs_amp"]])
                nb_raw = np.array([v[1] for v in res["nbar_vs_amp"]])
                ax_nb.scatter(A_raw, nb_raw, s=30, color="C0", zorder=5, label="data")

            if res["success"]:
                A_fit = np.linspace(0, amps.max(), 200)
                k = res["k"]
                ax_nb.plot(A_fit, k * A_fit ** 2, color="C1", lw=2, label=f"k·A²  (k={k:.2f})")
                ax_nb.axvline(
                    res["amp_for_one_photon"], color="C2", ls="--", lw=1.5,
                    label=f"A₁ph={res['amp_for_one_photon']:.4f}"
                )
            else:
                ax_nb.set_title("FIT FAILED", color="red", fontsize=9)

        ax_nb.set_xlabel("Displacement amplitude scale A (rel. to max_amp)")
        ax_nb.set_ylabel("Mean photon number n̄")
        ax_nb.set_title(f"{q_name} — n̄ vs A")
        ax_nb.legend(fontsize=8)
        ax_nb.set_xlim(left=0)
        ax_nb.set_ylim(bottom=0)

    fig.suptitle(f"Displacement Calibration (PNRS) — {mode_name} (28)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_spectrum_at_power(
    ds: xr.Dataset,
    qubits,
    selected_power_dbm: float,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot the 1D qubit spectroscopy spectrum at the nearest available displacement power point.

    Vertical dashed lines mark the expected photon-number peak positions from the fit.
    The estimated mean photon number n̄ at that power is shown in the title.
    """
    num_qubits = len(list(qubits))
    fig, axes = plt.subplots(1, num_qubits, figsize=(6 * num_qubits, 4), squeeze=False)

    for qubit_idx, qubit in enumerate(qubits):
        q_name = qubit.name
        ax = axes[0, qubit_idx]
        ds_q = ds.sel(qubit=q_name)

        # Nearest available power point
        power_dbm = ds_q.power.values
        idx = int(np.argmin(np.abs(power_dbm - selected_power_dbm)))
        actual_power = float(power_dbm[idx])

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        spectrum = ds_q.isel(power=idx)[signal_name].values
        dfs_khz = ds_q.detuning.values * 1e-3

        if signal_name == "I":
            spectrum = spectrum * 1e3
            ylabel = "I (mV)"
        else:
            ylabel = "State population"

        ax.plot(dfs_khz, spectrum, color="C0", lw=1.5)

        # Overlay expected peak positions from the global fit
        nbar_str = ""
        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["success"] and np.isfinite(res["chi_hz"]) and res["chi_hz"] > 0:
                chi_khz = res["chi_hz"] * 1e-3
                n_overlay = max(res.get("num_peaks_used", 4), 4)
                for n_idx in range(n_overlay):
                    peak_khz = -n_idx * chi_khz
                    if dfs_khz.min() <= peak_khz <= dfs_khz.max():
                        ax.axvline(
                            peak_khz, color=f"C{n_idx + 1}", ls="--", lw=1.2,
                            label=f"n={n_idx}"
                        )
                ax.legend(fontsize=8)
                # Estimated n̄ at this power
                amp_scale = float(10 ** ((actual_power - power_dbm.max()) / 20))
                nbar = res["k"] * amp_scale ** 2
                nbar_str = f"  (n̄ ≈ {nbar:.2f})"

        ax.set_xlabel("Qubit detuning (kHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{q_name} — {mode_name}: {actual_power:.1f} dBm{nbar_str}")

    fig.suptitle(
        f"PNRS Spectrum at {selected_power_dbm:.1f} dBm — {mode_name} (28)", fontsize=11
    )
    fig.tight_layout()
    return fig
