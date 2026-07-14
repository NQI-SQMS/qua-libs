from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def _add_rf_secondary_axis(ax: Axes, center_ghz: float):
    """Add a top x-axis showing absolute RF frequency in GHz."""
    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            lambda x, c=center_ghz: x / 1e3 + c,
            lambda x, c=center_ghz: (x - c) * 1e3,
        ),
    )
    ax2.set_xlabel("RF frequency [GHz]")
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))


def plot_gef_fidelity_map(ds_fit: xr.Dataset, qubits: List, fit_results: Dict) -> Figure:
    """2D colour-map of GEF fidelity vs readout frequency and amplitude, with the optimum marked."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_single_fidelity_map(ax, ds_fit, qubit, fit_results)
    grid.fig.suptitle("GEF readout 2D optimisation — fidelity map")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_iq_blobs_at_optimal(ds_fit: xr.Dataset, qubits: List, fit_results: Dict) -> Figure:
    """Scatter plot of the g / e / f IQ blobs at the optimal (frequency, amplitude) point."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_single_blobs(ax, ds_fit, qubit, fit_results)
    grid.fig.suptitle("GEF IQ blobs at optimal operating point")
    grid.fig.set_size_inches(12, 8)
    grid.fig.tight_layout()
    return grid.fig


def plot_fidelity_vs_frequency(ds_fit: xr.Dataset, qubits: List, fit_results: Dict) -> Figure:
    """GEF fidelity vs frequency at the optimal amplitude, to show the frequency sensitivity."""
    qubit_map = {q.name: q for q in qubits}
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        res = fit_results[q_name]
        ds_q = ds_fit.sel(qubit=q_name)

        opt_ai = int(np.argmin(np.abs(ds_q.coords["amplitude"].values - res["optimal_amp_scale"])))
        fid_vs_freq = ds_q.fidelity_gef.isel(amplitude=opt_ai).values
        freqs_mhz = ds_q.coords["frequency"].values / 1e6

        ax.plot(freqs_mhz, 100 * fid_vs_freq, color="C0", lw=2)
        ax.axvline(
            res["optimal_gef_detuning"] / 1e6, color="r", ls="--", lw=1.5,
            label=f"opt {res['optimal_gef_detuning']/1e6:.2f} MHz",
        )
        ax.set_xlabel("Detuning from current GEF freq [MHz]")
        ax.set_ylabel("GEF fidelity [%]")
        ax.set_title(q_name)
        ax.legend(fontsize=8)

        q_obj = qubit_map.get(q_name)
        if q_obj is not None:
            gef_shift = q_obj.resonator.GEF_frequency_shift or 0
            center_ghz = (q_obj.resonator.RF_frequency + gef_shift) / 1e9
            _add_rf_secondary_axis(ax, center_ghz)

    grid.fig.suptitle("GEF fidelity vs frequency at optimal amplitude")
    grid.fig.set_size_inches(12, 6)
    grid.fig.tight_layout()
    return grid.fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_single_fidelity_map(ax: Axes, ds_fit: xr.Dataset, qubit: dict, fit_results: Dict):
    q_name = qubit["qubit"]
    res = fit_results[q_name]
    ds_q = ds_fit.sel(qubit=q_name)

    fidelity = ds_q.fidelity_gef.transpose("amplitude", "frequency").values
    freq_ghz = ds_q.coords["abs_frequency"].values / 1e9
    amp_mv = ds_q.coords["readout_amplitude_v"].values * 1e3

    c = ax.pcolor(freq_ghz, amp_mv, 100 * fidelity, shading="auto", cmap="viridis")
    ax.figure.colorbar(c, ax=ax, label="GEF fidelity [%]")

    opt_freq_ghz = res["optimal_frequency"] / 1e9
    opt_amp_mv = res["optimal_amplitude_v"] * 1e3
    opt_fid = res["max_fidelity"] * 100
    ax.plot(
        opt_freq_ghz, opt_amp_mv, "r*", markersize=14,
        label=f"opt: {opt_freq_ghz:.5f} GHz / {opt_amp_mv:.1f} mV / {opt_fid:.1f}%",
    )
    ax.set_xlabel("Readout frequency [GHz]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(q_name)
    ax.legend(fontsize=7, loc="lower center")


def _plot_single_blobs(ax: Axes, ds_fit: xr.Dataset, qubit: dict, fit_results: Dict):
    q_name = qubit["qubit"]
    res = fit_results[q_name]
    ds_q = ds_fit.sel(qubit=q_name)

    opt_fi = int(np.argmin(np.abs(ds_q.coords["frequency"].values - res["optimal_gef_detuning"])))
    opt_ai = int(np.argmin(np.abs(ds_q.coords["amplitude"].values - res["optimal_amp_scale"])))

    Ig = ds_q["Ig"].isel(frequency=opt_fi, amplitude=opt_ai).values * 1e3
    Qg = ds_q["Qg"].isel(frequency=opt_fi, amplitude=opt_ai).values * 1e3
    Ie = ds_q["Ie"].isel(frequency=opt_fi, amplitude=opt_ai).values * 1e3
    Qe = ds_q["Qe"].isel(frequency=opt_fi, amplitude=opt_ai).values * 1e3
    If = ds_q["If"].isel(frequency=opt_fi, amplitude=opt_ai).values * 1e3
    Qf = ds_q["Qf"].isel(frequency=opt_fi, amplitude=opt_ai).values * 1e3

    alpha = min(1.0, max(0.05, 2000 / max(len(Ig), 1)))
    ax.scatter(Ig, Qg, s=2, alpha=alpha, color="C0", label="|g⟩")
    ax.scatter(Ie, Qe, s=2, alpha=alpha, color="C1", label="|e⟩")
    ax.scatter(If, Qf, s=2, alpha=alpha, color="C2", label="|f⟩")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(f"{q_name}  —  {res['max_fidelity'] * 100:.1f}% GEF fidelity")
    ax.legend(fontsize=8, markerscale=5)
    ax.set_aspect("equal")
