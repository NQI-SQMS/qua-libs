from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def _pclip(x: np.ndarray, lo: float = 5.0, hi: float = 95.0):
    """Return percentile-based (vmin, vmax) for robust colour scaling."""
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """
    4-panel DRAG calibration figure for each qubit.

    Panels
    ------
    (0,0) I heatmap  — plasma colormap, 5–95th percentile colour limits
    (0,1) Q heatmap  — PuOr colormap,   5–95th percentile colour limits
    (1,0) State heatmap — viridis, [0, 1]
    (1,1) Sum(I) vs alpha: raw + Gaussian-smoothed, optimal alpha marked

    For a single qubit the figure is 13 × 8 inches.
    For multiple qubits each qubit gets its own 4-panel figure returned in a list;
    callers expecting a single Figure receive the first entry.
    """
    if len(qubits) == 1:
        fig = _plot_single_qubit(ds, fits, qubits[0])
        return fig

    # Multi-qubit: return figures as a dict; callers can iterate node.results["figures"]
    figs = {}
    for q in qubits:
        figs[q.name] = _plot_single_qubit(ds.sel(qubit=q.name), fits.sel(qubit=q.name), q)
    # Return the first figure so existing callers that expect a single Figure still work
    return list(figs.values())[0]


def _plot_single_qubit(ds: xr.Dataset, fits: xr.Dataset, qubit: AnyTransmon) -> Figure:
    """Build the 4-panel figure for one qubit."""
    q_name = qubit.name

    # ── select per-qubit data ────────────────────────────────────────────────
    ds_q = ds.sel(qubit=q_name) if "qubit" in ds.dims else ds
    fits_q = fits.sel(qubit=q_name) if "qubit" in fits.dims else fits

    alpha_vals = ds_q.alpha.values        # shape: (n_alpha,)
    nb_pulses = ds_q.nb_of_pulses.values  # shape: (n_pulses,)
    optimal_alpha = float(fits_q["optimal_alpha"])

    # ── build 2-D arrays for heatmaps ───────────────────────────────────────
    has_state = "state" in ds_q.data_vars
    has_I = "I" in ds_q.data_vars
    has_Q = "Q" in ds_q.data_vars

    # xarray dims: (nb_of_pulses, alpha_prefactor)
    I_2d = ds_q["I"].values if has_I else None       # (n_pulses, n_alpha)
    Q_2d = ds_q["Q"].values if has_Q else None
    state_2d = ds_q["state"].values if has_state else None

    # sum(I) curves for optimisation panel
    sum_arr = fits_q["summed_data"].values if "summed_data" in fits_q else (
        I_2d.sum(axis=0) if I_2d is not None else state_2d.sum(axis=0)
    )
    sm_arr = fits_q["smoothed_data"].values if "smoothed_data" in fits_q else sum_arr

    # ── create figure ────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), dpi=110, constrained_layout=True)
    fig.suptitle(f"DRAG calibration — {q_name}", fontsize=13)

    # (0,0) I heatmap
    ax = axs[0, 0]
    if I_2d is not None:
        vmin, vmax = _pclip(I_2d)
        im = ax.pcolormesh(alpha_vals, nb_pulses, I_2d * 1e3,
                           cmap="plasma", vmin=vmin * 1e3, vmax=vmax * 1e3, shading="auto")
        fig.colorbar(im, ax=ax, label="I [mV]")
        ax.axvline(optimal_alpha, color="w", linestyle="--", linewidth=1.2, label=f"opt α={optimal_alpha:.4g}")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No I data", transform=ax.transAxes, ha="center")
    ax.set_xlabel(r"DRAG coefficient $\alpha$")
    ax.set_ylabel("Number of pulses")
    ax.set_title("I quadrature [mV]")

    # (0,1) Q heatmap
    ax = axs[0, 1]
    if Q_2d is not None:
        vmin, vmax = _pclip(Q_2d)
        im = ax.pcolormesh(alpha_vals, nb_pulses, Q_2d * 1e3,
                           cmap="PuOr", vmin=vmin * 1e3, vmax=vmax * 1e3, shading="auto")
        fig.colorbar(im, ax=ax, label="Q [mV]")
        ax.axvline(optimal_alpha, color="k", linestyle="--", linewidth=1.2, label=f"opt α={optimal_alpha:.4g}")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No Q data", transform=ax.transAxes, ha="center")
    ax.set_xlabel(r"DRAG coefficient $\alpha$")
    ax.set_ylabel("Number of pulses")
    ax.set_title("Q quadrature [mV]")

    # (1,0) State heatmap
    ax = axs[1, 0]
    if state_2d is not None:
        im = ax.pcolormesh(alpha_vals, nb_pulses, state_2d,
                           cmap="viridis", vmin=0, vmax=1, shading="auto")
        fig.colorbar(im, ax=ax, label="Excited state population")
        ax.axvline(optimal_alpha, color="w", linestyle="--", linewidth=1.2, label=f"opt α={optimal_alpha:.4g}")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No state data", transform=ax.transAxes, ha="center")
    ax.set_xlabel(r"DRAG coefficient $\alpha$")
    ax.set_ylabel("Number of pulses")
    ax.set_title("State population")

    # (1,1) Optimisation curve
    ax = axs[1, 1]
    ax.plot(alpha_vals, sum_arr, "o-", markersize=3, linewidth=1.0,
            alpha=0.6, label="Sum (raw)")
    ax.plot(alpha_vals, sm_arr, "-", linewidth=2.0, alpha=0.9,
            label=f"Smoothed (σ={fits_q.get('smooth_sigma', '')})")
    ax.axvline(optimal_alpha, linestyle="--", linewidth=1.5, color="C2",
               label=f"Optimal α = {optimal_alpha:.4g}")
    best_idx = int(np.argmin(sm_arr))
    ax.scatter([optimal_alpha], [sum_arr[best_idx]], s=60, zorder=5, color="C2")
    ax.set_xlabel(r"DRAG coefficient $\alpha$")
    ax.set_ylabel("Sum I [a.u.]")
    ax.set_title("Optimisation curve")
    ax.legend(fontsize=8)

    return fig
