from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


# ---------------------------------------------------------------------------
# IQ blob scatter plots
# ---------------------------------------------------------------------------

def plot_iq_blobs(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Scatter plot of raw GEF IQ blobs with Gaussian-fitted centres."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_iq_blobs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=3)
    for h in leg.legend_handles:
        h.set_markersize(6)
    grid.fig.suptitle("GEF IQ blobs (volts)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_iq_blobs(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset):
    ax.plot(1e3 * fit.Ig, 1e3 * fit.Qg, ".", alpha=0.2, label="|g⟩", markersize=6)
    ax.plot(1e3 * fit.Ie, 1e3 * fit.Qe, ".", alpha=0.2, label="|e⟩", markersize=6)
    ax.plot(1e3 * fit.If, 1e3 * fit.Qf, ".", alpha=0.2, label="|f⟩", markersize=6)
    ax.plot(1e3 * float(fit.I_g_center), 1e3 * float(fit.Q_g_center), "o", c="C0", ms=8, mec="k")
    ax.plot(1e3 * float(fit.I_e_center), 1e3 * float(fit.Q_e_center), "o", c="C1", ms=8, mec="k")
    ax.plot(1e3 * float(fit.I_f_center), 1e3 * float(fit.Q_f_center), "o", c="C2", ms=8, mec="k")
    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(qubit["qubit"])


# ---------------------------------------------------------------------------
# Confusion matrices (LDA)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot LDA 3×3 confusion matrices for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_confusion_matrix(ax, qubit, fits.sel(qubit=qubit["qubit"]))
    grid.fig.suptitle("GEF readout fidelity — LDA classifier")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_confusion_matrix(ax: Axes, qubit: dict, fit: xr.Dataset):
    # Use LDA confusion matrix stored in ds_fit if available, else compute nearest-centroid
    if "confusion_matrix_lda" in fit:
        confusion = np.asarray(fit.confusion_matrix_lda.values, dtype=float)
    else:
        # Fallback: nearest-centroid from raw blob data
        confusion = _compute_nc_confusion(fit)

    im = ax.imshow(confusion, vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2], labels=["|g⟩", "|e⟩", "|f⟩"])
    ax.set_yticks([0, 1, 2], labels=["|g⟩", "|e⟩", "|f⟩"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    for prep in range(3):
        for meas in range(3):
            color = "k" if prep == meas else "w"
            ax.text(meas, prep, f"{100 * confusion[prep, meas]:.1f}%",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_title(qubit["qubit"])


def _compute_nc_confusion(fit: xr.Dataset) -> np.ndarray:
    """Nearest-centroid fallback confusion matrix (from raw shots in fit dataset)."""
    confusion = np.zeros((3, 3))
    centers = np.array([
        [float(fit.I_g_center), float(fit.Q_g_center)],
        [float(fit.I_e_center), float(fit.Q_e_center)],
        [float(fit.I_f_center), float(fit.Q_f_center)],
    ])
    for p, prep_state in enumerate(["g", "e", "f"]):
        shots = np.column_stack([fit[f"I{prep_state}"].values, fit[f"Q{prep_state}"].values])
        d2 = np.sum((shots[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        counts = np.argmin(d2, axis=1)
        for j in range(3):
            confusion[p, j] = np.mean(counts == j)
    return confusion


# ---------------------------------------------------------------------------
# LDA decision-boundary plot
# ---------------------------------------------------------------------------

def plot_lda_boundaries(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot LDA 2-D decision boundaries with GEF blob scatter for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_lda_boundaries(ax, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=3)
    for h in leg.legend_handles:
        h.set_markersize(6)
    grid.fig.suptitle("GEF LDA decision boundaries")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_lda_boundaries(ax: Axes, qubit: dict, fit: xr.Dataset):
    Ig = fit.Ig.values.astype(float)
    Qg = fit.Qg.values.astype(float)
    Ie = fit.Ie.values.astype(float)
    Qe = fit.Qe.values.astype(float)
    If = fit.If.values.astype(float)
    Qf = fit.Qf.values.astype(float)

    mu_g = np.array([float(fit.I_g_center), float(fit.Q_g_center)])
    mu_e = np.array([float(fit.I_e_center), float(fit.Q_e_center)])
    mu_f = np.array([float(fit.I_f_center), float(fit.Q_f_center)])
    mus = [mu_g, mu_e, mu_f]

    Sigma = np.asarray(fit.lda_sigma.values, dtype=float)  # shape (2, 2)
    priors = np.asarray(fit.lda_priors.values, dtype=float)
    Sigma_inv = np.linalg.inv(Sigma + 1e-9 * np.eye(2))

    # Build decision boundary grid
    all_I = np.concatenate([Ig, Ie, If])
    all_Q = np.concatenate([Qg, Qe, Qf])
    pad_I = 0.15 * max(all_I.max() - all_I.min(), 1e-9)
    pad_Q = 0.15 * max(all_Q.max() - all_Q.min(), 1e-9)
    xx, yy = np.meshgrid(
        np.linspace(all_I.min() - pad_I, all_I.max() + pad_I, 300),
        np.linspace(all_Q.min() - pad_Q, all_Q.max() + pad_Q, 300),
    )
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    scores = []
    for k in range(3):
        mu = mus[k]
        score = grid_pts @ (Sigma_inv @ mu) - 0.5 * (mu @ Sigma_inv @ mu) + np.log(priors[k])
        scores.append(score)
    zz = np.argmax(np.column_stack(scores), axis=1).reshape(xx.shape)

    ax.contourf(1e3 * xx, 1e3 * yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.18)
    ax.plot(1e3 * Ig, 1e3 * Qg, ".", alpha=0.2, label="|g⟩", markersize=4)
    ax.plot(1e3 * Ie, 1e3 * Qe, ".", alpha=0.2, label="|e⟩", markersize=4)
    ax.plot(1e3 * If, 1e3 * Qf, ".", alpha=0.2, label="|f⟩", markersize=4)
    ax.plot(1e3 * mu_g[0], 1e3 * mu_g[1], "o", c="C0", ms=8, mec="k")
    ax.plot(1e3 * mu_e[0], 1e3 * mu_e[1], "o", c="C1", ms=8, mec="k")
    ax.plot(1e3 * mu_f[0], 1e3 * mu_f[1], "o", c="C2", ms=8, mec="k")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(qubit["qubit"])


# Keep old name for backward compatibility
def plot_individual_iq_blobs(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None):
    _plot_individual_iq_blobs(ax, ds, qubit, fit)


def plot_individual_confusion_matrix(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None):
    _plot_individual_confusion_matrix(ax, qubit, fit)
