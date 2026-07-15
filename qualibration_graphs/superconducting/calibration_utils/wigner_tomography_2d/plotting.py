"""Visualisation utilities for 2D Wigner tomography results."""
from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt

from .analysis import wigner_matrix_and_grad


# ─────────────────────────────────────────────────────────────────────────────
# Shared style constants
# ─────────────────────────────────────────────────────────────────────────────

_W_TICKS = [-2 / np.pi, -1 / np.pi, 0, 1 / np.pi, 2 / np.pi]
_W_LABELS = [r"$-2/\pi$", r"$-1/\pi$", "0", r"$1/\pi$", r"$2/\pi$"]
_LABEL_FS = 12
_TITLE_FS = 12
_SUPTITLE_FS = 16


def _add_wigner_colorbar(fig, mappable, ax) -> None:
    cb = fig.colorbar(mappable, ax=ax)
    cb.set_label(r"$\mathcal{W}$", fontsize=_LABEL_FS)
    cb.set_ticks(_W_TICKS)
    cb.set_ticklabels(_W_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# Wigner function from density matrix
# ─────────────────────────────────────────────────────────────────────────────

def wigner_from_rho(rho: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    """Evaluate the Wigner function on a 2D grid using the Laguerre expansion.

    Returns W of shape (len(y_vals), len(x_vals)).
    """
    N_ph = rho.shape[0]
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_disps = (xx + 1j * yy).ravel()
    M, _, _ = wigner_matrix_and_grad(grid_disps, N_ph)
    W_flat = (2.0 / np.pi) * (M @ rho.ravel()).real
    return W_flat.reshape(len(y_vals), len(x_vals))


# ─────────────────────────────────────────────────────────────────────────────
# Probe-displacement scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_wigner_disps(
    disps: np.ndarray,
    kappa_opt: float,
    N_ph: int,
    N_exp: int,
    title: str = "",
) -> plt.Figure:
    """Scatter the optimised probe displacements and compare against a square grid."""
    n_side = int(math.ceil(math.sqrt(N_exp)))
    r_max = 2.08 * math.sqrt(N_ph / 14)
    xs = np.linspace(-r_max, r_max, n_side)
    xx, yy = np.meshgrid(xs, xs)
    grid_disps = (xx + 1j * yy).ravel()[:N_exp]

    M_grid, _, _ = wigner_matrix_and_grad(grid_disps, N_ph)
    S_grid = np.linalg.svd(M_grid, compute_uv=False)
    kappa_grid = S_grid[0] / S_grid[-1] if S_grid[-1] > 1e-15 else np.inf

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    ax.scatter(disps.real, disps.imag, s=20, color="C0", label="optimised")
    ax.scatter(grid_disps.real, grid_disps.imag, s=10, color="C1", alpha=0.5,
               label=f"square grid ({n_side}²)")
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re$(\alpha)$", fontsize=_LABEL_FS)
    ax.set_ylabel(r"Im$(\alpha)$", fontsize=_LABEL_FS)
    ax.set_title(
        f"Displacement sets  (N_ph={N_ph}, N_exp={N_exp})\n"
        f"κ_opt={kappa_opt:.2f}  κ_grid={kappa_grid:.1f}",
        fontsize=_TITLE_FS,
    )
    ax.legend(fontsize=8)
    ax.grid(linestyle="dashed")

    ax = axes[1]
    ax.scatter(disps.real, disps.imag, s=20, color="C0")
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re$(\alpha)$", fontsize=_LABEL_FS)
    ax.set_ylabel(r"Im$(\alpha)$", fontsize=_LABEL_FS)
    ax.set_title(
        f"Optimised displacements\n{title}" if title else "Optimised displacements",
        fontsize=_TITLE_FS,
    )
    ax.grid(linestyle="dashed")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3-panel result plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_wigner_result(
    rho: np.ndarray,
    disps: np.ndarray,
    kappa: float,
    parity: np.ndarray | None = None,
    fidelity: float | None = None,
    wigner_range: float | None = None,
    title: str = "",
    n_grid: int = 101,
) -> plt.Figure:
    """3-panel figure: measured W(α) | reconstructed W(β) | photon-number bar.

    Parameters
    ----------
    rho          : (N_ph, N_ph) reconstructed density matrix
    disps        : (N_exp,) complex probe displacements (photon units)
    kappa        : tomography matrix condition number
    parity       : (N_exp,) raw parity measurements W_exp = P_- − P_+ (optional)
    fidelity     : optional fidelity to display in the title
    wigner_range : plot half-range (default: sqrt(N_ph) + 1.5)
    title        : appended to figure suptitle
    n_grid       : grid size for the reconstructed Wigner heatmap
    """
    N_ph = rho.shape[0]
    n_pts = len(disps)
    alpha_max = float(np.max(np.abs(disps)))

    if wigner_range is None or wigner_range <= 0:
        wigner_range = max(2.0, math.sqrt(N_ph) + 1.5)

    x_vals = np.linspace(-wigner_range, wigner_range, n_grid)
    y_vals = np.linspace(-wigner_range, wigner_range, n_grid)
    W = wigner_from_rho(rho, x_vals, y_vals)
    pn_diag = np.real(np.diag(rho))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: measured Wigner (tricontourf) or probe scatter ──────────────
    ax = axes[0]
    plot_x = disps.real
    plot_y = disps.imag
    if parity is not None:
        tcf = ax.tricontourf(
            plot_x, plot_y, (2.0 / np.pi) * parity,
            np.linspace(-2 / np.pi, 2 / np.pi, 32),
            cmap="RdBu_r", extend="both",
        )
        ax.plot(plot_x, plot_y, "ko", ms=1)
        _add_wigner_colorbar(fig, tcf, ax)
        ax.set_title(f"W(α) — measured  ({n_pts} pts)", fontsize=_TITLE_FS)
    else:
        ax.scatter(plot_x, plot_y, s=15, color="C0")
        ax.set_title(
            f"Probe displacements  κ={kappa:.2f}  ({n_pts} pts)", fontsize=_TITLE_FS
        )
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re$(\alpha)$", fontsize=_LABEL_FS)
    ax.set_ylabel(r"Im$(\alpha)$", fontsize=_LABEL_FS)
    ax.set_xlim(-alpha_max, alpha_max)
    ax.set_ylim(-alpha_max, alpha_max)
    ax.grid(linestyle="dashed")

    # ── Panel 2: reconstructed Wigner from ρ ─────────────────────────────────
    ax = axes[1]
    c = ax.pcolormesh(x_vals, y_vals, W, cmap="RdBu_r", vmin=-2 / np.pi, vmax=2 / np.pi)
    _add_wigner_colorbar(fig, c, ax)
    ax.set_xlabel(r"Re$(\beta)$", fontsize=_LABEL_FS)
    ax.set_ylabel(r"Im$(\beta)$", fontsize=_LABEL_FS)
    ftitle = f"  F={fidelity:.3f}" if fidelity is not None else ""
    ax.set_title(f"Reconstructed W(β){ftitle}", fontsize=_TITLE_FS)
    ax.set_aspect("equal")
    ax.grid(linestyle="dashed")

    # ── Panel 3: photon-number distribution ──────────────────────────────────
    ax = axes[2]
    ns = np.arange(N_ph)
    ax.bar(ns, pn_diag, color="C0", alpha=0.8)
    ax.set_xlabel("Photon number n", fontsize=_LABEL_FS)
    ax.set_ylabel(r"$P(n) = \rho_{nn}$", fontsize=_LABEL_FS)
    ax.set_title("Photon-number distribution", fontsize=_TITLE_FS)
    ax.set_xticks(ns)
    ax.grid(linestyle="dashed", axis="y")

    suptitle = "Wigner Tomography Result"
    if fidelity is not None:
        suptitle += f"  F={fidelity * 100:.2f}"
    if title:
        suptitle += f" — {title}"
    fig.suptitle(suptitle, fontsize=_SUPTITLE_FS)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Density matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_density_matrix(rho: np.ndarray, title: str = "", fidelity: float | None = None) -> plt.Figure:
    """2-panel figure: |ρ_nm| magnitude | arg(ρ_nm) phase, both as pcolormesh."""
    N_ph = rho.shape[0]
    photon_idx = np.arange(N_ph)
    mag = np.abs(rho)
    phase = np.angle(rho)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    c0 = ax.pcolormesh(photon_idx, photon_idx, mag, cmap="RdBu_r", vmin=-1, vmax=1)
    cb = fig.colorbar(c0, ax=ax)
    cb.set_label(r"$|\rho_{nm}|$", fontsize=_LABEL_FS)
    ax.set_title(r"$|\rho_{nm}|$", fontsize=_TITLE_FS)
    ax.set_xlabel("m", fontsize=_LABEL_FS)
    ax.set_ylabel("n", fontsize=_LABEL_FS)
    ax.set_xticks(photon_idx)
    ax.set_yticks(photon_idx)
    ax.grid(linestyle="dashed")

    ax = axes[1]
    c1 = ax.pcolormesh(photon_idx, photon_idx, phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    cb = fig.colorbar(c1, ax=ax)
    cb.set_label("rad", fontsize=_LABEL_FS)
    ax.set_title(r"$\arg(\rho_{nm})$", fontsize=_TITLE_FS)
    ax.set_xlabel("m", fontsize=_LABEL_FS)
    ax.set_ylabel("n", fontsize=_LABEL_FS)
    ax.set_xticks(photon_idx)
    ax.set_yticks(photon_idx)
    ax.grid(linestyle="dashed")

    suptitle = title
    if fidelity is not None:
        f_tag = f"F={fidelity * 100:.2f}"
        suptitle = f"{suptitle}  {f_tag}" if suptitle else f_tag
        axes[0].set_title(fr"$|\rho_{{nm}}|$  {f_tag}", fontsize=_TITLE_FS)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_SUPTITLE_FS)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    return fig
