"""
Plotting utilities for the BO qubit bootstrap node.

Three panels per qubit:
  1. Convergence — cost vs. global iteration (LHS grey, BO blue, best starred)
  2. Landscape   — (ω_d, power_dBm) scatter coloured by log10(cost)
  3. Best trace  — time-Rabi signal at the converged point with fit overlay
"""

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr


def plot_bo_results(
    ds_raw: Dict[str, xr.Dataset],
    ds_best_trace: Dict[str, xr.Dataset],
    fit_results: Dict[str, dict],
) -> plt.Figure:
    """
    Build a three-column figure (one row per qubit) summarising BO results.

    Parameters
    ──────────
    ds_raw        : {qubit_name: xr.Dataset} with dims (iteration,) and
                    variables omega_d_ghz, power_dbm, cost, phase, fit_success
    ds_best_trace : {qubit_name: xr.Dataset} with dims (time_ns,) and
                    variables signal, fit
    fit_results   : {qubit_name: dict} with omega_q_hz, power_dbm, best_cost, n_evaluations
    """
    qubit_names = list(ds_raw.keys())
    n_qubits = len(qubit_names)
    fig, axes = plt.subplots(
        n_qubits, 3,
        figsize=(15, 4 * max(n_qubits, 1)),
        squeeze=False,
    )

    for row, qname in enumerate(qubit_names):
        ds   = ds_raw[qname]
        bt   = ds_best_trace.get(qname)
        info = fit_results.get(qname, {})

        ax_conv, ax_land, ax_trace = axes[row]

        _plot_convergence(ax_conv, ds, qname, info)
        _plot_landscape(ax_land, ds, qname, info)
        if bt is not None:
            _plot_best_trace(ax_trace, bt, qname, info)
        else:
            ax_trace.set_visible(False)

    fig.suptitle("BO Qubit Bootstrap — Results", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _plot_convergence(ax: plt.Axes, ds: xr.Dataset, qname: str, info: dict):
    costs     = ds["cost"].values.astype(float)
    phases    = ds["phase"].values
    n_iter    = len(costs)
    iters     = np.arange(n_iter)

    lhs_mask = np.array([p == "lhs" for p in phases])
    bo_mask  = ~lhs_mask

    # Clip sentinel values for display
    costs_display = np.where(costs > 1e5, np.nan, costs)

    if lhs_mask.any():
        ax.scatter(iters[lhs_mask], costs_display[lhs_mask],
                   color="grey", s=40, label="LHS", zorder=3)
    if bo_mask.any():
        ax.scatter(iters[bo_mask], costs_display[bo_mask],
                   color="steelblue", s=40, label="BO-EI", zorder=3)

    # Running best
    running_best = np.minimum.accumulate(np.nan_to_num(costs_display, nan=np.inf))
    running_best[running_best == np.inf] = np.nan
    ax.plot(iters, running_best, "r--", linewidth=1.2, label="running best", zorder=2)

    # Mark global best
    valid = np.isfinite(costs_display)
    if valid.any():
        best_idx = int(np.nanargmin(costs_display))
        ax.scatter([best_idx], [costs_display[best_idx]],
                   marker="*", s=200, color="red", zorder=5, label=f"best (C={costs_display[best_idx]:.2f})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost  C")
    ax.set_title(f"{qname} — convergence")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_landscape(ax: plt.Axes, ds: xr.Dataset, qname: str, info: dict):
    omega_ghz = ds["omega_d_ghz"].values.astype(float)
    power_dbm = ds["power_dbm"].values.astype(float)
    costs     = ds["cost"].values.astype(float)
    phases    = ds["phase"].values

    valid = np.isfinite(costs) & (costs < 1e5)
    if not valid.any():
        ax.text(0.5, 0.5, "no valid evaluations", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{qname} — landscape")
        return

    log_cost = np.where(valid, np.log10(np.clip(costs, 1e-3, None)), np.nan)

    sc = ax.scatter(
        omega_ghz, power_dbm,
        c=log_cost,
        cmap="plasma_r",
        s=50, edgecolors="none", alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label="log₁₀(cost)")

    # Mark best point
    best_omega = info.get("omega_q_hz", np.nan)
    best_power = info.get("power_dbm", np.nan)
    if np.isfinite(best_omega) and np.isfinite(best_power):
        ax.scatter([best_omega / 1e9], [best_power],
                   marker="*", s=250, color="red", zorder=5,
                   label=f"best\n{best_omega/1e9:.4f} GHz\n{best_power:.1f} dBm")
        ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("ω_d  (GHz)")
    ax.set_ylabel("Power  (dBm)")
    ax.set_title(f"{qname} — search landscape")
    ax.grid(True, alpha=0.3)


def _plot_best_trace(ax: plt.Axes, bt: xr.Dataset, qname: str, info: dict):
    t_ns   = bt.coords["time_ns"].values.astype(float)
    signal = bt["signal"].values.astype(float)

    ax.plot(t_ns, signal, "o", markersize=3, color="steelblue", label="data")

    if "fit" in bt and not np.all(np.isnan(bt["fit"].values)):
        fit_curve = bt["fit"].values.astype(float)
        ax.plot(t_ns, fit_curve, "r-", linewidth=1.5, label="fit")

    # Annotate with fit info
    omega_q = info.get("omega_q_hz", np.nan)
    power   = info.get("power_dbm", np.nan)
    cost    = info.get("best_cost", np.nan)
    lines = []
    if np.isfinite(omega_q):
        lines.append(f"ω_q = {omega_q/1e9:.4f} GHz")
    if np.isfinite(power):
        lines.append(f"P = {power:.1f} dBm")
    if np.isfinite(cost):
        lines.append(f"C = {cost:.3f}")
    if lines:
        ax.text(0.97, 0.97, "\n".join(lines),
                transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlabel("Pulse duration  (ns)")
    ax.set_ylabel("ADC signal  (a.u.)")
    ax.set_title(f"{qname} — best time-Rabi")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
