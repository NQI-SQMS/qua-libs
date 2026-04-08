"""Plotting utilities for the T1 + thermal population monitor (node 34)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Tuple

from calibration_utils.T1_thermal_monitor.analysis import IterResult


def _scaled_time(t_sec: np.ndarray) -> Tuple[np.ndarray, str]:
    """Auto-scale elapsed time array to the most readable unit.

    Rules:
      total < 120 s    → seconds
      120 s – 7200 s   → minutes
      > 7200 s         → hours
    """
    total = float(np.nanmax(t_sec)) if t_sec.size > 0 else 0.0
    if total < 120.0:
        return t_sec, "Elapsed time (s)"
    elif total < 7200.0:
        return t_sec / 60.0, "Elapsed time (min)"
    else:
        return t_sec / 3600.0, "Elapsed time (h)"


def plot_T1_thermal_monitor(
    iter_results: Dict[str, IterResult],
    qubits,
) -> Figure:
    """Two-panel time-trace figure per qubit: T1 on top, P_th on bottom.

    The time axis is automatically scaled to seconds / minutes / hours
    depending on the total experiment duration.
    """
    n_qubits = max(len(iter_results), 1)
    fig, axes = plt.subplots(
        2, n_qubits,
        figsize=(6 * n_qubits, 7),
        squeeze=False,
    )

    for col, qubit in enumerate(qubits):
        q = qubit.name
        ax_T1 = axes[0, col]
        ax_Pth = axes[1, col]
        res = iter_results.get(q)

        ax_T1.set_title(q, fontsize=11)

        if res is None or len(res.t_sec) == 0:
            for ax in (ax_T1, ax_Pth):
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
            continue

        t_raw = np.array(res.t_sec)
        T1 = np.array(res.T1_us)
        T1_err = np.array(res.T1_error_us)
        P_th_pct = np.array(res.P_th) * 100.0

        t_scaled, t_label = _scaled_time(t_raw)

        # ── T1 panel ──────────────────────────────────────────────────────────
        valid_T1 = np.isfinite(T1) & np.isfinite(T1_err)
        if valid_T1.any():
            ax_T1.errorbar(
                t_scaled[valid_T1], T1[valid_T1], yerr=T1_err[valid_T1],
                fmt="o", ms=3, lw=0.8, capsize=2, color="C0", alpha=0.8,
                label=f"T1 (n={valid_T1.sum()})",
            )
            mean_T1 = float(np.nanmean(T1[valid_T1]))
            ax_T1.axhline(
                mean_T1, color="C3", ls="--", lw=1.2,
                label=f"mean = {mean_T1:.1f} µs",
            )
        ax_T1.set_ylabel("T1 (µs)")
        ax_T1.set_xlabel(t_label)
        ax_T1.legend(fontsize=8)

        # ── P_th panel ────────────────────────────────────────────────────────
        valid_Pth = np.isfinite(P_th_pct)
        if valid_Pth.any():
            ax_Pth.plot(
                t_scaled[valid_Pth], P_th_pct[valid_Pth],
                "s", ms=4, color="C1", alpha=0.8,
                label=f"P_th (n={valid_Pth.sum()})",
            )
            mean_Pth = float(np.nanmean(P_th_pct[valid_Pth]))
            ax_Pth.axhline(
                mean_Pth, color="C3", ls="--", lw=1.2,
                label=f"mean = {mean_Pth:.3f}%",
            )
        ax_Pth.set_ylabel("Thermal population (%)")
        ax_Pth.set_xlabel(t_label)
        ax_Pth.legend(fontsize=8)

    fig.suptitle("T1 & Thermal Population Monitor", fontsize=13)
    fig.tight_layout()
    return fig
