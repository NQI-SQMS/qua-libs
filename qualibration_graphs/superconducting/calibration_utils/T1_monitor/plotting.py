"""Plotting utilities for the T1 monitor node (node 29)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict

from calibration_utils.T1_monitor.analysis import T1MonitorResult


def plot_T1_monitor(monitor_results: Dict[str, T1MonitorResult], qubits) -> Figure:
    """Plot T1 vs elapsed time for all monitored qubits.

    Parameters
    ----------
    monitor_results : dict
        Mapping qubit name → T1MonitorResult.
    qubits :
        Qubit objects (used only for qubit name labels).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_qubits = len(monitor_results)
    fig, axes = plt.subplots(1, max(n_qubits, 1), figsize=(6 * max(n_qubits, 1), 4), squeeze=False)
    axes = axes[0]

    for ax, qubit in zip(axes, qubits):
        q_name = qubit.name
        res = monitor_results.get(q_name)
        if res is None or len(res.t_min) == 0:
            ax.set_title(q_name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        t = np.array(res.t_min)
        T1 = np.array(res.T1_us)
        T1_err = np.array(res.T1_error_us)

        # Mask invalid (nan) points
        valid = np.isfinite(T1) & np.isfinite(T1_err)

        if valid.any():
            ax.errorbar(
                t[valid], T1[valid], yerr=T1_err[valid],
                fmt="o", ms=3, lw=0.8, capsize=2, color="C0", alpha=0.7,
                label=f"T1 (n={valid.sum()})",
            )
            mean_T1 = float(np.mean(T1[valid]))
            ax.axhline(mean_T1, color="r", ls="--", lw=1.2, label=f"mean = {mean_T1:.1f} µs")

        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("T1 (µs)")
        ax.set_title(q_name)
        ax.legend(fontsize=8)

    fig.suptitle("T1 Monitor")
    fig.tight_layout()
    return fig
