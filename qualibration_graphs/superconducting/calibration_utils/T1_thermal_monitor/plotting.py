"""Plotting utilities for the T1 + T2* + T2 echo + thermal population monitor (node 32)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Tuple

from calibration_utils.T1_thermal_monitor.analysis import IterResult


def _scaled_time(t_sec: np.ndarray) -> Tuple[np.ndarray, str]:
    """Auto-scale elapsed time array to the most readable unit."""
    total = float(np.nanmax(t_sec)) if t_sec.size > 0 else 0.0
    if total < 120.0:
        return t_sec, "Elapsed time (s)"
    elif total < 7200.0:
        return t_sec / 60.0, "Elapsed time (min)"
    else:
        return t_sec / 3600.0, "Elapsed time (h)"


def _errorbar_panel(
    ax,
    t_scaled: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    ylabel: str,
    mean_label_fmt: str,
    color: str,
    marker: str = "o",
):
    """Helper: plot values vs time with error bars and a mean line.

    Points whose error is NaN are plotted without error bars (open markers).
    """
    valid = np.isfinite(values)
    if not valid.any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_ylabel(ylabel)
        return

    has_err = valid & np.isfinite(errors)
    no_err  = valid & ~np.isfinite(errors)

    if has_err.any():
        ax.errorbar(
            t_scaled[has_err], values[has_err], yerr=errors[has_err],
            fmt=marker, ms=3, lw=0.8, capsize=2, color=color, alpha=0.8,
            label=f"n={has_err.sum()}",
        )
    if no_err.any():
        ax.plot(
            t_scaled[no_err], values[no_err],
            marker, ms=4, color=color, alpha=0.5, mfc="none",
            label=f"(no err, n={no_err.sum()})",
        )

    mean_val = float(np.nanmean(values[valid]))
    ax.axhline(
        mean_val, color="C3", ls="--", lw=1.2,
        label=mean_label_fmt.format(mean_val),
    )
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_T1_thermal_monitor(
    iter_results: Dict[str, IterResult],
    qubits,
) -> Figure:
    """Five-panel time-trace figure per qubit:
      Row 0 – T1
      Row 1 – T2* (Ramsey)
      Row 2 – T2 echo
      Row 3 – Qubit frequency offset
      Row 4 – Thermal population P_th

    Each quantity is plotted with fit error bars where available.
    """
    n_qubits = max(len(iter_results), 1)
    n_rows = 5
    fig, axes = plt.subplots(
        n_rows, n_qubits,
        figsize=(6 * n_qubits, 4 * n_rows),
        squeeze=False,
    )

    for col, qubit in enumerate(qubits):
        q = qubit.name
        res = iter_results.get(q)

        axes[0, col].set_title(q, fontsize=11)

        if res is None or len(res.t_sec) == 0:
            for row in range(n_rows):
                axes[row, col].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row, col].transAxes, fontsize=12,
                )
            continue

        t_raw    = np.array(res.t_sec)
        t_scaled, t_label = _scaled_time(t_raw)

        # ── Row 0: T1 ─────────────────────────────────────────────────────────
        _errorbar_panel(
            axes[0, col],
            t_scaled,
            np.array(res.T1_us),
            np.array(res.T1_error_us),
            ylabel="T1 (µs)",
            mean_label_fmt="mean = {:.1f} µs",
            color="C0",
            marker="o",
        )
        axes[0, col].set_xlabel(t_label)

        # ── Row 1: T2* ────────────────────────────────────────────────────────
        _errorbar_panel(
            axes[1, col],
            t_scaled,
            np.array(res.T2star_us),
            np.array(res.T2star_error_us),
            ylabel="T2* (µs)",
            mean_label_fmt="mean = {:.1f} µs",
            color="C2",
            marker="s",
        )
        axes[1, col].set_xlabel(t_label)

        # ── Row 2: T2 echo ────────────────────────────────────────────────────
        _errorbar_panel(
            axes[2, col],
            t_scaled,
            np.array(res.T2echo_us),
            np.array(res.T2echo_error_us),
            ylabel="T2 echo (µs)",
            mean_label_fmt="mean = {:.1f} µs",
            color="C4",
            marker="^",
        )
        axes[2, col].set_xlabel(t_label)

        # ── Row 3: Qubit frequency offset ─────────────────────────────────────
        freq_hz  = np.array(res.freq_offset_hz)
        freq_khz = freq_hz / 1e3
        valid_f  = np.isfinite(freq_khz)
        if valid_f.any():
            axes[3, col].plot(
                t_scaled[valid_f], freq_khz[valid_f],
                "D", ms=3, color="C5", alpha=0.8,
                label=f"n={valid_f.sum()}",
            )
            mean_f = float(np.nanmean(freq_khz[valid_f]))
            axes[3, col].axhline(
                mean_f, color="C3", ls="--", lw=1.2,
                label=f"mean = {mean_f:+.2f} kHz",
            )
            axes[3, col].axhline(0, color="gray", ls=":", lw=0.8)
            axes[3, col].legend(fontsize=8)
        else:
            axes[3, col].text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=axes[3, col].transAxes, fontsize=12,
            )
        axes[3, col].set_ylabel("Δf (kHz)")
        axes[3, col].set_xlabel(t_label)

        # ── Row 4: Thermal population ─────────────────────────────────────────
        P_th_pct     = np.array(res.P_th) * 100.0
        P_th_err_pct = np.array(res.P_th_err) * 100.0
        valid_Pth    = np.isfinite(P_th_pct)

        if valid_Pth.any():
            has_err = valid_Pth & np.isfinite(P_th_err_pct)
            no_err  = valid_Pth & ~np.isfinite(P_th_err_pct)

            if has_err.any():
                axes[4, col].errorbar(
                    t_scaled[has_err], P_th_pct[has_err],
                    yerr=P_th_err_pct[has_err],
                    fmt="v", ms=4, lw=0.8, capsize=2, color="C1", alpha=0.8,
                    label=f"P_th ± 1σ (n={has_err.sum()})",
                )
            if no_err.any():
                axes[4, col].plot(
                    t_scaled[no_err], P_th_pct[no_err],
                    "v", ms=4, color="C1", alpha=0.5, mfc="none",
                    label=f"P_th (err N/A, n={no_err.sum()})",
                )
            mean_Pth = float(np.nanmean(P_th_pct[valid_Pth]))
            axes[4, col].axhline(
                mean_Pth, color="C3", ls="--", lw=1.2,
                label=f"mean = {mean_Pth:.3f}%",
            )
            axes[4, col].legend(fontsize=8)
        else:
            axes[4, col].text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=axes[4, col].transAxes, fontsize=12,
            )
        axes[4, col].set_ylabel("Thermal population (%)")
        axes[4, col].set_xlabel(t_label)

    fig.suptitle("T1, T2*, T2 echo, Qubit Frequency & Thermal Population Monitor", fontsize=13)
    fig.tight_layout()
    return fig
