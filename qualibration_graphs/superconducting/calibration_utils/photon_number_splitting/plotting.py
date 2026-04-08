"""Plotting for photon number splitting (node 29).

Two-panel layout (like Taeyoon's approach):
  Left  — raw spectrum with Gaussian fit overlaid and peak-position lines.
  Right — normalised photon-number distribution P(n) as a bar chart.
"""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualibration_libs.plotting import QubitGrid, grid_iter


def _gaussian_sum_shared_sigma(x, sigma, offset, *amp_pos_pairs):
    result = np.full_like(x, float(offset), dtype=float)
    for i in range(len(amp_pos_pairs) // 2):
        a = amp_pos_pairs[2 * i]
        x0 = amp_pos_pairs[2 * i + 1]
        result += a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return result


def _normalize_1d(y: np.ndarray, y_fit: np.ndarray):
    """Min-max normalize y to [0, 1]; apply same scaling to y_fit."""
    y_min, y_max = y.min(), y.max()
    span = y_max - y_min
    if span == 0:
        return y, y_fit
    return (y - y_min) / span, (y_fit - y_min) / span


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
    displacement_scale: float = 1.0,
    displacement_alpha: float = 1.0,
    normalize_plot: bool = False,
) -> Figure:
    n_qubits = len(list(qubits))
    fig, axes = plt.subplots(n_qubits, 2, figsize=(14, 5 * n_qubits), squeeze=False)

    for row, qubit in enumerate(qubits):
        q_name = qubit.name
        ax_raw = axes[row, 0]
        ax_pn = axes[row, 1]

        ds_q = ds.sel(qubit=q_name)
        x_hz = ds_q.detuning.values
        x_khz = x_hz * 1e-3

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        if signal_name == "state":
            y = ds_q.state.values
            ylabel = "State population"
        else:
            y = ds_q.I.values * 1e3
            ylabel = "I (mV)"

        should_normalize = normalize_plot and signal_name == "I"

        # --- Left panel: raw spectrum ---
        res = (fit_results or {}).get(q_name)

        # Build fit curve before normalization so we can normalize both together
        y_fit = None
        x_fine = None
        if res and res["success"]:
            chi_khz = res["chi_hz"] * 1e-3
            sigma_est = chi_khz / 4.0
            offset_est = float(np.min(y))
            popt_approx = [sigma_est, offset_est]
            for amp, pos in zip(res["peak_amplitudes"], [p * 1e-3 for p in res["peak_positions_hz"]]):
                popt_approx += [amp, pos]
            x_fine = np.linspace(x_khz[0], x_khz[-1], 500)
            y_fit = _gaussian_sum_shared_sigma(x_fine, *popt_approx)

        if should_normalize and y_fit is not None:
            y, y_fit = _normalize_1d(y, y_fit)
            ylabel = "I (normalized)"
        elif should_normalize:
            y_min, y_max = y.min(), y.max()
            span = y_max - y_min
            if span != 0:
                y = (y - y_min) / span
            ylabel = "I (normalized)"

        ax_raw.plot(x_khz, y, ".", ms=3, color="C0", label="data")

        if res and res["success"]:
            peaks_khz = [p * 1e-3 for p in res["peak_positions_hz"]]
            chi_khz = res["chi_hz"] * 1e-3
            ax_raw.plot(x_fine, y_fit, "-", color="red", lw=1.5, label="fit")

            for n_idx, pk in enumerate(peaks_khz):
                ax_raw.axvline(pk, color=f"C{n_idx + 2}", ls="--", lw=1.0, alpha=0.7)

            ax_raw.set_title(f"{q_name} — chi = {chi_khz:.3f} kHz", fontsize=9)
        else:
            ax_raw.set_title(f"{q_name} — FAILED", fontsize=9, color="red")

        ax_raw.set_xlabel("Qubit detuning (kHz)")
        ax_raw.set_ylabel(ylabel)
        ax_raw.legend(fontsize=7)
        ax_raw.grid(True, alpha=0.3)

        # --- Right panel: P(n) bar chart ---
        if res and res["success"] and res["peak_probabilities"]:
            n_vals = list(range(len(res["peak_probabilities"])))
            bars = ax_pn.bar(n_vals, res["peak_probabilities"], color="C0", alpha=0.8,
                             edgecolor="navy", linewidth=0.8)
            for bar, prob in zip(bars, res["peak_probabilities"]):
                ax_pn.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{prob:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )
            ax_pn.set_title(f"{q_name} — P(n)", fontsize=9)
        else:
            ax_pn.set_title(f"{q_name} — P(n) (no fit)", fontsize=9, color="red")

        ax_pn.set_xlabel("Photon number n")
        ax_pn.set_ylabel("P(n)")
        ax_pn.set_xticks(range(max(len((res or {}).get("peak_probabilities", [])), 1)))
        ax_pn.set_ylim(0, 1.1)
        ax_pn.grid(True, alpha=0.3, axis="y")

    actual_alpha = displacement_scale * displacement_alpha
    fig.suptitle(
        f"Photon Number Splitting — {mode_name}  "
        f"[scale={displacement_scale:.2f} × α={displacement_alpha:.1f} = {actual_alpha:.2f}]",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
