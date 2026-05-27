"""Plotting for photon number resolved spectroscopy (node 24).

Two-panel layout:
  Left  — raw PNRS spectrum (scatter) with shared-sigma Gaussian fit overlaid,
           fitted peak-position lines and expected-position guides.
  Right — normalised photon-number distribution P(n) as a bar chart with
           Poisson fit overlay.
"""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.special import factorial as _factorial


def _gaussian_sum_shared_sigma(x, sigma, offset, *amp_pos_pairs):
    result = np.full_like(x, float(offset), dtype=float)
    for i in range(len(amp_pos_pairs) // 2):
        a = amp_pos_pairs[2 * i]
        x0 = amp_pos_pairs[2 * i + 1]
        result += a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return result


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
    displacement_alpha: float = 1.0,
    normalize_plot: bool = False,
) -> Figure:
    n_qubits = len(list(qubits))
    fig, axes = plt.subplots(n_qubits, 2, figsize=(14, 5 * n_qubits), squeeze=False)

    for row, qubit in enumerate(qubits):
        q_name = qubit.name
        ax_raw = axes[row, 0]
        ax_pn  = axes[row, 1]

        ds_q = ds.sel(qubit=q_name)
        x_hz  = ds_q.detuning.values
        x_khz = x_hz * 1e-3

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        if signal_name == "state":
            y      = ds_q.state.values
            ylabel = "P(e)"
        else:
            y      = ds_q.I.values * 1e3
            ylabel = "I (mV)"

        res = (fit_results or {}).get(q_name)

        # ── Reconstruct fit curve ──────────────────────────────────────────
        x_fine = y_fit = None
        if res and res["success"]:
            chi_khz   = res["chi_hz"] * 1e-3
            sigma_hz  = res.get("sigma_hz", float("nan"))
            offset_v  = res.get("offset",   float("nan"))
            sigma_khz = float(sigma_hz) * 1e-3 if np.isfinite(sigma_hz) else chi_khz / 4.0
            offset_est = float(offset_v) if np.isfinite(offset_v) else float(np.min(y))

            popt_approx = [sigma_khz, offset_est]
            for amp, pos in zip(res["peak_amplitudes"],
                                [p * 1e-3 for p in res["peak_positions_hz"]]):
                popt_approx += [amp, pos]

            x_fine = np.linspace(x_khz[0], x_khz[-1], 500)
            y_fit  = _gaussian_sum_shared_sigma(x_fine, *popt_approx)

        # ── Left panel: PNRS spectrum ──────────────────────────────────────
        ax_raw.scatter(x_khz, y, s=25, color="C0", alpha=0.8, label="data", zorder=3)

        if res and res["success"]:
            chi_khz   = res["chi_hz"] * 1e-3
            sigma_hz  = res.get("sigma_hz", float("nan"))
            offset_v  = res.get("offset",   float("nan"))
            offset_est = float(offset_v) if np.isfinite(offset_v) else float(np.min(y))

            n_peaks = res["num_peaks_used"]
            ax_raw.plot(x_fine, y_fit, "--", color="C0", lw=2,
                        label=f"fit  ({n_peaks} peak{'s' if n_peaks > 1 else ''})")

            # Dotted lines at fitted peak positions
            peaks_khz = [p * 1e-3 for p in res["peak_positions_hz"]]
            for pk in peaks_khz:
                ax_raw.axvline(pk, color="C0", lw=1.0, ls=":", alpha=0.7)

            # Dashed lines at expected positions (n=0 at 0, n=k at -2k·χ)
            for n in range(n_peaks + 1):
                exp_pos = -n * 2.0 * chi_khz
                ax_raw.axvline(exp_pos, color="grey", lw=0.8, ls="--", alpha=0.5,
                               label=(f"n={n} expected" if n < 3 else None))

            ax_raw.set_title(f"{q_name} — chi = {chi_khz:.3f} kHz", fontsize=9)

        else:
            ax_raw.set_title(f"{q_name} — FAILED", fontsize=9, color="red")

        # Lock x-axis to the data range so axvlines outside it don't stretch the plot
        x_margin = 0.02 * (x_khz[-1] - x_khz[0])
        ax_raw.set_xlim(x_khz[0] - x_margin, x_khz[-1] + x_margin)
        ax_raw.set_xlabel("Qubit detuning (kHz)")
        ax_raw.set_ylabel(ylabel)
        ax_raw.legend(fontsize=7)
        ax_raw.grid(True, alpha=0.3)

        # ── Right panel: P(n) bar chart with Poisson overlay ──────────────
        if res and res["success"] and res["peak_probabilities"]:
            probs = res["peak_probabilities"]
            n_vals = list(range(len(probs)))
            bars = ax_pn.bar(n_vals, probs, color="C0", alpha=0.8,
                             edgecolor="navy", linewidth=0.8)
            for bar, prob in zip(bars, probs):
                ax_pn.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{prob:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )
            # Poisson fit overlay
            nbar = res.get("nbar", float("nan"))
            if np.isfinite(nbar):
                n_plot = np.arange(len(probs) + 2)
                p_poisson = np.exp(-nbar) * nbar ** n_plot / _factorial(n_plot)
                ax_pn.plot(n_plot, p_poisson, "o--", color="C1", lw=1.5, ms=5,
                           label=f"Poisson  n̄={nbar:.2f}", zorder=4)
                ax_pn.legend(fontsize=8)
            ax_pn.set_title(f"{q_name} — P(n)", fontsize=9)
        else:
            ax_pn.set_title(f"{q_name} — P(n) (no fit)", fontsize=9, color="red")

        ax_pn.set_xlabel("Photon number n")
        ax_pn.set_ylabel("P(n)")
        ax_pn.set_xticks(range(max(len((res or {}).get("peak_probabilities", [])), 1)))
        ax_pn.set_ylim(0, 1.1)
        ax_pn.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Photon Number Resolved Spectroscopy — {mode_name}  "
        f"[\u03b1={displacement_alpha:.2f}]",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
