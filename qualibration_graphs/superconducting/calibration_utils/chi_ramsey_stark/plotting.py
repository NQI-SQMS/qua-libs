"""Plotting routines for the Ramsey Stark-shift chi calibration (node 25)."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .analysis import RamseyStarkFit


def plot_ramsey_stark(
    dataset,
    qubits,
    fit_results: Dict[str, RamseyStarkFit],
    mode_name: str = "alice",
) -> plt.Figure:
    """
    Plot the Ramsey Stark-shift experiment.

    Left panel  — Ramsey oscillations for every cavity drive amplitude (colour-
                  coded by amplitude).
    Right panel — Frequency shift Δf vs A² with linear fit; χ annotation.

    Parameters
    ----------
    dataset      : xarray.Dataset  Raw/processed dataset.
    qubits       : Qubits          Active qubit collection.
    fit_results  : dict            Output of ``fit_raw_data``.
    mode_name    : str             Cavity mode label for figure titles.
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        n_qubits, 2,
        figsize=(13, 4.5 * max(n_qubits, 1)),
        squeeze=False,
    )

    amplitudes = np.asarray(dataset.coords["drive_amplitude"].values, dtype=float)
    tau_ns = np.asarray(dataset.coords["delay"].values, dtype=float)
    tau_us = tau_ns / 1e3

    cmap = plt.cm.viridis
    amp_colors = [cmap(j / max(len(amplitudes) - 1, 1)) for j in range(len(amplitudes))]

    # XarrayDataFetcher groups state1/state2/... → "state" with qubit dim
    use_state = "state" in dataset or any(f"state{i+1}" in dataset for i in range(len(amplitudes)))

    for i, qubit in enumerate(qubits):
        ax_traces = axes[i, 0]
        ax_chi    = axes[i, 1]
        res = fit_results.get(qubit.name)

        # ── Left: Ramsey traces ───────────────────────────────────────────────
        # Try grouped key first ("state"/"I"), then fall back to indexed ("state1"/"I1")
        key_candidates = (["state", f"state{i+1}"] if use_state else ["I", f"I{i+1}"])
        raw = None
        for key in key_candidates:
            if key in dataset:
                try:
                    da = dataset[key]
                    if "qubit" in da.dims:
                        raw = da.sel(qubit=qubit.name).values  # (n_amp, n_tau)
                    else:
                        raw = da.values
                    break
                except Exception:
                    pass
        if raw is None:
            raw = np.full((len(amplitudes), len(tau_ns)), np.nan)

        for j, (amp, color) in enumerate(zip(amplitudes, amp_colors)):
            ax_traces.plot(
                tau_us, raw[j, :],
                color=color, alpha=0.75, linewidth=0.9,
                label=f"A={amp:.2f}",
            )

        ax_traces.set_xlabel("Ramsey delay (µs)")
        ax_traces.set_ylabel("P(|e⟩)" if use_state else "I (a.u.)")
        ax_traces.set_title(
            f"{qubit.name} · {mode_name}: Ramsey traces\n"
            f"(colour = cavity drive amplitude)"
        )
        # Compact legend with two columns
        ax_traces.legend(fontsize=7, ncol=2, loc="upper right")

        # ── Right: Δf vs A² with fit ──────────────────────────────────────────
        if res is None or not res.success:
            ax_chi.set_title(f"{qubit.name}: fit failed")
            continue

        amp_sq = np.array(res.amplitudes) ** 2
        df_khz = np.array(res.delta_freq_hz) / 1e3  # → kHz
        valid  = np.isfinite(df_khz)

        ax_chi.scatter(
            amp_sq[valid], df_khz[valid],
            color="steelblue", zorder=5, s=60, label="Data",
        )

        # Fit line
        x_fit = np.linspace(0.0, amp_sq.max() * 1.05, 200)
        y_fit = res.chi_slope_hz_per_amp2 * x_fit / 1e3   # kHz
        ax_chi.plot(x_fit, y_fit, color="tomato", lw=2.0, label="Linear fit")

        # Annotation
        slope_mhz = res.chi_slope_hz_per_amp2 / 1e6
        ann_lines = [f"Slope: {slope_mhz:.3f} MHz/A²"]
        if res.chi_hz is not None:
            ann_lines.append(f"χ = {res.chi_hz / 1e6:.3f} MHz")
        else:
            ann_lines.append("χ: displacement_k not calibrated")
        ax_chi.text(
            0.05, 0.95, "\n".join(ann_lines),
            transform=ax_chi.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )

        ax_chi.set_xlabel("Drive amplitude² (A²)")
        ax_chi.set_ylabel("Ramsey frequency shift Δf (kHz)")
        ax_chi.set_title(f"{qubit.name} · {mode_name}: Δf vs A²  →  χ")
        ax_chi.legend()
        ax_chi.axhline(0, color="grey", lw=0.7, linestyle="--")

    fig.tight_layout()
    return fig
