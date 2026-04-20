"""Plotting routines for the parity-time calibration (node 30)."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .analysis import ParityTimeFit


def plot_parity_time(
    dataset,
    qubits,
    fit_results: Dict[str, ParityTimeFit],
    mode_name: str = "alice",
) -> plt.Figure:
    """Plot P_e(τ) Ramsey traces with FFT spectrum.

    Left panel  — raw signal trace with τ_parity marked.
    Right panel — FFT magnitude spectrum with χ_eff peak marked.

    Parameters
    ----------
    dataset      : xr.Dataset         Dataset with delay coordinate.
    qubits       : Qubits             Active qubit collection.
    fit_results  : dict               Output of ``fit_raw_data``.
    mode_name    : str                Cavity mode label for titles.
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        n_qubits, 2,
        figsize=(13, 4.5 * max(n_qubits, 1)),
        squeeze=False,
    )

    tau_ns = np.asarray(dataset.coords["delay"].values, dtype=float)
    tau_us = tau_ns / 1e3

    for i, qubit in enumerate(qubits):
        ax_trace = axes[i, 0]
        ax_fft   = axes[i, 1]
        res = fit_results.get(qubit.name)

        # ── Extract measured signal ───────────────────────────────────────────
        raw = None
        signal_key = None
        for key in [f"state{i + 1}", "state", f"I{i + 1}", "I"]:
            if key in dataset:
                try:
                    da = dataset[key]
                    raw = (
                        da.sel(qubit=qubit.name).values
                        if "qubit" in da.dims
                        else da.values
                    )
                    signal_key = key
                    break
                except Exception:
                    pass
        if raw is None:
            raw = np.full(len(tau_ns), np.nan)
        raw = np.asarray(raw, dtype=float).ravel()[: len(tau_ns)]
        using_iq = signal_key is not None and signal_key.startswith("I")

        # ── Left: trace ───────────────────────────────────────────────────────
        ax_trace.plot(tau_us, raw, "o", ms=3, color="steelblue",
                      alpha=0.7, label="Data")

        if res is not None and res.success:
            tau_parity_us = res.parity_time_s * 1e6
            ax_trace.axvline(tau_parity_us, color="forestgreen", lw=1.5, ls="--",
                             label=f"τ_parity = {tau_parity_us * 1e3:.0f} ns")
            for k in range(2, 10):
                t_k = k * tau_parity_us
                if t_k <= tau_us[-1]:
                    ax_trace.axvline(t_k, color="forestgreen", lw=0.6,
                                     ls=":", alpha=0.5)

            ann = (
                f"χ_eff/(2π) = {res.chi_eff_hz / 1e3:.1f} kHz\n"
                f"τ_parity   = {res.parity_time_s * 1e9:.0f} ns"
            )
            ax_trace.text(
                0.97, 0.97, ann,
                transform=ax_trace.transAxes, va="top", ha="right",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
            )
        elif res is not None:
            ax_trace.text(
                0.5, 0.5, f"FFT peak not found\n{res.message}",
                transform=ax_trace.transAxes, ha="center", va="center",
                fontsize=10, color="firebrick",
            )

        ax_trace.set_xlabel("Ramsey wait time (µs)")
        ax_trace.set_ylabel("I (V)" if using_iq else "P(|e⟩)")
        ax_trace.set_title(
            f"{qubit.name} · {mode_name}: Parity-time Ramsey\n"
            f"(selective_y90 → wait(τ) → selective_y90, cavity displaced)"
        )
        ax_trace.legend(fontsize=9, loc="lower right")
        if not using_iq:
            ax_trace.set_ylim(-0.05, 1.05)

        # ── Right: FFT spectrum (computed from raw signal) ────────────────────
        if np.any(np.isfinite(raw)) and len(tau_ns) > 1:
            dt_ns = float(tau_ns[1] - tau_ns[0])
            sig_c = raw - np.nanmean(raw)
            sig_c = np.where(np.isfinite(sig_c), sig_c, 0.0)
            fft_mag_plot = np.abs(np.fft.rfft(sig_c))
            freqs_plot = np.fft.rfftfreq(len(sig_c), d=dt_ns * 1e-9)
            freqs_khz = freqs_plot[1:] / 1e3
            ax_fft.plot(freqs_khz, fft_mag_plot[1:], "-", lw=1.2, color="slategrey")
            if res is not None and res.success:
                chi_khz = res.chi_eff_hz / 1e3
                ax_fft.axvline(chi_khz, color="tomato", lw=1.5, ls="--",
                               label=f"χ_eff/(2π) = {chi_khz:.1f} kHz")
                ax_fft.legend(fontsize=9)
                ax_fft.set_xlim(0, min(3.0 * chi_khz, float(freqs_khz[-1])))
        else:
            ax_fft.text(0.5, 0.5, "No signal data", transform=ax_fft.transAxes,
                        ha="center", va="center", color="grey")

        ax_fft.set_xlabel("Frequency (kHz)")
        ax_fft.set_ylabel("FFT magnitude (a.u.)")
        ax_fft.set_title(f"{qubit.name} · {mode_name}: FFT spectrum")

    fig.tight_layout()
    return fig
