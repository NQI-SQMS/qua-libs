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
    """Plot P_e(τ) Ramsey traces with damped-cosine fit overlay.

    Left panel  — raw P(e) trace with fit curve; parity time marked.
    Right panel — residuals (data − fit).

    Parameters
    ----------
    dataset      : xr.Dataset         Fit dataset (contains ``P_e_fit``).
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
        ax_resid = axes[i, 1]
        res = fit_results.get(qubit.name)

        # ── Extract measured signal ───────────────────────────────────────────
        raw = None
        for key in [f"state{i + 1}", "state", f"I{i + 1}", "I"]:
            if key in dataset:
                try:
                    da = dataset[key]
                    raw = (
                        da.sel(qubit=qubit.name).values
                        if "qubit" in da.dims
                        else da.values
                    )
                    break
                except Exception:
                    pass
        if raw is None:
            raw = np.full(len(tau_ns), np.nan)
        raw = np.asarray(raw, dtype=float).ravel()[: len(tau_ns)]

        # ── Extract fitted curve ──────────────────────────────────────────────
        fit_curve = np.full(len(tau_ns), np.nan)
        if "P_e_fit" in dataset:
            try:
                da_fit = dataset["P_e_fit"]
                fit_curve = (
                    da_fit.sel(qubit=qubit.name).values
                    if "qubit" in da_fit.dims
                    else da_fit.values
                ).ravel()[: len(tau_ns)]
            except Exception:
                pass

        # ── Left: trace + fit ─────────────────────────────────────────────────
        ax_trace.plot(tau_us, raw, "o", ms=3, color="steelblue",
                      alpha=0.7, label="Data")
        if np.any(np.isfinite(fit_curve)):
            ax_trace.plot(tau_us, fit_curve, "-", lw=2.0, color="tomato",
                          label="Damped-cosine fit")

        # Mark parity time
        if res is not None and res.success:
            tau_parity_us = res.parity_time_s * 1e6
            ax_trace.axvline(tau_parity_us, color="forestgreen", lw=1.5, ls="--",
                             label=f"τ_parity = {tau_parity_us * 1e3:.0f} ns")
            # Draw vertical lines at all integer multiples within the sweep range
            for k in range(2, 10):
                t_k = k * tau_parity_us
                if t_k <= tau_us[-1]:
                    ax_trace.axvline(t_k, color="forestgreen", lw=0.6,
                                     ls=":", alpha=0.5)

        # Annotation box
        if res is not None and res.success:
            ann = (
                f"χ_eff/(2π) = {res.chi_eff_hz / 1e3:.1f} kHz\n"
                f"τ_parity   = {res.parity_time_s * 1e9:.0f} ns\n"
                f"T2*        = {res.T2_star_ns / 1e3:.1f} µs"
            )
            ax_trace.text(
                0.97, 0.97, ann,
                transform=ax_trace.transAxes, va="top", ha="right",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
            )
        elif res is not None:
            ax_trace.text(
                0.5, 0.5, f"Fit failed\n{res.message}",
                transform=ax_trace.transAxes, ha="center", va="center",
                fontsize=10, color="firebrick",
            )

        ax_trace.set_xlabel("Ramsey wait time (µs)")
        ax_trace.set_ylabel("P(|e⟩)")
        ax_trace.set_title(
            f"{qubit.name} · {mode_name}: Parity-time Ramsey\n"
            f"(y90 → wait(τ) → y90, cavity displaced)"
        )
        ax_trace.legend(fontsize=9, loc="lower right")
        ax_trace.set_ylim(-0.05, 1.05)

        # ── Right: residuals ──────────────────────────────────────────────────
        resid = raw - fit_curve
        ax_resid.plot(tau_us, resid, "o-", ms=2, lw=0.8, color="slategrey")
        ax_resid.axhline(0.0, color="k", lw=0.7, ls="--")
        ax_resid.set_xlabel("Ramsey wait time (µs)")
        ax_resid.set_ylabel("Residual (data − fit)")
        ax_resid.set_title(
            f"{qubit.name} · {mode_name}: Fit residuals"
        )
        rms = float(np.sqrt(np.nanmean(resid ** 2)))
        ax_resid.text(
            0.97, 0.97, f"RMS = {rms:.4f}",
            transform=ax_resid.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    return fig
