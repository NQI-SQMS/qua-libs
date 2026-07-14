"""Plotting routines for the Fock |1> qubit Ramsey chi calibration (node 25)."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .analysis import FockChiFit, _damped_cosine


def plot_ramsey_stark(
    dataset,
    qubits,
    fit_results: Dict[str, FockChiFit],
    mode_name: str = "alice",
) -> plt.Figure:
    """Plot the Fock |1> qubit Ramsey experiment used to extract chi.

    One panel per qubit: Ramsey oscillation vs idle time with decaying cosine
    fit overlay, annotated with chi and T2*.

    Parameters
    ----------
    dataset      : xarray.Dataset  Raw/processed dataset.
    qubits       : Qubits          Active qubit collection.
    fit_results  : dict            Output of ``fit_raw_data``.
    mode_name    : str             Cavity mode label for figure titles.
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        1, n_qubits,
        figsize=(6 * max(n_qubits, 1), 4.5),
        squeeze=False,
    )

    tau_ns = np.asarray(dataset.coords["idle_time"].values, dtype=float)
    tau_us = tau_ns / 1e3

    use_state = "state" in dataset or any(f"state{i+1}" in dataset for i in range(n_qubits))

    for i, qubit in enumerate(qubits):
        ax = axes[0, i]
        res = fit_results.get(qubit.name)

        # Raw signal
        key_candidates = ["state", f"state{i+1}"] if use_state else ["I", f"I{i+1}"]
        raw = None
        for key in key_candidates:
            if key in dataset:
                try:
                    da = dataset[key]
                    if "qubit" in da.dims:
                        raw = da.sel(qubit=qubit.name).values
                    else:
                        raw = da.values
                    break
                except Exception:
                    pass
        if raw is None:
            raw = np.full(len(tau_ns), np.nan)

        ax.plot(tau_us, raw, color="steelblue", alpha=0.8, lw=1.0, label="Data")

        if res is not None and res.success:
            # Fit overlay
            tau_fit = np.linspace(tau_ns[0], tau_ns[-1], 500)
            A0 = (np.nanmax(raw) - np.nanmin(raw)) / 2.0
            c0 = np.nanmean(raw)
            fit_curve = _damped_cosine(
                tau_fit,
                A0,
                res.ramsey_T2_ns,
                res.ramsey_freq_hz,
                0.0,
                c0,
            )
            ax.plot(tau_fit / 1e3, fit_curve, color="tomato", lw=2.0, label="Fit")

            chi_mhz = res.chi_hz / 1e6
            t2_us = res.ramsey_T2_ns / 1e3
            ann = (
                f"χ = {chi_mhz:.4f} MHz\n"
                f"T2* = {t2_us:.2f} µs\n"
                f"f_osc = {res.ramsey_freq_hz / 1e6:.4f} MHz"
            )
            ax.text(
                0.97, 0.97, ann,
                transform=ax.transAxes, va="top", ha="right", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            )
            title_suffix = f"χ = {chi_mhz:.4f} MHz"
        else:
            title_suffix = "fit failed"

        ax.set_xlabel("Idle time (µs)")
        ax.set_ylabel("P(|e⟩)" if use_state else "I (a.u.)")
        ax.set_title(f"{qubit.name} · {mode_name} Fock|1⟩ Ramsey\n{title_suffix}")
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig
