"""Plotting utilities for the cavity coherent T1 node (33)."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

import xarray as xr

from .analysis import CoherentT1Fit, coherent_T1_model


def _normalize(signal: np.ndarray, fit_curve: np.ndarray):
    """Min-max normalize signal to [0, 1]; apply same scaling to fit_curve."""
    y_min = signal.min()
    y_max = signal.max()
    span = y_max - y_min
    if span == 0:
        return signal, fit_curve
    return (signal - y_min) / span, (fit_curve - y_min) / span


def plot_coherent_T1(
    ds: xr.Dataset,
    fit_results: Dict,
    mode_name: str = "alice",
    normalize_plot: bool = False,
) -> plt.Figure:
    """Plot the coherent T1 Gumbel decay and the inferred |α(t)|² for each qubit.

    The figure has two rows per qubit:
      - Top:    Measured signal P_e(t) with Gumbel fit overlay.
      - Bottom: Inferred photon number |α(t)|² = nbar(t) with exponential
                decay fit nbar0 * exp(-t / T1).

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with dim (qubit, idle_time).  Must contain a
        'nbar_data' variable added by fit_raw_data().
    fit_results : dict
        Per-qubit fit results — either CoherentT1Fit dataclasses or plain dicts
        (after asdict()).
    mode_name : str
        Cavity mode name (shown in title).
    normalize_plot : bool
        When True and signal is raw I, normalize P_e panel to [0, 1].
    """
    qubit_names = list(ds.qubit.values)
    n_qubits = len(qubit_names)
    signal_name = "state" if "state" in ds else "I"
    has_nbar = "nbar_data" in ds

    fig, axes = plt.subplots(
        2, n_qubits,
        figsize=(7 * n_qubits, 9),
        squeeze=False,
    )
    fig.suptitle(
        f"Cavity Coherent T1 — mode '{mode_name}' (node 33)",
        fontsize=13,
    )

    should_normalize = normalize_plot and signal_name == "I"

    for col, q_name in enumerate(qubit_names):
        ax_pe = axes[0][col]
        ax_nb = axes[1][col]

        ds_q = ds.sel(qubit=q_name)
        t_ns = ds_q.idle_time.values.astype(float)
        signal = ds_q[signal_name].values.astype(float)

        res = fit_results.get(str(q_name))
        # Normalise to plain dict (may arrive as dataclass or dict)
        if isinstance(res, CoherentT1Fit):
            res = {
                "T1_ns": res.T1_ns,
                "T1_error_ns": res.T1_error_ns,
                "nbar0": res.nbar0,
                "amplitude": res.amplitude,
                "offset": res.offset,
                "success": res.success,
            }

        fit_curve = None
        t_fine = np.linspace(t_ns[0], t_ns[-1], 500)
        if res is not None and res.get("success"):
            fit_curve = coherent_T1_model(
                t_fine,
                res["amplitude"],
                res["nbar0"],
                res["T1_ns"],
                res["offset"],
            )

        # --- Top panel: P_e(t) ---
        plot_signal = signal.copy()
        plot_fit = fit_curve.copy() if fit_curve is not None else None
        if should_normalize and plot_fit is not None:
            plot_signal, plot_fit = _normalize(plot_signal, plot_fit)
        elif should_normalize:
            plot_signal, _ = _normalize(plot_signal, plot_signal)

        ax_pe.plot(t_ns * 1e-3, plot_signal, "o", ms=4, color="steelblue", label="data")

        if res is not None and res.get("success"):
            T1_us = res["T1_ns"] * 1e-3
            T1_err_us = res["T1_error_ns"] * 1e-3
            ax_pe.plot(
                t_fine * 1e-3,
                plot_fit,
                "-",
                lw=2,
                color="tomato",
                label=f"Gumbel fit\nT₁ = {T1_us:.1f} ± {T1_err_us:.1f} µs\n|α₀|² = {res['nbar0']:.2f}",
            )
            ax_pe.axvline(T1_us, color="gray", linestyle="--", lw=1, alpha=0.6)
        elif res is not None:
            ax_pe.text(0.5, 0.5, "Fit failed", transform=ax_pe.transAxes,
                       ha="center", va="center", fontsize=12, color="red")

        ylabel_pe = "State population" if signal_name == "state" else ("I (normalized)" if should_normalize else "I (V)")
        ax_pe.set_xlabel("Wait time (µs)", fontsize=11)
        ax_pe.set_ylabel(ylabel_pe, fontsize=11)
        ax_pe.set_title(q_name, fontsize=11)
        ax_pe.legend(fontsize=9)
        ax_pe.grid(True, alpha=0.3)

        # --- Bottom panel: |α(t)|² = nbar(t) ---
        if has_nbar and res is not None and res.get("success"):
            nbar_data = ds_q["nbar_data"].values.astype(float)
            nbar_fit_curve = res["nbar0"] * np.exp(-t_fine / res["T1_ns"])
            T1_us = res["T1_ns"] * 1e-3

            ax_nb.plot(t_ns * 1e-3, nbar_data, "o", ms=4, color="steelblue", label="|α(t)|² data")
            ax_nb.plot(
                t_fine * 1e-3,
                nbar_fit_curve,
                "-",
                lw=2,
                color="tomato",
                label=f"|α₀|²·exp(−t/T₁)\nT₁ = {T1_us:.1f} µs\n|α₀|² = {res['nbar0']:.2f}",
            )
            ax_nb.axvline(T1_us, color="gray", linestyle="--", lw=1, alpha=0.6,
                          label=f"T₁ = {T1_us:.1f} µs")
            ax_nb.set_ylim(bottom=0)
        elif res is not None and not res.get("success"):
            ax_nb.text(0.5, 0.5, "Fit failed", transform=ax_nb.transAxes,
                       ha="center", va="center", fontsize=12, color="red")
        else:
            ax_nb.text(0.5, 0.5, "No nbar data", transform=ax_nb.transAxes,
                       ha="center", va="center", fontsize=11, color="gray")

        ax_nb.set_xlabel("Wait time (µs)", fontsize=11)
        ax_nb.set_ylabel("|α(t)|²  (photons)", fontsize=11)
        ax_nb.set_title(f"{q_name} — photon number decay", fontsize=11)
        ax_nb.legend(fontsize=9)
        ax_nb.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
