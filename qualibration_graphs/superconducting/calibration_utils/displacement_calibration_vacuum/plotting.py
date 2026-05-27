"""Plotting for the displacement vacuum-population calibration node (22)."""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Dict, Optional

from .analysis import vacuum_population

_MAX_VOLTAGE = 0.5
_AMP_SCALE_LIMIT = 1.9


def _normalize(y: np.ndarray, y_fit: np.ndarray):
    """Min-max normalize y to [0, 1]; apply same scaling to y_fit."""
    y_min, y_max = y.min(), y.max()
    span = y_max - y_min
    if span == 0:
        return y, y_fit
    return (y - y_min) / span, (y_fit - y_min) / span


def plot_vacuum_calibration(
    ds: xr.Dataset,
    fit_results: Dict,
    mode_name: str = "alice",
    qubit_pulse: str = "selective_x180",
    normalize_plot: bool = False,
    base_amplitude: Optional[float] = None,
    current_alpha_max: float = 1.0,
) -> plt.Figure:
    """Plot P_e(a) data and Gaussian fit for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw (processed) dataset with 'state' or 'I' variable, coords: qubit, amp.
        The 'amp' coordinate is in amplitude_scale units.
    fit_results : dict
        {qubit_name: dict} with keys sigma, amplitude, offset, success.
    mode_name : str
        Cavity mode label for the title.
    qubit_pulse : str
        Qubit pulse name used (for title annotation).
    normalize_plot : bool
        When True and signal is I, normalize to [0, 1].
    base_amplitude : float, optional
        Displacement pulse base amplitude [V] (before this calibration update).
        Used to compute and display the resulting alpha_max.
    current_alpha_max : float
        The alpha_max from the QuAM state used to build the amplitude sweep.
        Converts the 'amp' coordinate (amplitude_scale) to photon units for
        the primary x-axis: alpha = amplitude_scale * current_alpha_max.
    """
    qubits = ds.qubit.values
    n_qubits = len(qubits)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)

    signal_name = "state" if "state" in ds else "I"
    should_normalize = normalize_plot and signal_name == "I"

    for ax, q in zip(axes[0], qubits):
        ds_q = ds.sel(qubit=q)
        a_arr = ds_q.amp.values.astype(float)       # amplitude_scale
        alpha_arr = a_arr * current_alpha_max        # photon amplitude
        signal = ds_q[signal_name].values.astype(float)
        res = fit_results.get(str(q), {})

        # Build fit curve before normalization
        fit_curve = None
        a_fine = None
        if res.get("success"):
            a_fine = np.linspace(a_arr.min(), a_arr.max(), 400)
            alpha_fine = a_fine * current_alpha_max
            fit_curve = vacuum_population(a_fine, res["amplitude"], res["sigma"], res["offset"])

        if should_normalize and fit_curve is not None:
            signal, fit_curve = _normalize(signal, fit_curve)
            ylabel = "I (normalized)"
        elif should_normalize:
            y_min, y_max = signal.min(), signal.max()
            span = y_max - y_min
            if span != 0:
                signal = (signal - y_min) / span
            ylabel = "I (normalized)"
        else:
            ylabel = "P(e)" if signal_name == "state" else "Signal (V)"

        # Compute alpha_max result for annotation
        alpha_max_new = None
        if res.get("success") and base_amplitude is not None and base_amplitude > 0:
            alpha_max_new = _MAX_VOLTAGE / (base_amplitude * res["sigma"] * _AMP_SCALE_LIMIT)

        # --- Primary x-axis: α (photon amplitude) ---
        ax.plot(alpha_arr, signal, "o", ms=4, label="data")

        if res.get("success"):
            sigma_alpha = res["sigma"] * current_alpha_max   # sigma in photon units
            alpha_fine = a_fine * current_alpha_max
            ax.plot(alpha_fine, fit_curve, "-", color="red",
                    label=f"fit  A₁ph = {sigma_alpha:.3f} α")
            ax.axvline(sigma_alpha, color="red", linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="red")

        ax.set_xlabel("Displacement amplitude  α  (photons)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        title = f"{q} — mode: {mode_name} | pulse: {qubit_pulse}"
        if res.get("success"):
            sigma_alpha = res["sigma"] * current_alpha_max
            title += f"\nA₁ph = {res['sigma']:.4f} (scale) = {sigma_alpha:.3f} α"
            if alpha_max_new is not None:
                title += f"  →  α_max = {alpha_max_new:.2f} photons"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Secondary x-axis: amplitude_scale (what QUA sees)
        ax2 = ax.twiny()
        ax2.set_xlim(alpha_arr.min(), alpha_arr.max())
        if current_alpha_max != 0:
            scale_ticks = np.linspace(a_arr.min(), a_arr.max(), 5)
            alpha_ticks = scale_ticks * current_alpha_max
            ax2.set_xticks(alpha_ticks)
            ax2.set_xticklabels([f"{s:.3f}" for s in scale_ticks], fontsize=8)
        ax2.set_xlabel("Amplitude scale (QUA)", fontsize=9, color="steelblue")
        ax2.tick_params(axis="x", labelcolor="steelblue", labelsize=8)

    fig.suptitle(
        f"Displacement vacuum calibration — {mode_name}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
