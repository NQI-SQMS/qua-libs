"""Plotting for the displacement vacuum-population calibration node (35)."""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Dict, Optional

from .analysis import vacuum_population


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
) -> plt.Figure:
    """Plot P_e(a) data and Gaussian fit for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw (processed) dataset with 'state' or 'I' variable, coords: qubit, amp.
    fit_results : dict
        {qubit_name: dict} with keys sigma, amplitude, offset, success.
    mode_name : str
        Cavity mode label for the title.
    qubit_pulse : str
        Qubit pulse name used (for title annotation).
    normalize_plot : bool
        When True and signal is I, normalize to [0, 1].
    base_amplitude : float, optional
        Displacement pulse base amplitude [V].  When provided, a second x-axis
        is drawn showing the total pulse voltage = amplitude_scale × base_amplitude.
    """
    qubits = ds.qubit.values
    n_qubits = len(qubits)
    fig, axes = plt.subplots(1, n_qubits, figsize=(5 * n_qubits, 4.5), squeeze=False)

    signal_name = "state" if "state" in ds else "I"
    should_normalize = normalize_plot and signal_name == "I"

    for ax, q in zip(axes[0], qubits):
        ds_q = ds.sel(qubit=q)
        a_arr = ds_q.amp.values.astype(float)
        signal = ds_q[signal_name].values.astype(float)
        res = fit_results.get(str(q), {})

        # Build fit curve before normalization
        fit_curve = None
        a_fine = None
        if res.get("success"):
            a_fine = np.linspace(a_arr.min(), a_arr.max(), 400)
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

        ax.plot(a_arr, signal, "o", ms=4, label="data")

        if res.get("success"):
            ax.plot(a_fine, fit_curve, "-", color="red", label=f"fit  A₁ph = {res['sigma']:.4f}")
            ax.axvline(res["sigma"], color="red", linestyle="--", alpha=0.5)
        else:
            ax.set_title(f"{q} — fit failed", color="red")

        ax.set_xlabel("Displacement amplitude scale")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{q} — {mode_name} | pulse: {qubit_pulse}"
            + (f"\nA₁ph = {res['sigma']:.4f}" if res.get("success") else "")
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Second x-axis: total pulse voltage in V
        if base_amplitude is not None and base_amplitude > 0:
            ax2 = ax.twiny()
            ax2.set_xlim(a_arr.min() * base_amplitude, a_arr.max() * base_amplitude)
            ax2.set_xlabel("Pulse voltage amplitude (V)", fontsize=9, color="steelblue")
            ax2.tick_params(axis="x", labelcolor="steelblue", labelsize=8)

    fig.suptitle(
        f"Displacement vacuum calibration — {mode_name}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
