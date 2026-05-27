"""Plotting utilities for the displacement Wigner calibration node (32)."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

import xarray as xr

from .analysis import FitParameters, wigner_gaussian

_MAX_VOLTAGE = 0.5
_AMP_SCALE_LIMIT = 1.9


def plot_wigner_1d(
    ds: xr.Dataset,
    qubits,
    fit_results: Dict[str, FitParameters],
    parity_time_ns: int,
    base_amplitude: Optional[float] = None,
    current_alpha_max: float = 1.0,
) -> plt.Figure:
    """Plot the 1D Wigner slice P_e(a) with Gaussian fit for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with dims (qubit, amp).  The 'amp' coordinate is in
        amplitude_scale units.
    qubits : qubit list from node.namespace["qubits"]
    fit_results : dict
        Per-qubit FitParameters from fit_raw_data.
    parity_time_ns : int
        Parity time used in the experiment [ns], shown in the title.
    base_amplitude : float, optional
        Displacement pulse base amplitude [V] (before this calibration update).
        Used to compute and annotate the resulting alpha_max.
    current_alpha_max : float
        The alpha_max from the QuAM state used to build the amplitude sweep.
        Converts the 'amp' coordinate (amplitude_scale) to photon units for
        the primary x-axis: alpha = amplitude_scale * current_alpha_max.
    """
    qubit_names = list(ds.qubit.values)
    n_qubits = len(qubit_names)
    signal_name = "state" if "state" in ds else "I"

    fig, axes = plt.subplots(1, n_qubits, figsize=(7 * n_qubits, 5), squeeze=False)
    fig.suptitle(
        f"1D Wigner Slice — Displacement Wigner Calibration (32)\n"
        f"parity time = {parity_time_ns} ns",
        fontsize=13,
    )

    for ax, q_name in zip(axes[0], qubit_names):
        ds_q = ds.sel(qubit=q_name)
        a_arr = ds_q.amp.values.astype(float)       # amplitude_scale
        alpha_arr = a_arr * current_alpha_max        # photon amplitude
        signal = ds_q[signal_name].values.astype(float)

        ax.plot(alpha_arr, signal, "o", ms=4, label="data")

        res = fit_results.get(str(q_name))
        if isinstance(res, FitParameters):
            res = {"sigma": res.sigma, "amplitude": res.amplitude,
                   "offset": res.offset, "success": res.success}

        alpha_max_new = None
        if res is not None and res.get("success"):
            sigma = res["sigma"]
            sigma_alpha = sigma * current_alpha_max   # sigma in photon units

            a_fine = np.linspace(a_arr[0], a_arr[-1], 500)
            alpha_fine = a_fine * current_alpha_max
            fit_curve = wigner_gaussian(a_fine, res["amplitude"], sigma, res["offset"])

            ax.plot(alpha_fine, fit_curve, "-", lw=2,
                    label=f"fit: A₁ph = {sigma_alpha:.3f} α")
            ax.axvline(sigma_alpha, color="green", linestyle="--", lw=1, alpha=0.7,
                       label=f"|α|=1  (scale={sigma:.4f})")
            ax.axvline(-sigma_alpha, color="green", linestyle="--", lw=1, alpha=0.7)

            if base_amplitude is not None and base_amplitude > 0:
                alpha_max_new = _MAX_VOLTAGE / (base_amplitude * sigma * _AMP_SCALE_LIMIT)

        title = q_name
        if res is not None and res.get("success"):
            sigma_alpha = res["sigma"] * current_alpha_max
            title += f"\nA₁ph = {res['sigma']:.4f} (scale) = {sigma_alpha:.3f} α"
            if alpha_max_new is not None:
                title += f"  →  α_max = {alpha_max_new:.2f} photons"
        ax.set_title(title, fontsize=10)

        ax.set_xlabel("Displacement amplitude  α  (photons)", fontsize=11)
        ax.set_ylabel("State population", fontsize=11)
        ax.legend(fontsize=9)
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

    fig.tight_layout()
    return fig
