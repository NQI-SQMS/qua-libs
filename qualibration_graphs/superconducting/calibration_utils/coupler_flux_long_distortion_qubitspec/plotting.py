"""Plotting utilities for coupler flux long distortion (qubitspec variant)."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .analysis import _load_coupler_spectroscopy_curve, _load_ramseyflux_curve_from_param


def plot_spectroscopy_curve(spec_run_id, qubit_pairs, qubits, node) -> Optional[plt.Figure]:
    """Plot qubit freq vs coupler flux from a qubit-spectroscopy-vs-coupler-flux run.

    Returns a Figure, or None if spec_run_id is None or loading fails for all pairs.
    """
    if spec_run_id is None or not qubit_pairs:
        return None

    n = len(qubit_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    any_plotted = False
    for ax, qp, qubit in zip(axes[0], qubit_pairs, qubits):
        curve = _load_coupler_spectroscopy_curve(spec_run_id, qubit, qp.coupler, node)
        if curve is not None:
            flux_bias, qubit_freq = curve
            ax.plot(flux_bias, np.array(qubit_freq) / 1e9, marker=".", linestyle="-")
            any_plotted = True
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
        ax.set_title(f"{qubit.name} / {qp.coupler.name}")
        ax.grid(True)
    fig.suptitle(f"Qubit spectroscopy vs coupler flux — run #{spec_run_id}")
    fig.tight_layout()
    return fig if any_plotted else None


def plot_ramsey_curve(ramsey_run_id, qubit_pairs, qubits, node) -> Optional[plt.Figure]:
    """Plot qubit freq vs coupler flux from a Ramsey-vs-coupler-flux run.

    If ``ramsey_run_id`` is None, each subplot loads from
    ``qubit.extras[f"{coupler.name}_dispersion_load_id"]`` when present.

    Returns a Figure, or None if no run ID is available for any pair or loading
    fails for all pairs.
    """
    if not qubit_pairs:
        return None

    n = len(qubit_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    any_plotted = False
    used_run_ids: set[int] = set()
    for ax, qp, qubit in zip(axes[0], qubit_pairs, qubits):
        rid = ramsey_run_id or (
            qubit.extras.get(f"{qp.coupler.name}_dispersion_load_id")
            if hasattr(qubit, "extras")
            else None
        )
        if rid is not None:
            used_run_ids.add(int(rid))
            curve = _load_ramseyflux_curve_from_param(rid, qubit, qp.coupler, node)
        else:
            curve = None
        if curve is not None:
            flux_bias, qubit_freq = curve
            ax.plot(flux_bias, np.array(qubit_freq) / 1e9, marker=".", linestyle="-")
            any_plotted = True
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
        ax.set_title(f"{qubit.name} / {qp.coupler.name}")
        ax.grid(True)
    if used_run_ids:
        runs_txt = ", ".join(str(r) for r in sorted(used_run_ids))
        fig.suptitle(f"Ramsey vs coupler flux ??run(s) #{runs_txt}")
    else:
        fig.suptitle("Ramsey vs coupler flux ??no run ID (param or extras)")
    fig.tight_layout()
    return fig if any_plotted else None

