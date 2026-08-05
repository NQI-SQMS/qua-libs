"""Plotting functions for the SNZ t_phi_eff scan.

Generates line plots of f-state control population vs amplitude for
each t_phi_eff value, per qubit pair.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.snz_b_over_a_2.parameters import decompose_t_phi_eff, snz_factory


def plot_snz_waveforms(
    t_phi_eff_values,
    A,
    length,
    padding=4,
) -> plt.Figure:
    """Debug plot: 2-D colormap of SNZ waveform samples.

    Each row is one waveform corresponding to a ``t_phi_eff`` value.
    Waveforms of different lengths are zero-padded on the right so they
    align in a rectangular array.

    Parameters
    ----------
    t_phi_eff_values : array-like
        Array of effective idle times to visualise.
    A : float
        Flat-section amplitude (volts).
    length : int
        Total flat duration (CZ pulse length).
    padding : int
        Zero-padding per side passed to ``snz_factory``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    waveforms = []
    for tpe in t_phi_eff_values:
        t_phi, ratio = decompose_t_phi_eff(tpe)
        wf = snz_factory(A, ratio, length, t_phi, padding)
        waveforms.append(wf)

    max_len = max(len(wf) for wf in waveforms)
    matrix = np.zeros((len(waveforms), max_len))
    for i, wf in enumerate(waveforms):
        matrix[i, : len(wf)] = wf

    time_ns = np.arange(max_len + 1) - 0.5
    tpe_edges = np.zeros(len(t_phi_eff_values) + 1)
    tpe_arr = np.asarray(t_phi_eff_values)
    if len(tpe_arr) > 1:
        step = tpe_arr[1] - tpe_arr[0]
    else:
        step = 1.0
    tpe_edges[:-1] = tpe_arr - step / 2
    tpe_edges[-1] = tpe_arr[-1] + step / 2

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(time_ns, tpe_edges, matrix, shading="flat", cmap="RdBu_r")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("t_phi_eff (ns)")
    ax.set_title("SNZ waveform samples vs t_phi_eff")
    fig.colorbar(mesh, ax=ax, label="Amplitude (V)")
    fig.tight_layout()
    return fig


def plot_snz_raw(
    ds: xr.Dataset,
    qubit_pairs,
    opt_points=None,
) -> plt.Figure:
    """Plot f_state_control vs amplitude for each qubit pair.

    Each t_phi_eff value appears as a separate line/trace so the full
    2-D landscape is visible in a single panel per qubit pair.

    Parameters
    ----------
    ds : xr.Dataset
        Raw (or processed) dataset with ``amplitude`` and ``t_phi_eff``
        dimensions and an ``f_state_control`` variable.
    qubit_pairs : list
        Qubit-pair objects.
    opt_points : dict, optional
        Mapping ``{qubit_pair_name: (amp_full_opt, t_phi_eff_opt)}`` of the
        selected leakage-minimum seed; drawn as a marker on each panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs,
        1,
        figsize=(10, 5 * n_pairs),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
        qp_ds = ds.sel(qubit_pair=qp.name)

        if "f_state_control" in qp_ds.data_vars:
            qp_ds.f_state_control.plot(ax=ax, x="amp_full", vmin=0.0, cmap="magma")
        elif "I_control" in qp_ds.data_vars:
            qp_ds.I_control.plot(ax=ax, x="amp_full")

        if opt_points and qp.name in opt_points:
            amp_opt, tpe_opt = opt_points[qp.name]
            if amp_opt is not None and tpe_opt is not None:
                ax.plot(
                    amp_opt,
                    tpe_opt,
                    marker="*",
                    color="cyan",
                    markersize=18,
                    markeredgecolor="k",
                    zorder=5,
                    label="min-leakage seed",
                )
                ax.legend(loc="best", fontsize=8)

        ax.set_title(f"{qp.name} — Control |f⟩ population")
        ax.set_xlabel("Amplitude (V)")
        ax.set_ylabel("t_phi_eff (ns)")

    fig.suptitle("SNZ t_phi_eff scan")
    fig.tight_layout()
    return fig
