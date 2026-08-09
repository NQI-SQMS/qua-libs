"""Plotting utilities for cryoscope experiment visualizations."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def _qubit_names(qubits) -> List[str]:
    """Return a list of qubit name strings regardless of input type."""
    if hasattr(qubits, "get_names"):
        return qubits.get_names()
    return [q.name if hasattr(q, "name") else str(q) for q in qubits]


def plot_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: dict):
    """Plot cryoscope flux response with exponential decay fits for each qubit.

    Each qubit gets its own figure showing the measured flux response alongside
    the fitted sum-of-exponentials curve on both linear and log scales.

    The function expects a per-qubit ``fit_results`` dictionary (as returned by
    ``fit_raw_data`` and stored in ``node.results["fit_results"]``).  Pattern
    follows ``pi_flux/plotting.py``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a ``flux`` variable with dimensions
        ``(qubit, time)``.  Use ``ds.flux.sel(qubit=name)`` to extract
        each qubit's 1-D flux response.
    qubits : list of AnyTransmon
        Qubit objects to plot.
    fit_results : dict
        Dictionary mapping qubit name to its fit result (either a
        ``FitParameters`` dataclass or a plain dict with keys
        ``"components"`` and ``"a_dc"``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The last generated figure (one figure is created per qubit).
    """
    fig = None
    for q in qubits:
        t_data = ds.time.values
        y_data = ds.flux.sel(qubit=q.name).values
        if np.all(np.isnan(y_data)):
            continue

        # Retrieve per-qubit fit parameters from the results dictionary.
        # Supports both FitParameters dataclass and plain dict formats.
        q_fit = fit_results.get(q.name, {})
        if hasattr(q_fit, "components"):
            # FitParameters dataclass
            components = q_fit.components if q_fit.components is not None else []
            a_dc = getattr(q_fit, "a_dc", np.nan)
        else:
            # Plain dict (e.g. from asdict(FitParameters))
            components = q_fit.get("components", [])
            a_dc = q_fit.get("a_dc", np.nan)

        # Guard against NaN or None DC term for formatting & model building
        if a_dc is None or (isinstance(a_dc, (float, np.floating)) and np.isnan(a_dc)):
            # If we can't determine DC term, approximate from tail of data
            a_dc = float(y_data[-5:].mean()) if len(y_data) >= 5 else float(y_data.mean())

        fig, _ = plot_individual_fit(t_data, y_data, components=components, a_dc=a_dc)

    return fig


def plot_individual_fit(t_data: np.ndarray, y_data: np.ndarray, components: List[Tuple[float, float]], a_dc: float):
    """Plot exponential fit results with both linear and log scales.

    Args:
        t_data (np.ndarray): Time points in nanoseconds
        y_data (np.ndarray): Measured flux response data
        components (List[Tuple[float, float]]): List of (amplitude, tau) pairs for each fitted component
        a_dc (float): Constant term

    Returns:
        tuple: (fig, axs) where:
            - fig: Figure object
            - axs: List of axes objects
    """

    fit_text = f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot - linear scale
    axs[0].plot(t_data, y_data, ".--", label="Data")
    axs[0].plot(t_data, y_fit, label="Fit")
    axs[0].text(
        0.98,
        0.5,
        fit_text,
        transform=axs[0].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
    )
    axs[0].set_xlabel("Time (ns)")
    axs[0].set_ylabel("Flux Response")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Second subplot - log scale
    axs[1].plot(t_data, y_data, ".--", label="Data")
    axs[1].plot(t_data, y_fit, label="Fit")
    axs[1].text(
        0.98,
        0.5,
        fit_text,
        transform=axs[1].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
    )
    axs[1].set_xlabel("Time (ns)")
    axs[1].set_ylabel("Flux Response")
    axs[1].set_xscale("log")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    fig.tight_layout()

    return fig, axs


def plot_unwrapped_phase(ds_fit: xr.Dataset, qubits) -> plt.Figure:
    """Plot unwrapped phase vs time for all qubits on a single figure.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing a ``phase`` variable with dimensions
        ``(qubit, time)``.
    qubits : list
        Qubit objects to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = _qubit_names(qubits)
    fig, ax = plt.subplots(figsize=(8, 4))
    for qname in names:
        ds_fit["phase"].sel(qubit=qname).plot(ax=ax, label=qname, marker=".")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.set_title("Unwrapped phase vs time")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_cryoscope_freq(ds_fit: xr.Dataset, qubits, log_scale: bool = False) -> plt.Figure:
    """Plot cryoscope frequency (GHz) vs time, one panel per qubit.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing a ``freq`` variable with dimensions
        ``(qubit, time)``.
    qubits : list
        Qubit objects to plot.
    log_scale : bool, optional
        Use logarithmic x-axis. Default False.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, qname in zip(axes[0], names):
        ds_fit["freq"].sel(qubit=qname).plot(ax=ax, marker=".", label=qname)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(qname)
        if log_scale:
            ax.set_xscale("log")
            ax.grid(True, which="both")
        else:
            ax.grid(True)
    scale = " [log]" if log_scale else ""
    fig.suptitle(f"Cryoscope frequency vs time{scale}", y=1.02)
    fig.tight_layout()
    return fig


def plot_flux_response(ds_fit: xr.Dataset, qubits, log_scale: bool = False) -> plt.Figure:
    """Plot flux step response (V) vs time, one panel per qubit.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing a ``flux`` variable with dimensions
        ``(qubit, time)``.
    qubits : list
        Qubit objects to plot.
    log_scale : bool, optional
        Use logarithmic x-axis. Default False.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, qname in zip(axes[0], names):
        ds_fit["flux"].sel(qubit=qname).plot(ax=ax, marker=".", label=qname)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Flux (V)")
        ax.set_title(qname)
        if log_scale:
            ax.set_xscale("log")
            ax.grid(True, which="both")
        else:
            ax.grid(True)
    scale = " [log]" if log_scale else ""
    fig.suptitle(f"Flux response vs time{scale}", y=1.02)
    fig.tight_layout()
    return fig


def plot_spectroscopy_curve(ds_fit: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Plot the freq-vs-flux spectroscopy curve embedded in *ds_fit*, if present.

    Returns a Figure when ``ds_fit`` contains ``spec_curve_flux`` and
    ``spec_curve_freq`` variables (written by analysis when
    ``use_spectroscopy_data=True``), or ``None`` otherwise.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset, optionally containing ``spec_curve_flux`` and
        ``spec_curve_freq`` variables.
    qubits : list
        Qubit objects to plot.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if "spec_curve_flux" not in ds_fit or "spec_curve_freq" not in ds_fit:
        return None

    run_id = ds_fit.attrs.get("spectroscopy_run_id", "?")
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    spec_qubits = ds_fit["spec_curve_flux"].spec_qubit.values.tolist()
    for ax, qname in zip(axes[0], names):
        if qname not in spec_qubits:
            ax.set_title(f"{qname} — no curve")
            continue
        flux_arr = ds_fit["spec_curve_flux"].sel(spec_qubit=qname).values
        freq_arr = ds_fit["spec_curve_freq"].sel(spec_qubit=qname).values / 1e9
        ax.plot(flux_arr, freq_arr, lw=1.5)
        ax.set_xlabel("Flux bias (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
        ax.set_title(qname)
        ax.grid(True)
    fig.suptitle(f"Spectroscopy curve used (run #{run_id})")
    fig.tight_layout()
    return fig


def plot_raw_data(ds_raw: xr.Dataset, qubits) -> dict:
    """Plot raw measurement data per qubit: frame slices and 2D heatmap.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with a ``state`` or ``I`` variable and dimensions
        ``(qubit, time, frame)``.
    qubits : list
        Qubit objects to plot.

    Returns
    -------
    dict
        ``{"raw_<qname>": fig, ...}`` — one figure per qubit.
    """
    figures = {}
    data_key = "state" if "state" in ds_raw.data_vars else "I"
    time_vals = ds_raw.time.values

    for q in qubits:
        qname = q.name
        q_data = ds_raw[data_key].sel(qubit=qname)
        sample_idx = [0, len(time_vals) // 4, len(time_vals) // 2, len(time_vals) - 1]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for idx in sample_idx:
            t_sel = time_vals[idx]
            q_data.sel(time=t_sel).plot(ax=axes[0], label=f"t={t_sel} ns")
        axes[0].set_title(f"{qname}: {data_key} vs frame")
        axes[0].legend(fontsize=8)
        axes[0].set_xlabel("Frame")
        axes[0].grid(True)
        frame_vals = q_data.frame.values if "frame" in ds_raw.dims else np.linspace(0, 1, q_data.shape[1])
        im = axes[1].pcolormesh(
            q_data.time.values,
            frame_vals,
            q_data.values.T,
            shading="auto",
            cmap="viridis",
        )
        fig.colorbar(im, ax=axes[1]).set_label(data_key.capitalize())
        axes[1].set_title(f"{qname}: {data_key}(time, frame)")
        axes[1].set_xlabel("Time (ns)")
        axes[1].set_ylabel("Frame")
        fig.suptitle(f"Raw {data_key} — {qname}", y=1.02)
        fig.tight_layout()
        figures[f"raw_{qname}"] = fig

    return figures


def plot_phase_freq_flux(ds_fit: xr.Dataset, qubits) -> plt.Figure:
    """Plot unwrapped phase, cryoscope frequency (MHz), and flux side by side.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset with ``phase``, ``freq``, and ``flux`` variables and
        dimensions ``(qubit, time)``.
    qubits : list
        Qubit objects to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for q in qubits:
        qname = q.name
        ds_fit["phase"].sel(qubit=qname).plot(ax=axes[0], label=qname, marker=".")
        freq_mhz = ds_fit["freq"].sel(qubit=qname).values * 1e3
        axes[1].plot(ds_fit.time.values, freq_mhz, ".-", label=qname)
        ds_fit["flux"].sel(qubit=qname).plot(ax=axes[2], label=qname, marker=".")
    for ax, ylabel, title in zip(
        axes,
        ["Phase (rad)", "Frequency (MHz)", "Flux (V)"],
        ["Unwrapped phase", "Cryoscope frequency", "Flux response"],
    ):
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    return fig


def plot_fir_figures(ds_fit: xr.Dataset, qubits, fir_results: dict) -> dict:
    """Plot all FIR diagnostic figures for each qubit.

    Generates the following per qubit (when FIR analysis succeeded):
      - ``fir_resampled_<qname>``: 1 GS/s vs 2 GS/s comparison.
      - ``fir_fit_diagnostic_<qname>``: forward FIR fit 2×2 (from ``analyze_and_plot_inverse_fir``).
      - ``fir_inverse_diagnostic_<qname>``: inverse FIR 3×2 (from ``analyze_and_plot_inverse_fir``).
      - ``fir_corrected_<qname>``: corrected response validation at 1 GS/s.
      - ``fir_stem_<qname>``: coefficient stem plots (h and h_inv).

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset (used for time axis).
    qubits : list
        Qubit objects to plot.
    fir_results : dict
        Per-qubit FIR results as returned by ``fit_fir_data``.

    Returns
    -------
    dict
        Figure name → ``matplotlib.figure.Figure``.
    """
    figures = {}
    for q in qubits:
        qname = q.name
        res = fir_results.get(qname)
        if res is None or not res.get("success"):
            continue

        t1 = np.array(res["time_1gs"])
        t2 = np.array(res["time_2gs"])

        # Resampled flux: 1 GS/s vs 2 GS/s
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(t1, res["normalized_1gs"], "b.-", label="1 GS/s (original)", alpha=0.6)
        ax4.plot(t2, res["normalized_2gs"], "r.-", ms=3, label="2 GS/s (resampled)", alpha=0.6)
        ax4.axhline(1.0, color="k", ls="--", lw=0.8)
        ax4.set_xlabel("Time (ns)")
        ax4.set_ylabel("Normalized amplitude")
        ax4.set_title(f"Normalized flux response — {qname}")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        figures[f"fir_resampled_{qname}"] = fig4

        if res.get("fig_fir_fit") is not None:
            figures[f"fir_fit_diagnostic_{qname}"] = res["fig_fir_fit"]

        if res.get("fig_fir_inverse") is not None:
            figures[f"fir_inverse_diagnostic_{qname}"] = res["fig_fir_inverse"]

        # Corrected response validation at 1 GS/s
        fig7, ax7 = plt.subplots(figsize=(10, 5))
        ax7.plot(t1, res["normalized_1gs"], label="data (normalized)")
        ax7.plot(t1, res["corrected_1gs"], "--", label="expected corrected response")
        ax7.axhline(1.001, color="k", lw=0.8, ls="--", label="±0.1% tolerance")
        ax7.axhline(0.999, color="k", lw=0.8, ls="--")
        ax7.set_ylim([0.95, 1.05])
        # Noise-floor status badge (always shown so user can confirm tail is settled)
        sigma_disp = res.get("noise_sigma_displayed")
        noise_msg = res.get("noise_estimate_msg")
        if sigma_disp is not None and noise_msg is not None:
            ax7.plot([], [], " ", label=f"noise σ≈{sigma_disp:.1e} [{noise_msg}]")
        ax7.legend()
        ax7.set_xlabel("Time (ns)")
        ax7.set_ylabel("Normalized amplitude")
        ax7.set_title(f"FIR Final Result — {qname}")
        ax7.grid(True, alpha=0.3)
        fig7.tight_layout()
        figures[f"fir_corrected_{qname}"] = fig7

        # FIR coefficient stem plots
        h_fir_arr = np.array(res["forward_fir"])
        h_inv_arr = np.array(res["inverse_fir"])
        fig8, axes8 = plt.subplots(1, 2, figsize=(14, 4))
        axes8[0].stem(np.arange(len(h_fir_arr)), h_fir_arr, linefmt="b-", markerfmt="bo", basefmt="k-")
        axes8[0].set_xlabel("Tap Index")
        axes8[0].set_ylabel("Coefficient")
        axes8[0].set_title(f"Forward FIR h (L={len(h_fir_arr)}) — {qname}")
        axes8[0].grid(True, alpha=0.3)
        axes8[1].stem(np.arange(len(h_inv_arr)), h_inv_arr, linefmt="r-", markerfmt="rs", basefmt="k-")
        axes8[1].set_xlabel("Tap Index")
        axes8[1].set_ylabel("Coefficient")
        axes8[1].set_title(f"Inverse FIR h_inv (M={len(h_inv_arr)}) — {qname}")
        axes8[1].grid(True, alpha=0.3)
        fig8.tight_layout()
        figures[f"fir_stem_{qname}"] = fig8

    return figures
