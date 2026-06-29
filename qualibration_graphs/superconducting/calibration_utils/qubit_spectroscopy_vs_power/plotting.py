from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


# ======================================================================
# Public API
# ======================================================================

def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset | None = None,
    signal_source: str = "I_rot",
) -> Figure:
    """
    Plot for each qubit:
      • 2D spectroscopy vs power
      • 1D spectroscopy at selected power

    Args:
        signal_source: "I_rot" (default), "IQ_abs", or "phase" — selects which
                       signal is shown in the colour map and 1D slice.
    """

    # fit_raw_data returns a new Dataset (I_rot, working_signal, selected_power,
    # rough_qubit_frequency are computed there). Merge those variables in.
    if fits is not None:
        for var in ["I_rot", "working_signal", "selected_power", "rough_qubit_frequency"]:
            if var in fits and var not in ds:
                ds = ds.assign({var: fits[var]})

    ds = _ensure_iq_magnitude_mV(ds, signal_source)

    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        _plot_combined_cell(ax, ds, qubit, signal_source)

    signal_label = {"IQ_abs": "IQ_abs", "phase": "phase"}.get(signal_source, "I_rot")
    grid.fig.suptitle(f"Qubit spectroscopy vs power ({signal_label})")
    grid.fig.set_size_inches(15, 10)
    grid.fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.97, hspace=0.40)

    return grid.fig


# ======================================================================
# Combined cell (2D + 1D)
# ======================================================================

def _plot_combined_cell(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    signal_source: str = "I_rot",
):
    """
    Replace a single grid axis with two stacked sub-axes:
      - top: 2D spectroscopy vs power
      - bottom: 1D slice at selected power
    """

    fig = ax.figure
    gs = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[3, 1],
        hspace=0.25,
    )

    ax_2d = fig.add_subplot(gs[0])
    ax_1d = fig.add_subplot(gs[1], sharex=ax_2d)

    ax.remove()

    plot_qubit_spectro_vs_power(ax_2d, ds, qubit, signal_source)
    plot_qubit_spectro_at_selected_power(ax_1d, ds, qubit, signal_source)

    plt.setp(ax_2d.get_xticklabels(), visible=False)


# ======================================================================
# 2D plot
# ======================================================================

def plot_qubit_spectro_vs_power(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    signal_source: str = "I_rot",
):
    ds_q = ds.loc[qubit]

    ds_q = ds_q.assign_coords(freq_GHz=ds_q.full_freq / u.GHz)

    plot_var = {"IQ_abs": "IQ_abs_mV", "phase": "phase"}.get(signal_source, "I_rot_mV")
    ds_q[plot_var].plot(
        ax=ax,
        x="freq_GHz",
        y="power",
        add_colorbar=False,
        robust=True,
    )

    ax.set_ylabel("Drive power [dBm]")

    # Top axis: detuning (via coordinate transform — avoids twiny+tight_layout stretch bug)
    if "detuning" in ds_q and ds_q.detuning.size > 0:
        det0 = float(ds_q.detuning[0])
        ff0 = float(ds_q.full_freq[0])
        rf_freq_GHz = (ff0 - det0) / u.GHz  # constant RF carrier in GHz
        ax_top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda f: (f - rf_freq_GHz) * 1e3,   # GHz → MHz detuning
                lambda d: d / 1e3 + rf_freq_GHz,      # MHz detuning → GHz
            ),
        )
        ax_top.set_xlabel("Detuning [MHz]")

    # Selected power
    if "selected_power" in ds_q:
        p_sel = float(ds_q.selected_power)
        ax.axhline(p_sel, color="red", linewidth=2, zorder=10)

    # Selected frequency
    if "rough_qubit_frequency" in ds_q:
        f_sel = float(ds_q.rough_qubit_frequency) / u.GHz
        ax.axvline(
            f_sel,
            color="red",
            linestyle="--",
            linewidth=2,
            zorder=10,
        )



# ======================================================================
# 1D slice at selected power
# ======================================================================

def plot_qubit_spectro_at_selected_power(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    signal_source: str = "I_rot",
):
    ds_q = ds.loc[qubit]

    if "selected_power" in ds_q and np.isfinite(float(ds_q.selected_power)):
        p_sel = float(ds_q.selected_power)
        label = f"Slice at {p_sel:.1f} dBm"
    else:
        # No optimal power found — fall back to the highest power in the sweep
        p_sel = float(ds_q.power.max())
        label = f"Slice at max power: {p_sel:.1f} dBm"

    ds_slice = ds_q.sel(power=p_sel, method="nearest")

    x = ds_slice.full_freq / u.GHz

    if signal_source == "IQ_abs":
        plot_var, ylabel = "IQ_abs_mV", "IQ_abs [mV]"
    elif signal_source == "phase":
        plot_var, ylabel = "phase", "Phase [rad]"
    else:
        plot_var, ylabel = "I_rot_mV", "I_rot [mV]"

    ax.plot(x, ds_slice[plot_var], linewidth=2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_title(label, fontsize=9)

    if "rough_qubit_frequency" in ds_q:
        f_sel = float(ds_q.rough_qubit_frequency) / u.GHz
        ax.axvline(
            f_sel,
            color="limegreen",
            linestyle="--",
            linewidth=2,
        )


# ======================================================================
# PCA IQ-rotation diagnostic plots
# ======================================================================

def plot_iq_pca_diagnostics(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    subsample: int = 500,
) -> Figure:
    """
    Three-panel diagnostic figure per qubit showing the PCA IQ rotation:

      Panel 1 — Original IQ scatter (subsampled): the raw cloud in the IQ plane.
                The PCA principal axis is drawn as a red arrow.
      Panel 2 — Rotated IQ scatter: after applying the PCA rotation the cloud
                should be elongated along the horizontal (I_rot) axis.
      Panel 3 — Heatmap of I_rot vs (power, detuning): the 2D spectroscopy map
                in the rotated quadrature.  The detected qubit frequency is
                marked with a dashed red line.

    Args:
        ds        : Dataset returned by fit_raw_data (must contain I, Q, I_rot,
                    Q_rot, iw_angle, pca_variance_ratio, rough_qubit_frequency).
        qubits    : List of qubit objects (for grid layout and names).
        subsample : Maximum number of scatter points to plot (random subsample).
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        n_qubits, 3,
        figsize=(14, 4.5 * n_qubits),
        squeeze=False,
    )
    fig.suptitle("PCA IQ rotation diagnostics", fontsize=13)

    rng = np.random.default_rng(0)

    for row, q in enumerate(qubits):
        q_name = q.name
        ds_q = ds.sel(qubit=q_name)

        I_flat  = ds_q.I.values.ravel()
        Q_flat  = ds_q.Q.values.ravel()
        valid   = np.isfinite(I_flat) & np.isfinite(Q_flat)
        I_v, Q_v = I_flat[valid] * 1e3, Q_flat[valid] * 1e3   # → mV

        # Subsample for scatter speed
        n = len(I_v)
        idx = rng.choice(n, size=min(subsample, n), replace=False)

        angle_deg = float(ds_q.iw_angle.values) * 180 / np.pi
        var_ratio = float(ds_q.pca_variance_ratio.values) if "pca_variance_ratio" in ds_q else float("nan")

        # ── Panel 1: original IQ scatter + PCA axis ──────────────────────────
        ax1 = axes[row, 0]
        ax1.scatter(I_v[idx], Q_v[idx], s=4, alpha=0.4, color="steelblue", rasterized=True)
        # Draw principal axis arrow
        i_scale = (I_v.max() - I_v.min()) * 0.4
        angle_rad = float(ds_q.iw_angle.values)
        ax1.annotate(
            "", xy=(i_scale * np.cos(angle_rad), i_scale * np.sin(angle_rad)),
            xytext=(-i_scale * np.cos(angle_rad), -i_scale * np.sin(angle_rad)),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )
        ax1.set_xlabel("I [mV]")
        ax1.set_ylabel("Q [mV]")
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.set_title(
            f"{q_name} — original IQ  (θ = {angle_deg:.1f}°, VR = {var_ratio:.2f})",
            fontsize=9,
        )

        # ── Panel 2: rotated IQ scatter ──────────────────────────────────────
        ax2 = axes[row, 1]
        if "I_rot" in ds_q and "Q_rot" in ds_q:
            I_rot_flat = ds_q.I_rot.values.ravel()[valid] * 1e3
            Q_rot_flat = ds_q.Q_rot.values.ravel()[valid] * 1e3
            ax2.scatter(I_rot_flat[idx], Q_rot_flat[idx], s=4, alpha=0.4,
                        color="darkorange", rasterized=True)
        ax2.axvline(0, color="gray", lw=0.8, linestyle=":")
        ax2.axhline(0, color="gray", lw=0.8, linestyle=":")
        ax2.set_xlabel("I_rot [mV]")
        ax2.set_ylabel("Q_rot [mV]")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.set_title(f"{q_name} — rotated IQ  (elongated along I_rot)", fontsize=9)

        # ── Panel 3: I_rot heatmap vs (power, detuning) ──────────────────────
        ax3 = axes[row, 2]
        if "I_rot" in ds_q:
            freq_ghz  = ds_q.full_freq.values / u.GHz
            power_vals = ds_q.power.values
            i_rot_2d  = ds_q.I_rot.transpose("power", "detuning").values * 1e3  # mV
            im = ax3.pcolormesh(
                freq_ghz, power_vals, i_rot_2d,
                cmap="RdBu_r",
                vmin=np.nanpercentile(i_rot_2d, 2),
                vmax=np.nanpercentile(i_rot_2d, 98),
                shading="auto",
            )
            plt.colorbar(im, ax=ax3, label="I_rot [mV]")
            if "rough_qubit_frequency" in ds_q:
                f_q = float(ds_q.rough_qubit_frequency) / u.GHz
                ax3.axvline(f_q, color="red", linestyle="--", linewidth=1.5,
                            label=f"f = {f_q:.4f} GHz")
                ax3.legend(fontsize=8, loc="upper right")
        ax3.set_xlabel("RF frequency [GHz]")
        ax3.set_ylabel("Drive power [dBm]")
        ax3.set_title(f"{q_name} — I_rot heatmap", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ======================================================================
# Gradient-score diagnostic plots
# ======================================================================

def plot_gradient_score_diagnostics(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    signal_source: str = "I_rot",
) -> Figure:
    """
    Four-panel diagnostic figure per qubit showing the multi-detector
    frequency detection:

      Panel 1 — Smoothed, offset-subtracted heatmap (data_norm): highlights
                vertical features after Gaussian smoothing and row-mean removal.
                The detected frequency is marked with a white dashed line.
      Panel 2 — Ridge score vs frequency: response of the vertical-ridge
                convolution kernel; peaks where the signal column stands above
                its immediate frequency neighbours across all power levels.
      Panel 3 — Persistence score vs frequency: number of power rows that show
                a gradient magnitude above 1σ at each frequency.  High values
                mean the feature is present at many drive powers (not a fluke).
      Panel 4 — Combined score vs frequency (0.5×ridge + 0.3×persistence +
                0.2×variance) with a red dashed line at the detected frequency.
                The detected frequency is printed to stdout.
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        n_qubits, 4,
        figsize=(20, 4 * n_qubits),
        squeeze=False,
    )
    fig.suptitle("Multi-detector qubit frequency detection", fontsize=13)

    for row, q in enumerate(qubits):
        q_name = q.name
        ds_q = ds.sel(qubit=q_name)

        freq_ghz  = ds_q.full_freq.values / u.GHz   # (n_freq,)
        power_vals = ds_q.power.values               # (n_power,)

        # Detected frequency (shared across panels)
        f_det = float(ds_q.gradient_frequency) / u.GHz if "gradient_frequency" in ds_q else None

        # ── Panel 1: smoothed, offset-subtracted heatmap ─────────────────────
        ax1 = axes[row, 0]
        if "data_norm" in ds_q:
            data_norm = ds_q.data_norm.transpose("power", "detuning").values
            vmax = np.nanpercentile(np.abs(data_norm), 98)
            im = ax1.pcolormesh(
                freq_ghz, power_vals, data_norm,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
            )
            plt.colorbar(im, ax=ax1, label="Norm. signal [a.u.]")
        else:
            ax1.text(0.5, 0.5, "data_norm not available",
                     ha="center", va="center", transform=ax1.transAxes)
        if f_det is not None:
            ax1.axvline(f_det, color="white", linestyle="--", linewidth=1.5,
                        label=f"f = {f_det:.4f} GHz")
            ax1.legend(fontsize=8, loc="upper right")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("Drive power [dBm]")
        ax1.set_title(f"{q_name} — smoothed map", fontsize=10)

        # ── Panel 2: ridge score ──────────────────────────────────────────────
        ax2 = axes[row, 1]
        if "ridge_score" in ds_q:
            s = ds_q.ridge_score.values
            ax2.plot(freq_ghz, s, linewidth=2, color="steelblue")
            ax2.fill_between(freq_ghz, s, alpha=0.25, color="steelblue")
        else:
            ax2.text(0.5, 0.5, "ridge_score not available",
                     ha="center", va="center", transform=ax2.transAxes)
        if f_det is not None:
            ax2.axvline(f_det, color="red", linestyle="--", linewidth=1.5)
        ax2.set_xlabel("RF frequency [GHz]")
        ax2.set_ylabel("Score [a.u.]")
        ax2.set_title(f"{q_name} — ridge score", fontsize=10)

        # ── Panel 3: persistence score ────────────────────────────────────────
        ax3 = axes[row, 2]
        if "persistence_score" in ds_q:
            s = ds_q.persistence_score.values
            ax3.plot(freq_ghz, s, linewidth=2, color="darkorange")
            ax3.fill_between(freq_ghz, s, alpha=0.25, color="darkorange")
        else:
            ax3.text(0.5, 0.5, "persistence_score not available",
                     ha="center", va="center", transform=ax3.transAxes)
        if f_det is not None:
            ax3.axvline(f_det, color="red", linestyle="--", linewidth=1.5)
        ax3.set_xlabel("RF frequency [GHz]")
        ax3.set_ylabel("Score [a.u.]")
        ax3.set_title(f"{q_name} — persistence score", fontsize=10)

        # ── Panel 4: combined score ───────────────────────────────────────────
        ax4 = axes[row, 3]
        if "gradient_score" in ds_q:
            s = ds_q.gradient_score.values
            ax4.plot(freq_ghz, s, linewidth=2, color="forestgreen")
            ax4.fill_between(freq_ghz, s, alpha=0.25, color="forestgreen")
        else:
            ax4.text(0.5, 0.5, "gradient_score not available",
                     ha="center", va="center", transform=ax4.transAxes)
        if f_det is not None:
            ax4.axvline(f_det, color="red", linestyle="--", linewidth=2,
                        label=f"f = {f_det:.4f} GHz")
            ax4.legend(fontsize=8)
            print(f"[{q_name}] Detected qubit frequency: {f_det:.6f} GHz")
        ax4.set_xlabel("RF frequency [GHz]")
        ax4.set_ylabel("Score [a.u.]")
        ax4.set_title(f"{q_name} — combined score (0.4·ridge + 0.4·persist + 0.2·var)", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ======================================================================
# Power-broadening fit diagnostic
# ======================================================================

# ======================================================================
# Utilities
# ======================================================================

def _ensure_iq_magnitude_mV(ds: xr.Dataset, signal_source: str = "I_rot") -> xr.Dataset:
    if "IQ_abs_mV" not in ds:
        ds["IQ_abs_mV"] = 1e3 * ds.IQ_abs
    if "I_rot_mV" not in ds and "I_rot" in ds:
        ds["I_rot_mV"] = 1e3 * ds.I_rot
    return ds
