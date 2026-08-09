"""Plotting module for the JAZZ2-N SNZ amplitude / t_phi_eff scan."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from qualibration_libs.core import BatchableList


def _plot_heatmap(ax, amp_scale, tpe_vals, p_values, *, cmap="magma"):
    """Helper: plot a single (amp_scale, t_phi_eff) heatmap on ``ax``."""
    xg, yg = np.meshgrid(amp_scale, tpe_vals, indexing="xy")
    return ax.pcolormesh(xg, yg, p_values.T, cmap=cmap, shading="auto")


def plot_raw_data_with_fit(ds_fit: xr.Dataset, qubit_pairs: BatchableList) -> plt.Figure:
    """Plot the JAZZ2-N SNZ result per qubit pair.

    For each qubit pair the figure shows

    * Top (full width): the N-averaged map ``<P_|00>>_N(amp, t_phi_eff)`` with
      a star at the fitted optimum and a top secondary x-axis converting the
      amplitude scale to Volts.
    * Bottom (only when more than one N value was swept): a strip of small
      per-N heatmaps ``P_|00>(amp, t_phi_eff)`` for inspection.
    """
    n_pairs = len(qubit_pairs)
    # Detect whether the dataset has a non-trivial N axis.
    has_N = ("N" in ds_fit.dims) and (int(ds_fit.sizes.get("N", 1)) > 1)
    num_N = int(ds_fit.sizes.get("N", 1)) if has_N else 0

    # Each pair occupies two rows when has_N (main + per-N strip), otherwise just one.
    rows_per_pair = 2 if has_N else 1
    cols = max(num_N, 1)
    fig = plt.figure(figsize=(4.5 * cols, 4.0 + (2.5 if has_N else 0)) if has_N else (4.5, 4.0))
    if has_N:
        fig = plt.figure(figsize=(max(4.5, 2.5 * cols), (4.0 + 2.5) * n_pairs))
    else:
        fig = plt.figure(figsize=(4.5 * min(n_pairs, 4), 4.0 * ((n_pairs + 3) // 4)))

    gs = GridSpec(rows_per_pair * n_pairs, cols, figure=fig,
                  height_ratios=[3, 1] * n_pairs if has_N else [1] * n_pairs)

    for i, qp in enumerate(qubit_pairs):
        qp_name = qp.name
        fr = ds_fit.sel(qubit_pair=qp_name)

        amp_scale = fr.amplitude.values
        amp_abs = fr["amp_full"].values if "amp_full" in fr.coords else amp_scale
        tpe_vals = fr.t_phi_eff.values

        # ---- Averaged heatmap (main) ----
        if has_N:
            ax_main = fig.add_subplot(gs[rows_per_pair * i, :])
        else:
            ax_main = fig.add_subplot(gs[i, 0])

        p_avg = fr["p00_avg"].transpose("amplitude", "t_phi_eff").values
        pcm = _plot_heatmap(ax_main, amp_scale, tpe_vals, p_avg)

        opt_amp_scale = float(fr.optimal_amplitude_scale.values)
        opt_tpe = float(fr.optimal_t_phi_eff.values)
        method = str(fr.fit_method.values)
        success = bool(fr.success.values)
        if np.isfinite(opt_amp_scale) and np.isfinite(opt_tpe):
            color = "red" if success else "white"
            ax_main.plot(
                opt_amp_scale,
                opt_tpe,
                marker="*",
                color=color,
                markersize=14,
                label=f"opt = ({opt_amp_scale:.4f}, {opt_tpe:.3f}) [{method}]",
            )

        def amp_scale_to_abs(s, abs_values=amp_abs, scale_values=amp_scale):
            return np.interp(s, scale_values, abs_values)

        def amp_abs_to_scale(a, abs_values=amp_abs, scale_values=amp_scale):
            return np.interp(a, abs_values, scale_values)

        secax = ax_main.secondary_xaxis("top", functions=(amp_scale_to_abs, amp_abs_to_scale))
        secax.set_xlabel("Amplitude (V)")
        title = f"{qp_name}"
        if has_N:
            title = f"{qp_name}  $\\langle P_{{|00\\rangle}}\\rangle_N$  (N \u2208 [{int(fr.N.values[0])}, {int(fr.N.values[-1])}])"
        ax_main.set_title(title)
        ax_main.set_xlabel("Amplitude scale (a.u.)")
        ax_main.set_ylabel(r"$t_{\phi,\mathrm{eff}}$ (ns)")
        ax_main.legend(loc="upper right", fontsize=8)
        cbar = fig.colorbar(pcm, ax=ax_main, shrink=0.85)
        cbar.set_label("$P_{|00\\rangle}$")

        # ---- Per-N strip (only when num_N > 1) ----
        if has_N:
            n_values = fr.N.values
            p_per_N = fr["p"].transpose("N", "amplitude", "t_phi_eff").values
            for j, n_val in enumerate(n_values):
                ax_sub = fig.add_subplot(gs[rows_per_pair * i + 1, j])
                _plot_heatmap(ax_sub, amp_scale, tpe_vals, p_per_N[j])
                if np.isfinite(opt_amp_scale) and np.isfinite(opt_tpe):
                    ax_sub.plot(opt_amp_scale, opt_tpe, marker="*", color="lime", markersize=8)
                ax_sub.set_title(f"N = {int(n_val)}", fontsize=8)
                if j == 0:
                    ax_sub.set_ylabel(r"$t_{\phi,\mathrm{eff}}$ (ns)", fontsize=7)
                else:
                    ax_sub.set_yticklabels([])
                ax_sub.set_xlabel("amp (a.u.)", fontsize=7)
                ax_sub.tick_params(labelsize=7)

    fig.suptitle("JAZZ2-N SNZ amplitude / $t_{\\phi,\\mathrm{eff}}$ scan")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
