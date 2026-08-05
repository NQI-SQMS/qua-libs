"""Plotting functions for the SNZ conditional phase measurement.

Generates 2-D heatmaps of the conditional phase difference and
f-state leakage vs (t_phi_eff, amplitude) for each qubit pair.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_snz_conditional_phase(
    ds: xr.Dataset,
    qubit_pairs,
    fit_results=None,
) -> plt.Figure:
    """Plot conditional phase difference and leakage heatmaps.

    For each qubit pair two panels are produced:

    * **Top**: conditional phase difference
      ``(phase[ctrl=0] - phase[ctrl=1]) % 1`` vs (t_phi_eff, amplitude).
      Colormap is ``twilight_shifted`` with range [0, 1] (0.5 = pi).
    * **Bottom**: f-state leakage of the control qubit (with control
      prepared in |e>, averaged over frame) vs (t_phi_eff, amplitude).

    The optimal operating point (from fit_results) is marked with a red
    star on both panels.

    Parameters
    ----------
    ds : xr.Dataset
        Fitted dataset containing ``phase_diff`` and optionally
        ``f_state_control`` data variables.
    qubit_pairs : list
        Qubit-pair objects.
    fit_results : dict, optional
        Mapping of qubit-pair name to fit-result dict (or FitResults).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        2 * n_pairs,
        1,
        figsize=(10, 10 * n_pairs),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        qp_name = qp.name
        qp_ds = ds.sel(qubit_pair=qp_name)

        fr = None
        if fit_results is not None:
            fr = fit_results.get(qp_name, None)

        def _get_fr(field):
            if fr is None:
                return None
            return getattr(fr, field, None) if hasattr(fr, field) else fr.get(field, None)

        opt_amp_rel = _get_fr("optimal_amplitude")
        opt_tpe = _get_fr("optimal_t_phi_eff")

        opt_amp_abs = None
        if opt_amp_rel is not None and "amp_full" in qp_ds.coords:
            amp_full = qp_ds.amp_full.values
            amp_rel = qp_ds.amplitude.values
            opt_amp_abs = float(np.interp(opt_amp_rel, amp_rel, amp_full))

        # --- Phase difference panel ---
        ax_phase = axes[2 * i]
        if "phase_diff" in qp_ds.data_vars:
            phase = qp_ds.phase_diff
            amp_coord = qp_ds.amp_full if "amp_full" in qp_ds.coords else qp_ds.amplitude
            tpe_coord = qp_ds.t_phi_eff

            X, Y = np.meshgrid(amp_coord.values, tpe_coord.values)
            pcm = ax_phase.pcolormesh(
                X,
                Y,
                phase.transpose("t_phi_eff", "amplitude").values,
                cmap="twilight_shifted",
                shading="auto",
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(pcm, ax=ax_phase, label="Phase diff (2\u03c0 units)")
            ax_phase.set_xlabel("Amplitude (V)")
            ax_phase.set_ylabel("t_phi_eff (ns)")
            ax_phase.set_title(f"{qp_name} \u2014 Conditional phase")

            if opt_amp_abs is not None and opt_tpe is not None:
                ax_phase.plot(opt_amp_abs, opt_tpe, marker="*", color="red", markersize=14)
        else:
            ax_phase.set_title(f"{qp_name} \u2014 No phase_diff data")

        # --- Leakage panel ---
        ax_leak = axes[2 * i + 1]
        if "f_state_control" in qp_ds.data_vars:
            leak = qp_ds.f_state_control.sel(control_axis=1).mean(dim="frame")
            amp_coord = qp_ds.amp_full if "amp_full" in qp_ds.coords else qp_ds.amplitude
            tpe_coord = qp_ds.t_phi_eff

            X, Y = np.meshgrid(amp_coord.values, tpe_coord.values)
            pcm2 = ax_leak.pcolormesh(
                X,
                Y,
                leak.transpose("t_phi_eff", "amplitude").values,
                cmap="magma",
                shading="auto",
                vmin=0.0,
                vmax=0.2,
            )
            fig.colorbar(pcm2, ax=ax_leak, label="|f\u27e9 population", extend="max")
            ax_leak.set_xlabel("Amplitude (V)")
            ax_leak.set_ylabel("t_phi_eff (ns)")
            ax_leak.set_title(f"{qp_name} \u2014 Control |f\u27e9 leakage")

            if opt_amp_abs is not None and opt_tpe is not None:
                ax_leak.plot(opt_amp_abs, opt_tpe, marker="*", color="red", markersize=14)
        else:
            ax_leak.set_title(f"{qp_name} \u2014 No leakage data")

    fig.suptitle("SNZ conditional phase scan")
    fig.tight_layout()
    return fig
