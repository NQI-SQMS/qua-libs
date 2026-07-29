"""Plotting for resonator spectroscopy at Fock |N⟩ vs vacuum."""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from typing import Optional


def plot_results(
    ds_fock: xr.Dataset,
    ds_vac: xr.Dataset,
    fit_results_fock: dict,
    fit_results_vac: dict,
    fock_level: int,
    mode_name: str,
) -> Figure:
    """Overlay Fock and vacuum resonator spectroscopy traces on the same axes.

    Each qubit gets two panels (amplitude top, phase bottom) showing both the
    Fock |k+1⟩ and vacuum |0⟩ traces with their fitted dip frequencies marked.
    The dispersive shift Δf is annotated in the amplitude panel title.
    """
    k = fock_level
    n_qubits = len(ds_fock.qubit.values)
    has_phase = "phase" in ds_fock and "phase" in ds_vac

    n_rows = 2 if has_phase else 1
    fig, axes = plt.subplots(n_rows, n_qubits, figsize=(7 * n_qubits, 4 * n_rows), squeeze=False)

    for col, q_name in enumerate(ds_fock.qubit.values):
        ax_amp = axes[0][col]
        use_rf = "full_freq" in ds_fock.coords
        xlabel = "RF frequency (GHz)" if use_rf else "Detuning (MHz)"

        x_fock = (ds_fock.full_freq.sel(qubit=q_name).values * 1e-9
                  if use_rf else ds_fock.detuning.values * 1e-6)
        x_vac  = (ds_vac.full_freq.sel(qubit=q_name).values * 1e-9
                  if ("full_freq" in ds_vac.coords)
                  else ds_vac.detuning.values * 1e-6)

        y_fock = ds_fock.IQ_abs.sel(qubit=q_name).values
        y_vac  = ds_vac.IQ_abs.sel(qubit=q_name).values

        ax_amp.plot(x_fock, y_fock, ".", ms=3, color="C0", label=f"Fock |{k + 1}⟩")
        ax_amp.plot(x_vac,  y_vac,  ".", ms=3, color="C1", label="Vacuum |0⟩")

        freq_fock = _get_freq(fit_results_fock.get(q_name, {}))
        freq_vac  = _get_freq(fit_results_vac.get(q_name, {}))

        if freq_fock is not None:
            fp = freq_fock * 1e-9 if use_rf else 0.0
            ax_amp.axvline(fp, color="C0", ls="--", lw=1.2,
                           label=f"|{k+1}⟩: {freq_fock * 1e-9:.6f} GHz")

        if freq_vac is not None:
            fp = freq_vac * 1e-9 if use_rf else 0.0
            ax_amp.axvline(fp, color="C1", ls="--", lw=1.2,
                           label=f"|0⟩: {freq_vac * 1e-9:.6f} GHz")

        if freq_fock is not None and freq_vac is not None:
            shift_khz = (freq_fock - freq_vac) * 1e-3
            ax_amp.set_title(f"{q_name}  —  Δf = {shift_khz:+.1f} kHz")
        else:
            ax_amp.set_title(q_name)

        ax_amp.set_ylabel("|S21| (a.u.)")
        ax_amp.legend(fontsize=8)

        if has_phase:
            ax_phase = axes[1][col]
            ph_fock = np.degrees(ds_fock.phase.sel(qubit=q_name).values)
            ph_vac  = np.degrees(ds_vac.phase.sel(qubit=q_name).values)
            ax_phase.plot(x_fock, ph_fock, ".", ms=3, color="C0", label=f"Fock |{k + 1}⟩")
            ax_phase.plot(x_vac,  ph_vac,  ".", ms=3, color="C1", label="Vacuum |0⟩")
            ax_phase.set_xlabel(xlabel)
            ax_phase.set_ylabel("Phase (deg)")
            ax_phase.legend(fontsize=8)
        else:
            ax_amp.set_xlabel(xlabel)

    fig.suptitle(
        f"Resonator spectroscopy: Fock |{k + 1}⟩ vs vacuum — {mode_name}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def _get_freq(res) -> Optional[float]:
    """Return the fitted frequency if the fit succeeded, else None."""
    if isinstance(res, dict):
        return float(res["frequency"]) if res.get("success") else None
    return float(res.frequency) if getattr(res, "success", False) else None
