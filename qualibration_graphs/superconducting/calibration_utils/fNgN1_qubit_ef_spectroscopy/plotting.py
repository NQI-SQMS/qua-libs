"""Plotting utilities for qubit ef spectroscopy at Fock |N⟩."""
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit_results: dict,
    rf_center_hz: float,
    fock_level: int,
    mode_name: str,
) -> Figure:
    """Plot ef spectroscopy at Fock N with Lorentzian fit overlay."""
    k = fock_level
    signal_name = "state" if "state" in ds.data_vars else "I"
    signal = getattr(ds, signal_name, ds.I)

    n_qubits = len(ds.qubit.values)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    for ax, q_name in zip(axes[0], ds.qubit.values):
        x = ds.detuning.values * 1e-6
        y = signal.sel(qubit=q_name).values
        ax.plot(x, y, ".", ms=3, label="data")
        if "fit" in ds.data_vars:
            ax.plot(x, ds.fit.sel(qubit=q_name).values, "-", lw=1.5, label="fit")
        res = fit_results.get(q_name, {})
        if res.get("success"):
            ax.axvline(
                res["frequency_hz"] * 1e-6,
                color="r",
                ls="--",
                lw=1,
                label=f"Δf={res['frequency_hz'] * 1e-3:.0f} kHz",
            )
        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("State")
        ax.set_title(q_name)
        ax.legend(fontsize=8)
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda d, c=rf_center_hz: (c + d * 1e6) * 1e-9,
                lambda f, c=rf_center_hz: (f * 1e9 - c) * 1e-6,
            ),
        )
        secax.set_xlabel("RF frequency (GHz)")
    fig.suptitle(f"ef spectroscopy at Fock {k} — {mode_name}")
    fig.tight_layout()
    return fig
