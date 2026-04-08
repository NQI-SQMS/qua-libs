from typing import List

import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_distances_with_fit(ds: xr.Dataset, qubits: List, ds_fit: xr.Dataset) -> Figure:
    """Plot D_ge, D_ef, D_gf and the worst-case Distance vs readout amplitude, with optimum marked."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        ds_q = ds.sel(qubit=q_name)
        fit_q = ds_fit.sel(qubit=q_name)

        amps = ds_q.amp_prefactor.values
        ax.plot(amps, ds_q.Dge.values, "--", color="C1", lw=1.5, label="D_ge")
        ax.plot(amps, ds_q.Def.values, "--", color="C2", lw=1.5, label="D_ef")
        ax.plot(amps, ds_q.Dgf.values, "--", color="C3", lw=1.5, label="D_gf")
        ax.plot(amps, ds_q.Distance.values, color="C0", lw=2, label="min distance")

        opt = float(fit_q.optimal_amp_factor.values)
        opt_amp_mv = 1e3 * float(fit_q.optimal_amp.values)
        ax.axvline(opt, color="k", ls="--", lw=1.5,
                   label=f"opt ×{opt:.2f} ({opt_amp_mv:.1f} mV)")

        ax.set_xlabel("Amplitude scale factor")
        ax.set_ylabel("IQ distance (V)")
        ax.set_title(q_name)
        ax.legend(fontsize=7)

    grid.fig.suptitle("GEF readout power optimisation")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig
