from typing import List

import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_distances_with_fit(ds: xr.Dataset, qubits: List, ds_fit: xr.Dataset) -> Figure:
    """Plot GEF discrimination fidelity vs readout amplitude with the optimum marked."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        fit_q = ds_fit.sel(qubit=q_name)

        amps = fit_q.amp_prefactor.values
        ax.plot(amps, 100 * fit_q.fidelity_gef.values, color="C0", lw=2, label="GEF fidelity")

        opt = float(fit_q.optimal_amp_factor.values)
        opt_amp_mv = 1e3 * float(fit_q.optimal_amp.values)
        opt_fid = 100 * float(fit_q.max_fidelity.values)
        ax.axvline(opt, color="r", ls="--", lw=1.5,
                   label=f"opt ×{opt:.2f} ({opt_amp_mv:.1f} mV, {opt_fid:.1f}%)")

        ax.set_xlabel("Amplitude scale factor")
        ax.set_ylabel("GEF fidelity [%]")
        ax.set_title(q_name)
        ax.legend(fontsize=7)

    grid.fig.suptitle("GEF readout power optimisation — fidelity")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig
